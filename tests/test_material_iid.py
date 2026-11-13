import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from box_nodes import material_iid as iid


class MaterialIIDTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        folder_paths = SimpleNamespace(get_folder_paths=lambda name: [str(self.root / 'first'), str(self.root)],
                                       get_output_directory=lambda: str(self.root / 'output'))
        self.mm = Mock()
        self.mm.get_torch_device.return_value = torch.device('cpu')
        self.mm.should_use_fp16.return_value = False
        self.pipeline = Mock()
        self.pipeline.target_properties = {
            'target_names': ['albedo', 'material'], 'albedo': {'prediction_space': 'srgb'},
            'material': {'sub_target_names': ['roughness', 'metallicity', None]},
            'roughness': {'prediction_space': 'linear'}, 'metallicity': {'prediction_space': 'linear'},
        }
        self.loader = Mock()
        self.loader.from_pretrained.return_value = self.pipeline
        modules = {
            'folder_paths': folder_paths, 'comfy': SimpleNamespace(model_management=self.mm),
            'diffusers': SimpleNamespace(MarigoldIntrinsicsPipeline=self.loader),
        }
        self.enterContext(patch.dict(sys.modules, modules))
        self.checkpoint = self.root / iid.CHECKPOINT
        self.checkpoint.mkdir()
        (self.checkpoint / 'model_index.json').write_text('{}')

    def test_linear_data_layout_face_retries_and_offline_load(self):
        seed = 42
        def predict(image, num_inference_steps, generator, match_input_resolution, output_type):
            self.assertEqual((num_inference_steps, generator.initial_seed(), match_input_resolution, output_type),
                             (3, seed, True, 'pt'))
            _, _, height, width = image.shape
            material = torch.tensor([0.04, 0.73, 0.16]).reshape(1, 3, 1, 1).expand(1, 3, height, width)
            return SimpleNamespace(prediction=torch.cat((image, material)))

        self.pipeline.side_effect = predict
        source = torch.rand(2, 11, 13, 4)
        albedo, rma = iid.EstimateFaceMaterialIID().estimate(source, seed, 3)
        expected_rgb = torch.lerp(torch.full_like(source[..., :3], 0.5), source[..., :3], source[..., 3:])
        torch.testing.assert_close(albedo, expected_rgb)
        torch.testing.assert_close(rma, torch.tensor([0.04, 0.73, 1.]).expand_as(albedo))
        args, kwargs = self.loader.from_pretrained.call_args
        self.assertEqual(args, (str(self.checkpoint),))
        self.assertTrue(kwargs['local_files_only'])
        self.assertTrue(kwargs['use_safetensors'])
        self.assertEqual(self.mm.throw_exception_if_processing_interrupted.call_count, 2)
        self.mm.soft_empty_cache.assert_called_once()

    def test_loaded_pipeline_serves_separate_face_calls_once(self):
        self.pipeline.return_value = SimpleNamespace(prediction=torch.zeros(2, 3, 8, 8))
        with iid.loaded_estimator() as estimate:
            for _ in range(2):
                estimate(torch.zeros(1, 8, 8, 3), seed=1)
            self.mm.soft_empty_cache.assert_not_called()
        self.loader.from_pretrained.assert_called_once()
        self.assertEqual(self.pipeline.call_count, 2)
        self.mm.soft_empty_cache.assert_called_once()

    def test_incompatible_checkpoint_releases_pipeline(self):
        self.pipeline.target_properties = {'target_names': ['albedo', 'shading', 'residual']}
        with self.assertRaisesRegex(ValueError, 'albedo and linear material targets'):
            iid.EstimateFaceMaterialIID().estimate(torch.zeros((1, 8, 8, 3)))
        self.pipeline.assert_not_called()
        self.mm.soft_empty_cache.assert_called_once()

    def test_weights_change_and_missing_checkpoint_do_not_reuse_cache(self):
        before = iid.checkpoint_identity()
        weights = self.checkpoint / 'unet' / 'diffusion_pytorch_model.safetensors'
        weights.parent.mkdir()
        weights.write_bytes(b'v1')
        after = iid.checkpoint_identity()
        self.assertNotEqual(before, after)
        weights.write_bytes(b'v2 replacement')
        self.assertNotEqual(iid.checkpoint_identity(), after)
        (self.checkpoint / 'model_index.json').unlink()
        self.assertNotEqual(iid.EstimateFaceMaterialIID.IS_CHANGED(None, 0, 4), after)
        with self.assertRaisesRegex(FileNotFoundError, 'models/diffusers/'):
            iid.EstimateFaceMaterialIID().estimate(torch.zeros(1, 8, 8, 3))
        self.loader.from_pretrained.assert_not_called()
