import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from box_nodes import material_iid, material_pipeline as pipeline


class MaterialPreflightTests(unittest.TestCase):
    def test_native_mixed_providers_use_their_own_model_files(self):
        h3 = pipeline.NATIVE_MODELS['h3']
        paths = SimpleNamespace(get_full_path=lambda folder, name: '/some/model' if name in h3 else None)
        with patch.dict(sys.modules, {'folder_paths': paths}):
            self.assertTrue(pipeline.material_availability('h3')['available'])
            expected = [f'{group}/{name}' for group, name in zip(pipeline.MODEL_FOLDERS, pipeline.NATIVE_MODELS['qwen'])]
            self.assertEqual(pipeline.material_availability('qwen')['missing'], expected)
        self.assertTrue(pipeline.material_availability('keep')['available'])
        with self.assertRaises(ValueError):
            pipeline.material_availability('unknown')

    def test_iid_missing_package_and_partial_fp16_snapshot(self):
        with tempfile.TemporaryDirectory() as location:
            root = Path(location)
            model = root / material_iid.CHECKPOINT
            paths = SimpleNamespace(get_folder_paths=lambda group: [root])
            with patch.dict(sys.modules, {'folder_paths': paths}), patch.object(pipeline, 'find_spec', return_value=True):
                self.assertIn('model_index.json', pipeline.material_availability('marigold')['missing'][0])
                (model / 'unet').mkdir(parents=True)
                (model / 'model_index.json').write_text('{}')
                (model / 'unet/diffusion_pytorch_model.fp16.safetensors').touch()
                missing = pipeline.material_availability('marigold')['missing']
                self.assertTrue(any(file.endswith('vae/diffusion_pytorch_model.fp16.safetensors') for file in missing))
                for file in missing:
                    target = root / Path(file).relative_to('diffusers')
                    target.parent.mkdir(exist_ok=True, parents=True)
                    target.touch()
                self.assertTrue(pipeline.material_availability('marigold')['available'])
                with patch.object(pipeline, 'find_spec', side_effect=lambda dep: None if dep == 'diffusers' else True):
                    self.assertEqual(pipeline.material_availability('marigold')['missing'], ['Python package diffusers'])
