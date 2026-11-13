import sys
import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from box_nodes.material_estimate import MaterialEstimate, _qwen_sampling_model


class MaterialEstimateTests(unittest.TestCase):
    def test_each_backend_uses_the_face_first_and_generates_two_distinct_outputs(self):
        rgba = torch.zeros(1, 32, 64, 4)
        guide = torch.ones(1, 32, 64, 3) * .25
        for backend in ('qwen_enhanced', 'h3_enhanced'):
            with self.subTest(backend=backend):
                qwen = backend == 'qwen_enhanced'
                encoded = Mock(return_value=SimpleNamespace(
                    result=('positive', 'negative', {}) if qwen else ('positive', {})))
                native = SimpleNamespace(execute=encoded)
                sample = Mock(return_value=({},))
                albedo = torch.ones(1, 32, 64, 3) * .37
                rma = torch.tensor([.12, .5, .84]).expand_as(albedo)
                decode = Mock(side_effect=[(albedo,), (rma,)])
                modules = {
                    'nodes': SimpleNamespace(
                        common_ksampler=sample, VAEDecode=lambda: SimpleNamespace(decode=decode),
                        ConditioningZeroOut=lambda: SimpleNamespace(zero_out=lambda pos: ('negative',))),
                    'comfy_extras.nodes_qwen': SimpleNamespace(TextEncodeQwenImage21=native),
                    'comfy_extras.nodes_minimax_h3': SimpleNamespace(MiniMaxH3ImageConditioning=native),
                }
                with patch.dict(sys.modules, modules):
                    output = MaterialEstimate().estimate(
                        rgba, 'model', 'clip', 'vae', backend, (1 << 64) - 1, guide=guide, resolution=0)
                self.assertTrue(torch.equal(output[0], albedo))
                self.assertTrue(torch.equal(output[1], rma))
                color_prompt, data_prompt = [call.args[1] for call in encoded.call_args_list]
                self.assertIn('sRGB base-color', color_prompt)
                self.assertIn('physically based renderer', color_prompt)
                self.assertIn('Remove cast shadows', color_prompt)
                self.assertIn('linear data texture', data_prompt)
                self.assertIn('roughness in red, metalness in green, and ambient occlusion in blue', data_prompt)
                self.assertIn('0 for dielectrics', data_prompt)
                self.assertIn('1 on open surfaces', data_prompt)
                refs = encoded.call_args.kwargs['images']
                self.assertTrue(torch.all(refs['image_1'] == 1))
                self.assertTrue(torch.equal(refs['image_2'], guide))
                if qwen:
                    self.assertEqual(encoded.call_args.kwargs['resolution'], 0)
                else:
                    self.assertEqual(encoded.call_args.args[2:4], (64, 32))
                self.assertEqual(sample.call_args_list[1].args[1:4], (0, 24, 3.) if qwen else (0, 6, 1.))

    def test_batch_cannot_silently_drop_a_face(self):
        with self.assertRaisesRegex(ValueError, 'one RGB or RGBA'):
            MaterialEstimate().estimate(torch.zeros(2, 32, 32, 3), None, None, None, 'qwen_enhanced', 0)

    def test_qwen_2k_chunks_only_independent_rows_on_the_cloned_model(self):
        class Denoiser:
            transformer_blocks = [object()]

        original = Mock()
        original.load_device = torch.device('cuda')
        original.model_options = {'transformer_options': {'qwen_image21_cache': {'dtype': 'default'}}}
        patched = original.clone.return_value
        patched.model_options = deepcopy(original.model_options)
        forward = Mock(side_effect=lambda x: x.square())
        original.get_model_object.side_effect = [Denoiser(), forward]
        memory = Mock()
        memory.get_total_memory.return_value = 16 * 1024 ** 3
        loaders = (SimpleNamespace(patcher='clip'), SimpleNamespace(patcher='vae'))
        modules = {
            'comfy': SimpleNamespace(model_management=memory),
            'comfy.ldm.qwen_image21.model': SimpleNamespace(QwenImage21Transformer2DModel=Denoiser),
        }
        with patch.dict(sys.modules, modules):
            self.assertIs(_qwen_sampling_model(original, 256, 256, *loaders), original)
            self.assertIs(_qwen_sampling_model(original, 2048, 2048, *loaders), patched)
        name, run = patched.add_object_patch.call_args.args
        self.assertEqual(name, 'diffusion_model.transformer_blocks.0.img_mlp.forward')
        values = torch.arange(2053., dtype=torch.float32).view(1, -1, 1)
        self.assertTrue(torch.equal(run(values), values.square()))
        self.assertEqual([call.args[0].shape[1] for call in forward.call_args_list], [1024, 1024, 5])
        self.assertEqual(original.model_options['transformer_options']['qwen_image21_cache'], {'dtype': 'default'})
        self.assertEqual(patched.model_options['transformer_options']['qwen_image21_cache']['device'], 'cpu')
        self.assertEqual([c.args for c in memory.unload_model_and_clones.call_args_list], [('clip',), ('vae',)])
