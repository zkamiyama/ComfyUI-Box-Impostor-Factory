import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from box_nodes.material_estimate import MaterialEstimate


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
