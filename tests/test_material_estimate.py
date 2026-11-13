import sys
import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from box_nodes.material_estimate import MaterialEstimate, _qwen_sampling_model


class MaterialEstimateTests(unittest.TestCase):
    def test_each_backend_estimates_separate_maps_and_packs_scalar_values(self):
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
                roughness = torch.tensor([.125, .25, .375]).expand_as(albedo)
                metalness = torch.zeros_like(albedo)
                metalness[:, :, 1, :] = .499
                metalness[:, :, 2, :] = .5
                metalness[:, :, 3, :] = 1
                ao = torch.full_like(albedo, .875)
                decode = Mock(side_effect=[(image,) for image in (albedo, roughness, metalness, ao)])
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
                rma = output[1]
                self.assertEqual(rma.shape, albedo.shape)
                self.assertTrue(torch.all(rma[..., 0] == .25))
                expected_metal = torch.zeros_like(rma[..., 1])
                expected_metal[:, :, 2:4] = 1
                self.assertTrue(torch.equal(rma[..., 1], expected_metal))
                self.assertTrue(torch.all(rma[..., 2] == .875))
                color_prompt, roughness_prompt, metalness_prompt, ao_prompt = [
                    call.args[1] for call in encoded.call_args_list]
                self.assertIn('sRGB base-color', color_prompt)
                self.assertIn('physically based renderer', color_prompt)
                self.assertIn('Remove cast shadows', color_prompt)
                self.assertIn('grayscale roughness map', roughness_prompt)
                self.assertIn('binary black-and-white metalness mask', metalness_prompt)
                self.assertIn('grayscale ambient occlusion map', ao_prompt)
                for call in encoded.call_args_list:
                    refs = call.kwargs['images']
                    self.assertTrue(torch.all(refs['image_1'] == 1))
                    self.assertTrue(torch.equal(refs['image_2'], guide))
                    if qwen:
                        self.assertEqual(call.kwargs['resolution'], 0)
                    else:
                        self.assertEqual(call.args[2:4], (64, 32))
                expected_sampling = (24, 3.) if qwen else (6, 1.)
                self.assertEqual([call.args[1:4] for call in sample.call_args_list],
                                 [(seed, *expected_sampling) for seed in ((1 << 64) - 1, 0, 1, 2)])

                # Verify the full no-guide prompts and pass seeds used in the short-prompt trial.
                decode.side_effect = [(image,) for image in (albedo, roughness, metalness, ao)]
                encoded.reset_mock()
                sample.reset_mock()
                with patch.dict(sys.modules, modules):
                    MaterialEstimate().estimate(rgba, 'model', 'clip', 'vae', backend, 0,
                                                resolution=0, prompt_preset='short')
                source, reference = ('<image1>', '<image2>') if qwen else ('<Picture 1>', '<Picture 2>')
                prefix = f'Edit {source}. Match its camera, framing, silhouette and position of every feature exactly. '
                guide_prefix = f'Use {reference} only as a geometry guide for the same view. '
                self.assertEqual([call.args[1] for call in encoded.call_args_list], [
                    color_prompt.replace(guide_prefix, ''),
                    prefix + 'Create a roughness map. Fill the image white. Only make glossy, perfectly smooth material '
                             'parts darker, using a gray gradient. Matte materials remain white.',
                    prefix + 'Black image. Paint ONLY the exposed metal parts white. All other parts black. No gray.',
                    ao_prompt.replace(guide_prefix, ''),
                ])
                self.assertEqual([call.args[1:4] for call in sample.call_args_list],
                                 [(seed, *expected_sampling) for seed in range(4)])

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
