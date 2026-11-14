import json
from contextlib import nullcontext
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from box_nodes import canvas, faces, material_pipeline as pipeline, store
from test_nodes import fixture


class MaterialPipelineTests(unittest.TestCase):
    def test_registration_keeps_binary_metalness_and_continuous_roughness_and_ao(self):
        source = np.ones((8, 8, 4), dtype='f4')
        report = {'tileSize': [8, 8], 'registration': {'scale': 1., 'centerShiftYX': [0., .25]}}
        maps = np.zeros((1, 8, 8, 3), dtype='f4')
        maps[:, :, 4:] = 1
        _, continuous = pipeline.register_estimates(maps, maps, source, report)
        _, classified = pipeline.register_estimates(maps, maps, source, report, binary_metalness=True)
        self.assertTrue(np.any((continuous[..., 1] > 0.) & (continuous[..., 1] < 1.)))
        np.testing.assert_array_equal(classified[..., 1], continuous[..., 1] >= .5)
        np.testing.assert_array_equal(classified[..., [0, 2]], continuous[..., [0, 2]])

    def test_keeps_native_material_detail_above_face_image_resolution(self):
        source = np.ones((16, 16, 4), dtype='f4')
        source[:, :3, 3] = 0
        crop = {'version': 1, 'size': 16, 'faceId': '+Z', 'guideRect': [0, 4, 16, 8]}
        report = {'guideCanvas': crop, 'tileSize': [64, 32],
                  'registration': {'scale': 1., 'centerShiftYX': [0., 0.]}}
        stripes = np.tile(np.arange(64, dtype='f4') % 2, (1, 64, 1))
        model = np.repeat(stripes[..., None], 3, axis=-1)
        base, rma = pipeline.register_estimates(model, 1. - model, source, report)
        # Every stripe lies well above the source grid's spatial bandwidth.
        np.testing.assert_allclose(base[:, :, 18:], model[:, 16:48, 18:], atol=1e-6)
        np.testing.assert_allclose(rma[:, :, 18:], 1. - model[:, 16:48, 18:], atol=1e-6)
        np.testing.assert_array_equal(rma[:, :, :8], 0.)

    def test_reprojects_maps_through_the_same_crop_and_registration_as_appearance(self):
        rng = np.random.default_rng(581)
        source = rng.random((32, 32, 4), dtype='f4')
        source[..., 3] = 0
        source[9:20, 6:24, 3] = 1
        crop = {'version': 1, 'size': 32, 'faceId': '+X', 'guideRect': [0, 4, 32, 24]}
        guide = np.zeros((24, 24), 'f4')
        guide[8:20, 8:22] = 1
        aligned, report = faces.align_one(canvas.undo_guide_canvas(source, crop), guide, 0, 1, 1)
        self.assertEqual(report['status'], 'accepted')
        self.assertNotEqual(report['registration']['centerShiftYX'], [0, 0])
        report['guideCanvas'] = crop
        albedo, _ = pipeline.register_estimates(source[None], source[None], source, report)
        linear = np.where(albedo[0] <= .04045, albedo[0] / 12.92, ((albedo[0] + .055) / 1.055) ** 2.4)
        mask = aligned[..., 3] > 0
        np.testing.assert_allclose(linear[mask], aligned[..., :3][mask], atol=1e-6)

    def test_batch_reuses_single_face_and_replacement_invalidates_just_that_face(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(patch.object(store, 'root', return_value=root))
        self.enterContext(patch.dict(sys.modules, torch=SimpleNamespace(from_numpy=lambda value: value)))
        identity = self.enterContext(patch.object(pipeline, 'model_identity', return_value='weights-v1'))
        node = pipeline.GenerateFaceMaterials()
        base = 'a' * 64
        fixture(root / base, base)
        self.assertEqual(node.generate(base, 'keep'), (base,))
        identity.assert_not_called()
        # When the requested edit was rejected, preserve the geometry Box.
        self.assertEqual(node.generate(base, 'marigold', '+X'), (base,))
        self.assertEqual(node.generate(base, 'supermat', '+X'), (base,))
        original = np.zeros((32, 32, 4), dtype='f4')
        original[4:-4, 4:-4] = (.8, .3, .5, 1)
        ref = faces.save_view(original)
        box = faces.compose(base, faces.align(dict.fromkeys(('+X', '-Z'), ref), base, .85, .25, .25))
        infer = Mock(side_effect=lambda image, **kw: (image, image))
        loader = self.enterContext(patch.object(pipeline.material_iid, 'loaded_estimator', create=True,
                                                side_effect=lambda: nullcontext(infer)))
        one, = node.generate(box, 'marigold', '+X')
        self.assertEqual(infer.call_count, 1)
        self.assertEqual(infer.call_args.args[0].shape, (1, 32, 32, 3))
        both, = node.generate(one, 'marigold')
        self.assertEqual(infer.call_count, 2)
        self.assertEqual(node.generate(both, 'marigold'), (both,))
        self.assertEqual(node.generate(both, 'marigold', prompt_preset='short'), (both,))
        self.assertEqual(infer.call_count, 2)
        self.assertEqual(loader.call_count, 2)
        original[..., 0] = .15
        changed = faces.compose(both, faces.align({'+X': faces.save_view(original)}, both, .85, .25, .25))
        parent = store.validate(root / changed, changed)
        self.assertNotIn('materialEdit', parent['faces']['+X'])
        self.assertEqual(parent['faces']['-Z']['materialEdit']['albedo']['provider'], 'marigold')
        final, = node.generate(changed, 'marigold')
        self.assertEqual(infer.call_count, 3)
        manifest = json.loads((root / final / 'manifest.json').read_text())
        self.assertEqual(manifest['maps'], ['roughness', 'metalness'])
        self.assertNotIn('promptPreset', manifest['faces']['+X']['materialEdit']['surface']['recipe'])
        self.assertEqual(manifest['faces']['-Z']['materialEdit'], parent['faces']['-Z']['materialEdit'])

    def test_generated_materials_record_preset_and_reuse_only_matching_results(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(patch.object(store, 'root', return_value=root))
        self.enterContext(patch.dict(sys.modules, torch=SimpleNamespace(from_numpy=lambda value: value)))
        self.enterContext(patch.object(pipeline, 'model_identity', return_value='weights-v1'))
        self.enterContext(patch.object(pipeline, 'load_native', return_value=(None, None, None)))
        infer = self.enterContext(patch.object(pipeline.material_estimate.MaterialEstimate, 'estimate',
                                              side_effect=lambda image, *args, **kwargs: (image, image)))
        node = pipeline.GenerateFaceMaterials()
        base = 'a' * 64
        fixture(root / base, base)
        original = np.zeros((32, 32, 4), dtype='f4')
        original[4:-4, 4:-4] = (.8, .3, .5, 1)
        box = faces.compose(base, faces.align({'+X': faces.save_view(original)}, base, .85, .25, .25))
        for source in ('qwen', 'h3'):
            with self.subTest(source=source):
                infer.reset_mock()
                detailed, = node.generate(box, source, '+X')
                short, = node.generate(box, source, '+X', prompt_preset='short')
                self.assertNotEqual(detailed, short)
                self.assertEqual(node.generate(box, source, '+X', prompt_preset='short'), (short,))
                self.assertEqual([call.kwargs['prompt_preset'] for call in infer.call_args_list], ['detailed', 'short'])
                for asset, preset in ((detailed, 'detailed'), (short, 'short')):
                    manifest = store.validate(root / asset, asset)
                    recipe = manifest['faces']['+X']['materialEdit']['surface']['recipe']
                    self.assertEqual(recipe['promptPreset'], preset)
