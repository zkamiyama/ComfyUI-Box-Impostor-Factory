import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from box_nodes import faces, nodes, store
from box_nodes.silhouette import InspectFaceSilhouette
from test_nodes import fixture


class SilhouetteTests(unittest.TestCase):
    def setUp(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(patch.object(store, 'root', return_value=root))
        self.enterContext(patch.object(nodes, 'tensor', side_effect=lambda a: a))
        self.asset = 'c' * 64
        self.directory = root / self.asset
        self.manifest = fixture(self.directory, self.asset)
        self.inspect = InspectFaceSilhouette().inspect

    def edited(self, data, provenance=None, face_id='+X'):
        source = {**faces.save_view(data), 'provenance': provenance or {}}
        aligned = faces.align({face_id: source}, self.asset, .95, 0., 0.)
        return faces.compose(self.asset, aligned)

    def test_baseline_has_no_comparison_and_does_not_change_saved_asset(self):
        before = (self.directory / 'manifest.json').read_bytes()
        overlay, extra, missing, raw = self.inspect(self.asset, '+X')
        qa = json.loads(raw)
        self.assertEqual(qa['status'], 'no-source')
        self.assertIsNone(qa['metrics'])
        self.assertEqual(overlay.shape, (1, 8, 8, 3))
        self.assertEqual(extra.shape, (1, 8, 8))
        self.assertFalse(extra.any() or missing.any())
        self.assertEqual((self.directory / 'manifest.json').read_bytes(), before)
        self.assertTrue(Path(qa['reportPath']).is_file())

    def test_fallback_is_inspected_before_alignment_and_masks_are_reused(self):
        data = np.zeros((8, 8, 4), dtype=np.float32)
        # Shift the 6 x 6 foreground one pixel up. The strict admission rule rejects it.
        data[:6, 1:7, 3] = 1
        asset = self.edited(data)
        overlay, extra, missing, raw = self.inspect(asset, '+X')
        qa = json.loads(raw)
        self.assertEqual((qa['stage'], qa['status'], qa['admission']),
                         ('before-registration', 'compared', 'fallback'))
        self.assertEqual(qa['metrics']['extraPixels'], 6)
        self.assertEqual(qa['metrics']['missingPixels'], 6)
        self.assertAlmostEqual(qa['metrics']['iou'], 30 / 42)
        np.testing.assert_array_equal(np.nonzero(extra[0])[0], np.zeros(6))
        np.testing.assert_array_equal(np.nonzero(missing[0])[0], np.full(6, 6))
        np.testing.assert_array_equal(np.rint(overlay[0, 0, 2] * 255), [255, 94, 32])
        np.testing.assert_array_equal(np.rint(overlay[0, 6, 2] * 255), [0, 190, 255])
        stage = Path(qa['reportPath']).parent
        with Image.open(stage / 'extra.png') as image:
            np.testing.assert_array_equal(np.array(image), extra[0] * 255)
        with patch('box_nodes.silhouette.measure', side_effect=AssertionError('unexpected recompute')):
            self.assertEqual(self.inspect(asset, '+X')[3], raw)

    def test_float_samples_survive_threshold_and_recorded_canvas_is_reversed(self):
        # Fill the padding, then place one camera-view shape solely within the capture rectangle.
        data = np.ones((16, 16, 4), dtype=np.float32)
        data[4:12, :, 3] = 0
        data[5:11, 2:14, 3] = 1
        data[4, :, 3] = .49999  # Sub-threshold coverage along the capture edge.
        canvas = {'version': 1, 'size': 32, 'faceId': '+Z', 'guideRect': [0, 8, 32, 16]}
        asset = self.edited(data, {'backend': 'qwen_enhanced', 'canvas': canvas}, '+Z')
        _, extra, missing, raw = self.inspect(asset, '+Z')
        qa = json.loads(raw)
        self.assertEqual((qa['coordinateSpace'], qa['sourceFormat']), ('reversed-guide-letterbox', '.npy'))
        self.assertEqual(qa['metrics']['iou'], 1)
        self.assertFalse(extra.any() or missing.any())

    def test_invalid_recorded_canvas_is_not_treated_as_a_perfect_match(self):
        data = np.ones((8, 16, 4), dtype=np.float32)
        canvas = {'version': 1, 'size': 16, 'faceId': '-Z', 'guideRect': [0, 0, 16, 16]}
        asset = self.edited(data, {'backend': 'h3_enhanced', 'canvas': canvas}, '-Z')
        _, extra, missing, raw = self.inspect(asset, '-Z')
        qa = json.loads(raw)
        self.assertEqual((qa['status'], qa['reason'], qa['admission']), ('unavailable', 'invalid-canvas', 'fallback'))
        self.assertIsNone(qa['metrics'])
        self.assertFalse(extra.any() or missing.any())

    def test_portable_png_only_face_and_legacy_model_with_unknown_camera(self):
        data = np.zeros((8, 8, 4), dtype=np.float32)
        data[1:7, 1:7, 3] = 1
        asset = self.edited(data)
        (store.asset_dir(asset) / 'views/px/source.npy').unlink()
        qa = json.loads(self.inspect(asset, '+X')[3])
        self.assertEqual((qa['sourceFormat'], qa['metrics']['iou']), ('.png', 1))
        legacy = self.edited(data, {'backend': 'qwen_enhanced'})
        qa = json.loads(self.inspect(legacy, '+X')[3])
        self.assertEqual(qa['reason'], 'generated-face-without-camera-canvas')
        self.assertIsNone(qa['metrics'])
