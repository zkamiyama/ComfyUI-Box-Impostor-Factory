import sys
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from box_nodes.canvas import ModelCanvas, letterbox_rect, place, undo_guide_canvas


class CanvasTests(unittest.TestCase):
    def test_different_connected_guide_or_reference_changes_generation_identity(self):
        from types import SimpleNamespace
        model = ModelCanvas()
        meta = {'bakeBox': {'min': [0,0,0], 'max': [1,1,1]}}
        pixels = np.zeros((1,8,8,3), dtype=np.float32)
        changed = pixels.copy()
        changed[0, 4, 5, 1] = 1
        def canvas(ref, guide):
            return model.prepare(ref, guide, 'a' * 64, '+Z', 32)[2]
        with patch.dict(sys.modules, torch=SimpleNamespace(from_numpy=lambda a:a)), \
             patch('box_nodes.store.asset_dir', return_value=Path('.')), patch('box_nodes.store.validate', return_value=meta):
            original = canvas(pixels, pixels)
            self.assertEqual(original, canvas(pixels.copy(), pixels.copy()))
            self.assertNotEqual(original['referenceInputSha256'], canvas(changed,pixels)['referenceInputSha256'])
            self.assertNotEqual(original['guideInputSha256'], canvas(pixels,changed)['guideInputSha256'])
            self.assertNotEqual(original['guideInputSha256'], canvas(pixels,pixels.reshape(1,4,16,3))['guideInputSha256'])

    def test_non_cubic_top_guide_is_reversible_at_multiple_resolutions(self):
        rect = letterbox_rect(4, 2, 64)
        self.assertEqual(rect, [0, 16, 64, 32])
        image = np.ones((17, 17, 4), dtype=np.float32)
        image[:8, :, 0] = 0
        made = place(image, rect, 64)[0]
        np.testing.assert_array_equal(made[:16], .5)
        canvas = {'size': 64, 'faceId': '+Y', 'version': 1, 'guideRect': rect}
        cropped = undo_guide_canvas(made, canvas)
        self.assertEqual(cropped.shape,(32,64,3))
        self.assertEqual(cropped[0,0,0], 0)
        self.assertEqual(cropped[-1,0,0], 1)
        self.assertEqual(undo_guide_canvas(np.zeros((128,128,4)), canvas).shape, (64,128,4))

    def test_aspect_change_cannot_be_silently_accepted(self):
        canvas = {'size': 32, 'faceId': '-Z', 'version': 1, 'guideRect': [0,0,32,32]}
        with self.assertRaisesRegex(ValueError, 'square'):
            undo_guide_canvas(np.zeros((32,64,4)), canvas)
