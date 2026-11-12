import sys
import unittest
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from box_nodes.canvas import letterbox_rect, place, undo_guide_canvas


class CanvasTests(unittest.TestCase):
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
