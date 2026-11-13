import unittest

import numpy as np

from box_nodes.depth_refine import guided_depth


class GuidedDepthTests(unittest.TestCase):
    def test_geometry_units_delta_and_color_guidance(self):
        z = np.full((19, 17), .42, dtype=np.float32)
        z[9, 8] += .012
        rgb = np.full((*z.shape, 3), .1, dtype=np.float32)
        alpha = np.ones((*z.shape, 1))
        smooth = guided_depth(z, rgb, alpha)
        self.assertLess(smooth[9, 8], z[9, 8] - .002)
        self.assertLessEqual(float(np.abs(smooth - z).max()), .012)
        rgb[9, 8] = 1
        guarded = guided_depth(z, rgb, alpha)
        self.assertEqual(guarded[9, 8], z[9, 8])
        np.testing.assert_array_equal(guarded, z)

    def test_silhouettes_disconnected_depth_layers_and_nans_stay_fixed(self):
        z = np.full((29, 21), .23, dtype=np.float32)
        z[:, 12:] = .78
        rgb = np.ones((*z.shape, 3), dtype=np.float32)
        alpha = np.ones(z.shape)
        z[4, 4] = np.nan
        rgb[4, 4] = np.nan
        alpha[16, :] = 0
        z[15, 9] += .006
        z[17, 9] -= .006
        alpha[24, 8] = .125
        z[24, 8] += .008
        z[6, 5] -= .005
        z[8, 11] += .004
        z[8, 12] -= .004
        z[0, 8] += .003
        out = guided_depth(z, rgb, alpha)
        for y, x in [(15,9), (17,9), (24,8), (6,5), (8,11), (8,12), (0,8)]:
            self.assertEqual(out[y, x], z[y, x])
        np.testing.assert_array_equal(np.isnan(out), np.isnan(z))
        np.testing.assert_array_equal(guided_depth(z, rgb, alpha, max_delta=0), z)

    def test_strip_boundary_and_invalid_shapes(self):
        z = np.full((168, 13), .3, dtype=np.float32)
        z[129:132, 6] += .009
        rgb = np.zeros((*z.shape, 3), dtype=np.float32)
        alpha = np.ones(z.shape)
        a = guided_depth(z, rgb, alpha)
        b = guided_depth(z[::-1], rgb[::-1], alpha[::-1])[::-1]
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-7)
        with self.assertRaises(ValueError):
            guided_depth(z, rgb[:, :-1], alpha)
        with self.assertRaises(ValueError):
            guided_depth(z, rgb, alpha, color_sigma=0)
