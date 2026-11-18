import unittest
import numpy as np

from box_nodes.material_detail import restore_detail


def linear(rgb):
    return np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)


class MaterialDetailTests(unittest.TestCase):
    def test_disabled_preserves_exact_cached_output(self):
        base = np.random.default_rng(3).random((1, 32, 48, 3), dtype='f4')
        self.assertIs(restore_detail(base, None, 1., 0.), base)

    def test_matching_source_needs_no_correction(self):
        base = np.random.default_rng(7).random((1, 48, 48, 3), dtype='f4')
        candidate = np.dstack((linear(base[0]), np.ones((48, 48), 'f4')))
        result = restore_detail(base, candidate, .4, .75)
        np.testing.assert_allclose(result, base, atol=2e-7)

    def test_gain_is_bounded_and_alpha_edges_are_unchanged(self):
        h, w = 128, 96
        base = np.full((1, h, w, 3), .5, dtype='f4')
        candidate = np.zeros((h, w, 4), 'f4')
        candidate[10:-10, 10:-10, 3] = 1
        candidate[..., :3] = np.where(np.indices((h, w))[1, ..., None] % 2, 1., .001)
        result = linear(restore_detail(base, candidate, .4, 1.)[0])
        gain = result / linear(base[0])
        self.assertGreaterEqual(gain.min(), 1 / 1.5 - 1e-6)
        self.assertLessEqual(gain.max(), 1.5 + 1e-6)
        np.testing.assert_allclose(result[:12], linear(base[0, :12]), atol=1e-7)
        np.testing.assert_allclose(result[:, :12], linear(base[0, :, :12]), atol=1e-7)
        self.assertGreater(float(gain[20:-20, 20:-20].std()), .01)

    def test_restoration_keeps_channel_ratios_and_large_scale_tone(self):
        h, w = 256, 128
        base = np.broadcast_to(np.array([.50, .35, .20], dtype='f4'), (1, h, w, 3)).copy()
        candidate = np.ones((h, w, 4), 'f4')
        x = np.indices((h, w))[1]
        candidate[..., :3] = linear(base[0]) * (1 + .1 * np.sin(x[..., None] * np.pi / 2))
        result = linear(restore_detail(base, candidate, .5, .75)[0])
        before = linear(base[0])
        np.testing.assert_allclose(result[..., 0] / result[..., 1], before[..., 0] / before[..., 1], rtol=1e-6)
        self.assertLess(float(np.abs(result.mean((0, 1)) - before.mean((0, 1))).max()), .003)


if __name__ == '__main__':
    unittest.main()
