import unittest

import numpy as np

from scripts.fit_splat_guides import FACES, fit_silhouettes


class SplatFitTests(unittest.TestCase):
    def test_one_transform_fits_opposing_faces_and_uses_native_axis_signs(self):
        r = 128
        coord = (np.arange(r) + .5) / r - .5
        xx, yy = np.meshgrid(coord, coord)
        center, spans = np.array([.2, -.3, .1]), np.array([1.4, .9, .5])
        scale, shift = np.array([.83, 1.15, .72]), np.array([.055, -.032, .073])
        dims = np.array([.18, .27, .23])
        source_center = np.array([.06, -.1, -.035])
        manifest = {'resolution': r, 'bakeBox': {'min': center - spans / 2, 'max': center + spans / 2}, 'faces': {}}
        moving, target = {}, {}
        for f in FACES:
            axis, sign = 'XYZ'.index(f[1]), 1 if f[0] == '+' else -1
            side = np.eye(3)[axis] * sign
            up = np.array([0, 1, 0]) if axis != 1 else np.array([0, 0, -sign])
            manifest['faces'][f] = {'side': side, 'up': up}
            right = np.cross(up, side)

            def project(radius, position):
                a = (xx - right @ position) / (abs(right) @ radius)
                b = (-yy - up @ position) / (abs(up) @ radius)
                return a**2 + b**2 <= 1.

            moving[f] = project(dims, source_center)
            target[f] = project(scale * dims, scale * source_center + shift)
        result = fit_silhouettes(manifest, moving, target)
        xf = result['transformSplat']
        np.testing.assert_allclose([xf['scale_' + a] for a in 'xyz'], scale, atol=.025)
        np.testing.assert_allclose([xf['translate_' + a] for a in 'xyz'],
                                   ((1 - scale) * center + shift * spans) * [1, -1, -1], atol=.01)
        self.assertGreater(min(result['predictedSilhouetteIoU'].values()), .94)

        identical = fit_silhouettes(manifest, moving, moving)
        self.assertEqual(list(identical['transformSplat'].values()), [0.] * 6 + [1.] * 3)


if __name__ == '__main__':
    unittest.main()
