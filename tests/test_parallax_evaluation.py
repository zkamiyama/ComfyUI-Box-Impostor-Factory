"""Independent CPU fixtures for depth bytes, projection, visibility and atlas orientation."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

PATH = Path(__file__).resolve().parents[1] / 'scripts/evaluate_parallax_sweep.py'
spec = importlib.util.spec_from_file_location('parallax_evaluation', PATH)
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def camera(angle=0):
    radians = np.deg2rad(angle)
    c, s = np.cos(radians), np.sin(radians)
    view = np.array([[c, 0, -s, 0], [0, 1, 0, 0], [s, 0, c, -3], [0, 0, 0, 1.]])
    near, far = .1, 6.
    projection = np.array([[2., 0, 0, 0], [0, 2., 0, 0],
                           [0, 0, -2 / (far - near), -(far + near) / (far - near)], [0, 0, 0, 1.]])
    return {'angleDegrees': angle, 'near': near, 'far': far,
            'viewMatrix': view.T.ravel().tolist(), 'projectionMatrix': projection.T.ravel().tolist()}


def textured_plane(angle, size=64):
    frame = camera(angle)
    y, x = np.mgrid[:size, :size]
    radians = np.deg2rad(angle)
    wx = (((x + .5) / size - .5) + np.sin(radians) * .2) / np.cos(radians)
    wy = .5 - (y + .5) / size
    world = np.stack((wx, wy, np.full_like(wx, .2)), axis=-1)
    _, _, depth, _ = evaluation.project(frame, world, size, size)
    alpha = (abs(wx) < .35) & (abs(wy) < .35)
    linear = np.stack((wx + .5, wy + .5, np.full_like(wx, .3), alpha), axis=-1)
    frame['_rgba'] = np.asarray(evaluation.rgba_png(linear))
    frame['_depth'] = np.where(alpha, depth, 1).astype('<f4')
    frame['headROI'] = [0, 0, size, size]
    return frame


class ParallaxEvaluationTests(unittest.TestCase):
    def test_packed_float_png_preserves_all_four_bytes_including_nonopaque_fourth_byte(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'depth.png'
            depth = np.array([[0., .1, .5], [1., np.nan, -.2]], dtype='<f4')
            raw = depth.tobytes()
            Image.fromarray(np.frombuffer(raw, dtype='u1').reshape(2, 3, 4)).save(path)
            actual = evaluation.read_depth(path, hashlib.sha256(raw).hexdigest(), 3, 2)
            self.assertEqual(actual.tobytes(), raw)
            with self.assertRaisesRegex(ValueError, 'checksum'):
                evaluation.read_depth(path, '0' * 64, 3, 2)

    def test_world_reprojection_has_depth_dependent_horizontal_motion_and_reversible_pixels(self):
        world = np.array([[0., .1, .1], [0., .1, .3]])
        a, b = camera(0), camera(15)
        x, y, depth, _ = evaluation.project(a, world, 64, 64)
        np.testing.assert_allclose(x, [31.5, 31.5], atol=1e-12)
        restored = evaluation.unproject(a, x, y, depth, 64, 64)
        np.testing.assert_allclose(restored, world, atol=1e-12)
        tx, ty, _, _ = evaluation.project(b, restored, 64, 64)
        np.testing.assert_allclose(tx, 31.5 - np.sin(np.deg2rad(15)) * world[:, 2] * 64, atol=1e-12)
        np.testing.assert_allclose(ty, y, atol=1e-12)
        self.assertGreater(abs(tx[1] - tx[0]), 3.)

    def test_true_camera_motion_is_compensated_but_a_frozen_screen_image_is_not(self):
        a, b = textured_plane(0), textured_plane(15)
        measured = evaluation.motion_metrics(a, b, [0, 0, 64, 64], 64, 64, .002,
                                             evaluation.rgba_linear(a['_rgba']))
        self.assertGreater(measured['mutuallyVisiblePixels'], 1000)
        actual = measured['frozenFrontPixelControl']['actualVariant']['linearRGBAbsoluteError']['mean']
        frozen = measured['frozenFrontPixelControl']['frozenFrontPixels']['linearRGBAbsoluteError']['mean']
        self.assertLess(actual, .002)
        self.assertGreater(frozen, actual * 8)
        self.assertGreater(measured['projectedMotionPixels']['p50'], 2)
        self.assertLess(measured['motionBeyondConstantZPlane']['motionResidualAgainstConstantZPlanePixels']['max'], .00001)

    def test_a_closer_occluder_is_excluded_before_color_error_is_computed(self):
        a, b = textured_plane(0), textured_plane(0)
        baseline = evaluation.correspondence(a, b, [0, 0, 64, 64], 64, 64, .002)
        changed = copy.deepcopy(b)
        changed['_depth'][20:40, 20:40] -= .05
        changed['_rgba'][20:40, 20:40, :3] = [255, 0, 0]
        checked = evaluation.correspondence(a, changed, [0, 0, 64, 64], 64, 64, .002)
        self.assertEqual(int(baseline['valid'].sum() - checked['valid'].sum()), 400)
        self.assertFalse(checked['valid'][20:40, 20:40].any())
        self.assertLess(evaluation.error_metrics(checked['a'], checked['b'], checked['valid'])['sRGB8AbsoluteError']['max'], .0001)

    def test_front_reference_decodes_plus_z_bottom_up_premultiplied_atlas_independently(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            r = 8
            y, x = np.mgrid[:r, :r]
            straight = np.stack((x / 8, y / 8, np.full_like(x, .25, dtype='f4'), np.full_like(x, .5, dtype='f4')), axis=-1)
            premultiplied = straight.copy()
            premultiplied[..., :3] *= premultiplied[..., 3:4]
            atlas = np.zeros((2 * r, 3 * r, 4), dtype='<f2')
            atlas[r:2 * r, r:2 * r] = premultiplied[::-1]
            texture = root / 'color.bin'
            atlas.tofile(texture)
            manifest = {'resolution': r, 'asset_id': 'test',
                        'bakeBox': {'min': [-.5, -.5, -.5], 'max': [.5, .5, .5]},
                        'textures': {'colorMap': {'dtype': 'float16', 'width': 3 * r, 'height': 2 * r,
                                                  'path': 'color.bin', 'sha256': evaluation.sha(texture)}}}
            manifest_path = root / 'manifest.json'
            manifest_path.write_text(json.dumps(manifest))
            sweep = {'source': {'manifestPath': str(manifest_path), 'manifestSHA256': evaluation.sha(manifest_path)}}
            actual, proof, _ = evaluation.native_atlas_reference(sweep, root, camera(0), r, r)
            np.testing.assert_allclose(actual, straight, atol=1e-7)
            self.assertEqual(proof['tile'], [1, 1])
            self.assertTrue(proof['notOriginalGeneratedPNG'])


if __name__ == '__main__':
    unittest.main()
