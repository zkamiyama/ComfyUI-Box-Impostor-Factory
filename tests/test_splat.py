import unittest
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from box_nodes.splat import atlas_pixels, camera_plan, depth_agreement, splat_hash


def matrix(m):
    return np.asarray(m).ravel(order='F').tolist()


class SplatTests(unittest.TestCase):
    def test_six_registered_cameras_with_pixel_centers_and_non_square_proxy(self):
        frame = np.eye(4)
        frame[:3, :3] = .8 * Rotation.from_euler('zyx', [28, -73, 15], degrees=True).as_matrix()
        frame[:3, 3] = [-1, 2, .3]
        spans = np.array([1., .7, .4])
        center = np.array([.03, -.08, .15])
        point = np.array([.08, .04, .11]) + center
        r = 64
        for axis, axis_id in enumerate('XYZ'):
            for sign in (-1, 1):
                face_id = ('-' if sign < 0 else '+') + axis_id
                side = np.eye(3)[axis] * sign
                up = np.array([0, 1, 0]) if axis != 1 else np.array([0, 0, -sign])
                right = np.cross(up, side)
                basis = np.column_stack([right, up, side])
                camera_world = np.eye(4)
                camera_world[:3, :3] = basis
                camera_world[:3, 3] = center + side
                span = np.abs(spans @ basis)[:2]
                w, h = span
                m = {'bakeBox': {'min': center-spans/2, 'max': center+spans/2}, 'resolution': r,
                     'faces': {face_id: {'viewMatrix': matrix(np.linalg.inv(camera_world)),
                                         'frustum': [-w/2, w/2, -h/2, h/2]}}, 'inverseTransform': matrix(frame)}
                for space in ('source_glb', 'canonical_box'):
                    with self.subTest(face=face_id, space=space):
                        camera, size, content = camera_plan(m, face_id, space)
                        pp = frame @ np.append(point, 1.) if space == 'source_glb' else np.append(point, 1.)
                        eye = np.array(list(camera['position'].values()))
                        target = np.array(list(camera['target'].values()))
                        basis2 = Rotation.from_quat(list(camera['quaternion'].values())).as_matrix()
                        scale = min(size)/(2*np.linalg.norm(eye-target)*np.tan(np.radians(camera['fov']/2)))
                        xy = ((pp[:3]-eye) @ basis2)[:2] * [1, -1] * scale + np.array(size)/2
                        x, y = (xy - np.asarray(size)/2) * r/content + r/2 - .5
                        expected = ((point-center) @ basis)[:2] * [1, -1]/span*r + r/2-.5
                        np.testing.assert_allclose([x,y], expected, atol=1e-6)
                        yy, xx = np.indices((size[1], size[0]), dtype=float)
                        grid = atlas_pixels(np.stack((xx, yy), axis=-1), content, r)
                        np.testing.assert_allclose(grid[r//2,r//2], np.asarray(size)/2+content/(2*r))

    def test_rank_diagnostic_accepts_unshared_depth_units(self):
        mesh = np.arange(9, dtype=float).reshape(3,3)
        mask = np.ones((3,3), bool)
        self.assertAlmostEqual(depth_agreement(mesh, .9-mesh*.07, mask)['spearman'], 1.)
        self.assertAlmostEqual(depth_agreement(mesh, mesh*.05, mask)['spearman'], -1.)
        self.assertEqual(depth_agreement(mesh, np.ones_like(mesh), mask)['status'], 'constant-depth')
        self.assertEqual(depth_agreement(mesh, mesh, ~mask)['status'], 'insufficient-overlap')

    def test_digest_honors_real_batch_count_and_splat_values(self):
        s = SimpleNamespace(positions=np.ones((1,3,3)), scales=np.ones((1,3,3)),
                            opacities=np.ones((1,3,1)), rotations=np.ones((1,3,4)),
                            sh=np.ones((1,3,1,3)), counts=np.array([2]))
        digest = splat_hash(s)
        s.positions[0,2,1] = np.nan
        self.assertEqual(splat_hash(s), digest)
        s.sh[0,0,0,2] += .1
        self.assertNotEqual(splat_hash(s), digest)
        s.positions[0,0,1] = np.nan
        with self.assertRaisesRegex(ValueError, 'non-finite'):
            splat_hash(s)
