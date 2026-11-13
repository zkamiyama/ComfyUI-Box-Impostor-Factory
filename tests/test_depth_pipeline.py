import unittest

import numpy as np

from box_nodes.depth_pipeline import bound_paired_depth, vertex_displacements


class ProjectionTests(unittest.TestCase):
    def test_opposite_envelope_blocks_large_motion_and_keeps_unpaired_rays_fixed(self):
        front = np.full((3, 3), .4, dtype='f4')
        opposite = np.full_like(front, .4)
        opposite[2, 0] = np.nan
        opposite[1, 0] = .599
        delta = np.full_like(front, .01)
        x = bound_paired_depth(delta, front, opposite, '+X')
        self.assertEqual(x[2, 2], 0.)
        self.assertAlmostEqual(float(x[1, 2]), .00045, places=6)
        y = bound_paired_depth(delta, front, opposite, '-Y')
        self.assertEqual(y[0, 0], 0.)
        self.assertAlmostEqual(float(y[1, 0]), .00045, places=6)

    def test_only_front_visible_vertices_move_with_the_measured_face(self):
        bounds = {'min': [-1., -1., -1.], 'max': [1., 1., 1.]}
        r = 16
        direct = np.full((r, r), .25, np.float32)
        change = np.full_like(direct, .005)
        change[:r//2, :] = 0.
        view = (np.eye(4), np.array([0, 0, 1]), direct, change)
        points = [[0, -.5, .5], [0, -.5, 0.], [0, .5, .5]]
        shift = vertex_displacements(points, [view], bounds, r)
        np.testing.assert_allclose(shift, [[0, 0, -.01], [0, 0, 0], [0, 0, 0]], atol=1e-6)

    def test_nan_background_and_opposite_view_do_not_pull_hidden_surfaces(self):
        bounds = {'min': [0] * 3, 'max': [1] * 3}
        depth = np.full((8, 8), np.nan, np.float32)
        estimates = [(np.eye(4), np.asarray(side), depth, np.zeros_like(depth))
                     for side in ([1, 0, 0], [-1, 0, 0])]
        displacement = vertex_displacements([[.4, .1, .2]], estimates, bounds, 8)
        np.testing.assert_array_equal(displacement, np.zeros((1, 3)))
