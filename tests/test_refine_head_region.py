"""CPU checks for head clipping, physical image framing and approximate mapping."""
import copy
import importlib.util
from pathlib import Path
import unittest

import numpy as np
from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/refine_head_region.py'
spec = importlib.util.spec_from_file_location('refine_head_region', SCRIPT)
head = importlib.util.module_from_spec(spec)
spec.loader.exec_module(head)


def cube():
    vertices = np.array([[-1, 0, -.5], [1, 0, -.5], [1, 1, -.5], [-1, 1, -.5],
                         [-1, 0, .5], [1, 0, .5], [1, 1, .5], [-1, 1, .5]], dtype='f8')
    faces = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
                      [0, 1, 5], [0, 5, 4], [3, 7, 6], [3, 6, 2],
                      [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5]], dtype='i8')
    return vertices, faces


def source_camera():
    projection = np.diag([1., .5, -1., 1.])
    return ({'side': [0, 0, 1], 'up': [0, 1, 0], 'imageRowOrder': 'top-to-bottom',
             'viewMatrix': np.eye(4).T.ravel().tolist(),
             'projectionMatrix': projection.T.ravel().tolist()},
            {'version': 1, 'size': 200, 'faceId': '+Z', 'guideRect': [50, 0, 100, 200],
             'physicalWidthHeight': [2., 4.], 'imageRowOrder': 'top-to-bottom'},
            {'scale': 1., 'centerShiftYX': [0., 0.]})


class HeadRegionTests(unittest.TestCase):
    def test_crossing_triangles_form_exact_open_plane_without_erasing_top_or_flipping_faces(self):
        vertices, faces = cube()
        cropped, clipped, proof = head.clip_above_y(vertices, faces, .5)
        self.assertEqual(len(clipped), 14)
        self.assertEqual(proof['boundary']['boundaryEdges'], 8)
        self.assertEqual(proof['boundary']['boundaryEdgesOnCutPlane'], 8)
        self.assertEqual(proof['boundary']['nonmanifoldEdges'], 0)
        self.assertFalse(proof['syntheticCap'])
        np.testing.assert_array_equal(cropped.min(axis=0), [-1, .5, -.5])
        np.testing.assert_array_equal(cropped.max(axis=0), [1, 1, .5])
        triangles = cropped[clipped]
        self.assertFalse(np.any(np.all(triangles[..., 1] == .5, axis=1)))
        area_vectors = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        self.assertAlmostEqual(float(np.linalg.norm(area_vectors, axis=1).sum() / 2), 5.)
        # Every face remains outward, including the newly clipped side triangles.
        center = np.array([0., .75, 0.])
        self.assertTrue(np.all(np.einsum('ij,ij->i', area_vectors, triangles.mean(axis=1) - center) > 0))

    def test_cut_through_existing_vertex_does_not_create_zero_area_or_duplicate_corners(self):
        vertices = np.array([[-1., .5, -.5], [1, 1, -.5], [1, 0, .5], [-1, 1, .5]])
        faces = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]])
        points, triangles, _ = head.clip_above_y(vertices, faces, .5)
        self.assertTrue(np.all(points[:, 1] >= .5))
        corners = points[triangles]
        self.assertTrue(np.all(np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0],
                                                      corners[:, 2] - corners[:, 0]), axis=1) > 0))
        for face in triangles:
            self.assertEqual(len(set(face)), 3)

    def test_empty_cut_and_nonfinite_cut_fail_before_writing(self):
        vertices, faces = cube()
        for value in [0., 1., 2., float('nan')]:
            with self.subTest(cut=value), self.assertRaises(ValueError):
                head.clip_above_y(vertices, faces, value)

    def test_scene_nodes_are_applied_once_and_geometry_only_output_has_no_added_normals(self):
        vertices, faces = cube()
        data = head.geometry_glb(vertices, faces)
        doc, binary = head.depth_mesh._decode(data)
        doc['nodes'][0]['translation'] = [.2, -.3, .4]
        doc['nodes'][0]['scale'] = [2., 2., 2.]
        transformed = head.depth_mesh._encode(doc, bytearray(binary))
        points, triangles = head.mesh_arrays(transformed)
        expected = np.unique(vertices * 2 + [.2, -.3, .4], axis=0)
        np.testing.assert_allclose(points, expected, atol=1e-12)
        result = head.geometry_glb(points, triangles)
        actual_doc, _ = head.depth_mesh._decode(result)
        attrs = actual_doc['meshes'][0]['primitives'][0]['attributes']
        self.assertEqual(set(attrs), {'POSITION'})
        self.assertEqual(actual_doc['nodes'], [{'mesh': 0}])
        self.assertEqual(len(triangles), 12)

    def test_original_physical_pixel_projection_preserves_axes_and_is_depth_independent(self):
        face, canvas, registration = source_camera()
        matrix = head.camera_source_affine(face, canvas, (200, 200), registration)
        for z in [-100., 0., 100.]:
            np.testing.assert_allclose(matrix @ [.2, 1., z, 1.], [109.5, 49.5, 1.])
        np.testing.assert_allclose(matrix @ [-1., 2., 0., 1.], [49.5, -.5, 1.])
        np.testing.assert_allclose(matrix @ [1., -2., 0., 1.], [149.5, 199.5, 1.])

    def test_reference_uses_existing_nonsquare_crop_pixels_and_masks_the_same_world_cut(self):
        face, canvas, registration = source_camera()
        y, x = np.mgrid[:200, :200]
        data = np.stack([x, y, np.full_like(x, 100),
                         ((x >= 70) & (x < 130) & (y >= 20) & (y < 180)) * 255], axis=-1).astype('u1')
        original = data.copy()
        cropped, proof = head.crop_reference(Image.fromarray(data), face, canvas, registration,
                                             [[-.3, .5, -.2], [.3, 1.5, .2]], .5, 3)
        self.assertEqual(proof['sourceCropRectXYXY'], [67, 17, 133, 78])
        self.assertEqual(cropped.size, (66, 61))
        self.assertEqual(proof['subjectSizePixels'], [60, 55])
        self.assertFalse(proof['resampled'])
        self.assertFalse(proof['newImageInference'])
        expected = original[17:78, 67:133].copy()
        expected[75 - 17:] = 0
        np.testing.assert_array_equal(np.asarray(cropped), expected)
        np.testing.assert_array_equal(data, original)

    def test_rejects_registration_perspective_and_wrong_physical_aspect(self):
        face, canvas, registration = source_camera()
        with self.assertRaisesRegex(ValueError, 'identity'):
            head.camera_source_affine(face, canvas, (200, 200), {'scale': 1.01, 'centerShiftYX': [0, 0]})
        perspective = copy.deepcopy(face)
        matrix = np.array(perspective['projectionMatrix']).reshape(4, 4).T
        matrix[3, 2] = -1
        perspective['projectionMatrix'] = matrix.T.ravel().tolist()
        with self.assertRaisesRegex(ValueError, 'orthographic'):
            head.camera_source_affine(perspective, canvas, (200, 200), registration)
        stretched = copy.deepcopy(canvas)
        stretched['guideRect'] = [0, 0, 200, 200]
        with self.assertRaisesRegex(ValueError, 'letterbox'):
            head.camera_source_affine(face, stretched, (200, 200), registration)

    def test_nominal_voxel_transform_has_documented_center_margin_and_reversible_world_map(self):
        vertices, _ = cube()
        proof = head.nominal_voxel_mapping(vertices, {'requestedResolution': 64, 'pad': 3, 'eps': 1e-6})
        forward = np.array(proof['canonicalWorldToNominalVoxelRowMajor'])
        inverse = np.array(proof['nominalVoxelToCanonicalWorldRowMajor'])
        np.testing.assert_allclose(forward @ [0., .5, 0., 1.], [32., 32., 32., 1.], atol=2e-6)
        homogeneous = np.column_stack([vertices, np.ones(len(vertices))])
        np.testing.assert_allclose(homogeneous @ forward.T @ inverse.T, homogeneous, atol=1e-12)
        self.assertAlmostEqual((forward @ [-1., 0., -.5, 1.])[0], 3., delta=.0001)
        self.assertFalse(proof['observedVoxelization'])

    def test_output_fit_has_one_scale_and_exposes_width_depth_mismatch(self):
        source, _ = cube()
        raw = source * [1.3, 2., .8] + [2, 3, -4]
        mapped, proof = head.fit_isotropic_y(source, raw)
        self.assertAlmostEqual(proof['scale'], .5)
        np.testing.assert_allclose(proof['extentRatioXYZ'], [.65, 1., .4])
        np.testing.assert_allclose(mapped.min(axis=0) + mapped.max(axis=0),
                                   source.min(axis=0) + source.max(axis=0), atol=1e-12)
        forward, inverse = np.array(proof['rawToCanonicalWorldRowMajor']), np.array(proof['canonicalWorldToRawRowMajor'])
        np.testing.assert_allclose(forward @ inverse, np.eye(4), atol=1e-12)
        self.assertFalse(proof['exactInverse'])
        self.assertFalse(proof['rotationEstimated'])
        self.assertFalse(proof['axisSpecificStretch'])


if __name__ == '__main__':
    unittest.main()
