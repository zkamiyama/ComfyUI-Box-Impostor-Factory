from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from box_nodes import multiview, view_reproject
from test_multiview import sphere


class ViewReprojection(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.manifest = sphere(self.root, r=128)
        self.r = self.manifest['resolution']
        for descriptor in self.manifest['textures'].values():
            descriptor['dtype'] = 'float16'
        self.original = {name: np.array(multiview.atlas(self.root, descriptor, '<f2'))
                         for name, descriptor in self.manifest['textures'].items()}
        self.expected = self.original['colorMap'].copy()
        colors = self.original['colorMap'].copy()
        r = self.r
        colors[:r, :r, :3] = 0  # Wrong +X appearance on the same analytic sphere.
        colors.tofile(self.root/'colorMap.bin')
        material = np.rint(np.clip(self.expected, 0, 1)*255).astype(np.uint8)
        material[:r, :r, :3] = 0
        material.tofile(self.root/'material.bin')
        self.manifest['textures']['materialMap'] = {'path': 'material.bin', 'dtype': 'uint8',
                                                   'width': 3*r, 'height': 2*r}
        self.manifest['maps'] = ['roughness', 'metalness']
        self.before = {'colorMap': colors, 'normalMap': self.original['normalMap'], 'materialMap': material}

    def atlas(self, name):
        descriptor = self.manifest['textures'][name]
        return np.array(multiview.atlas(self.root, descriptor, '<f2' if descriptor['dtype'] == 'float16' else 'u1'))

    def test_visible_front_surface_replaces_color_and_material_together(self):
        report = view_reproject._reproject(self.root, self.manifest, '+Z', 2.)
        r = self.r
        color = self.atlas('colorMap')[:r, :r][::-1]
        expected = self.expected[:r, :r][::-1]
        # The analytic sphere stores RGB = position. z>.65 is the front-visible hemisphere.
        visible = (expected[..., 2] > .65) & (expected[..., 3] > .99)
        changed = visible & (color[..., :3].sum(axis=-1) > 0)
        self.assertGreater(changed.sum(), visible.sum()*.85)
        self.assertLess(np.abs(color[changed, :3]-expected[changed, :3]).mean(), .005)
        material = self.atlas('materialMap')[:r, :r][::-1].astype(np.float32)/255
        np.testing.assert_allclose(material[changed, :3], color[changed, :3], atol=.006)
        self.assertGreater(report['faces']['+X']['reprojectedPixels'], 0)

    def test_grazing_target_pixels_are_not_discarded_by_normal_cosine(self):
        view_reproject._reproject(self.root, self.manifest, '+Z', 2.)
        r = self.r
        color = self.atlas('colorMap')[:r, :r][::-1]
        expected = self.expected[:r, :r][::-1]
        normals = self.original['normalMap'][:r, :r][::-1, :, :3]*2-1
        grazing = ((normals[..., 0] > .05) & (normals[..., 0] < .25) &
                   (expected[..., 2] > .7) & (expected[..., 3] > .99))
        self.assertGreater(grazing.sum(), 20)
        self.assertGreater((color[grazing, :3].sum(axis=-1) > 0).mean(), .65)

    def test_hidden_surface_and_opposite_capture_remain_unchanged(self):
        view_reproject._reproject(self.root, self.manifest, '+Z', 2.)
        r = self.r
        color = self.atlas('colorMap')[:r, :r][::-1]
        expected = self.expected[:r, :r][::-1]
        hidden = (expected[..., 2] < .3) & (expected[..., 3] > .99)
        np.testing.assert_array_equal(color[hidden, :3], 0)
        for name, original in self.before.items():
            actual = self.atlas(name)
            np.testing.assert_array_equal(actual[..., 3], original[..., 3])
            np.testing.assert_array_equal(actual[r:, r:2*r], original[r:, r:2*r])
            np.testing.assert_array_equal(actual[r:, 2*r:], original[r:, 2*r:])
            self.assertTrue(np.isfinite(actual).all())

    def test_occluder_in_anchor_depth_prevents_color_transfer(self):
        f = self.root/self.manifest['faces']['+Z']['depth']
        depth = np.fromfile(f, '<f2')
        depth[np.isfinite(depth)] -= .25
        depth.tofile(f)
        report = view_reproject._reproject(self.root, self.manifest, '+Z', 2.)
        self.assertEqual(sum(x['reprojectedPixels'] for x in report['faces'].values()), 0)
        np.testing.assert_array_equal(self.atlas('colorMap'), self.before['colorMap'])

    def test_rejected_anchor_is_an_explicit_noop(self):
        manifest = {'faces': {'+Z': {'edit': {'status': 'fallback'}}}}
        with patch.object(view_reproject.store, 'asset_dir', return_value=self.root), \
             patch.object(view_reproject.store, 'validate', return_value=manifest), \
             patch.object(view_reproject.store, 'publish') as publish:
            with self.assertWarnsRegex(UserWarning, 'not accepted'):
                self.assertEqual(view_reproject.reconcile_views('source'), 'source')
            publish.assert_not_called()


if __name__ == '__main__':
    unittest.main()
