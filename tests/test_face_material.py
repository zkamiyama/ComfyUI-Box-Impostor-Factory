import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from box_nodes import archive, faces, store
from box_nodes.face_material import BILoadGeneratedFace, BIApplyFaceMaterial, apply_face_materials
from test_nodes import fixture


class FaceMaterialTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(patch.object(store, 'root', return_value=self.root))
        self.base = 'a' * 64
        self.m = fixture(self.root / self.base, self.base)
        self.rgba = np.zeros((8, 8, 4), dtype='f4')
        self.rgba[..., :3] = (.2, .7, .4)
        self.rgba[1:-1, 1:-1, 3] = 1
        self.rgba[1, 1, 3] = .75
        self.srgb = self.image((.5, .1, .7))
        self.orm = self.image((.2, .4, .8))
        self.apply = BIApplyFaceMaterial().apply

    def image(self, rgb):
        return np.broadcast_to(np.array(rgb, 'f4'), (1, 8, 8, 3)).copy()

    def compose(self, base, *ids):
        saved = faces.save_view(self.rgba)
        return faces.compose(base, faces.align(dict.fromkeys(ids, saved), base, .9, .25, .25))

    def read(self, asset, name):
        directory = store.asset_dir(asset)
        m = store.validate(directory, asset)
        desc = m['textures'][name]
        return np.fromfile(directory / desc['path'], 'u1' if desc['dtype'] == 'uint8' else '<f2').reshape(
            desc['height'], desc['width'], 4)

    def add_baked(self):
        directory = self.root / self.base
        material = np.empty((16, 24, 4), 'u1')
        material[:] = [9, 10, 45, 128]
        p = directory / 'maps/original.rgba8.bin'
        material.tofile(p)
        self.m['textures']['materialMap'] = {
            **self.m['textures']['colorMap'], 'path': 'maps/' + p.name, 'dtype': 'uint8', 'byteLength': material.nbytes,
            'rowBytes': 24*4, 'sha256': store.file_hash(p), 'alpha': 'coverage', 'colorSpace': 'data'}
        self.m['maps'] = ['roughness', 'metalness', 'ao']
        (directory / 'manifest.json').write_text(json.dumps(self.m))

    def test_input_immutable_source_masks_gamma_orientation_and_fallback(self):
        base = self.compose(self.base, '+X', '-Z')
        load = BILoadGeneratedFace().load
        before, mask = load(base, '+X')
        np.testing.assert_allclose(before[0, 3, 3].numpy(), self.rgba[3, 3, :3], atol=1e-6)
        np.testing.assert_allclose(before[0, 0, 0].numpy(), .5, atol=1e-6)
        np.testing.assert_array_equal(mask[0].numpy(), self.rgba[..., 3])
        self.srgb[0, 2, 3] = [1, 0, 0]
        child, = self.apply(base, '+X', 'ORM', self.srgb, self.orm, 'test')
        color, initial = self.read(child, 'colorMap'), self.read(base, 'colorMap')
        np.testing.assert_array_equal(faces.tile_view(color, '+X')[::-1][2, 3], [.5, 0, 0, .5])
        self.assertAlmostEqual(float(faces.tile_view(color, '+X')[::-1][3, 3, 0]), .214041 * .5, places=4)
        np.testing.assert_array_equal(color[..., 3].view('u2'), initial[..., 3].view('u2'))
        for fid in store.FACES[1:]:
            np.testing.assert_array_equal(faces.tile_view(color, fid).view('u2'), faces.tile_view(initial, fid).view('u2'))
        after, newmask = load(child, '+X')
        np.testing.assert_array_equal(before, after)
        np.testing.assert_array_equal(mask, newmask)
        surface = self.read(child, 'materialMap')
        np.testing.assert_array_equal(faces.tile_view(surface, '+X')[5, 3], [51, 102, 26, 128])
        np.testing.assert_array_equal(faces.tile_view(surface, '-Z')[3, 3], [128, 0, 128, 128])
        m = store.validate(self.root / child, child)
        for name in ('normalMap', 'intervalMap', 'rangeMap'):
            self.assertEqual(m['textures'][name], self.m['textures'][name])
        with self.assertRaisesRegex(ValueError, 'accepted generated'):
            load(child, '-Y')
        self.assertEqual(self.apply(child, '+X', 'ORM', self.srgb, self.orm, 'test'), (child,))

    def test_partial_channels_baked_ao_and_face_only_invalidation_after_roundtrip(self):
        self.add_baked()
        base = self.compose(self.base, '+X', '-Z')
        req = dict(albedo=self.srgb, rma=self.orm, provider='marigold', packing='RM', recipe={'seed': 5})
        child = apply_face_materials(base, {'+X': req, '-Z': req})
        old_rma = self.read(child, 'materialMap')
        np.testing.assert_array_equal(old_rma[..., 2:], self.read(self.base, 'materialMap')[..., 2:])
        new_manifest = store.validate(self.root / child, child)
        self.assertEqual(new_manifest['faces']['+X']['materialEdit']['albedo']['recipe']['seed'], 5)
        qa = json.loads((self.root / child / new_manifest['qa']).read_bytes())
        self.assertEqual(len(qa['crossView']['pairs']), 12)
        self.assertEqual(qa['faces']['+X']['status'], 'accepted')
        archive_path = archive.pack_asset(self.root / child, self.root / 'test.box.zip')
        shutil.rmtree(self.root / child)
        shutil.rmtree(self.root / base)
        archive.unpack_asset(archive_path, self.root / child)
        self.rgba[..., :3] = (.3, .2, .1)
        updated = self.compose(child, '+X')
        updated_manifest = store.validate(self.root / updated, updated)
        self.assertNotIn('materialEdit', updated_manifest['faces']['+X'])
        self.assertEqual(updated_manifest['faces']['-Z']['materialEdit'], new_manifest['faces']['-Z']['materialEdit'])
        current = self.read(updated, 'materialMap')
        np.testing.assert_array_equal(faces.tile_view(current, '+X'), faces.tile_view(self.read(self.base, 'materialMap'), '+X'))
        np.testing.assert_array_equal(faces.tile_view(current, '-Z'), faces.tile_view(old_rma, '-Z'))
        self.rgba[..., 3] = 0
        last = self.compose(updated, '-Z')
        last_manifest = store.validate(self.root / last, last)
        self.assertNotIn('faceMaterialBase', last_manifest)
        self.assertEqual(last_manifest['textures']['materialMap'], self.m['textures']['materialMap'])
        self.assertEqual(last_manifest['faces']['-Z']['edit']['status'], 'fallback')
        baked_path = self.root / child / new_manifest['faceMaterialBase']['materialMap']['path']
        baked_path.write_bytes(bytes(baked_path.stat().st_size))
        with self.assertRaisesRegex(ValueError, 'digest mismatch'):
            store.validate(self.root / child, child)

    def test_unlit_only_and_albedo_retry_preserve_other_properties(self):
        base = self.compose(self.base, '-Y')
        a, = self.apply(base, '-Y', albedo=self.srgb)
        self.assertNotIn('materialMap', store.validate(self.root / a, a)['textures'])
        b, = self.apply(a, '-Y', 'RMA', rma=self.orm)
        c, = self.apply(b, '-Y', 'RMA', albedo=self.image((1, 1, 1)))
        np.testing.assert_array_equal(self.read(b, 'materialMap'), self.read(c, 'materialMap'))
        only, = self.apply(base, '-Y', albedo=self.image((1, 1, 1)))
        np.testing.assert_array_equal(self.read(c, 'colorMap').view('u2'), self.read(only, 'colorMap').view('u2'))
        c2 = self.compose(c, '-Y')
        m = store.validate(self.root / c2, c2)
        self.assertNotIn('materialMap', m['textures'])
        self.assertEqual(m['maps'], [])
        with self.assertRaisesRegex(ValueError, 'Supply an albedo'):
            self.apply(c, '-Y')
        with self.assertRaisesRegex(ValueError, 'registered'):
            self.apply(c, '-Y', albedo=self.srgb[:, :7])
        with self.assertRaisesRegex(ValueError, 'accepted generated'):
            self.apply(self.base, '-Y', albedo=self.srgb)

    def test_square_map_at_different_resolution_filters_albedo_in_linear_space(self):
        base = self.compose(self.base, '+Y')
        rgb = np.ones((1, 32, 32, 3), 'f4')
        rgb[:, ::2] = 0
        surface = np.full_like(rgb, .5)
        applied, = self.apply(base, '+Y', 'RM', rgb, surface)
        color = faces.tile_view(self.read(applied, 'colorMap'), '+Y')
        self.assertAlmostEqual(float(color[3, 3, 0]), .25, places=3)
        material = faces.tile_view(self.read(applied, 'materialMap'), '+Y')
        np.testing.assert_array_equal(material[3, 3], [64, 64, 128, 128])

    def test_binary_metalness_keeps_one_class_through_soft_weights_and_tile_resizing(self):
        self.add_baked()
        self.rgba[1, 2, 3] = .25
        base = self.compose(self.base, '+X')
        packed = np.ones((1, 32, 32, 3), 'f4')
        packed[..., 1] = 0
        packed[:, :, :15, 1] = 1
        req = dict(packing='RMA', rma=packed, binaryMetalness=True)
        out = apply_face_materials(base, {'+X': req})
        original = self.read(base, 'materialMap')
        updated = self.read(out, 'materialMap')
        tile = faces.tile_view(updated, '+X')[::-1]
        edited = self.rgba[..., 3] >= .5
        alpha = tile[..., 3]
        np.testing.assert_array_equal(tile[..., 3], faces.tile_view(original, '+X')[::-1, :, 3])
        self.assertTrue(np.all((tile[..., 1] == 0)[edited] | (tile[..., 1] == alpha)[edited]))
        self.assertEqual(tile[1, 1, 1], alpha[1, 1])  # .75 accepted, .5 baked coverage
        self.assertEqual(tile[3, 5, 1], 0)
        np.testing.assert_array_equal(tile[..., 1][~edited], faces.tile_view(original, '+X')[::-1, :, 1][~edited])
        np.testing.assert_array_equal(faces.tile_view(updated, '-X'), faces.tile_view(original, '-X'))
