"""Color-transfer regressions: linear interpolation, precision and no normal changes."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image
import trimesh

from box_nodes import depth_mesh
from scripts import hi3d_postprocess as post


def texture_source(pixels, uv, factor=(1, 1, 1, 1)):
    pixels = np.asarray(pixels, dtype='u1')
    material = SimpleNamespace(baseColorTexture=Image.fromarray(pixels),
                               baseColorFactor=np.asarray(factor, dtype='f4'))
    return SimpleNamespace(vertices=np.zeros((len(uv), 3)), visual=SimpleNamespace(
        kind='texture', uv=np.asarray(uv), material=material))


def accessor(doc, binary, index):
    spec = doc['accessors'][index]
    view = doc['bufferViews'][spec['bufferView']]
    dtype = np.dtype({5121:'u1', 5123:'<u2', 5125:'<u4', 5126:'<f4'}[spec['componentType']])
    dims = {'SCALAR':1, 'VEC3':3, 'VEC4':4}[spec['type']]
    return np.ndarray((spec['count'], dims), dtype, buffer=binary,
                      offset=view.get('byteOffset', 0) + spec.get('byteOffset', 0),
                      strides=(view.get('byteStride', dims*dtype.itemsize), dtype.itemsize)).copy()


def primitive(doc):
    return doc['meshes'][0]['primitives'][0]


def mesh_with_colors(colors, normals=False):
    base = trimesh.creation.icosphere(subdivisions=1)
    mesh = trimesh.Trimesh(vertices=base.vertices, faces=base.faces, process=False)
    colors = np.broadcast_to(np.asarray(colors, 'f4'), (len(mesh.vertices), 4)).copy()
    mesh.visual = trimesh.visual.ColorVisuals(
        mesh=mesh, vertex_colors=np.rint(colors*255).astype('u1'))
    mesh.vertex_attributes[post.LINEAR_COLOR_ATTRIBUTE] = colors
    if normals:
        # Authored directions intentionally differ from computed geometry normals.
        mesh.vertex_normals = np.tile([0., 1., 0.], (len(mesh.vertices), 1))
    return mesh


class Hi3DColorTransferTests(unittest.TestCase):
    def test_texture_midgray_is_linear_but_alpha_is_not_gamma_decoded(self):
        source = texture_source([[[128, 128, 128, 128]]], [[.5, .5]])
        result = post.sample_source_basecolor(source)
        expected_gray = ((128/255 + .055)/1.055)**2.4
        np.testing.assert_allclose(result[0, :3], expected_gray, atol=6e-8, rtol=0)
        self.assertAlmostEqual(float(result[0, 3]), 128/255, places=7)
        self.assertEqual(result.dtype, np.dtype('f4'))

    def test_texture_decodes_before_bilinear_interpolation_and_linear_factor(self):
        source = texture_source([[[0, 0, 0, 0], [255, 255, 255, 255]]],
                                [[.5, .5]], factor=(.25, .5, 1, .4))
        # Interpolation in linear space yields .5; decoding a gamma-space
        # interpolated .5 would instead yield .214041.
        np.testing.assert_allclose(post.sample_source_basecolor(source),
                                   [[.125, .25, .5, .2]], atol=1e-7, rtol=0)

    def test_vertex_colors_are_already_linear_for_byte_and_float_sources(self):
        for values, expected in (
            (np.array([[128, 64, 32, 128]], dtype='u1'),
             np.array([[128, 64, 32, 128]], dtype='f4')/255),
            (np.array([[.123456789, .333333343, .75, .23456789]], dtype='f4'),
             np.array([[.123456789, .333333343, .75, .23456789]], dtype='f4')),
        ):
            with self.subTest(dtype=values.dtype):
                source = SimpleNamespace(vertices=np.zeros((1, 3)),
                    visual=SimpleNamespace(kind='vertex', vertex_colors=values))
                np.testing.assert_array_equal(post.sample_source_basecolor(source), expected)

    def test_texture_and_vertex_linear_rgba_multiply_without_alpha_decode(self):
        source = texture_source([[[128, 128, 128, 128]]], [[0, 0]])
        vertex = np.array([[.25, .5, 1, .5]], 'f4')
        result = post.sample_source_basecolor(source, vertex_rgba=vertex)
        linear_gray = ((128/255 + .055)/1.055)**2.4
        np.testing.assert_allclose(result, [[linear_gray*.25, linear_gray*.5,
                                            linear_gray, (128/255)*.5]], atol=6e-8, rtol=0)

    def test_float_color_export_preserves_geometry_and_existing_normal_policy(self):
        for has_normals in (False, True):
            with self.subTest(has_normals=has_normals):
                mesh = mesh_with_colors([.123456789, .23456789, .3456789, .456789], has_normals)
                base_doc, base_binary = depth_mesh._decode(
                    mesh.export(file_type='glb', include_normals=has_normals))
                result_doc, result_binary = depth_mesh._decode(post.export_prepared_glb(mesh))
                before = primitive(base_doc)
                after = primitive(result_doc)
                self.assertEqual(base_binary, result_binary)
                self.assertEqual(result_doc['nodes'], base_doc['nodes'])
                self.assertEqual(result_doc.get('materials'), base_doc.get('materials'))
                for name in ('POSITION', 'NORMAL'):
                    self.assertEqual(name in before['attributes'], name in after['attributes'])
                    if name in before['attributes']:
                        np.testing.assert_array_equal(
                            accessor(base_doc, base_binary, before['attributes'][name]),
                            accessor(result_doc, result_binary, after['attributes'][name]))
                self.assertEqual('NORMAL' in after['attributes'], has_normals)
                self.assertEqual('vertex_normals' in mesh._cache, has_normals)
                np.testing.assert_array_equal(accessor(base_doc, base_binary, before['indices']),
                                              accessor(result_doc, result_binary, after['indices']))
                spec = result_doc['accessors'][after['attributes']['COLOR_0']]
                self.assertEqual((spec['componentType'], spec['type']), (5126, 'VEC4'))
                self.assertFalse(spec.get('normalized', False))
                self.assertNotIn(post.LINEAR_COLOR_ATTRIBUTE, after['attributes'])
                np.testing.assert_array_equal(
                    accessor(result_doc, result_binary, after['attributes']['COLOR_0']),
                    mesh.vertex_attributes[post.LINEAR_COLOR_ATTRIBUTE])

    def test_prepare_roundtrip_recovers_raw_float_source_colors_without_quantization(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            source = mesh_with_colors([.123456789, .333333343, .75, .23456789])
            source_path, raw_path = folder/'source.glb', folder/'raw.glb'
            post.export_prepared_glb(source, source_path)
            raw = trimesh.Trimesh(vertices=source.vertices*2, faces=source.faces, process=False)
            raw.export(raw_path, include_normals=False)
            prepared, report = post.prepare(source_path, raw_path)
            doc, binary = depth_mesh._decode(post.export_prepared_glb(prepared))
            attrs = primitive(doc)['attributes']
            self.assertEqual(report['color_transfer_version'], post.COLOR_TRANSFER_VERSION)
            self.assertNotIn('NORMAL', attrs)
            np.testing.assert_array_equal(accessor(doc, binary, attrs['COLOR_0']),
                                          source.vertex_attributes[post.LINEAR_COLOR_ATTRIBUTE])
            np.testing.assert_array_equal(accessor(doc, binary, attrs['POSITION']),
                                          source.vertices.astype('f4'))
            self.assertGreater(float(np.max(np.abs(
                accessor(doc, binary, attrs['COLOR_0'])*255 -
                np.rint(accessor(doc, binary, attrs['COLOR_0'])*255)))), .1)

    def test_prepare_preserves_exact_linear_material_factor_from_glb_json(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            source = trimesh.creation.icosphere(subdivisions=1)
            image = Image.fromarray(np.array([[[128, 128, 128, 128]]], dtype='u1'))
            material = trimesh.visual.material.PBRMaterial(baseColorTexture=image)
            source.visual = trimesh.visual.TextureVisuals(
                uv=np.zeros((len(source.vertices), 2)), material=material)
            document, binary = depth_mesh._decode(source.export(file_type='glb', include_normals=False))
            factor = [.314159265, .271828182, .618033989, .23456789]
            document['materials'][0]['pbrMetallicRoughness']['baseColorFactor'] = factor
            source_path, raw_path = folder/'source.glb', folder/'raw.glb'
            source_path.write_bytes(depth_mesh._encode(document, bytearray(binary)))
            raw = trimesh.Trimesh(vertices=source.vertices, faces=source.faces, process=False)
            raw.export(raw_path, include_normals=False)
            prepared, _ = post.prepare(source_path, raw_path)
            doc, binary = depth_mesh._decode(post.export_prepared_glb(prepared))
            rgba = accessor(doc, binary, primitive(doc)['attributes']['COLOR_0'])
            gray = ((128/255 + .055)/1.055)**2.4
            expected = np.array([gray, gray, gray, 128/255])*factor
            np.testing.assert_allclose(rgba, np.tile(expected, (len(rgba), 1)), atol=6e-8, rtol=0)


if __name__ == '__main__':
    unittest.main()
