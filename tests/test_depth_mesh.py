import json
import unittest

import numpy as np

from box_nodes.depth_mesh import _accessor, _decode, _encode, deform_glb


def fixture():
    # Two non-indexed triangles have duplicated positions at a UV seam. Normals,
    # tangents and UVs share a strided bufferView, as glTF allows.
    xyz = np.array([[0,0,0],[1,0,0],[0,1,0], [1,0,0],[1,1,0],[0,1,0]], dtype='<f4')
    normal = np.tile([0,0,1], (len(xyz),1))
    tangent = np.tile([1,0,0,-1], (len(xyz),1))
    uv = np.arange(12, dtype='<f4').reshape(6,2) / 12
    packed = np.column_stack((xyz, normal, tangent, uv)).astype('<f4')
    material = {'pbrMetallicRoughness':{'baseColorTexture':{'index':0}, 'roughnessFactor':.32}}
    return _encode({
        'asset': {'version':'2.0'}, 'scene':0, 'scenes':[{'nodes':[0]}],
        'nodes':[{'translation':[3,4,5], 'scale':[2,3,1], 'children':[1]}, {'mesh':0}],
        'meshes':[{'primitives':[{'attributes':{'POSITION':0,'NORMAL':1,'TANGENT':2,'TEXCOORD_0':3},'material':0}]}],
        'buffers':[{}], 'bufferViews':[{'buffer':0,'byteLength':packed.nbytes,'byteStride':48},
                                       {'buffer':0,'byteOffset':packed.nbytes,'byteLength':4}],
        'accessors':[{'bufferView':0,'byteOffset':offset,'count':6,'componentType':5126,'type':kind}
                     for offset,kind in [(0,'VEC3'),(12,'VEC3'),(24,'VEC4'),(40,'VEC2')]],
        'materials':[material],'textures':[{'source':0}],'images':[{'bufferView':1,'mimeType':'image/png'}]
    }, bytearray(packed.tobytes() + b'test'))


class DeformMeshTests(unittest.TestCase):
    def test_world_space_welds_seams_preserves_other_views_and_transports_shading(self):
        data = fixture()
        seen = []
        def warp(world):
            seen.append(world)
            offset = np.zeros_like(world)
            offset[:, 2] = .5 * (world[:, 0]-3)
            return offset
        warped, stats = deform_glb(data, warp)
        self.assertEqual(len(seen),1)
        np.testing.assert_array_equal(seen[0].min(0), [3,4,5])
        np.testing.assert_array_equal(seen[0].max(0), [5,7,5])
        self.assertEqual((stats['uniquePositions'],stats['vertices'],stats['triangles']), (4,6,2))
        self.assertEqual(stats['stepScale'],1)
        original, baseline = _decode(data)
        doc, binary = _decode(warped)
        self.assertTrue(binary.startswith(baseline))
        self.assertEqual(doc['bufferViews'][:2],original['bufferViews'])
        for name in ['nodes','materials','textures','images','scenes']:
            self.assertEqual(doc[name],original[name])
        attrs = doc['meshes'][0]['primitives'][0]['attributes']
        self.assertEqual(attrs['TEXCOORD_0'],3)
        xyz = _accessor(doc,binary,attrs['POSITION'],'VEC3',{5126})
        np.testing.assert_array_equal(xyz[1],xyz[3])
        np.testing.assert_array_equal(xyz[2],xyz[5])
        np.testing.assert_allclose(xyz[:,2],xyz[:,0])
        self.assertEqual(doc['accessors'][attrs['POSITION']]['max'],[1,1,1])
        normal = _accessor(doc,binary,attrs['NORMAL'],'VEC3',{5126})
        np.testing.assert_allclose(normal[0],np.array([-1,0,1])/np.sqrt(2),atol=1e-6)
        tangent = _accessor(doc,binary,attrs['TANGENT'],'VEC4',{5126})
        np.testing.assert_allclose(tangent[0,:3],np.array([1,0,1])/np.sqrt(2),atol=1e-6)
        np.testing.assert_array_equal(tangent[:,3],-1)
        np.testing.assert_allclose(np.sum(normal*tangent[:,:3],axis=1),0,atol=1e-6)

    def test_identity_foldover_backtracking_and_invalid_shapes(self):
        data = fixture()
        same, stats = deform_glb(data,np.zeros_like)
        self.assertEqual(same,data)
        self.assertEqual(stats['maxDisplacement'],0)
        def reverse(world):
            out = np.zeros_like(world)
            out[:,0] = -2*(world[:,0]-3)
            return out
        _, stats = deform_glb(data,reverse)
        self.assertEqual(stats['stepScale'],.25)
        for incorrect in (lambda w: np.zeros(3), lambda w: w*np.nan):
            with self.assertRaisesRegex(ValueError,'finite world-space'):
                deform_glb(data, incorrect)

    def test_refuses_unsupported_mesh_features_without_modifying_input(self):
        initial, binary = _decode(fixture())
        for feature,edit in [('morph',lambda d:d['meshes'][0]['primitives'][0].update(targets=[{}])),
                        ('skinned',lambda d:d.update(skins=[{}])),
                        ('quantized',lambda d:d.update(extensionsUsed=['KHR_mesh_quantization'])),
                        ('instanced',lambda d:d['nodes'].append({'mesh':0})),
                        ('sparse',lambda d:d['accessors'][0].update(sparse={'count':1}))]:
            with self.subTest(feature=feature):
                doc = json.loads(json.dumps(initial))
                edit(doc)
                with self.assertRaises(ValueError):
                    deform_glb(_encode(doc, bytearray(binary)),np.zeros_like)
