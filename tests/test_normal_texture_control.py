"""Small CPU tests for the explicit normalTexture control, without Comfy/GPU."""
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/run_normal_texture_control.py'
spec = importlib.util.spec_from_file_location('normal_texture_control', SCRIPT)
control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(control)


def example_glb():
    binary = struct.pack('<18f', 0, 0, 0, 1, 0, 0, 0, 1, 0,
                         0, 0, 1, 0, 0, 1, 0, 0, 1)
    document = {
        'asset': {'version': '2.0'}, 'scene': 0, 'scenes': [{'nodes': [0]}],
        'nodes': [{'mesh': 0}], 'buffers': [{'byteLength': len(binary)}],
        'bufferViews': [{'buffer': 0, 'byteOffset': 0, 'byteLength': 36},
                        {'buffer': 0, 'byteOffset': 36, 'byteLength': 36}],
        'accessors': [{'bufferView': 0, 'componentType': 5126, 'count': 3, 'type': 'VEC3'},
                      {'bufferView': 1, 'componentType': 5126, 'count': 3, 'type': 'VEC3'}],
        'meshes': [{'primitives': [{'attributes': {'POSITION': 0, 'NORMAL': 1}, 'material': 0}]}],
        'materials': [{'normalTexture': {'index': 1, 'scale': .7},
                       'pbrMetallicRoughness': {'baseColorTexture': {'index': 0},
                                                'roughnessFactor': .6},
                       'extensions': {'KHR_materials_clearcoat': {
                           'clearcoatNormalTexture': {'index': 2}}}}],
        'textures': [{'source': 0}, {'source': 1}, {'source': 2}],
        'images': [{'uri': 'data:image/png;base64,AAAA'}] * 3,
    }
    # Unknown chunks are legal to ignore; the control must preserve them too.
    return control.glb_bytes(document, [(control.JSON_CHUNK, b''),
                                        (control.BIN_CHUNK, binary), (0x12345678, b'abcd')])


class NormalTextureControlTests(unittest.TestCase):
    def test_only_requested_material_field_changes(self):
        original = example_glb()
        changed, proof = control.strip_normal_texture(original)
        self.assertTrue(proof['jsonOnlyNormalTextureRemoved'])
        self.assertTrue(proof['nonJsonChunksByteIdentical'])
        self.assertTrue(proof['geometrySignatureEqual'])
        document, chunks = control.read_glb(changed)
        self.assertNotIn('normalTexture', document['materials'][0])
        self.assertIn('clearcoatNormalTexture',
                      document['materials'][0]['extensions']['KHR_materials_clearcoat'])
        self.assertEqual(document['meshes'][0]['primitives'][0]['attributes']['NORMAL'], 1)
        self.assertEqual(chunks[-1], (0x12345678, b'abcd'))
        self.assertEqual(control.strip_normal_texture(original)[0], changed)

    def test_geometry_or_other_material_mutation_is_detected(self):
        original = example_glb()
        changed, _ = control.strip_normal_texture(original)
        document, chunks = control.read_glb(changed)
        binary = bytearray(chunks[1][1])
        struct.pack_into('<f', binary, 0, .01)
        altered = control.glb_bytes(document, [chunks[0], (control.BIN_CHUNK, bytes(binary)), chunks[2]])
        self.assertFalse(control.glb_comparison(original, altered)['geometrySignatureEqual'])
        document['materials'][0]['pbrMetallicRoughness']['roughnessFactor'] = 1
        altered = control.glb_bytes(document, chunks)
        self.assertFalse(control.glb_comparison(original, altered)['jsonOnlyNormalTextureRemoved'])

    def test_rejects_malformed_and_empty_ablation(self):
        original = example_glb()
        for data in (original[:10], original[:-1], b'xxxx' + original[4:]):
            with self.assertRaises(ValueError):
                control.strip_normal_texture(data)
        changed, _ = control.strip_normal_texture(original)
        with self.assertRaisesRegex(ValueError, 'no material.normalTexture'):
            control.strip_normal_texture(changed)

    def test_native_verification_rejects_color_or_depth_drift(self):
        with tempfile.TemporaryDirectory() as folder:
            before, after = Path(folder) / 'before', Path(folder) / 'after'
            before.mkdir()
            after.mkdir()
            original = example_glb()
            changed, _ = control.strip_normal_texture(original)
            (before / 'proxy.glb').write_bytes(original)
            (after / 'proxy.glb').write_bytes(changed)
            manifest = {'resolution': 1, 'proxy': 'proxy.glb', 'textures': {},
                        'baseColor': {'path': 'color.bin'}, 'faceOrder': ['+Z'],
                        'faces': {'+Z': {'guide': 'guide.png', 'depth': 'depth.bin'}}}
            for name, value in [('colorMap', b'color'), ('normalMap', struct.pack('<4e', .5, .5, 1, 1)),
                                ('intervalMap', b'interval'), ('rangeMap', b'range')]:
                filename = 'color.bin' if name == 'colorMap' else name + '.bin'
                manifest['textures'][name] = {'path': filename}
                if name == 'normalMap':
                    manifest['textures'][name]['dtype'] = 'float16'
                (before / filename).write_bytes(value)
                (after / filename).write_bytes(struct.pack('<4e', 1, .5, .5, 1) if name == 'normalMap' else value)
            for name, value in [('guide.png', b'guide'), ('depth.bin', b'depth')]:
                (before / name).write_bytes(value)
                (after / name).write_bytes(value)
            self.assertTrue(control.compare_assets(before, manifest, after, manifest)['passed'])
            (after / 'color.bin').write_bytes(b'wrong-color')
            result = control.compare_assets(before, manifest, after, manifest)
            self.assertFalse(result['passed'])
            self.assertFalse(result['checks']['colorMapEqual'])
            (after / 'color.bin').write_bytes(b'color')
            (after / 'depth.bin').write_bytes(b'wrong-depth')
            result = control.compare_assets(before, manifest, after, manifest)
            self.assertFalse(result['passed'])
            self.assertFalse(result['checks']['+ZdepthEqual'])
            (after / 'depth.bin').write_bytes(b'depth')
            (after / 'normalMap.bin').write_bytes(struct.pack('<4e', 1, .5, .5, .5))
            result = control.compare_assets(before, manifest, after, manifest)
            self.assertFalse(result['passed'])
            self.assertFalse(result['checks']['normalCoverageEqual'])

    def test_prepare_cache_does_not_depend_on_rgb_source(self):
        with tempfile.TemporaryDirectory() as folder:
            baseline, work = Path(folder) / 'baseline', Path(folder) / 'work'
            baseline.mkdir()
            data = example_glb()
            (baseline / 'proxy.glb').write_bytes(data)
            manifest = {'asset_id': 'a' * 64, 'proxy': 'proxy.glb',
                        'proxySha256': control.sha(data), 'resolution': 1024,
                        'backend': 'webgl2', 'maps': [], 'tightBounds': {},
                        'bakeBox': {}, 'transform': [], 'inverseTransform': []}
            (baseline / 'manifest.json').write_text(json.dumps(manifest))
            _, first, directory = control.prepare(baseline, work)
            _, second, repeated = control.prepare(baseline, work)
            self.assertEqual(first, second)
            self.assertEqual(directory, repeated)
            self.assertEqual(Path(first['proxy']).read_bytes(),
                             control.strip_normal_texture(data)[0])


if __name__ == '__main__':
    unittest.main()
