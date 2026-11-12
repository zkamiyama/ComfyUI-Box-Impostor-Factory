import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from scripts import run_lightweight_smoke as smoke


class LightweightSmokeTests(unittest.TestCase):
    def test_native_graph_samples_all_guides_and_submits_both_box_renders(self):
        graph = smoke.build_graph('fixture.glb', 'sd.safetensors', 256, 4, 101, .6, 'webgl2',
                                  positive_prompt='painted lantern', material_maps='material+emissive')
        self.assertEqual(graph['4']['inputs']['text'], 'painted lantern')
        for index in ('2', '6', '9'):
            self.assertEqual(graph[index]['inputs']['material_maps'], 'material+emissive')
        for i, (name, socket) in enumerate(zip(smoke.FACES, smoke.SOCKETS)):
            capture = graph[graph['7']['inputs'][socket][0]]['inputs']
            decoder = graph[capture['image'][0]]
            sampler = graph[decoder['inputs']['samples'][0]]
            encoder = graph[sampler['inputs']['latent_image'][0]]
            split = graph[capture['mask'][0]]
            self.assertEqual([decoder['class_type'], sampler['class_type'], encoder['class_type'], split['class_type']],
                             ['VAEDecode', 'KSampler', 'VAEEncode', 'SplitImageWithAlpha'])
            self.assertEqual((encoder['inputs']['pixels'], capture['mask_mode'], capture['mask'][1]),
                             ([capture['mask'][0], 0], 'load_image', 1))
            self.assertEqual(split['inputs']['image'], ['2', i])
            self.assertEqual(sampler['inputs']['seed'], 101 + i)
            self.assertEqual(sampler['inputs']['model'], ['3', 0])
            provenance = json.loads(capture['provenance'])
            self.assertEqual(provenance['positive_prompt'], 'painted lantern')
            self.assertEqual((provenance['face'], provenance['checkpoint'], provenance['backend']), (name, 'sd.safetensors', 'manual_faces'))
        for index, asset in [('13', '6'), ('16', '9')]:
            join = graph[graph[index]['inputs']['images'][0]]
            render = graph[join['inputs']['image'][0]]
            self.assertEqual(render['class_type'], 'BIRenderBox')
            self.assertEqual(render['inputs']['asset'], [asset, 0])
            self.assertEqual(join['inputs']['alpha'], [join['inputs']['image'][0], 1])

    def test_preflight_reads_only_glb_header_and_finds_both_checkpoint_schemas(self):
        graph = smoke.build_graph('props/one.glb', 'test.safetensors', 256, 4, 0, .6, 'webgl2')
        for spec in [(['test.safetensors'],), ('COMBO', {'options': ['test.safetensors']})]:
            registry = {item['class_type']: {} for item in graph.values()}
            registry['CheckpointLoaderSimple'] = {'input': {'required': {'ckpt_name': spec}}}
            def response(origin, path, *, limit=None):
                if path == '/object_info':
                    return json.dumps(registry).encode()
                if path == '/system_stats':
                    return b'{"devices":[]}'
                self.assertEqual(path, '/view?filename=one.glb&subfolder=props&type=input')
                self.assertEqual(limit, 12)
                return b'glTF' + (2).to_bytes(4, 'little') + (128).to_bytes(4, 'little')
            with patch.object(smoke, 'request', side_effect=response):
                self.assertEqual(smoke.preflight('http://localhost', graph, 'test.safetensors'), [])
        with patch.object(smoke, 'request', side_effect=[json.dumps(registry).encode(), b'wrong file']):
            with self.assertRaisesRegex(ValueError, 'GLB v2'):
                smoke.preflight('http://localhost', graph, 'test.safetensors')

    def test_all_six_tiles_must_change_inside_original_coverage(self):
        r = 2
        before = np.zeros((r * 2, r * 3, 4), dtype='<f2')
        before[::2, ::2, 3] = 1
        after = before.copy()
        after[..., 0] = .5
        counts = smoke.changed_faces(before.tobytes(), after.tobytes(), r)
        self.assertEqual(counts, dict.fromkeys(smoke.FACES, 1))
        after[2, 4, 0] = 0
        with self.assertRaisesRegex(ValueError, 'face -Z'):
            smoke.changed_faces(before.tobytes(), after.tobytes(), r)
        after[0, 1, 3] = .75
        with self.assertRaisesRegex(ValueError, 'coverage changed'):
            smoke.changed_faces(before.tobytes(), after.tobytes(), r)

    def test_proof_reads_generated_png_and_rejects_unchanged_final_render(self):
        r = 4
        baseline_id, final_id = 'a' * 64, 'b' * 64
        base_dir = f'/box-impostor/assets/{baseline_id}/'
        final_dir = f'/box-impostor/assets/{final_id}/'
        baseline = np.zeros((r*2, r*3, 4), dtype='<f2')
        baseline[..., 3] = 1
        atlas = baseline.copy()
        atlas[..., 2] = .75
        media = {}
        def descriptor(directory, name, data):
            media[directory + name] = data
            return dict(path=name, sha256=hashlib.sha256(data).hexdigest())
        maps = {}
        for name in ('normalMap', 'intervalMap', 'rangeMap', 'materialMap', 'emissiveMap'):
            maps[name] = descriptor(base_dir, name, name.encode())
            media[final_dir + name] = name.encode()
        base = dict(maps=['roughness', 'metalness', 'ao', 'emissive'], textures=dict(maps, colorMap=descriptor(base_dir, 'color', baseline.tobytes())))
        faces = {}
        for i, (face, slug) in enumerate(zip(smoke.FACES, smoke.SOCKETS)):
            faces[face] = dict(edit=dict(status='accepted', view=slug + '.png', provenance=dict(
                generator=smoke.GENERATOR, seed=i, face=face, checkpoint='model', positive_prompt='painted lantern')))
            media[final_dir + slug + '.png'] = b'png'
        final = dict(maps=list(base['maps']), parentAsset=baseline_id, faceOrder=smoke.FACES, resolution=r, faces=faces,
                     textures=dict(maps, colorMap=descriptor(final_dir, 'color', atlas.tobytes())))
        media[base_dir + 'manifest.json'] = json.dumps(base).encode()
        media[final_dir + 'manifest.json'] = json.dumps(final).encode()
        preview = '/box-impostor/web/viewer.html?manifest=' + final_dir + 'manifest.json'
        outputs = {'10': dict(box_impostor=[preview])}
        frames = []
        for n, color, node_id in [('baseline', 0, '13'), ('generated', 128, '16')]:
            rgba = np.full((r,r,4), 255, np.uint8)
            rgba[..., 0] = color
            buffer = io.BytesIO()
            Image.fromarray(rgba).save(buffer, 'PNG')
            outputs[node_id] = dict(images=[dict(filename=n, type='output', subfolder='')])
            path = '/view?filename=' + n + '&type=output&subfolder='
            media[path] = buffer.getvalue()
            frames.append(path)
        args = SimpleNamespace(resolution=r, checkpoint='model', seed=0, positive_prompt='painted lantern',
                               material_maps='material+emissive')
        with tempfile.TemporaryDirectory() as folder, patch.object(smoke, 'request', side_effect=lambda origin, path: media[path]):
            proof = smoke.verify('http://localhost', outputs, Path(folder), args)
            self.assertEqual(proof['renderedPixelsChanged'], r*r)
            self.assertEqual(set(proof['materialSha256']), {'emissiveMap', 'materialMap'})
            media[final_dir + 'emissiveMap'] = b'corrupted'
            with self.assertRaisesRegex(ValueError, 'Digest mismatch: emissiveMap'):
                smoke.verify('http://localhost', outputs, Path(folder), args)
            media[final_dir + 'emissiveMap'] = media[base_dir + 'emissiveMap']
            for manifest, directory in ((base, base_dir), (final, final_dir)):
                manifest['maps'].remove('emissive')
                del manifest['textures']['emissiveMap']
                media[directory + 'manifest.json'] = json.dumps(manifest).encode()
            proof = smoke.verify('http://localhost', outputs, Path(folder), args)
            self.assertEqual(set(proof['materialSha256']), {'materialMap'})
            media[frames[1]] = media[frames[0]]
            with self.assertRaisesRegex(ValueError, 'final Box render'):
                smoke.verify('http://localhost', outputs, Path(folder), args)


if __name__ == '__main__':
    unittest.main()
