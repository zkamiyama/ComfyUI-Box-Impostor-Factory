import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'run_geometry_ab.py'
spec = importlib.util.spec_from_file_location('run_geometry_ab', SCRIPT)
ab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ab)


class GeometryABTests(unittest.TestCase):
    def test_structure_maps_keep_depth_and_normal_rows_registered(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            size = 8
            alpha = np.zeros((size, size), dtype='u1')
            alpha[1:-1, 1:-1] = 255
            guide = np.full((size, size, 4), 128, dtype='u1')
            guide[..., 3] = alpha
            Image.fromarray(guide).save(root / 'guide.png')
            Image.fromarray(np.full((8, 4, 4), 255, dtype='u1')).save(root / 'source.png')
            depth = np.broadcast_to(np.linspace(.1, .8, size, dtype='f4')[:, None], (size, size)).copy()
            depth[alpha == 0] = np.nan
            depth.astype('<f2').tofile(root / 'depth.bin')
            normal = np.zeros((size, size, 3), dtype='f4')
            normal[..., 0] = np.arange(size)[:, None] / 8
            normal[..., 1:] = .5
            atlas = np.zeros((size * 2, size * 3, 4), dtype='<f2')
            atlas[size:, size:2*size, :3] = normal[::-1]
            atlas.tofile(root / 'normal.bin')
            manifest = {
                'resolution': size, 'bakeBox': {'min': [-.5, -.5, -.5], 'max': [.5, .5, .5]},
                'faces': {'+Z': {'guide': 'guide.png', 'depth': 'depth.bin',
                    'depthEncoding': 'linear-axis-0-near-1-far;nan-background'}},
                'reference': {'image': 'source.png'},
                'textures': {'normalMap': {'path': 'normal.bin', 'width': size*3, 'height': size*2}},
            }
            images, metadata = ab.structure_images(root, manifest, '+Z', size)
            self.assertEqual(metadata['guideRect'], [0, 0, size, size])
            self.assertEqual(metadata['sourceRect'], [2, 0, 4, 8])
            np.testing.assert_allclose(images['normals'][1:-1, 1:-1], normal[1:-1, 1:-1])
            expected = 1 - depth.astype('<f2').astype('f4')
            np.testing.assert_allclose(images['axis-depth'][1:-1, 1:-1, 0], expected[1:-1, 1:-1])
            np.testing.assert_array_equal(images['mask'][..., 0], alpha / 255)
            np.testing.assert_allclose(images['normals'][0], .5)
            self.assertGreater(images['axis-depth'][1, 2, 0], images['axis-depth'][-2, 2, 0])

    def recipe(self, root):
        folder = root / 'input'
        folder.mkdir()
        pictures = {}
        for index, role in enumerate(['guide', 'source', 'axis-depth', 'normals', 'mask']):
            path = folder / (role + '.png')
            Image.fromarray(np.full((8, 8, 3), index * 40, dtype='u1')).save(path)
            pictures[role] = {'path': str(path), 'image': path.name, 'sha256': ab.digest(path)}
        return {'branch': 'A', 'faces': {'+Z': {'pictures': pictures}},
                'pictureOrder': list(pictures), 'models': dict(ab.MODELS),
                'prompt': ab.PROMPT, 'seeds': {'+Z': 105},
                'sampler': {'steps': 6, 'cfg': 1, 'sampler_name': 'euler', 'scheduler': 'simple', 'denoise': 1}}

    def test_saved_result_must_match_current_prompt_and_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recipe = self.recipe(root)
            case = root / 'A' / 'pz'
            case.mkdir(parents=True)
            for name in ['generated.png', 'foreground.png']:
                Image.fromarray(np.full((8, 8, 3), 128, dtype='u1')).save(case / name)
            ab.write_json(case / 'prompt.json', ab.build_prompt(recipe, '+Z'))
            ab.write_json(case / 'result.json', {
                'promptSha256': ab.digest(case / 'prompt.json'),
                'rawSha256': ab.digest(case / 'generated.png'),
                'maskSha256': ab.digest(case / 'foreground.png'),
            })
            with patch.object(ab, 'request', side_effect=AssertionError('must not infer')):
                ab.run_face(root, recipe, '+Z', 'http://unused')
                recipe['prompt'] += ' Different view.'
                with self.assertRaisesRegex(ValueError, 'different inputs or settings'):
                    ab.run_face(root, recipe, '+Z', 'http://unused')

    def test_graph_uses_recipe_models_and_five_real_reference_sockets(self):
        with tempfile.TemporaryDirectory() as tmp:
            recipe = self.recipe(Path(tmp))
            recipe['models']['model'] = 'changed-model.safetensors'
            recipe['seeds']['+Z'] = 321
            prompt = ab.build_prompt(recipe, '+Z')
            self.assertEqual(prompt['6']['inputs']['unet_name'], 'changed-model.safetensors')
            self.assertEqual(prompt['12']['inputs']['seed'], 321)
            refs = prompt['10']['inputs']
            self.assertEqual([refs['images.image_' + str(n)] for n in range(1, 6)],
                             [[str(n), 0] for n in range(1, 6)])
            self.assertEqual((refs['width'], refs['height']), (1024, 1024))


if __name__ == '__main__':
    unittest.main()
