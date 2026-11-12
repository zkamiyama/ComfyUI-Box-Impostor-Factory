import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from box_nodes import faces, store
from box_nodes.revision import RebakeProxyRevision, compare_face
from test_nodes import fixture


class RevisionTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(patch.object(store, 'root', return_value=self.root))
        self.key = 'a' * 64
        self.baked_key = 'b' * 64
        self.old_root = self.root / self.key
        self.new_root = self.root / self.baked_key
        self.old = fixture(self.old_root, self.key)
        self.new = fixture(self.new_root, self.baked_key)
        corrected = self.new_root / 'geometry/proxy.glb'
        corrected.write_bytes(b'revised test geometry')
        self.new['proxySha256'] = store.file_hash(corrected)
        self.proxy = {'path': str(corrected), 'sha256': self.new['proxySha256']}
        (self.new_root / 'manifest.json').write_text(json.dumps(self.new))

    def test_original_units_with_independent_normalization_and_shifted_framing(self):
        transform = np.diag([2., 2., 2., 1.])
        transform[:3, 3] = [.5, 0., .125]
        self.new.update(transform=transform.flatten(order='F').tolist(),
                        bakeBox={'min': [-2.] * 3, 'max': [2.] * 3})
        # Revised image is displaced right by one pixel (0.25 NDC). Both sample planes
        # represent z=.25 in source units even though their encoded depths differ.
        shifted = np.eye(4)
        shifted[0, 3] = .25
        self.new['faces']['+Z']['projectionMatrix'] = (shifted @ np.linalg.inv(transform)).flatten(order='F').tolist()
        mask = np.zeros((8, 8, 4), np.uint8)
        mask[1:7, 2:8, 3] = 255
        Image.fromarray(mask).save(self.new_root / self.new['faces']['+Z']['guide'])
        depth_path = self.new_root / self.new['faces']['+Z']['depth']
        np.full((8, 8), .34375, '<f2').tofile(depth_path)
        compare = lambda: compare_face(self.old_root, self.old, self.new_root, self.new, '+Z')
        initial = compare()
        for direction in ('originalToRevised', 'revisedToOriginal'):
            result = initial[direction]
            self.assertEqual(result['sharedFrameIoU'], 1.)
            self.assertEqual(result['sharedFramePixels'], 56)
            self.assertEqual(result['depthSamples'], 36)
            self.assertEqual(result['maxAbsDepthSourceUnits'], 0.)
        np.full((8, 8), .40625, '<f2').tofile(depth_path)
        self.assertEqual(compare()['originalToRevised']['meanAbsDepthSourceUnits'], .125)

    def test_nonparallel_rays_and_missing_depth(self):
        del self.new['faces']['+Z']['depth']
        result = compare_face(self.old_root, self.old, self.new_root, self.new, '+Z')
        self.assertEqual(result['originalToRevised']['sharedFrameIoU'], 1.)
        self.assertIsNone(result['originalToRevised']['meanAbsDepthSourceUnits'])
        rotate = np.eye(4)
        rotate[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
        self.new['transform'] = rotate.flatten(order='F').tolist()
        result = compare_face(self.old_root, self.old, self.new_root, self.new, '+Z')
        self.assertEqual(result['status'], 'uncomparable')

    def test_new_asset_uses_only_rebaked_maps_and_input_resolution_and_is_cached(self):
        data = np.ones((8, 8, 4), np.float32)
        data[..., 3] = 0
        data[1:7, 1:7, 3] = 1
        manual = faces.compose(self.key, faces.align({'+Z': faces.save_view(data)}, self.key, .99, .1, .1))
        parent_map = store.validate(store.asset_dir(manual), manual)['textures']['colorMap']
        with patch.object(store, 'baseline', return_value=self.baked_key) as bake:
            revised, report = RebakeProxyRevision().revise(manual, self.proxy, 'webgl2')
            bake.assert_called_once_with(self.proxy, 8, 'webgl2', ())
            with patch('box_nodes.revision.compare_face', side_effect=AssertionError('unexpected repeat')):
                self.assertEqual(RebakeProxyRevision().revise(manual, self.proxy, 'webgl2'), (revised, report))
        result = store.validate(store.asset_dir(revised), revised)
        self.assertEqual(result['parentAsset'], manual)
        self.assertEqual(result['geometryAsset'], self.baked_key)
        self.assertEqual(result['textures'], self.new['textures'])
        self.assertNotEqual(result['textures']['colorMap'], parent_map)
        self.assertFalse(any(f.get('edit') for f in result['faces'].values()))
        self.assertEqual(set(json.loads(report)['guideComparisons']), set(store.FACES))
        with patch.object(store, 'baseline', side_effect=AssertionError('unexpected bake')):
            unchanged, reason = RebakeProxyRevision().revise(revised, self.proxy)
            self.assertEqual(unchanged, revised)
            self.assertEqual(json.loads(reason)['status'], 'unchanged-proxy')
