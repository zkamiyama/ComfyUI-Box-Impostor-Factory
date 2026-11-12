from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from box_nodes import generation, nodes, store
from test_nodes import fixture


class CacheIdentityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.home = Path(temporary.name)
        self.root = self.home / 'box'
        self.root.mkdir()
        self.asset = 'b' * 64
        self.directory = self.root / self.asset
        self.manifest = fixture(self.directory, self.asset)
        self.enterContext(patch.object(store, 'root', return_value=self.root))

    def test_root_output_slot_invalidates_masks_and_generation_requests(self):
        graph = {'1': {'class_type': 'TwoMasks', 'inputs': {}},
                 '2': {'class_type': 'BIPrepareReference', 'inputs': {'mask': ['1', 0]}}}
        image = np.full((1, 8, 8, 3), .5, dtype=np.float32)
        mask = np.eye(8, dtype=np.float32)[None]
        prepare = nodes.PrepareReference()
        kwargs = dict(image=image, mask_mode='foreground', prompt=graph, unique_id='2')
        with patch.object(nodes, 'tensor', side_effect=lambda data: data):
            first = prepare.prepare(**kwargs, mask=mask)
            self.assertEqual(prepare.check_lazy_status(**kwargs), [])
            graph['2']['inputs']['mask'][1] = 1
            self.assertEqual(prepare.check_lazy_status(**kwargs), ['mask'])
            second = prepare.prepare(**kwargs, mask=1 - mask)
            self.assertNotEqual(first[2]['sha256'], second[2]['sha256'])
            np.testing.assert_array_equal(second[1], 1 - mask)
        graph['1']['class_type'] = 'TwoImages'
        graph['2']['inputs'] = {'image': ['1', 0]}
        args = (self.asset, first[2], '+X', 'qwen_enhanced', '{}', 42, 'rgba', graph, '2')
        output0 = generation.generation_request(*args)[0]
        graph['2']['inputs']['image'][1] = 1
        self.assertNotEqual(output0, generation.generation_request(*args)[0])

    def test_viewer_code_controls_render_cache_independently_of_bake(self):
        repo = self.home / 'repo'
        for name in ('web', 'worker'):
            shutil.copytree(store.REPO / name, repo / name)
        shutil.copyfile(store.REPO / 'package-lock.json', repo / 'package-lock.json')
        with patch.object(store, 'REPO', repo):
            baked = store.worker_version()
            rendered = store.presentation_version()
            for name in ('web/viewer.html', 'web/viewer.mjs'):
                with self.subTest(file=name):
                    with (repo / name).open('a') as out:
                        out.write('\n')
                    new = store.presentation_version()
                    self.assertNotEqual(rendered, new)
                    self.assertEqual(baked, store.worker_version())
                    rendered = new
            # The browser bake bridge also constructs its renderer via asset_io.mjs.
            with (repo / 'web/asset_io.mjs').open('a') as out:
                out.write('\n')
            self.assertNotEqual(store.worker_version(), baked)
            self.assertNotEqual(store.presentation_version(), rendered)

    def test_rendered_image_and_saved_preview_follow_renderer_revisions(self):
        def draw(command, *args):
            path = Path(args[args.index('--output') + 1])
            Image.new('RGBA', (64, 64), '#123456').save(path)
        old = self.directory / 'previews' / 'hero.png'
        old.parent.mkdir()
        Image.new('RGB', (8, 8)).save(old)
        with patch.object(store, 'presentation_version', return_value='first') as version, \
                patch.object(nodes, 'tensor', side_effect=lambda data: data), \
                patch.object(store, 'run_worker', side_effect=draw) as worker:
            render = nodes.RenderBox()
            for revision, expected_calls in [('first', 2), ('first', 2), ('new', 4)]:
                version.return_value = revision
                render.render(self.asset, 64, 64)
                self.assertEqual(store.thumbnail(self.directory, self.manifest), old)
                self.assertEqual(worker.call_count, expected_calls)
                with Image.open(old) as image:
                    self.assertEqual(image.size, (64, 64))
                    self.assertEqual(image.info['BoxPresentation'],
                                     store.key('thumbnail-v3', self.asset, 'standard', revision))
            previous_bytes = old.read_bytes()
            version.return_value = 'unavailable'
            worker.side_effect = RuntimeError('renderer unavailable')
            with self.assertWarnsRegex(RuntimeWarning, 'saved Box preview'):
                self.assertEqual(store.thumbnail(self.directory, self.manifest), old)
            self.assertEqual(old.read_bytes(), previous_bytes)
            with self.assertRaisesRegex(RuntimeError, 'renderer unavailable'):
                store.thumbnail(self.directory, self.manifest, 'reduce_bleeding')


if __name__ == '__main__':
    unittest.main()
