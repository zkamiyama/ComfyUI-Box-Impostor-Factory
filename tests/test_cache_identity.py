from pathlib import Path
import copy
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

    def test_renderer_configuration_invalidates_bake_and_render_caches(self):
        def versions():
            return store.worker_version(), store.presentation_version()

        with patch.dict('os.environ', BI_NODE='', BI_CHROMIUM='', BI_SOFTWARE_GPU='0'):
            original = versions()
            for name, value in [('BI_NODE', '/opt/windows-node-wrapper'),
                                ('BI_CHROMIUM', '/opt/custom-browser'), ('BI_SOFTWARE_GPU', '1')]:
                with self.subTest(name=name), patch.dict('os.environ', {name:value}):
                    for previous, changed in zip(original, versions()):
                        self.assertNotEqual(previous, changed)
        with patch.dict('os.environ', BI_NODE='/no-such-factory-node'):
            with self.assertRaisesRegex(RuntimeError, 'BI_NODE'):
                store.run_worker('--help')

    def test_renumbered_graph_adopts_legacy_mesh_without_conflating_changed_edges(self):
        n = lambda kind, **pins: {'class_type':kind, 'inputs':pins}
        first = {'5:10':n('KSampler', seed=42), '5:11':n('Mesh', a=['5:10',0], b=['5:10',0]),
                 '6':n('BICanonicalizeProxy', model_3d=['5:11',0])}
        second = {'9:99':n('Mesh', b=['9:98',0], a=['9:98',0]), '9:98':n('KSampler', seed=42),
                  '8':n('BICanonicalizeProxy', model_3d=['9:99',0])}
        ref = {'sha256':'a'*64}
        proxy = nodes.CanonicalizeProxy()
        old_hash, _, _ = generation.lineage(first,'6',('model_3d',),legacy=True)
        old_key = store.key('generated-glb-v1',ref['sha256'],old_hash)
        legacy = self.root/'.stages/native-mesh'/old_key
        legacy.mkdir(parents=True)
        (legacy/'source.glb').write_bytes(b'cached native geometry')
        self.assertEqual(proxy.check_lazy_status(reference=ref,prompt=first,unique_id='6'),[])
        self.assertEqual(proxy.check_lazy_status(reference=ref,prompt=second,unique_id='8'),[])
        key, meta = proxy.cached_request(ref,first,'6')
        self.assertEqual((key,meta),proxy.cached_request(ref,second,'8'))
        self.assertEqual((legacy.parent/key/'source.glb').read_bytes(), b'cached native geometry')
        for variation in ('slot','literal','duplicate','root_slot','socket'):
            changed = copy.deepcopy(second)
            pins = changed['9:99']['inputs']
            if variation == 'slot': pins['a'][1] = 1
            if variation == 'literal': changed['9:98']['inputs']['seed'] = 43
            if variation == 'duplicate':
                changed['other'] = copy.deepcopy(changed['9:98'])
                pins['b'][0] = 'other'
            if variation == 'root_slot': changed['8']['inputs']['model_3d'][1] = 1
            if variation == 'socket': pins['c'] = pins.pop('a')
            with self.subTest(variation=variation):
                self.assertEqual(proxy.check_lazy_status(reference=ref,prompt=changed,unique_id='8'),['model_3d'])

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
                for cls in (nodes.PreviewBox, nodes.RenderBox):
                    self.assertEqual(cls.IS_CHANGED(asset=None), revision)
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
