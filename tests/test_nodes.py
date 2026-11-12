import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from box_nodes import faces, nodes, store
from box_nodes.generation import CachedGeneratedFace


def fixture(directory, asset_id, r=8):
    directory.mkdir(parents=True)
    geo = directory / 'geometry'
    geo.mkdir()
    (geo / 'proxy.glb').write_bytes(b'local proxy')
    map_dir = directory / 'maps'
    map_dir.mkdir()
    tex = {}
    for name in store.REQUIRED_MAPS:
        wh = ((3, 1) if name == 'rangeMap' else (3*r, r if name == 'intervalMap' else 2*r))
        wh = tuple(map(int, wh))
        a = np.zeros((wh[1], wh[0], 4), dtype='<f2')
        a[..., 3] = .5
        a[..., :3] = .125
        path = f'maps/{name}.bin'
        a.tofile(directory / path)
        f = 'nearest' if name == 'rangeMap' else 'linear'
        tex[name] = dict(path=path, width=wh[0], height=wh[1], channels=4, dtype='float16', endian='little', rowOrder='increasingShaderV', rowBytes=wh[0]*8, byteLength=a.nbytes, sha256=store.file_hash(directory / path), minFilter=f, magFilter=f, mipmaps=False, wrap='clamp-to-edge')
    bounds = dict(min=[-.5,-.5,-.5], max=[.5,.5,.5])
    metadata = {}
    (directory / 'guides').mkdir()
    for fid, slug in faces.SLUGS.items():
        path = f'guides/{slug}.png'
        mask = np.zeros((r,r,4), dtype=np.uint8)
        mask[1:-1,1:-1,3] = 255
        Image.fromarray(mask).save(directory / path)
        depth = f'guides/{slug}.depth16f.bin'
        np.full((r,r), .25, dtype='<f2').tofile(directory / depth)
        i = store.FACES.index(fid)
        metadata[fid] = dict(guide=path, depth=depth, depthEncoding='linear-axis-0-near-1-far;nan-background', imageRowOrder='top-to-bottom', up=([0,1,0] if i//2!=1 else [0,0,(-1)**(i+1)]), side=(np.eye(3,dtype=int)[i//2] * (-1)**i).tolist(), tile=[i%3, i//3], viewMatrix=np.eye(4).flatten().tolist(),projectionMatrix=np.eye(4).flatten().tolist())
    m = dict(schemaVersion=1, asset_id=asset_id, layout='box', resolution=r, proxy='geometry/proxy.glb', proxySha256=store.file_hash(geo / 'proxy.glb'), transform=np.eye(4).flatten().tolist(),inverseTransform=np.eye(4).flatten().tolist(), tightBounds=bounds, bakeBox=bounds, faceOrder=list(store.FACES), faces=metadata, maps=[], textures=tex, baseColor=tex['colorMap'])
    (directory / 'manifest.json').write_text(json.dumps(m))
    return m


class NodeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name) / 'box_impostor'
        self.base.mkdir()
        mock = patch.object(store, 'root', return_value=self.base)
        self.addCleanup(mock.stop)
        mock.start()
        self.asset = 'a' * 64
        self.m = fixture(self.base / self.asset, self.asset)

    def test_render_box_image_mask_cache_and_parameters(self):
        calls = []
        def fake_worker(*args):
            calls.append(args)
            image = np.zeros((80, 96, 4), dtype=np.uint8)
            image[12:66, 20:80, :3] = (40, 120, 220)
            image[12:66, 20:80, 3] = 255
            Image.fromarray(image, mode='RGBA').save(args[args.index('--output') + 1])
        with patch.object(store, 'run_worker', side_effect=fake_worker), patch.object(store, 'worker_version', return_value='unit-test-render'):
            rendered, mask = nodes.RenderBox().render(self.asset, 96, 80, '+Z',
                                                       'orthographic', 'reduce_bleeding', 'webgl2')
            self.assertEqual(tuple(rendered.shape), (1, 80, 96, 3))
            self.assertEqual(tuple(mask.shape), (1, 80, 96))
            self.assertEqual(float(mask[0, 0, 0]), 1.0)
            self.assertEqual(float(mask[0, 20, 30]), 0.0)
            self.assertAlmostEqual(float(rendered[0, 20, 30, 2]), 220 / 255, places=5)
            nodes.RenderBox().render(self.asset, 96, 80, '+Z',
                                     'orthographic', 'reduce_bleeding', 'webgl2')
            self.assertEqual(len(calls), 1, 'Identical render must reuse the stored PNG')
            nodes.RenderBox().render(self.asset, 96, 80, '+Z',
                                     'orthographic', 'standard', 'webgl2')
            self.assertEqual(len(calls), 2, 'Sampling choice must be part of the cache key')
            self.assertIn('--sampling', calls[0])
            self.assertIn('reduce_bleeding', calls[0])
        with self.assertRaises(ValueError):
            nodes.RenderBox().render(self.asset, 8, 80)
        with self.assertRaises(ValueError):
            nodes.RenderBox().render(self.asset, 96, 80, sampling='invalid')

    def test_foreground_conventions_and_placeholder(self):
        image = np.zeros((1,8,8,3), dtype=np.float32)
        mask = np.ones((1,8,8), dtype=np.float32)
        mask[:,2:6,1:5] = 0
        rgba = faces.rgba(image, 'load_image', mask)
        self.assertEqual(rgba[..., 3].sum(), 16)
        self.assertEqual(rgba[..., :3].sum(), 0)
        with self.assertRaises(ValueError):
            faces.rgba(image, 'load_image', np.zeros((1,64,64)))
        with self.assertRaises(ValueError):
            faces.rgba(image, 'rgba')
        self.assertEqual(faces.rgba(image, 'opaque')[..., 3].sum(), 64)

    def test_manual_compose_preserves_bits_other_faces_and_caches(self):
        a = np.zeros((8,8,4), dtype=np.float32)
        a[1:-1,1:-1,3] = 1
        a[1:4,1:-1,0] = 1
        face = faces.save_view(a)
        aligned = faces.align({'+X':face}, self.asset, .9, .1, .1)
        asset = faces.compose(self.asset, aligned)
        m = store.validate(store.asset_dir(asset), asset)
        self.assertEqual(m['textures']['normalMap'], self.m['textures']['normalMap'])
        src = np.fromfile(store.asset_dir(asset) / m['baseColor']['path'], '<f2').reshape(16,24,4)
        dst = np.fromfile(store.asset_dir(asset) / m['textures']['colorMap']['path'], '<f2').reshape(16,24,4)
        np.testing.assert_array_equal(src[..., 3].view('u2'), dst[..., 3].view('u2'))
        self.assertEqual(dst[6,3,0], .5)
        self.assertEqual(dst[1,3,0], 0.)
        np.testing.assert_array_equal(src[:,8:,:],dst[:,8:,:])
        np.testing.assert_array_equal(src[8:,:,:],dst[8:,:,:])
        with patch.object(faces,'align_one', side_effect=RuntimeError('cache miss')), patch.object(store, 'run_worker', side_effect=RuntimeError('rebake')):
            self.assertEqual(faces.compose(self.asset, faces.align({'+X':face}, self.asset,.9,.1,.1)), asset)
        self.assertEqual(nodes.LoadBox().load(asset), (asset,))

    def test_replace_one_saved_face_keeps_other_ai_face_and_baseline(self):
        image = np.zeros((8,8,4), dtype=np.float32)
        image[1:7,1:7,3] = 1.
        image[1:7,1:7,0] = 1.
        a = {**faces.save_view(image), 'provenance': {'backend': 'qwen_enhanced'}}
        image[..., :3] = [0,1,0]
        b = {**faces.save_view(image), 'provenance': {'backend': 'qwen_enhanced'}}
        replace = nodes.ReplaceFaces().replace
        parent, = replace(self.asset, {'+X': a, '-Z': b}, .95, .1, .1)
        image[..., :3] = [0,0,1]
        c = faces.save_view(image)
        with patch.object(store, 'run_worker', side_effect=RuntimeError('unnecessary worker')):
            child, = replace(parent, {'+X': c}, .95, .1, .1)
        old, new = [store.validate(store.asset_dir(k), k) for k in (parent, child)]
        self.assertEqual(new['parentAsset'], parent)
        self.assertEqual(new['geometryAsset'], self.asset)
        self.assertEqual(old['baseColor'], new['baseColor'])
        self.assertEqual(old['faces']['-Z']['edit'], new['faces']['-Z']['edit'])
        atlas = lambda k,m: np.fromfile(store.asset_dir(k) / m['textures']['colorMap']['path'], '<f2').reshape(16,24,4)
        x,y = atlas(parent,old), atlas(child,new)
        self.assertFalse(np.array_equal(faces.tile_view(x,'+X'), faces.tile_view(y,'+X')))
        for fid in store.FACES[1:]:
            np.testing.assert_array_equal(faces.tile_view(x,fid).view('u2'), faces.tile_view(y,fid).view('u2'))
        np.testing.assert_array_equal(x[...,3].view('u2'), y[...,3].view('u2'))
        fallback, = replace(child, {'+X': faces.save_view(np.zeros_like(image))}, .95, .1, .1)
        m = store.validate(store.asset_dir(fallback),fallback)
        z = atlas(fallback,m)
        np.testing.assert_array_equal(faces.tile_view(z,'+X'), faces.tile_view(atlas(self.asset,self.m),'+X'))
        np.testing.assert_array_equal(faces.tile_view(z,'-Z'), faces.tile_view(x,'-Z'))
        self.assertEqual(m['faces']['+X']['edit']['status'],'fallback')

    def test_threshold_changes_reuse_registration(self):
        image = np.zeros((8,8,4), dtype=np.float32); image[1:7,1:7,3] = 1.
        item = faces.save_view(image)
        a = faces.align({'+X': item}, self.asset, .85, .1, .15)
        with patch.object(faces, 'align_one', side_effect=RuntimeError('unnecessary registration')):
            b = faces.align({'+X': item}, self.asset,.95,.2,.2)
        self.assertEqual(a['faces']['+X']['measured'], b['faces']['+X']['measured'])
        self.assertNotEqual(a['faces']['+X']['key'], b['faces']['+X']['key'])

    def test_foreground_registration_and_fallback(self):
        guide = np.zeros((32,32), np.float32)
        guide[6:24,6:24] = 1
        a = np.zeros((32,32,4),np.float32)
        a[8:26,6:24,3]=1
        candidate, report = faces.align_one(a, guide, .95, .2, .1)
        self.assertEqual(report['status'], 'accepted')
        self.assertGreater(report['after']['iou'], report['before']['iou'])
        self.assertIsNone(faces.align_one(a,guide,.95,.01,.1)[0])
        self.assertIsNone(faces.align_one(a*0,guide,.95,.2,.1)[0])

    def test_load_save_and_corruption(self):
        node=nodes.SaveBox()
        exported=Path(node.save(self.asset,'sample')[0])
        self.assertEqual(exported,Path(self.tmp.name)/'box_impostor_exports/sample/manifest.json')
        self.assertEqual(nodes.LoadBox().load('sample'),(self.asset,))
        self.assertEqual(nodes.LoadBox().load(str(exported)), (self.asset,))
        with self.assertRaises(ValueError):
            node.save(self.asset,'../escape')
        path=self.base/self.asset/self.m['textures']['rangeMap']['path']
        path.write_bytes(b'x'*path.stat().st_size)
        with self.assertRaisesRegex(ValueError,'digest'):
            nodes.LoadBox().load(self.asset)

    def test_lazy_model_hit_and_prompt_change(self):
        graph={'1':{'class_type':'KSampler','inputs':{'seed':42,'steps':6,'sampler_name':'euler'}},'2':{'class_type':'VAEDecode','inputs':{'samples':['1',0]}},'3':{'class_type':'BICachedGeneratedFace','inputs':{'image':['2',0]}}}
        node=CachedGeneratedFace()
        kwargs=dict(guides=self.asset,reference={'sha256':'f'*64},face_id='+X',backend='qwen_enhanced',recipe='{}',seed=42,mask_mode='rgba',prompt=graph,unique_id='3')
        self.assertEqual(node.check_lazy_status(**kwargs), ['image'])
        ref=node.resolve(**kwargs,image=np.ones((1,8,8,4),np.float32))[0]
        self.assertEqual(CachedGeneratedFace().check_lazy_status(**kwargs),[])
        self.assertEqual(node.resolve(**kwargs)[0],ref)
        graph['1']['inputs']['steps']=20
        self.assertEqual(node.check_lazy_status(**kwargs),['image'])

    def test_reference_and_connected_mesh_skip_cached_upstream(self):
        mat = nodes.PrepareReference()
        rgb = np.ones((1,8,8,3), np.float32)
        mask = np.ones((1,8,8), np.float32)
        graph = {'1': {'class_type': 'MaskFromFixture', 'inputs': {}}, '2': {'class_type': 'BIPrepareReference', 'inputs': {'mask':['1',0]}}, '3': {'class_type': 'NativeMeshFromFixture', 'inputs': {'seed':42}}, '4': {'class_type': 'BICanonicalizeProxy', 'inputs': {'model_3d':['3',0], 'reference':['2',2]}}}
        with patch.object(nodes, 'tensor', side_effect=lambda a: a):
            common = dict(image=rgb, mask_mode='foreground', prompt=graph, unique_id='2')
            self.assertEqual(mat.check_lazy_status(**common), ['mask'])
            ref = mat.prepare(**common, mask=mask)[2]
            self.assertEqual(mat.check_lazy_status(**common), [])
            self.assertEqual(mat.prepare(**common)[2], ref)
        class NativeGLB:
            format = 'glb'
            def get_bytes(self):
                return b'fixture-bytes'
        def canonicalize(command, *args):
            self.assertEqual(command, 'canonicalize')
            stage = Path(args[args.index('--output') + 1]); stage.mkdir()
            glb = stage / 'canonical.glb'; glb.write_bytes(b'canonical fixture')
            (stage / 'canonical.json').write_text(json.dumps({'glbSha256':store.file_hash(glb)}))
        node = nodes.CanonicalizeProxy()
        common = dict(reference=ref, prompt=graph, unique_id='4')
        self.assertEqual(node.check_lazy_status(**common), ['model_3d'])
        with patch.object(store,'run_worker',side_effect=canonicalize) as run:
            proxy = node.canonicalize(**common,model_3d=NativeGLB())[0]
            self.assertEqual(node.check_lazy_status(**common), [])
            self.assertEqual(nodes.CanonicalizeProxy().canonicalize(**common)[0],proxy)
            self.assertEqual(run.call_count, 1)
        graph['3']['inputs']['seed'] = 100
        self.assertEqual(node.check_lazy_status(**common), ['model_3d'])

    def test_confinement(self):
        for bad in ('../proxy.glb','/etc/passwd','C:/Windows/x','a/../b','a\\b','./a'):
            with self.subTest(bad=bad),self.assertRaises(ValueError):
                store.contained(self.base,bad)


if __name__ == '__main__':
    unittest.main()
