import subprocess
import tempfile
import unittest
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from box_nodes import store
from box_nodes.splat_cache import CachedTripoSplat, FIELDS, decoder_request, load_decode, save_decode


class SplatCacheTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        models = self.root/'model_files'
        models.mkdir()
        for name in ('dino','unet','vae','decoder'):
            (models/name).write_bytes(name.encode())
        self.enterContext(patch.object(store, 'root', return_value=self.root))
        self.enterContext(patch.dict(sys.modules, {
            'folder_paths': SimpleNamespace(get_full_path=lambda folder, name: str(models/name)
                                           if (models/name).is_file() else None),
            'comfy': SimpleNamespace(model_management=SimpleNamespace(__file__=str(self.root/'comfy'/'model_management.py'))),
        }))
        def n(kind, **pins):
            return {'class_type': kind, 'inputs': pins}
        self.prompt = {
            '1': n('BIPrepareReference'), '2': n('TripoSplatPreprocessImage', image=['1',0],mask=['1',1]),
            '3': n('CLIPVisionLoader',clip_name='dino'),'4': n('VAELoader',vae_name='vae'),
            '5': n('TripoSplatConditioning',clip_vision=['3',0],vae=['4',0],image=['2',0]),
            '6': n('UNETLoader',unet_name='unet'),
            '7': n('KSampler',model=['6',0],positive=['5',0],negative=['5',1],latent_image=['5',2],seed=1),
            '8': n('VAELoader',vae_name='decoder'),
            '9': n('VAEDecodeTripoSplat',samples=['7',0],vae=['8',0],seed=2,num_gaussians=32768),
            '10': n('BICachedTripoSplat',reference=['1',2],splat=['9',0]),
        }
        self.ref = {'sha256': 'a'*64}

    def test_lazy_reuse_requires_same_reference_full_graph_and_weights(self):
        def request():
            return decoder_request(self.ref, self.prompt, '10')
        key, meta = request()
        self.assertEqual(set(meta['weights']), {'clip_vision/dino','vae/vae','vae/decoder','diffusion_models/unet'})
        self.assertEqual(CachedTripoSplat().check_lazy_status(self.ref, None, self.prompt, '10'), ['splat'])
        marker = self.root/'.stages'/'triposplat'/key/'result.json'
        marker.parent.mkdir(parents=True)
        marker.write_text('{}')
        self.assertEqual(CachedTripoSplat().check_lazy_status(self.ref, None, self.prompt, '10'), [])
        self.prompt['7']['inputs']['seed'] += 1
        self.assertNotEqual(key, request()[0])
        self.prompt['7']['inputs']['seed'] -= 1
        self.assertEqual(key, request()[0])
        self.ref['sha256'] = 'b'*64
        self.assertNotEqual(key, request()[0])
        self.ref['sha256'] = 'a'*64
        (self.root/'model_files'/'decoder').write_bytes(b'changed model')
        self.assertNotEqual(key, request()[0])
        (self.root/'model_files'/'decoder').unlink()
        self.assertIsNone(request())

    def test_different_prepared_reference_or_unknown_dynamic_input_is_not_reused(self):
        self.prompt['11'] = {'class_type':'BIPrepareReference','inputs':{}}
        self.prompt['2']['inputs']['mask'] = ['11',1]
        self.assertIsNone(decoder_request(self.ref, self.prompt, '10'))
        self.prompt['11']['class_type'] = 'UnknownDynamicMask'
        self.assertIsNone(decoder_request(self.ref, self.prompt, '10'))
        self.assertEqual(CachedTripoSplat().resolve(self.ref, 'live splat', self.prompt, '10'), ('live splat',))

    def test_native_tensor_storage_and_digest_validation(self):
        rng = np.random.default_rng(91)
        arrays = [rng.random(shape, dtype=np.float32) for shape in
                  ((1,5,3),(1,5,3),(1,5,4),(1,5,1),(1,5,1,3))]
        item = SimpleNamespace(**dict(zip(FIELDS,arrays)))
        save_decode(self.root, item, {'graphSha256':'abc'})
        loaded = load_decode(self.root)
        for before, after in zip(arrays, loaded):
            np.testing.assert_array_equal(before, after)
        with (self.root/'splat.npz').open('ab') as f:
            f.write(b'invalid')
        with self.assertRaisesRegex(ValueError,'digest mismatch'):
            load_decode(self.root)


class NativeTypeTests(unittest.TestCase):
    def test_native_ply_round_trip_when_comfy_checkout_is_present(self):
        here = Path(__file__).resolve().parent
        comfy = here.parent/'.test-comfy'
        if not (comfy/'main.py').is_file():
            self.skipTest('Pinned ComfyUI checkout is available in the native integration CI job.')
        subprocess.run([sys.executable, str(here/'splat_native_storage.py'), str(comfy)],
                       cwd=comfy, check=True, timeout=120)
