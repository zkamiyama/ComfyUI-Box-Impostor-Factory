"""Hi3D opt-in path: bypass, cache correctness and canonical rebake handoff."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import trimesh

from box_nodes import store
from box_nodes.hi3d_refine import RefineProxyHi3D


class Hi3DProxyTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        patcher=patch.object(store,'root',return_value=self.root/'store')
        patcher.start()
        self.addCleanup(patcher.stop)
        (self.root/'store').mkdir()
        self.source=self.root/'source.glb'
        self.raw=self.root/'raw.glb'
        mesh=trimesh.creation.icosphere(subdivisions=2)
        mesh.visual.vertex_colors=np.tile([140,120,80,255],(len(mesh.vertices),1)).astype('u1')
        mesh.export(self.source)
        mesh.vertices*=2
        mesh.export(self.raw)
        self.ref=self.root/'ref'
        self.ref.mkdir()
        Image.new('RGBA',(64,64),(125,110,90,255)).save(self.ref/'source.png')
        self.meta=self.root/'canonical.json'
        self.meta.write_text(json.dumps({'transform':np.eye(4).reshape(-1).tolist(),
            'inverseTransform':np.eye(4).reshape(-1).tolist(),
            'sourceSha256':'f'*64,'glbSha256':store.file_hash(self.source)}))
        self.proxy={'path':str(self.source),'sha256':store.file_hash(self.source),
                    'stageKey':'a'*64,'metadata':str(self.meta),
                    'reference':{'path':str(self.ref),'sha256':'b'*64}}

    def test_bypass_uses_original_proxy_without_importing_or_loading_gpu(self):
        node=RefineProxyHi3D()
        with patch.object(node,'resolve_python',side_effect=AssertionError('bypass accessed GPU')):
            self.assertIs(node.refine(self.proxy,mode='bypass')[0],self.proxy)

    def test_refinement_caches_and_preserves_original_proxy(self):
        node=RefineProxyHi3D()
        calls=[]
        def fake_run(command,log):
            calls.append(command)
            shutil.copyfile(self.raw,command[command.index('--output')+1])
            Path(command[command.index('--report')+1]).write_text('{"status":"stub"}')
            Path(log).write_text('stub')
        with patch.object(node,'resolve_python',return_value=sys.executable),patch.object(node,'invoke',side_effect=fake_run):
            out,=node.refine(self.proxy,mode='hi3d_512',steps=25,smoothing='none')
            again,=node.refine(self.proxy,mode='hi3d_512',steps=25,smoothing='none')
        self.assertEqual(len(calls),1)
        self.assertEqual(calls[0][calls[0].index('--resolution')+1],'512')
        self.assertEqual(out['sha256'],again['sha256'])
        self.assertEqual(out['stageKey'],again['stageKey'])
        self.assertNotEqual(out['sha256'],self.proxy['sha256'])
        self.assertEqual(out['reference'],self.proxy['reference'])
        self.assertTrue(Path(out['path']).is_file())
        meta=json.loads(Path(out['metadata']).read_text())
        self.assertEqual(meta['glbSha256'],store.file_hash(Path(out['path'])))
        self.assertEqual(meta['sourceSha256'],'f'*64)
        self.assertEqual(out['refinement']['inference']['status'],'stub')
        self.assertTrue(out['refinement']['has_vertex_color'])
        self.assertEqual(store.file_hash(self.source),self.proxy['sha256'])

    def test_1024_mode_is_explicit_and_does_not_smooth_by_default(self):
        node=RefineProxyHi3D()
        calls=[]
        def fake_run(command,log):
            calls.append(command)
            shutil.copyfile(self.raw,command[command.index('--output')+1])
            Path(command[command.index('--report')+1]).write_text('{"status":"stub"}')
            Path(log).write_text('stub')
        with patch.object(node,'resolve_python',return_value=sys.executable),patch.object(node,'invoke',side_effect=fake_run):
            result,=node.refine(self.proxy,mode='hi3d_1024')
        self.assertEqual(calls[0][calls[0].index('--resolution')+1],'1024')
        self.assertIn('--offload',calls[0], '1024 on 16GB should auto-enable CPU offload')
        self.assertEqual(result['refinement']['smooth_iterations'],0)
        self.assertEqual(json.loads(Path(result['metadata']).read_text())['refinement']['resolution'],1024)

    def test_requires_reference_and_explicit_valid_mode(self):
        node=RefineProxyHi3D()
        with self.assertRaisesRegex(ValueError,'canonical proxy'):
            node.refine({'path':'unused'},mode='hi3d_512')
        with self.assertRaisesRegex(ValueError,'Invalid'):
            node.refine(self.proxy,mode='hi3d_512',smoothing='unsupported')


if __name__=='__main__': unittest.main()
