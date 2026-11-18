"""AI depth inference fusion regression tests; GPU weights are not required for unit tests."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from box_nodes import nodes,store
from box_nodes.neural_depth import (NeuralDepthDetail, project_prediction,
                                     neural_detail_delta, freeze_original_appearance)
from tests.test_nodes import fixture


class NeuralDepthTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        patcher=patch.object(store,'root',return_value=self.root)
        patcher.start();self.addCleanup(patcher.stop)

    def test_ai_depth_is_new_detail_not_old_bilateral_smoothing(self):
        h=96
        y,x=np.mgrid[:h,:h]
        baseline=np.full((h,h),.3,np.float32)
        predicted=(2+.002*x+.16*np.exp(-((x-49)**2+(y-50)**2)/(2*4**2))).astype('f4')
        coverage=np.zeros_like(baseline);coverage[8:88,8:88]=1
        delta,stat=neural_detail_delta(baseline,predicted,coverage,max_delta=.02)
        self.assertGreater(float(delta[50,49]),0.005)
        self.assertAlmostEqual(float(delta[0,0]),0.)
        self.assertLess(abs(float(delta[8,20])),.003)
        self.assertLessEqual(float(np.max(np.abs(delta))),.0200001)
        self.assertGreater(stat['changedPixels'],200)

    def test_projection_undoes_original_generation_canvas(self):
        raw=np.repeat(np.arange(16,dtype='f4')[:,None],16,axis=1)
        meta={'guideRect':[4,2,8,12],'size':16}
        out=project_prediction(raw,meta,8)
        self.assertEqual(out.shape,(8,8))
        self.assertTrue(np.isfinite(out).all())
        self.assertGreater(float(out[7,3]),float(out[0,3]))
        with self.assertRaises(ValueError):
            project_prediction(raw,None,8)

    def test_predicted_depth_tracks_accepted_rgb_translation(self):
        y,x=np.mgrid[:64,:64]
        raw=np.exp(-((x-30)**2+(y-24)**2)/8).astype('f4')
        raw[:4]=np.nan
        report={'sampling':'source-affine-v1','registrationCenterYX':[31.5,31.5],
                'registration':{'scale':1.,'centerShiftYX':[5.,-3.]}}
        projected=project_prediction(raw,{'guideRect':[0,0,64,64],'size':64},64,report)
        self.assertEqual(np.unravel_index(np.nanargmax(projected),projected.shape),(29,27))
        self.assertTrue(np.isnan(projected[:9]).all())

    def test_moge_box_node_is_opt_in(self):
        box='a'*64
        node=NeuralDepthDetail()
        with patch('box_nodes.neural_depth.infer_and_rebake',side_effect=AssertionError('unexpected AI')) as predict:
            self.assertEqual(node.enhance(box,mode='bypass'),(box,))
            predict.assert_not_called()
        with patch('box_nodes.neural_depth.infer_and_rebake',return_value='b'*64) as predict:
            self.assertEqual(node.enhance(box,mode='moge2_detail',backend='webgl2'),('b'*64,))
            predict.assert_called_once_with(box,'webgl2')
        self.assertIn('moge2_detail',nodes.DEPTH_SOURCES)

    def test_preserved_apperance_changes_z_but_not_color_or_normal(self):
        before='a'*64; after='b'*64
        old_dir=self.root/before
        new_dir=self.root/after
        old=fixture(old_dir,before)
        newer=fixture(new_dir,after)
        # New mesh and first-hit depth are different, while source color must be identical.
        (new_dir/'geometry/proxy.glb').write_bytes(b'changed local 3D vertices')
        z=np.full((8,8),.66,dtype='<f2')
        z.tofile(new_dir/'guides/pz.depth16f.bin')
        newer['proxySha256']=store.file_hash(new_dir/'geometry/proxy.glb')
        newer['faces']['+Z']['depthSha256']=store.file_hash(new_dir/'guides/pz.depth16f.bin')
        # Frozen appearance must carry its own accepted source samples and registration.
        (old_dir/'views/pz').mkdir(parents=True)
        (new_dir/'views/pz').mkdir(parents=True)
        np.save(old_dir/'views/pz/aligned.npy',np.full((8,8,4),.25,dtype='f4'))
        np.save(new_dir/'views/pz/aligned.npy',np.full((8,8,4),.75,dtype='f4'))
        (old_dir/'qa').mkdir(exist_ok=True)
        (new_dir/'qa').mkdir(exist_ok=True)
        (old_dir/'qa/report.json').write_text('{"registration":"original"}')
        (new_dir/'qa/report.json').write_text('{"registration":"rebaked"}')
        old['qa']=newer['qa']='qa/report.json'
        (old_dir/'manifest.json').write_text(json.dumps(old))
        # Validate the source fixture. The patch preserves the original normals/colors.
        (new_dir/'manifest.json').write_text(json.dumps(newer))
        result=freeze_original_appearance(before,after)
        new=store.validate(self.root/result,result)
        self.assertEqual(new['resolution'],8)
        self.assertEqual(new['proxySha256'],newer['proxySha256'])
        self.assertEqual(new['textures']['colorMap']['sha256'],old['textures']['colorMap']['sha256'])
        self.assertEqual(new['textures']['normalMap']['sha256'],old['textures']['normalMap']['sha256'])
        self.assertEqual(new['faces']['+Z']['depthSha256'],newer['faces']['+Z']['depthSha256'])
        self.assertIn('appearancePreservation',new['provenance'])
        np.testing.assert_array_equal(np.load(self.root/result/'views/pz/aligned.npy'),
                                      np.load(old_dir/'views/pz/aligned.npy'))
        self.assertEqual((self.root/result/'qa/report.json').read_text(),
                         (old_dir/'qa/report.json').read_text())
        self.assertEqual(json.loads((self.root/result/'qa/rebaked-geometry.json').read_text()),
                         {'registration':'rebaked'})


if __name__=='__main__':unittest.main()
