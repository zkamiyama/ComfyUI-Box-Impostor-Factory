"""Frozen B RGB regression tests: CPU files only, no native renderer or models."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from box_nodes import store
from box_nodes.full_depth import reuse_fixed_rgb


class FixedRgbTests(unittest.TestCase):
    def assets(self, folder, shift=1):
        root=Path(folder)
        source, revised, work=root/'source',root/'revised',root/'work'
        source.mkdir();revised.mkdir()
        view=source/'views/pz';view.mkdir(parents=True)
        r=32
        y,x=np.mgrid[:r,:r]
        candidate=np.zeros((r,r,4),'f4')
        candidate[...,:3]=np.stack((x/r,y/r,(x+2*y)/(3*r)),axis=-1)
        candidate[6:26,6:26,3]=1
        np.save(view/'aligned.npy',candidate,allow_pickle=False)
        np.save(view/'source.npy',candidate,allow_pickle=False)
        Image.fromarray(np.uint8(candidate*255)).save(view/'source.png')
        rgba=np.full((r,r,4),255,'u1')
        rgba[...,3]=np.uint8(candidate[...,3]*255)
        Image.fromarray(rgba).save(source/'guide.png')
        shifted=rgba.copy();shifted[...,3]=np.roll(rgba[...,3],shift,axis=1)
        Image.fromarray(shifted).save(revised/'guide.png')
        qa={'status':'accepted','reason':'silhouette-passed',
            'before':{'iou':.9},'after':{'iou':1.},
            'registration':{'scale':1.03,'centerShiftYX':[.25,.5]},
            'registrationCenterYX':[15.5,15.5],'sampling':'source-affine-v1',
            'thresholds':{'minIoU':.85,'maxShiftFraction':.1,'maxScaleChange':.15}}
        (source/'qa.json').write_text(json.dumps({'faces':{'+Z':qa}}))
        ident=np.eye(4).ravel().tolist()
        frame={'side':[0,0,1],'up':[0,1,0],'tile':[1,1],
               'viewMatrix':ident,'projectionMatrix':ident,
               'frustum':[-1,1,-1,1,.1,2],'imageRowOrder':'top-to-bottom',
               'depthEncoding':'linear-axis-0-near-1-far;nan-background',
               'guide':'guide.png'}
        original={'asset_id':'a'*64,'resolution':r,
                  'bakeBox':{'min':[0,0,0],'max':[1,1,1]},
                  'tightBounds':{'min':[.1,.1,.1],'max':[.9,.9,.9]},
                  'transform':ident,'inverseTransform':ident,'faceOrder':['+Z'],
                  'coordinateSystem':'right-handed-y-up','qa':'qa.json',
                  'faces':{'+Z':{**frame,'edit':{
                      'status':'accepted','view':'views/pz/source.png',
                      'alignmentKey':'c'*64,'alignedSha256':store.file_hash(view/'aligned.npy'),
                      'provenance':{'backend':'h3_enhanced','canvas':{'size':32}}}}}}
        changed=copy.deepcopy(original)
        changed['asset_id']='b'*64
        del changed['faces']['+Z']['edit']
        return source,original,revised,changed,work,qa

    def test_exact_aligned_bytes_and_mapping_survive_changed_silhouette(self):
        with tempfile.TemporaryDirectory() as folder:
            source,original,revised,changed,work,old_qa=self.assets(folder)
            path=source/'views/pz/aligned.npy'
            before=path.read_bytes()
            with mock.patch('box_nodes.faces.align',side_effect=AssertionError('No re-registration')):
                aligned,proof=reuse_fixed_rgb(source,original,revised,changed,work)
            item=aligned['faces']['+Z']
            self.assertEqual(Path(item['measured'])/'aligned.npy',path)
            self.assertEqual(path.read_bytes(),before)
            report=json.loads((Path(item['directory'])/'report.json').read_text())
            self.assertEqual(report['registration'],old_qa['registration'])
            self.assertEqual(report['registrationCenterYX'],old_qa['registrationCenterYX'])
            self.assertEqual(report['sampling'],old_qa['sampling'])
            self.assertFalse(report['resampled'])
            self.assertTrue(proof['passed'])
            self.assertTrue(proof['faces']['+Z']['sameAlignedRGB'])
            self.assertLess(report['after']['iou'],1.)
            self.assertGreater(report['after']['iou'],.85)

    def test_bad_fixed_mapping_is_reported_not_re_registered_or_accepted(self):
        with tempfile.TemporaryDirectory() as folder:
            source,original,revised,changed,work,_=self.assets(folder,shift=13)
            with mock.patch('box_nodes.faces.align',side_effect=AssertionError('No re-registration')):
                with self.assertRaisesRegex(ValueError,'silhouette QA failed'):
                    reuse_fixed_rgb(source,original,revised,changed,work)
            proof=json.loads((work/'fixed-rgb-qa.json').read_text())
            self.assertFalse(proof['passed'])
            self.assertEqual(proof['faces']['+Z']['status'],'fallback')

    def test_changed_camera_frame_is_rejected_before_replay(self):
        with tempfile.TemporaryDirectory() as folder:
            source,original,revised,changed,work,_=self.assets(folder)
            changed['faces']['+Z']['projectionMatrix'][0]=2.
            with self.assertRaisesRegex(ValueError,'unchanged bounds/cameras'):
                reuse_fixed_rgb(source,original,revised,changed,work)
            proof=json.loads((work/'fixed-rgb-qa.json').read_text())
            self.assertFalse(proof['fixedFrame']['passed'])

    def test_modified_archived_aligned_samples_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            source,original,revised,changed,work,_=self.assets(folder)
            path=source/'views/pz/aligned.npy'
            data=np.load(path,allow_pickle=False)
            data[10,10,0]=.123
            np.save(path,data,allow_pickle=False)
            with self.assertRaisesRegex(ValueError,'do not match their hash'):
                reuse_fixed_rgb(source,original,revised,changed,work)


if __name__=='__main__':
    unittest.main()
