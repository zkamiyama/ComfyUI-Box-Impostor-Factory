"""Small CPU tests for known-camera RGB mapping; no model or native renderer."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from box_nodes import faces, store
from box_nodes.camera_rebase import camera_pixel_affine, reproject_samples, rebase_views


def camera(side=(0, 0, 1), center=(0, 0, 0), width=1., height=1., depth=.8):
    side = np.asarray(side, 'f8')
    up = np.array([0., 1., 0.]) if abs(side[1]) < .5 else np.array([0., 0., 1.])
    right = np.cross(up, side)
    rotation = np.stack([right, up, side])
    position = np.asarray(center) + side*(depth/2+.1)
    view = np.eye(4)
    view[:3, :3] = rotation
    view[:3, 3] = -rotation @ position
    near, far = .05, depth+.15
    projection = np.array([[2/width, 0, 0, 0], [0, 2/height, 0, 0],
                           [0, 0, -2/(far-near), -(far+near)/(far-near)], [0, 0, 0, 1.]])
    return {'side': side.tolist(), 'up': up.tolist(), 'tile': [1, 1],
            'imageRowOrder': 'top-to-bottom',
            'depthEncoding': 'linear-axis-0-near-1-far;nan-background',
            'viewMatrix': view.T.ravel().tolist(), 'projectionMatrix': projection.T.ravel().tolist(),
            'frustum': [-width/2, width/2, -height/2, height/2, near, far], 'guide': 'guide.png'}


def pixel(face, point, resolution):
    projection = np.array(face['projectionMatrix']).reshape(4, 4).T
    view = np.array(face['viewMatrix']).reshape(4, 4).T
    clip = projection @ view @ np.r_[point, 1.]
    return np.array([(clip[0]/clip[3]+1)*resolution/2-.5,
                     (1-clip[1]/clip[3])*resolution/2-.5, 1.])


class CameraRebaseTests(unittest.TestCase):
    def test_six_axes_agree_with_direct_world_projection_at_multiple_depths(self):
        for side in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
            with self.subTest(side=side):
                old = camera(side, center=(.04,-.02,.03), width=.8, height=1.1, depth=.6)
                new = camera(side, center=(-.01,.01,.02), width=.9, height=1.04, depth=.7)
                affine, proof = camera_pixel_affine(old, new, 32, 40)
                for point in [(-.12,.16,.18),(.21,-.09,-.2),(.0,.05,.1)]:
                    np.testing.assert_allclose(affine @ pixel(new, point, 40),
                                               pixel(old, point, 32), atol=1e-12)
                self.assertTrue(proof['depthIndependent'])

    def test_perspective_depth_coupling_direction_and_row_changes_rejected(self):
        old = camera()
        cases = []
        perspective = copy.deepcopy(old)
        p = np.array(perspective['projectionMatrix']).reshape(4, 4).T
        p[3] = [0,0,-1,0]
        perspective['projectionMatrix'] = p.T.ravel().tolist()
        cases.append(perspective)
        depth_coupled = copy.deepcopy(old)
        p = np.array(depth_coupled['projectionMatrix']).reshape(4, 4).T
        p[0,2] = .2
        depth_coupled['projectionMatrix'] = p.T.ravel().tolist()
        cases.append(depth_coupled)
        direction = camera((1,0,0));direction['side'] = old['side']
        cases.append(direction)
        bottom_rows = copy.deepcopy(old);bottom_rows['imageRowOrder'] = 'bottom-to-top'
        cases.append(bottom_rows)
        for bad in cases:
            with self.subTest(camera=bad):
                with self.assertRaises(ValueError):
                    camera_pixel_affine(old, bad, 32, 32)

    def test_identity_preserves_all_samples_exactly(self):
        rng = np.random.default_rng(4)
        data = rng.random((8,8,4), dtype=np.float32)
        data[0,:,3] = 0  # even invisible source RGB is preserved in an identity replay
        affine, _ = camera_pixel_affine(camera(), camera(), 8, 8)
        actual, resampled = reproject_samples(data, affine, 8)
        np.testing.assert_array_equal(actual, data)
        self.assertFalse(resampled)

    def test_premultiplied_interpolation_has_no_hidden_rgb_bleed_or_double_alpha(self):
        data = np.zeros((2,2,4), 'f4')
        data[:,0] = [1,0,0,1]
        data[:,1] = [0,0,1,0]  # hidden blue must not contaminate the half-covered red
        affine = np.eye(3);affine[0,2] = .5
        actual, resampled = reproject_samples(data, affine, 2)
        np.testing.assert_allclose(actual[:,0], [[1,0,0,.5]]*2, atol=1e-7)
        self.assertTrue(resampled)
        atlas = np.zeros((4,6,4), dtype='<f2');atlas[...,3] = .8
        faces.replace_tile(atlas, '+Z', actual)
        tile = faces.tile_view(atlas, '+Z')
        np.testing.assert_allclose(tile[:,0,0], .4, atol=.001)
        np.testing.assert_allclose(tile[:,0,3], .8, atol=.001)
        np.testing.assert_array_equal(tile[:,0,2], 0)

    def assets(self, folder, *, mismatched_silhouette=False, width=1.):
        source, revised = Path(folder)/'source', Path(folder)/'revised'
        view = source/'views/pz';view.mkdir(parents=True);revised.mkdir()
        r = 32
        y,x = np.mgrid[:r,:r]
        data = np.stack((x/(r-1), y/(r-1), np.full_like(x,.3,dtype='f4'),
                         ((x>=6)&(x<26)&(y>=6)&(y<26)).astype('f4')), axis=-1).astype('f4')
        np.save(view/'aligned.npy', data, allow_pickle=False)
        np.save(view/'source.npy', data, allow_pickle=False)
        Image.fromarray(np.uint8(data*255)).save(view/'source.png')
        source_guide = np.uint8(data*255)
        Image.fromarray(source_guide).save(source/'guide.png')
        old, new = camera(), camera(center=(1/r,0,0), width=width)
        affine, _ = camera_pixel_affine(old, new, r, r)
        transformed, _ = reproject_samples(data, affine, r)
        new_guide = np.uint8(transformed*255)
        if mismatched_silhouette:
            new_guide[...,3] = 0;new_guide[1:5,1:5,3] = 255
        Image.fromarray(new_guide).save(revised/'guide.png')
        thresholds = {'minIoU':.85, 'maxShiftFraction':.10, 'maxScaleChange':.15}
        qa = {'status':'accepted','reason':'silhouette-passed','after':{'iou':1.,'boundaryP95Px':0.},
              'thresholds':thresholds,'sourceSize':[32,32],
              'registration':{'scale':1.,'centerShiftYX':[0,0]}}
        (source/'qa.json').write_text(json.dumps({'faces':{'+Z':qa}}))
        ident = np.eye(4).ravel().tolist()
        original = {'asset_id':'a'*64,'resolution':r,'faces':{'+Z':{**old,'edit':{
            'status':'accepted','alignmentKey':'c'*64,'alignedSha256':store.file_hash(view/'aligned.npy'),
            'view':'views/pz/source.png','provenance':{'backend':'h3_enhanced'}}}},
            'faceOrder':['+Z'],'coordinateSystem':'right-handed-y-up','transform':ident,
            'inverseTransform':ident,'qa':'qa.json','tightBounds':{'min':[-.4]*3,'max':[.4]*3},
            'bakeBox':{'min':[-.5]*3,'max':[.5]*3}}
        target = copy.deepcopy(original)
        target.update(asset_id='b'*64,faces={'+Z':new},
                      tightBounds={'min':[-.36,-.4,-.4],'max':[.44,.4,.4]},
                      bakeBox={'min':[-.46,-.5,-.5],'max':[.54,.5,.5]})
        return source,original,revised,target,view,thresholds

    def test_known_camera_rebase_preserves_source_bytes_and_no_registration(self):
        with tempfile.TemporaryDirectory() as folder:
            source,old,revised,new,view,thresholds = self.assets(folder)
            before = {name:(view/name).read_bytes() for name in ('source.png','source.npy','aligned.npy')}
            with mock.patch('box_nodes.faces.align',side_effect=AssertionError('No image registration')):
                aligned, proof = rebase_views(source,old,revised,new,Path(folder)/'work')
            self.assertTrue(proof['passed'])
            report = proof['faces']['+Z']
            self.assertEqual(report['thresholds'],thresholds)
            self.assertEqual(report['after']['iou'],1.)
            self.assertTrue(report['alignedResampled'])
            self.assertFalse(report['sameAlignedBytes'])
            self.assertFalse(report['registrationOptimized'])
            self.assertEqual(Path(aligned['faces']['+Z']['source']['path']),view)
            for name,data in before.items():
                self.assertEqual((view/name).read_bytes(),data)

    def test_bad_shape_or_excessive_camera_transform_is_not_fitted_or_published(self):
        for kwargs in [{'mismatched_silhouette':True}, {'width':1.3}]:
            with self.subTest(case=kwargs), tempfile.TemporaryDirectory() as folder:
                source,old,revised,new,_,_ = self.assets(folder,**kwargs)
                with mock.patch('box_nodes.faces.align',side_effect=AssertionError('No image registration')):
                    with self.assertRaisesRegex(ValueError,'QA failed'):
                        rebase_views(source,old,revised,new,Path(folder)/'work')
                proof = json.loads((Path(folder)/'work/camera-rebase-proof.json').read_text())
                self.assertFalse(proof['passed'])
                self.assertEqual(proof['faces']['+Z']['status'],'fallback')

    def test_changed_world_transform_or_archived_aligned_hash_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            source,old,revised,new,view,_ = self.assets(folder)
            changed = copy.deepcopy(new);changed['transform'][12] = .1
            with self.assertRaisesRegex(ValueError,'canonical world frame'):
                rebase_views(source,old,revised,changed,Path(folder)/'work')
            raw = np.load(view/'aligned.npy');raw[8,8,0] += .1
            np.save(view/'aligned.npy',raw,allow_pickle=False)
            with self.assertRaisesRegex(ValueError,'aligned hash mismatch'):
                rebase_views(source,old,revised,new,Path(folder)/'work')


if __name__ == '__main__':
    unittest.main()
