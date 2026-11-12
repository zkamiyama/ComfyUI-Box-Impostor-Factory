from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from box_nodes import multiview

SIDE = np.array([(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)])
UP = np.array([(0,1,0),(0,1,0),(0,0,-1),(0,0,1),(0,1,0),(0,1,0)])
FACES = multiview.store.FACES


def sphere(root, r=40):
    manifest = {'resolution':r, 'bakeBox':{'min':[0,0,0],'max':[1,1,1]},'maps':[],
                'textures':{},'faces':{},'parentAsset':'sphere'}
    normal_atlas = np.zeros((2*r,3*r,4), dtype='<f2')
    color_atlas = normal_atlas.copy()
    iy,ix=np.indices((r,r))
    u,v=(ix+.5)/r-.5,.5-(iy+.5)/r
    l2=.42**2-u**2-v**2
    visible=l2>0
    depth_axis=np.sqrt(np.maximum(l2,0))
    for i,(face,side,up) in enumerate(zip(FACES,SIDE,UP)):
        points = u[...,None]*np.cross(up,side)+v[...,None]*up+depth_axis[...,None]*side
        row,col=divmod(i,3)
        for target,rgb in [(normal_atlas,.5+.5*points/.42),(color_atlas,.5+points)]:
            tile = target[row*r:(row+1)*r,col*r:(col+1)*r][::-1]
            tile[visible,:3] = rgb[visible]
            tile[visible,3]=1
        depth=np.where(visible,.5-depth_axis,np.nan).astype('<f2')
        file='depth'+str(i)+'.bin'
        depth.tofile(root/file)
        manifest['faces'][face]=dict(tile=[col,row],side=side.tolist(),up=up.tolist(),depth=file,
            imageRowOrder='top-to-bottom',depthEncoding='linear-axis-0-near-1-far;nan-background')
    for name,values in [('colorMap',color_atlas),('normalMap',normal_atlas)]:
        values.tofile(root/(name+'.bin'))
        manifest['textures'][name]={'path':name+'.bin','width':3*r,'height':2*r}
    return manifest


class MultiView(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.manifest=sphere(self.root)

    def test_same_surface_and_deliberate_color_error(self):
        report=multiview.evaluate_pairs(self.root,self.manifest)
        self.assertEqual(report['measuredPairs'],12)
        self.assertLess(max(p['medianLinearRGBMAE'] for p in report['pairs'].values()),.012)
        r=self.manifest['resolution']
        color=np.memmap(self.root/'colorMap.bin','<f2','r+',shape=(2*r,3*r,4))
        color[:r,:r,:3]=0
        color.flush()
        edited=multiview.evaluate_pairs(self.root,self.manifest)
        self.assertGreater(edited['pairs']['+X/+Z']['medianLinearRGBMAE'],.35)
        self.assertLess(edited['pairs']['-X/+Z']['medianLinearRGBMAE'],.012)
        self.assertIn('+X',edited['worstPair'])

    def test_occluded_target_reports_no_overlap_without_rejecting_other_faces(self):
        d_path=self.root/self.manifest['faces']['+X']['depth']
        d=np.fromfile(d_path,'<f2')
        d[np.isfinite(d)]=.96
        d.tofile(d_path)
        report=multiview.evaluate_pairs(self.root,self.manifest)
        self.assertEqual(report['pairs']['+X/+Z']['status'],'insufficient-overlap')
        self.assertIsNone(report['pairs']['+X/+Z']['p95LinearRGBMAE'])
        self.assertEqual(report['pairs']['-X/+Z']['status'],'measured')

    def test_legacy_zero_background_is_excluded_from_silhouette_stencil(self):
        r = 8
        face = {'side': [0, 0, 1], 'up': [0, 1, 0], 'tile': [0, 0]}
        color = np.zeros((r, r, 4), dtype='<f2')
        color[2:6, 2:6, 3] = 1
        normal = np.broadcast_to([.5, .5, 1., 1.], color.shape)
        for background in (0., np.nan):
            with self.subTest(background=background):
                depth = np.full((r, r), background, dtype='<f2')
                depth[2:6, 2:6] = .02  # close enough to zero to pass a depth-only stencil
                view = multiview.View('+Z', r, face, depth, color, normal, None)
                result = view.sample(np.array([2, 3]), np.array([3, 3]), np.ones(3))
                np.testing.assert_array_equal(result[3], [False, True])

    def test_edit_one_face_refreshes_only_its_four_adjacent_pairs(self):
        misses=[]
        def cache(_kind,key,write):
            directory=self.root/key
            if not directory.exists():
                directory.mkdir()
                write(directory)
                misses.append(key)
            return directory
        keys=dict.fromkeys(FACES,'baseline')
        multiview.evaluate_pairs(self.root,self.manifest,face_keys=keys,cache=cache)
        self.assertEqual(len(misses),12)
        keys['+Y']='edited'
        multiview.evaluate_pairs(self.root,self.manifest,face_keys=keys,cache=cache)
        self.assertEqual(len(misses),16)
