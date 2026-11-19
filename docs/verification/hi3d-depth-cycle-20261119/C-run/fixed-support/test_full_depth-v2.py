"""CPU-only regressions for full depth fitting; no neural weights or renderer needed."""
import unittest
import numpy as np

from box_nodes import depth_mesh
from box_nodes.full_depth import (DepthView, constraints, fit_depth, geometry_normals,
                                  joint_fit, mesh_arrays, vertex_normals, feasible_support_step)
from tests.test_depth_mesh import fixture


def sphere_mesh(radius=.3, subdivisions=3):
    points = [np.array(v,dtype='f8') for v in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]]
    triangles = [(0,2,4),(2,1,4),(1,3,4),(3,0,4),(2,0,5),(1,2,5),(3,1,5),(0,3,5)]
    for _ in range(subdivisions):
        cache = {}
        def midpoint(a,b):
            edge = tuple(sorted((a,b)))
            if edge not in cache:
                p = points[a]+points[b]
                cache[edge] = len(points)
                points.append(p/np.linalg.norm(p))
            return cache[edge]
        next_faces = []
        for a,b,c in triangles:
            ab,bc,ca = midpoint(a,b),midpoint(b,c),midpoint(c,a)
            next_faces.extend([(a,ab,ca),(ab,b,bc),(ca,bc,c),(ab,bc,ca)])
        triangles = next_faces
    return .5+radius*np.array(points),np.array(triangles,dtype=np.int32)


def sphere_views(radius=.3, target_radius=.31, r=96):
    y,x = np.mgrid[:r,:r]
    u,v = (x+.5)/r-.5,.5-(y+.5)/r
    mask = u*u+v*v < (radius-.001)**2
    coarse = np.where(mask,.5-np.sqrt(np.maximum(0,radius**2-u*u-v*v)),np.nan).astype('f4')
    target = np.where(mask,.5-np.sqrt(np.maximum(0,target_radius**2-u*u-v*v)),np.nan).astype('f4')
    sides = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
    ups = [(0,1,0),(0,1,0),(0,0,-1),(0,0,1),(0,1,0),(0,1,0)]
    return [DepthView(str(i),s,u,coarse.copy(),target.copy(),mask.copy())
            for i,(s,u) in enumerate(zip(sides,ups))]


class FullDepthTests(unittest.TestCase):
    def test_exact_bilinear_jacobian_matches_subpixel_finite_difference(self):
        r=32
        rng=np.random.default_rng(9)
        target=rng.uniform(.15,.35,(r,r)).astype('f4')
        mask=np.ones((r,r),dtype=bool)
        view=DepthView('+Z',[0,0,1],[0,1,0],target.copy(),target,mask)
        points=np.array([[.234,.527,.6],[.468,.319,.65],[.713,.824,.7]],'f8')
        _,jac,valid,_=view.sample(points)
        self.assertTrue(valid.all())
        numerical=np.zeros_like(points)
        epsilon=1e-4
        for axis in range(3):
            shift=np.zeros_like(points);shift[:,axis]=epsilon
            numerical[:,axis]=(view.sample(points+shift)[0]-view.sample(points-shift)[0])/(2*epsilon)
        np.testing.assert_allclose(jac,numerical,atol=.001,rtol=2e-4)

    def test_support_exiting_vertex_does_not_block_feasible_interior_update(self):
        r=32
        mask=np.zeros((r,r),dtype=bool);mask[:,8:]=True
        coarse=np.full((r,r),.3,'f4')
        view=DepthView('+Z',[0,0,1],[0,1,0],coarse,coarse.copy(),mask)
        current=np.array([[(8.001+.5)/r,.5,.6],[.6,.5,.6]],'f8')
        step=np.array([[-.05,0,0],[0,0,.02]],'f8')
        residual,jac,valid,_=view.sample(current)
        self.assertTrue(valid.all())
        self.assertFalse(view.covered(current+step)[0])
        weights=np.ones(2,'f4');saved=weights.copy()
        rows=[(residual,jac,weights)]
        feasible,report=feasible_support_step(current,step,[view],rows,lambda p:p)
        updated=current+feasible
        self.assertTrue(view.covered(updated).all())
        self.assertEqual(report['supportBacktrackedVertices'],1)
        np.testing.assert_allclose(feasible[1],step[1],atol=1e-12)
        self.assertLess(abs(feasible[0,0]),abs(step[0,0])/100)
        np.testing.assert_array_equal(weights,saved)
        new_residual,_,new_valid,_=view.sample(updated)
        # Same fixed observations and invalid penalty as the actual line search.
        self.assertLess(float(np.sum(weights*np.where(new_valid,new_residual**2,1))),
                        float(np.sum(weights*residual**2)))

    def test_affine_registration_retains_broad_shape_change(self):
        y,x = np.mgrid[:128,:128]
        coarse = (.3+.15*x/128+.08*y/128).astype('f4')
        broad = .025*np.exp(-((x-64)**2+(y-64)**2)/(2*17**2))
        predicted = (coarse+broad)*3.2+5.
        valid = np.ones_like(coarse,dtype=bool)
        # Outliers do not determine the common scale or offset.
        predicted[:3,:] += .8
        target,report = fit_depth(predicted,coarse,valid,max_change=.06)
        self.assertGreater(float(target[64,64]-coarse[64,64]),.017)
        self.assertLess(float(np.median(abs(target[8:20,8:20]-coarse[8:20,8:20]))),.004)
        self.assertGreater(report['scale'],0.)
        self.assertFalse(report['exactCameraCalibration'])

    def test_uninformative_prediction_is_reported_without_moving_the_prior(self):
        constant=np.ones((16,16),'f4')
        target,report=fit_depth(constant,constant*.3,np.ones_like(constant,dtype=bool),max_change=.04)
        self.assertTrue(np.isnan(target).all())
        self.assertEqual(report['status'],'unused')
        view=DepthView('+Z',[0,0,1],[0,1,0],constant*.3,target,np.ones_like(constant,dtype=bool))
        self.assertFalse(view.mask.any())

    def test_six_views_move_one_mesh_and_preserve_displacement_bound(self):
        points,triangles = sphere_mesh()
        views = sphere_views()
        moved,report = joint_fit(points,triangles,views,{'min':[0,0,0],'max':[1,1,1]},
                                max_displacement_height=.03,iterations=7,prior=.01,smoothness=.15)
        radius0 = np.linalg.norm(points-.5,axis=1)
        radius1 = np.linalg.norm(moved-.5,axis=1)
        self.assertGreater(float(np.mean(radius1-radius0)),.004)
        self.assertLessEqual(float(np.max(np.linalg.norm(moved-points,axis=1))),.03*.6+1e-7)
        self.assertGreater(report['iterations'][0]['multiViewVertices'],len(points)//3)
        self.assertLess(report['iterations'][-1]['energyAfter'],report['iterations'][0]['energyBefore']*.65)
        np.testing.assert_allclose(moved.min(axis=0),points.min(axis=0),atol=1e-7)
        np.testing.assert_allclose(moved.max(axis=0),points.max(axis=0),atol=1e-7)

    def test_unseen_rear_surface_stays_fixed(self):
        axis=np.linspace(.3,.7,15)
        x,y=np.meshgrid(axis,axis)
        front=np.column_stack((x.ravel(),y.ravel(),np.full(x.size,.7)))
        rear=front.copy();rear[:,2]=.3
        points=np.vstack((front,rear,[[.5,.5,.1],[.5,.5,.9]]))
        faces=[]
        for offset in (0,len(front)):
            for j in range(14):
                for i in range(14):
                    a=offset+j*15+i;faces.extend([(a,a+1,a+16),(a,a+16,a+15)])
        triangles=np.array(faces,dtype=np.int32)
        r=64;mask=np.ones((r,r),bool)
        view=DepthView('+Z',[0,0,1],[0,1,0],np.full((r,r),.3,'f4'),np.full((r,r),.28,'f4'),mask)
        moved,report=joint_fit(points,triangles,[view],{'min':[0,0,0],'max':[1,1,1]},iterations=6,max_displacement_height=.04,prior=.01)
        self.assertGreater(float(np.mean(moved[:len(front),2]-front[:,2])),.008)
        np.testing.assert_allclose(moved[len(front):],points[len(front):],atol=1e-8)

    def test_opposing_depth_disagreement_is_not_averaged_as_evidence(self):
        r=32;p=np.array([[.5,.5,.5]],'f4');mask=np.ones((r,r),bool)
        views=[DepthView('+Z',[0,0,1],[0,1,0],np.full((r,r),.5,'f4'),np.full((r,r),.35,'f4'),mask),
               DepthView('-Z',[0,0,-1],[0,1,0],np.full((r,r),.5,'f4'),np.full((r,r),.35,'f4'),mask)]
        rows,_=constraints(p,views,[np.ones(1,'f4'),np.ones(1,'f4')])
        self.assertTrue(all(float(row[2][0])==0 for row in rows))

    def test_zero_target_does_not_smooth_existing_geometry(self):
        points,triangles=sphere_mesh(subdivisions=2)
        # A nonuniform mesh still has exactly zero regularizer at zero displacement.
        # With no supported observations the solver must not shrink its position field.
        views=sphere_views()
        for v in views:
            v.coverage[:]=0
        moved,_=joint_fit(points,triangles,views,{'min':[0,0,0],'max':[1,1,1]},iterations=2)
        np.testing.assert_array_equal(moved,points)

    def test_recomputed_normals_follow_changed_geometry_and_keep_uv_materials(self):
        data=fixture()
        def warp(world):
            d=np.zeros_like(world);d[:,2]=.4*(world[:,0]-3)
            return d
        warped,_=depth_mesh.deform_glb(data,warp)
        rebuilt=geometry_normals(warped)
        doc,binary=depth_mesh._decode(rebuilt)
        before,_=depth_mesh._decode(data)
        self.assertEqual(doc['materials'],before['materials'])
        attrs=doc['meshes'][0]['primitives'][0]['attributes']
        self.assertEqual(attrs['TEXCOORD_0'],3)
        local=depth_mesh._accessor(doc,binary,attrs['NORMAL'],'VEC3',{5126})
        expected=np.array([-.8,0,1]);expected/=np.linalg.norm(expected)
        np.testing.assert_allclose(local,np.tile(expected,(len(local),1)),atol=1e-6)
        old,_=mesh_arrays(warped);new,_=mesh_arrays(rebuilt)
        np.testing.assert_array_equal(old,new)

    def test_cancelled_normals_use_largest_incident_face(self):
        points=np.array([[0,0,0],[2,0,0],[0,1,0],[1,0,0]],'f8')
        triangles=np.array([[0,2,3],[0,1,2],[0,2,3]],dtype=np.int32)
        normals,report=vertex_normals(points,triangles,return_report=True)
        np.testing.assert_allclose(normals[0],[0,0,1],atol=1e-7)
        np.testing.assert_allclose(np.linalg.norm(normals,axis=1),1,atol=1e-7)
        self.assertGreaterEqual(report['cancellationFallbackVertices'],1)

    def test_unused_and_degenerate_only_normals_have_documented_unit_placeholder(self):
        points=np.array([[0,0,0],[1,0,0],[0,1,0],[2,2,2],[3,3,3]],'f8')
        triangles=np.array([[0,1,2],[0,2,1],[3,3,3]],dtype=np.int32)
        normals,report=vertex_normals(points,triangles,return_report=True)
        np.testing.assert_allclose(normals[:3],np.tile([0,0,1],(3,1)),atol=1e-7)
        np.testing.assert_allclose(normals[3:],np.tile([0,1,0],(2,1)),atol=1e-7)
        self.assertEqual(report['cancellationFallbackVertices'],3)
        self.assertEqual(report['degenerateOnlyPlaceholderVertices'],1)
        self.assertEqual(report['unusedPlaceholderVertices'],1)


if __name__=='__main__':
    unittest.main()
