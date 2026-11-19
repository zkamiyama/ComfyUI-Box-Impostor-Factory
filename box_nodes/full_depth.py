"""Opt-in joint full-depth fitting in a shared orthographic Box frame.

MoGe depth is registered by a positive affine fit to the seed geometry. This is
an anchored approximation, not calibrated perspective-to-orthographic recovery.
No frequency band is removed. The regularizer acts on displacement, not position.
"""
import json
from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
from PIL import Image
from scipy import ndimage, sparse
from scipy.sparse.linalg import LinearOperator, cg

from . import depth_mesh, faces, store
from .canvas import undo_guide_canvas
from .neural_depth import project_prediction
from .revision import DEPTH_ENCODING, RebakeProxyRevision

METHOD = 'joint-full-depth-v1'
SLUGS = dict(zip(store.FACES, ('px', 'nx', 'py', 'ny', 'pz', 'nz')))


def fit_depth(predicted, coarse, valid, *, max_change):
    """Robust positive affine registration; retain the complete fitted residual."""
    valid = valid & np.isfinite(predicted) & np.isfinite(coarse)
    yy, xx = np.nonzero(valid)
    def unusable(reason):
        return np.full_like(coarse,np.nan,dtype='f4'), {'calibration':'seed-prior-only',
            'exactCameraCalibration':False,'status':'unused','reason':reason,'samples':len(xx)}
    if len(xx) < 3:
        return unusable('insufficient shared foreground')
    step = max(1, len(xx) // 65536)
    x = predicted[yy[::step], xx[::step]].astype('f8')
    y = coarse[yy[::step], xx[::step]].astype('f8')
    center, scale = float(np.median(x)), float(np.std(x))
    if scale < 1e-8:
        return unusable('predicted depth has no variation')
    design = np.column_stack(((x-center)/scale, np.ones(len(x))))
    coeff = np.linalg.lstsq(design, y, rcond=None)[0]
    for _ in range(8):
        residual = design @ coeff - y
        width = max(.001, 1.4826 * np.median(np.abs(residual-np.median(residual))))
        weight = np.minimum(1., 1.5*width/np.maximum(np.abs(residual), 1e-12))
        coeff = np.linalg.solve(design.T @ (design*weight[:, None]), design.T @ (y*weight))
    slope = float(coeff[0]/scale)
    if slope <= 0:
        return unusable('positive affine fit is not supported by this prediction')
    offset = float(coeff[1] - slope*center)
    fitted = predicted*slope + offset
    delta = np.clip(fitted-coarse, -max_change, max_change)
    target = np.where(valid, coarse+delta, np.nan).astype('f4')
    residual = fitted[valid]-coarse[valid]
    report = {'status':'fitted','calibration':'positive-affine-to-seed-orthographic-depth',
              'exactCameraCalibration':False, 'scale':slope, 'offset':offset,
              'samples':int(valid.sum()), 'medianAbsFitResidual':float(np.median(abs(residual))),
              'p95AbsFitResidual':float(np.quantile(abs(residual), .95)),
              'clippedFraction':float(np.mean(abs(residual)>max_change)),
              'maxNormalizedAxialChange':float(max_change),
              'residualFilter':'none; full low/mid/high frequency residual'}
    return target, report


@dataclass
class DepthView:
    face: str
    side: np.ndarray
    up: np.ndarray
    coarse: np.ndarray
    target: np.ndarray
    mask: np.ndarray

    def __post_init__(self):
        self.side = np.asarray(self.side, 'f4')
        self.up = np.asarray(self.up, 'f4')
        self.right = np.cross(self.up, self.side)
        self.r = self.coarse.shape[0]
        self.mask = self.mask & np.isfinite(self.coarse) & np.isfinite(self.target)
        if self.mask.any():
            indices = ndimage.distance_transform_edt(~self.mask, return_distances=False, return_indices=True)
            filled = self.target[tuple(indices)]
        else:
            filled = np.zeros_like(self.target,dtype='f4')
        self.gy, self.gx = [x.astype('f4') for x in np.gradient(filled)]
        self.target = np.where(self.mask, self.target, 0).astype('f4')
        self.coarse = np.where(np.isfinite(self.coarse), self.coarse, 0).astype('f4')
        self.coverage = self.mask.astype('f4')

    def project(self, points):
        centered = points-.5
        coords = np.stack(((.5-centered @ self.up)*self.r-.5,
                           (.5+centered @ self.right)*self.r-.5))
        return coords, .5-centered @ self.side

    def sample(self, points):
        coords, axial = self.project(points)
        kwargs = dict(order=1, mode='constant', cval=0., prefilter=False)
        covered = ndimage.map_coordinates(self.coverage, coords, **kwargs) > .999
        target = ndimage.map_coordinates(self.target, coords, **kwargs)
        gx = ndimage.map_coordinates(self.gx, coords, **kwargs)*self.r
        gy = ndimage.map_coordinates(self.gy, coords, **kwargs)*self.r
        jacobian = -self.side[None, :] - gx[:, None]*self.right + gy[:, None]*self.up
        return axial-target, jacobian.astype('f4'), covered, coords


def mesh_arrays(data):
    doc, binary = depth_mesh._decode(data)
    primitives = list(depth_mesh._primitives(doc, binary))
    positions = [p @ transform[:3, :3].T+transform[:3, 3] for _, p, _, transform in primitives]
    unique, inverse = np.unique(np.concatenate(positions), axis=0, return_inverse=True)
    offsets = np.cumsum([0, *(len(p) for p in positions)])
    triangles = np.concatenate([inverse[offsets[i]+tri] for i, (_, _, tri, _) in enumerate(primitives)])
    return unique, triangles.astype(np.int32)


def vertex_normals(points, triangles):
    normal = np.zeros_like(points, dtype='f8')
    a, b, c = points[triangles].transpose(1, 0, 2)
    area = np.cross(b-a, c-a)
    for corners in triangles.T:
        np.add.at(normal, corners, area)
    return depth_mesh._unit(normal).astype('f4')


def displacement_laplacian(triangles, count):
    row = triangles[:, [0, 1, 2]].ravel()
    col = triangles[:, [1, 2, 0]].ravel()
    a = sparse.coo_matrix((np.ones(len(row)*2, 'f4'),
                          (np.concatenate((row,col)), np.concatenate((col,row)))),
                         shape=(count,count)).tocsr()
    degree = np.asarray(a.sum(axis=1)).ravel()
    scale = float(np.mean(degree[degree>0])) if (degree>0).any() else 1.
    return (sparse.diags(degree/scale)-a/scale).tocsr()


def seed_visibility(points, normals, views, span):
    result = []
    for view in views:
        coords, axial = view.project(points)
        captured = ndimage.map_coordinates(view.coarse, coords, order=1, mode='constant', cval=0.)
        coverage = ndimage.map_coordinates(view.coverage, coords, order=1, mode='constant', cval=0.)
        cosine = normals @ view.side
        pitch = max(abs(span @ view.right), abs(span @ view.up))/view.r
        tolerance = 2*pitch/(abs(span @ view.side)*np.maximum(cosine, .15))+.001
        visible = (coverage>.999) & (cosine>.05) & (abs(axial-captured)<tolerance)
        result.append(np.where(visible, np.maximum(cosine, 0)**2, 0).astype('f4'))
    return result


def point_visibility(points, view, coords):
    """Conservative vertex-splat occlusion check; seed first-hit checks remain active."""
    _, axial = view.project(points)
    xy = np.rint(coords).astype(np.int32)
    inside = (xy[0]>=0)&(xy[0]<view.r)&(xy[1]>=0)&(xy[1]<view.r)
    near = np.full(view.r*view.r, np.inf, 'f4')
    np.minimum.at(near, xy[0,inside]*view.r+xy[1,inside], axial[inside])
    # Sampling the same pixel avoids treating a neighboring steep surface as an occluder.
    nearest = np.full(len(points), np.inf, 'f4')
    nearest[inside] = near[xy[0,inside]*view.r+xy[1,inside]]
    return inside & (axial-nearest <= 2/view.r+.001)


def constraints(points, views, seed_weights):
    sampled = [view.sample(points) for view in views]
    weights = [base*valid*point_visibility(points, view, coords)
               for base, (residual,jac,valid,coords),view in zip(seed_weights,sampled,views)]
    move = np.zeros_like(points)
    total = np.zeros_like(points)
    for view, (residual,_,_,_),weight in zip(views,sampled,weights):
        move += view.side*(weight*residual)[:,None]
        total += abs(view.side)*weight[:,None]
    proposal = points+move/np.maximum(total,1e-6)
    agree = []
    observed_axes = np.zeros_like(points, dtype=bool)
    supported_axes = np.zeros_like(points, dtype=bool)
    for view,weight in zip(views,weights):
        residual,_,valid,_ = view.sample(proposal)
        tolerance = 3/view.r+.003
        good = valid & (abs(residual)<tolerance) & (weight>0)
        agree.append(good)
        observed_axes |= (weight>0)[:,None] & (abs(view.side)>0)
        supported_axes |= good[:,None] & (abs(view.side)>0)
    observed = observed_axes.sum(axis=1)
    supported = supported_axes.sum(axis=1)
    rows = []
    for view,(residual,jac,valid,_),weight,good in zip(views,sampled,weights,agree):
        confidence = np.where(observed>=2, np.where((supported>=2)&good, 1., 0.),
                              np.where(good, .2, 0.)).astype('f4')
        width = 3/view.r+.003
        robust = np.minimum(1., width/np.maximum(abs(residual),1e-8))
        rows.append((residual,jac,(weight*confidence*robust).astype('f4')))
    return rows, {'multiViewVertices':int(np.count_nonzero(supported>=2)),
                  'singleViewVertices':int(np.count_nonzero((observed==1)&(supported==1))),
                  'rejectedCrossViewVertices':int(np.count_nonzero((observed>=2)&(supported<2)))}


def joint_fit(points, triangles, views, bounds, *, max_displacement_height=.015,
              iterations=8, smoothness=.3, prior=.025, cg_iterations=120):
    """IRLS/Gauss-Newton on one mesh with a matrix-free 3x3-block PCG system."""
    low, high = np.asarray(bounds['min']), np.asarray(bounds['max'])
    span = high-low
    height = float(np.ptp(points[:,1]))
    if height<=0:
        height = float(span[1])
    max_world = max_displacement_height*height
    base = ((points-low)/span).astype('f4')
    current = base.copy()
    normals = vertex_normals(points, triangles)
    seed = seed_visibility(base,normals,views,span)
    laplacian = displacement_laplacian(triangles,len(base))
    pins = np.isclose(base,base.min(axis=0),atol=1e-7)|np.isclose(base,base.max(axis=0),atol=1e-7)
    diagonal_prior = np.full_like(base,prior)+pins.astype('f4')*10000
    logs = []
    def energy(candidate, rows):
        displacement = candidate-base
        cost = float(np.sum(diagonal_prior*displacement**2))
        cost += smoothness*float(np.sum(displacement*(laplacian@displacement)))
        for view,(_,_,weight) in zip(views,rows):
            residual,_,valid,_ = view.sample(candidate)
            cost += float(np.sum(weight*np.where(valid,residual**2,1.)))
        return cost
    for iteration in range(iterations):
        rows, agreement = constraints(current,views,seed)
        hessian = np.zeros((len(base),3,3),'f4')
        rhs = np.zeros_like(base)
        for residual,jac,weight in rows:
            hessian += jac[:,:,None]*jac[:,None,:]*weight[:,None,None]
            rhs -= jac*(weight*residual)[:,None]
        displacement = current-base
        rhs -= diagonal_prior*displacement+smoothness*(laplacian@displacement)
        diag = np.diagonal(hessian,axis1=1,axis2=2)+diagonal_prior+smoothness*laplacian.diagonal()[:,None]
        def product(vector):
            v = vector.reshape(-1,3)
            return (np.einsum('nij,nj->ni',hessian,v)+diagonal_prior*v+smoothness*(laplacian@v)).ravel()
        operator = LinearOperator((base.size,base.size),matvec=product,dtype=np.float32)
        preconditioner = LinearOperator(operator.shape,matvec=lambda v:v/diag.ravel(),dtype=np.float32)
        step, status = cg(operator,rhs.ravel(),M=preconditioner,rtol=2e-4,atol=1e-8,maxiter=cg_iterations)
        step = step.reshape(-1,3)
        length = np.linalg.norm(step*span,axis=1)
        step *= np.minimum(1.,max_world/3/np.maximum(length,1e-12))[:,None]
        old_energy = energy(current,rows)
        accepted = False
        for backtrack in range(9):
            candidate = current+step*(.5**backtrack)
            delta = (candidate-base)*span
            length = np.linalg.norm(delta,axis=1)
            candidate = base+(candidate-base)*np.minimum(1.,max_world/np.maximum(length,1e-12))[:,None]
            candidate = np.clip(candidate,base.min(axis=0),base.max(axis=0))
            candidate[pins] = base[pins]
            new_energy = energy(candidate,rows)
            if new_energy <= old_energy+1e-10:
                accepted = True
                break
        logs.append({'iteration':iteration,'energyBefore':old_energy,'energyAfter':new_energy,
                     'pcgStatus':int(status),'stepAccepted':accepted,
                     'stepScale':float(.5**backtrack),**agreement})
        if not accepted:
            break
        moved = float(np.max(np.linalg.norm((candidate-current)*span,axis=1)))
        current = candidate
        if moved < height*1e-6:
            break
    output = points+(current-base).astype('f8')*span
    lengths = np.linalg.norm(output-points,axis=1)/height
    return output, {'solver':'joint-IRLS-GaussNewton-matrix-free-PCG',
        'visibility':'seed first-hit + current vertex-splat; not triangle rasterization',
        'regularization':'graph Laplacian of displacement only',
        'parameters':{'maxDisplacementHeight':max_displacement_height,'iterations':iterations,
                      'smoothness':smoothness,'prior':prior,'cgIterations':cg_iterations},
        'vertices':len(points),'triangles':len(triangles),'height':height,
        'normalizedDisplacement':{'p50':float(np.quantile(lengths,.5)),
                                  'p95':float(np.quantile(lengths,.95)),'max':float(lengths.max())},
        'iterations':logs}


def geometry_normals(data):
    """Rebuild vertex normals from the changed shared surface, preserving UV/materials."""
    points,triangles = mesh_arrays(data)
    normals = vertex_normals(points,triangles)
    doc,binary = depth_mesh._decode(data)
    original_binary = binary
    binary = bytearray(binary)
    primitives = list(depth_mesh._primitives(doc,original_binary))
    positions = [p@t[:3,:3].T+t[:3,3] for _,p,_,t in primitives]
    _,inverse = np.unique(np.concatenate(positions),axis=0,return_inverse=True)
    offsets = np.cumsum([0,*(len(p) for p in positions)])
    for i,(primitive,xyz,tri,transform) in enumerate(primitives):
        local = depth_mesh._unit(normals[inverse[offsets[i]:offsets[i+1]]]@transform[:3,:3])
        if 'NORMAL' not in primitive['attributes']:
            # POSITION has the same count and float32 VEC3 layout; the writer allocates new storage.
            primitive['attributes']['NORMAL'] = primitive['attributes']['POSITION']
        depth_mesh._write_attribute(doc,binary,primitive,'NORMAL',local)
        if 'TANGENT' in primitive['attributes']:
            tangents = depth_mesh._accessor(doc,original_binary,primitive['attributes']['TANGENT'],'VEC4',{5126}).astype('f8')
            tangent = tangents[:,:3]-local*np.sum(local*tangents[:,:3],axis=1,keepdims=True)
            depth_mesh._write_attribute(doc,binary,primitive,'TANGENT',
                                       np.column_stack((depth_mesh._unit(tangent),tangents[:,3])))
    return depth_mesh._encode(doc,binary)


def prepare_views(directory, manifest, predictions, max_displacement_height):
    r = manifest['resolution']
    height = manifest['tightBounds']['max'][1]-manifest['tightBounds']['min'][1]
    span = np.subtract(manifest['bakeBox']['max'],manifest['bakeBox']['min'])
    qa = json.loads(store.contained(directory,manifest['qa']).read_bytes())['faces']
    inference = json.loads((predictions/'inference.json').read_text())
    views,reports = [],{}
    for fid in store.FACES:
        info = manifest['faces'][fid]
        edit = info.get('edit',{})
        if edit.get('status')!='accepted':
            raise ValueError('Full-depth fitting needs an accepted generated RGB for '+fid)
        if info.get('depthEncoding')!=DEPTH_ENCODING:
            raise ValueError('Full-depth fitting needs native first-hit depth for '+fid)
        source = store.contained(directory,edit['view'])
        if inference['faces'][SLUGS[fid]]['source_sha256']!=store.file_hash(source):
            raise ValueError('MoGe prediction is from a different generated RGB: '+fid)
        raw = np.load(predictions/(SLUGS[fid]+'.npy'),allow_pickle=False)
        canvas = edit.get('provenance',{}).get('canvas')
        registration = dict(qa[fid])
        if 'registrationCenterYX' not in registration:
            source_rgba = np.load(source.with_suffix('.npy'),allow_pickle=False)
            alpha = undo_guide_canvas(source_rgba,canvas)[...,3:4]
            center,_ = faces.bounds(faces.resize(alpha,(r,r))[...,0]>=.5)
            registration['registrationCenterYX'] = center.tolist()
        predicted = project_prediction(raw,canvas,r,registration)
        coarse = np.fromfile(store.contained(directory,info['depth']),'<f2').reshape(r,r).astype('f4')
        with Image.open(store.contained(directory,info['guide'])) as image:
            coverage = np.asarray(image.getchannel('A'))>250
        aligned = np.load(source.parent/'aligned.npy',allow_pickle=False)
        valid = coverage & (aligned[...,3]>.99) & np.isfinite(coarse) & np.isfinite(predicted)
        valid = ndimage.binary_erosion(valid,iterations=2)
        axial_limit = max_displacement_height*height/abs(np.asarray(info['side'])@span)
        target,report = fit_depth(predicted,coarse,valid,max_change=axial_limit)
        views.append(DepthView(fid,info['side'],info['up'],coarse,target,valid))
        reports[fid] = {**report,'sourceSha256':store.file_hash(source),'predictionSha256':store.file_hash(predictions/(SLUGS[fid]+'.npy'))}
    return views,reports,inference


def refine_to_proxy(asset, prediction_dir, work_dir, **options):
    started = time.monotonic()
    directory = store.asset_dir(asset)
    manifest = store.validate(directory,asset)
    predictions,work = Path(prediction_dir),Path(work_dir)
    work.mkdir(parents=True,exist_ok=True)
    views,registration,inference = prepare_views(directory,manifest,predictions,options.get('max_displacement_height',.015))
    data = store.contained(directory,manifest['proxy']).read_bytes()
    points,triangles = mesh_arrays(data)
    tight = manifest['tightBounds']
    if not (np.allclose(points.min(0),tight['min'],atol=1e-5) and np.allclose(points.max(0),tight['max'],atol=1e-5)):
        raise ValueError('Proxy positions and orthographic captures are not in the same canonical frame.')
    revised,solver = joint_fit(points,triangles,views,manifest['bakeBox'],**options)
    def displacement(samples):
        if not np.array_equal(samples,points):
            raise ValueError('Mesh vertex ordering changed before displacement.')
        return revised-points
    changed,geometry = depth_mesh.deform_glb(data,displacement,local_backtracking=True)
    changed = geometry_normals(changed)
    # Re-read by primitive order: lexicographic unique ordering may change after deformation.
    old_doc,old_binary = depth_mesh._decode(data)
    new_doc,new_binary = depth_mesh._decode(changed)
    actual_lengths = []
    for old,new in zip(depth_mesh._primitives(old_doc,old_binary),depth_mesh._primitives(new_doc,new_binary)):
        actual_lengths.append(np.linalg.norm((new[1]-old[1])@old[3][:3,:3].T,axis=1)/solver['height'])
    actual = np.concatenate(actual_lengths)
    report = {'method':METHOD,'sourceAsset':asset,'registration':registration,'inference':inference,
              'solver':solver,'mesh':geometry,
              'actualDisplacementHeight':{'p50':float(np.quantile(actual,.5)),'p95':float(np.quantile(actual,.95)),'max':float(actual.max())},
              'normalSource':'area-weighted geometric normals recomputed from changed shared mesh',
              'limitations':['Depth-to-orthographic registration is an anchored affine approximation.',
                             'Current occlusion uses vertex splats; final native rebake supplies exact rasterized depth.'],
              'seconds':round(time.monotonic()-started,3)}
    path = work/'refined.glb'
    path.write_bytes(changed)
    metadata = {'glbSha256':store.file_hash(path),
                'sourceSha256':manifest.get('provenance',{}).get('originalGlbSha256',manifest['proxySha256']),
                'transform':manifest['transform'],'inverseTransform':manifest['inverseTransform'],
                'tightBounds':manifest['tightBounds'],'bakeBox':manifest['bakeBox']}
    (work/'canonical.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (work/'refinement.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    for view in views:
        np.save(work/(SLUGS[view.face]+'-full-target.npy'),np.where(view.mask,view.target,np.nan),allow_pickle=False)
    proxy = {'sha256':store.file_hash(path),'stageKey':store.key(METHOD,asset,store.file_hash(path),solver['parameters'],registration,store.file_hash(Path(__file__))),
             'path':str(path),'metadata':str(work/'canonical.json'),'depthRefinement':report,
             'rotationDegreesXYZ':manifest.get('provenance',{}).get('rotationDegreesXYZ',[0,0,0])}
    ref = manifest.get('reference')
    if ref:
        proxy['reference']={**ref,'path':str(store.contained(directory,ref['image']).parent)}
        if ref.get('camera'):
            proxy['camera']=ref['camera'];proxy['croppedImage']=str(store.contained(directory,ref['camera']['image']))
    if manifest.get('provenance',{}).get('meshGeneration'):
        proxy['provenance']=manifest['provenance']['meshGeneration']
    return proxy,report


def _residual_metrics(depth, target, valid, world_per_depth, height):
    residual = np.where(valid,(depth-target)*world_per_depth/height,0)
    sigma = max(2,depth.shape[0]/64)
    weight = ndimage.gaussian_filter(valid.astype('f4'),sigma)
    low = np.divide(ndimage.gaussian_filter(residual,sigma),weight,
                    out=np.zeros_like(residual),where=weight>.5)
    low_valid = valid & (weight>.5)
    return {'samples':int(valid.sum()),
            'rmsHeight':float(np.sqrt(np.mean(residual[valid]**2))) if valid.any() else None,
            'lowFrequencySamples':int(low_valid.sum()),
            'lowFrequencyRmsHeight':float(np.sqrt(np.mean(low[low_valid]**2))) if low_valid.any() else None,
            'lowFrequencySigmaPixels':sigma,
            'p95AbsHeight':float(np.quantile(abs(residual[valid]),.95)) if valid.any() else None}


def rebake_refinement(asset, proxy, work_dir, backend='webgl2'):
    directory = store.asset_dir(asset)
    original = store.validate(directory,asset,check_bytes=False)
    revised,_ = RebakeProxyRevision().revise(asset,proxy,backend)
    qa = json.loads(store.contained(directory,original['qa']).read_bytes())['faces']
    items = {}
    for fid,info in original['faces'].items():
        edit = info['edit']
        source_path = store.contained(directory,edit['view'])
        source = faces.save_view(np.load(source_path.with_suffix('.npy'),allow_pickle=False))
        source['provenance']={**edit['provenance'],'depthRefinementMethod':METHOD}
        limits = qa[fid].get('thresholds',{})
        aligned = faces.align({fid:source},revised,limits.get('minIoU',.85),
                              limits.get('maxShiftFraction',.1),limits.get('maxScaleChange',.15))
        items.update(aligned['faces'])
    final = faces.compose(revised,{'baseline':revised,'faces':items})
    after_dir = store.asset_dir(final)
    after = store.validate(after_dir,final)
    evaluation = {'sourceAsset':asset,'assetId':final,'geometryAsset':revised,'faces':{},
                  'normalMapChanged':after['textures']['normalMap']['sha256']!=original['textures']['normalMap']['sha256'],
                  'intervalMapChanged':after['textures']['intervalMap']['sha256']!=original['textures']['intervalMap']['sha256'],
                  'rangeMapChanged':after['textures']['rangeMap']['sha256']!=original['textures']['rangeMap']['sha256']}
    r = original['resolution'];span=np.subtract(original['bakeBox']['max'],original['bakeBox']['min'])
    height=original['tightBounds']['max'][1]-original['tightBounds']['min'][1]
    for fid in store.FACES:
        target=np.load(Path(work_dir)/(SLUGS[fid]+'-full-target.npy'),allow_pickle=False)
        before=np.fromfile(store.contained(directory,original['faces'][fid]['depth']),'<f2').reshape(r,r).astype('f4')
        now=np.fromfile(store.contained(after_dir,after['faces'][fid]['depth']),'<f2').reshape(r,r).astype('f4')
        valid=np.isfinite(target)&np.isfinite(before)&np.isfinite(now)
        depth_span=abs(np.asarray(original['faces'][fid]['side'])@span)
        evaluation['faces'][fid]={'targetPixels':int(np.count_nonzero(np.isfinite(target))),
                                'sharedPixels':int(valid.sum()),
                                'missingAfterPixels':int(np.count_nonzero(np.isfinite(target)&~np.isfinite(now))),
                                'before':_residual_metrics(before,target,valid,depth_span,height),
                                'after':_residual_metrics(now,target,valid,depth_span,height),
                                'sameSourceRGB':store.file_hash(store.contained(directory,original['faces'][fid]['edit']['view']))==store.file_hash(store.contained(after_dir,after['faces'][fid]['edit']['view'])),
                                'alignmentStatus':after['faces'][fid]['edit']['status']}
    (Path(work_dir)/'rebake-evaluation.json').write_text(json.dumps(evaluation,indent=2,allow_nan=False)+'\n')
    return final,evaluation
