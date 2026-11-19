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
from .mesh_normals import geometry_normals, mesh_arrays, vertex_normals

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
    changed,normal_report = geometry_normals(changed,return_report=True)
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
              'normalRecompute':normal_report,
              'limitations':['Depth-to-orthographic registration is an anchored affine approximation.',
                             'Current occlusion uses vertex splats; final native rebake supplies exact rasterized depth.'],
              'seconds':round(time.monotonic()-started,3)}
    for view in views:
        np.save(work/(SLUGS[view.face]+'-full-target.npy'),np.where(view.mask,view.target,np.nan),allow_pickle=False)
    proxy = _write_proxy(changed, report, manifest, directory, work,
                         {'solver':solver['parameters'], 'registration':registration})
    return proxy, report


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


FIXED_MAPPING_VERSION = 'fixed-b-rgb-v1'
NORMAL_CONTROL_METHOD = 'geometry-normal-only-v1'


def _write_proxy(changed, report, manifest, directory, work, parameters):
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    path = work/'refined.glb'
    path.write_bytes(changed)
    metadata = {
        'glbSha256':store.file_hash(path),
        'sourceSha256':manifest.get('provenance', {}).get('originalGlbSha256', manifest['proxySha256']),
        'transform':manifest['transform'], 'inverseTransform':manifest['inverseTransform'],
        'tightBounds':manifest['tightBounds'], 'bakeBox':manifest['bakeBox']}
    (work/'canonical.json').write_text(json.dumps(metadata, indent=2)+'\n')
    (work/'refinement.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    proxy = {
        'sha256':store.file_hash(path),
        'stageKey':store.key(report['method'], manifest['asset_id'], store.file_hash(path),
                             parameters, report.get('normalRecompute'),
                             store.file_hash(Path(__file__)),
                             store.file_hash(Path(__file__).with_name('mesh_normals.py'))),
        'path':str(path), 'metadata':str(work/'canonical.json'), 'depthRefinement':report,
        'rotationDegreesXYZ':manifest.get('provenance', {}).get('rotationDegreesXYZ', [0,0,0])}
    ref = manifest.get('reference')
    if ref:
        proxy['reference'] = {**ref, 'path':str(store.contained(directory, ref['image']).parent)}
        if ref.get('camera'):
            proxy['camera'] = ref['camera']
            proxy['croppedImage'] = str(store.contained(directory, ref['camera']['image']))
    if manifest.get('provenance', {}).get('meshGeneration'):
        proxy['provenance'] = manifest['provenance']['meshGeneration']
    return proxy


def normal_only_proxy(asset, work_dir):
    """Same B positions/materials with the identical normal policy used by C."""
    directory = store.asset_dir(asset)
    manifest = store.validate(directory, asset)
    data = store.contained(directory, manifest['proxy']).read_bytes()
    changed, normals = geometry_normals(data, return_report=True)
    before, old_binary = depth_mesh._decode(data)
    after, new_binary = depth_mesh._decode(changed)
    checks = {'originalBinaryPrefixEqual':old_binary == new_binary[:len(old_binary)]}
    for key in ('nodes','scenes','scene','materials','textures','images','samplers',
                'extensions','extensionsUsed','extensionsRequired'):
        checks[key+'Equal'] = before.get(key) == after.get(key)
    old_parts = list(depth_mesh._primitives(before, old_binary))
    new_parts = list(depth_mesh._primitives(after, new_binary))
    checks['primitiveCountEqual'] = len(old_parts) == len(new_parts)
    for i, (old, new) in enumerate(zip(old_parts, new_parts)):
        checks[str(i)+'PositionsEqual'] = np.array_equal(old[1], new[1])
        checks[str(i)+'IndicesEqual'] = np.array_equal(old[2], new[2])
        checks[str(i)+'TransformEqual'] = np.array_equal(old[3], new[3])
        for name, index in old[0]['attributes'].items():
            if name not in ('NORMAL','TANGENT'):
                checks[str(i)+name+'AccessorEqual'] = index == new[0]['attributes'].get(name)
    if not all(checks.values()):
        raise ValueError('Normal-only control changed data outside its normal frame.')
    report = {'method':NORMAL_CONTROL_METHOD, 'sourceAsset':asset,
              'normalRecompute':normals, 'invariants':checks,
              'actualDisplacementHeight':{'p50':0., 'p95':0., 'max':0.},
              'normalSource':'same robust area-weighted shared-surface normals as C',
              'newInference':False}
    proxy = _write_proxy(changed, report, manifest, directory, work_dir, {})
    return proxy, report


def fixed_capture_frame(original, revised):
    """Fixed RGB and depth fitting are valid only in the same six camera frames."""
    checks = {}
    for key in ('resolution','tightBounds','bakeBox','transform','inverseTransform',
                'faceOrder','coordinateSystem'):
        checks[key] = original.get(key) == revised.get(key)
    camera_keys = ('side','up','tile','viewMatrix','projectionMatrix','frustum',
                   'imageRowOrder','depthEncoding')
    for fid in original['faces']:
        current = revised['faces'].get(fid, {})
        for key in camera_keys:
            checks[fid+':'+key] = original['faces'][fid].get(key) == current.get(key)
    return {'passed':all(checks.values()), 'checks':checks}


def reuse_fixed_rgb(source_dir, original, revised_dir, revised, work_dir, method=METHOD):
    """Evaluate the frozen B samples on C silhouettes; never optimize their mapping."""
    source_dir, revised_dir, work = Path(source_dir), Path(revised_dir), Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    frame = fixed_capture_frame(original, revised)
    evidence = {'mappingPolicy':FIXED_MAPPING_VERSION, 'sourceAsset':original['asset_id'],
                'revisedAsset':revised['asset_id'], 'fixedFrame':frame, 'faces':{}}
    report_path = work/'fixed-rgb-qa.json'
    if not frame['passed']:
        report_path.write_text(json.dumps(evidence, indent=2)+'\n')
        raise ValueError('Fixed RGB requires unchanged bounds/cameras; see '+str(report_path))
    previous = json.loads(store.contained(source_dir, original['qa']).read_bytes())['faces']
    items = {}
    r = original['resolution']
    for fid, info in original['faces'].items():
        edit = info.get('edit', {})
        if edit.get('status') != 'accepted' or previous[fid].get('status') != 'accepted':
            raise ValueError('Fixed RGB needs an accepted original B view: '+fid)
        view = store.contained(source_dir, edit['view']).parent
        aligned_path = view/'aligned.npy'
        digest = store.file_hash(aligned_path)
        if digest != edit.get('alignedSha256'):
            raise ValueError('Archived B aligned samples do not match their hash: '+fid)
        candidate = np.load(aligned_path, allow_pickle=False)
        if candidate.shape != (r,r,4) or not np.isfinite(candidate).all():
            raise ValueError('Invalid archived B aligned sample dimensions or values: '+fid)
        with Image.open(store.contained(source_dir, info['guide'])) as image:
            before_mask = np.asarray(image.getchannel('A')) >= 128
        with Image.open(store.contained(revised_dir, revised['faces'][fid]['guide'])) as image:
            after_mask = np.asarray(image.getchannel('A')) >= 128
        candidate_mask = candidate[...,3] >= .5
        before_score = faces.scores(candidate_mask, before_mask)
        after_score = faces.scores(candidate_mask, after_mask)
        limits = previous[fid].get('thresholds', {})
        min_iou = limits.get('minIoU', .85)
        accepted = bool(after_mask.any() and candidate_mask.any() and after_score['iou'] >= min_iou)
        report = {
            **previous[fid], 'sourceAfter':previous[fid].get('after'),
            'before':before_score, 'after':after_score,
            'status':'accepted' if accepted else 'fallback',
            'reason':'fixed-mapping-silhouette-passed' if accepted else 'fixed-mapping-silhouette-iou',
            'mappingPolicy':FIXED_MAPPING_VERSION,
            'sourceAlignmentKey':edit['alignmentKey'],
            'registrationChanged':False, 'resampled':False,
            'newGeometryUncoveredPixels':int(np.count_nonzero(after_mask & ~candidate_mask))}
        folder = work/'fixed-rgb'/SLUGS[fid]
        folder.mkdir(parents=True, exist_ok=True)
        (folder/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        hashes = {name:store.file_hash(view/name) for name in ('source.png','source.npy','aligned.npy')}
        evidence['faces'][fid] = {
            'sourceAlignmentKey':edit['alignmentKey'], 'sourceFiles':hashes,
            'sourceSilhouette':before_score, 'revisedSilhouette':after_score,
            'minIoU':min_iou, 'status':report['status'], 'sameAlignedRGB':True,
            'newGeometryUncoveredPixels':report['newGeometryUncoveredPixels'],
            'registrationChanged':False, 'resampled':False}
        source = {'path':str(view), 'provenance':{
            **edit.get('provenance', {}), 'depthRefinementMethod':method,
            'mappingPolicy':FIXED_MAPPING_VERSION, 'sourceRGBAsset':original['asset_id'],
            'sourceAlignmentKey':edit['alignmentKey']}}
        if original.get('reference'):
            ref = original['reference']
            source['reference'] = {**ref, 'path':str(store.contained(source_dir, ref['image']).parent)}
        items[fid] = {
            'key':store.key(FIXED_MAPPING_VERSION, original['asset_id'], revised['asset_id'],
                            fid, edit['alignmentKey'], hashes, report),
            'source':source, 'directory':str(folder), 'measured':str(view)}
    evidence['passed'] = all(face['status'] == 'accepted' for face in evidence['faces'].values())
    report_path.write_text(json.dumps(evidence, indent=2, allow_nan=False)+'\n')
    if not evidence['passed']:
        bad = [fid for fid, face in evidence['faces'].items() if face['status'] != 'accepted']
        raise ValueError('Fixed RGB silhouette QA failed for '+', '.join(bad)+
                         '; no re-registration or fallback-color comparison published. See '+str(report_path))
    return {'baseline':revised['asset_id'], 'faces':items}, evidence


def rebake_refinement(asset, proxy, work_dir, backend='webgl2'):
    directory = store.asset_dir(asset)
    original = store.validate(directory, asset, check_bytes=False)
    revised_id, _ = RebakeProxyRevision().revise(asset, proxy, backend)
    revised_dir = store.asset_dir(revised_id)
    revised = store.validate(revised_dir, revised_id)
    method = proxy.get('depthRefinement', {}).get('method', METHOD)
    aligned, fixed = reuse_fixed_rgb(directory, original, revised_dir, revised, work_dir, method)
    final = faces.compose(revised_id, aligned)
    after_dir = store.asset_dir(final)
    after = store.validate(after_dir, final)
    evaluation = {
        'sourceAsset':asset, 'assetId':final, 'geometryAsset':revised_id, 'faces':{},
        'mappingPolicy':FIXED_MAPPING_VERSION, 'fixedFrame':fixed['fixedFrame'],
        'silhouetteQaPassed':fixed['passed'], 'method':method,
        'normalMapChanged':after['textures']['normalMap']['sha256'] != original['textures']['normalMap']['sha256'],
        'intervalMapChanged':after['textures']['intervalMap']['sha256'] != original['textures']['intervalMap']['sha256'],
        'rangeMapChanged':after['textures']['rangeMap']['sha256'] != original['textures']['rangeMap']['sha256']}
    r = original['resolution']
    span = np.subtract(original['bakeBox']['max'], original['bakeBox']['min'])
    height = original['tightBounds']['max'][1]-original['tightBounds']['min'][1]
    for fid in store.FACES:
        old_view = store.contained(directory, original['faces'][fid]['edit']['view']).parent
        new_view = store.contained(after_dir, after['faces'][fid]['edit']['view']).parent
        hashes = {name:{'before':store.file_hash(old_view/name), 'after':store.file_hash(new_view/name)}
                  for name in ('source.png','source.npy','aligned.npy')}
        entry = {
            'sameSourceRGB':hashes['source.png']['before'] == hashes['source.png']['after'],
            'sameSourceSamples':hashes['source.npy']['before'] == hashes['source.npy']['after'],
            'sameAlignedRGB':hashes['aligned.npy']['before'] == hashes['aligned.npy']['after'],
            'imageHashes':hashes, 'samplingMappingUnchanged':True,
            'alignmentStatus':after['faces'][fid]['edit']['status'],
            'silhouetteQA':fixed['faces'][fid]}
        if not all(entry[key] for key in ('sameSourceRGB','sameSourceSamples','sameAlignedRGB')):
            raise ValueError('Composition changed frozen source/aligned samples: '+fid)
        if method != NORMAL_CONTROL_METHOD:
            target = np.load(Path(work_dir)/(SLUGS[fid]+'-full-target.npy'), allow_pickle=False)
            before = np.fromfile(store.contained(directory, original['faces'][fid]['depth']), '<f2').reshape(r,r).astype('f4')
            now = np.fromfile(store.contained(after_dir, after['faces'][fid]['depth']), '<f2').reshape(r,r).astype('f4')
            valid = np.isfinite(target) & np.isfinite(before) & np.isfinite(now)
            depth_span = abs(np.asarray(original['faces'][fid]['side']) @ span)
            entry.update(targetPixels=int(np.count_nonzero(np.isfinite(target))),
                         sharedPixels=int(valid.sum()),
                         missingAfterPixels=int(np.count_nonzero(np.isfinite(target) & ~np.isfinite(now))),
                         before=_residual_metrics(before,target,valid,depth_span,height),
                         after=_residual_metrics(now,target,valid,depth_span,height))
        evaluation['faces'][fid] = entry
    if method == NORMAL_CONTROL_METHOD:
        checks = {}
        for name, spec in original['textures'].items():
            if name != 'normalMap':
                checks[name+'Equal'] = spec['sha256'] == after['textures'][name]['sha256']
        for fid in store.FACES:
            for key in ('guide','depth'):
                checks[fid+key+'Equal'] = (
                    store.file_hash(store.contained(directory, original['faces'][fid][key])) ==
                    store.file_hash(store.contained(after_dir, after['faces'][fid][key])))
        evaluation['normalOnlyInvariants'] = checks
        if not all(checks.values()):
            (Path(work_dir)/'rebake-evaluation.json').write_text(json.dumps(evaluation, indent=2)+'\n')
            raise ValueError('Normal-only control changed color, coverage or geometry maps.')
    (Path(work_dir)/'rebake-evaluation.json').write_text(json.dumps(evaluation,indent=2,allow_nan=False)+'\n')
    return final, evaluation
