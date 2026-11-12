"""Compare colors at jointly visible proxy surface points in adjacent capture views."""
from itertools import combinations
import json

import numpy as np

from . import store


MAX_GRID = 128
MIN_SAMPLES = 24
MIN_COSINE = 0.25
VERSION = 'cross-view-v1'


def atlas(directory, descriptor, dtype):
    return np.memmap(store.contained(directory, descriptor['path']), mode='r', dtype=dtype,
                     shape=(descriptor['height'], descriptor['width'], 4))


class View:
    def __init__(self, face_id, resolution, face, depths, color, normal, material):
        self.face_id = face_id
        self.r = resolution
        self.side = np.array(face['side'], dtype=float)
        self.up = np.array(face['up'], dtype=float)
        self.right = np.cross(self.up, self.side)
        self.depth = depths
        self.extent = None
        if not np.allclose(np.stack([self.right,self.up,self.side]) @ np.stack([self.right,self.up,self.side]).T,
                           np.eye(3), atol=1e-6):
            raise ValueError('Invalid capture axes for '+face_id)
        col, row = face['tile']

        def tile(data):
            return data[row*resolution:(row+1)*resolution,col*resolution:(col+1)*resolution][::-1]

        self.color = tile(color)
        self.normal = tile(normal)
        self.material = None if material is None else tile(material)

    def sample(self, x, y, physical_lengths):
        # Five-point depth stencil removes silhouettes, disocclusions and narrow foreground fragments.
        border = (x > 0) & (y > 0) & (x < self.r-1) & (y < self.r-1)
        x = np.clip(x,1,self.r-2)
        y = np.clip(y,1,self.r-2)
        z = self.depth[y,x].astype(np.float32)
        alpha = self.color[y,x,3].astype(np.float32)
        inv_alpha = 1 / np.maximum(alpha,1e-3)
        normal = 2 * self.normal[y,x,:3].astype(np.float32) * inv_alpha[:,None] - 1
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True),1e-4)
        cosine = normal @ self.side
        valid = border & np.isfinite(z) & (alpha > .99) & (cosine > MIN_COSINE)
        thickness = abs(float(physical_lengths @ self.side))
        pitch = max(abs(float(physical_lengths @ self.right)),abs(float(physical_lengths @ self.up)))/self.r
        # Accepted slopes can change depth by pitch*tan(theta) per texel; larger jumps span surfaces.
        max_delta = (pitch / MIN_COSINE) / thickness + .001
        for ox,oy in [(-1,0),(1,0),(0,-1),(0,1)]:
            neighbor = self.depth[y+oy,x+ox].astype(np.float32)
            valid &= np.isfinite(neighbor) & (np.abs(neighbor-z) < max_delta)
        if self.material is not None:
            surface = self.material[y,x,:2].astype(np.float32) / 255
            valid &= ~((surface[:,0] < .35) & (surface[:,1] > .5))
        rgb = self.color[y,x,:3].astype(np.float32) * inv_alpha[:,None]
        valid &= np.isfinite(rgb).all(axis=1)
        u = (x+.5)/self.r - .5
        v = .5 - (y+.5)/self.r
        points = .5 + u[:,None]*self.right + v[:,None]*self.up + (.5-z)[:,None]*self.side
        return points, rgb, normal, valid, z


def one_direction(source, target, lengths):
    step = max(1, (source.r+MAX_GRID-1)//MAX_GRID)
    grid = np.arange(1,source.r-1,step)
    xx,yy = np.meshgrid(grid,grid)
    p, colors, normals, valid, _ = source.sample(xx.ravel(),yy.ravel(),lengths)
    eligible = int(np.count_nonzero(valid))
    valid &= (normals @ target.side) > MIN_COSINE
    p, colors, normals = p[valid], colors[valid], normals[valid]
    if not len(p):
        return np.empty(0), eligible
    relative = p-.5
    x = np.floor((.5+relative @ target.right)*target.r).astype(int)
    y = np.floor((.5-relative @ target.up)*target.r).astype(int)
    _, other, target_normals, visible, depths = target.sample(x,y,lengths)
    # Reprojection lands within half a pixel of the recorded target sample. Scale the
    # tolerance by that pixel's footprint along the tangent plane, in physical GLB units.
    spans = np.abs(np.stack([target.right,target.up,target.side]) @ lengths)
    dx = np.abs(target_normals @ target.right) * spans[0]
    dy = np.abs(target_normals @ target.up) * spans[1]
    axial = np.maximum(target_normals @ target.side,MIN_COSINE)
    depth_tolerance = (dx+dy) / (2 * target.r * axial * spans[2]) + .0015
    visible &= np.abs((.5-relative @ target.side) - depths) <= depth_tolerance
    visible &= np.sum(normals*target_normals,axis=1) > .75
    return np.mean(np.abs(colors[visible]-other[visible]),axis=1), eligible


def pair_report(first, second, lengths):
    errors_ab, eligible_ab = one_direction(first,second,lengths)
    errors_ba, eligible_ba = one_direction(second,first,lengths)
    samples = np.concatenate((errors_ab,errors_ba))
    eligible = eligible_ab+eligible_ba
    result = {
        'faces':[first.face_id,second.face_id], 'samples':len(samples),
        'eligibleSamples':eligible, 'overlapFraction':len(samples)/eligible if eligible else 0.,
        'directions':[{'from':first.face_id,'samples':len(errors_ab),'eligible':eligible_ab},
                      {'from':second.face_id,'samples':len(errors_ba),'eligible':eligible_ba}],
        'status':'insufficient-overlap','medianLinearRGBMAE':None,'p95LinearRGBMAE':None,
    }
    if len(samples) >= MIN_SAMPLES:
        result.update(status='measured',medianLinearRGBMAE=float(np.median(samples)),
                      p95LinearRGBMAE=float(np.percentile(samples,95)))
    return result


def evaluate_pairs(directory, manifest, *, face_keys=None, cache=None):
    """Return diagnostic measurements; inconclusive overlap never rejects a generated face."""
    r = manifest['resolution']
    lengths = np.array(manifest['bakeBox']['max']) - np.array(manifest['bakeBox']['min'])
    if not np.isfinite(lengths).all() or np.any(lengths<=0):
        raise ValueError('Invalid capture extents')
    result = {'method':VERSION,'metric':'mean absolute linear RGB difference per visible surface point',
              'minSamples':MIN_SAMPLES,'maxGridPerAxis':MAX_GRID,'minViewCosine':MIN_COSINE,
              'pairs':{},'worstPair':None,'status':'unavailable'}
    if r < 4 or any(not manifest['faces'][f].get('depth') for f in store.FACES):
        return {**result,'reason':'guide-depth-unavailable'}
    tex = manifest['textures']
    colors = atlas(directory,tex['colorMap'],'<f2')
    normals = atlas(directory,tex['normalMap'],'<f2')
    mat = atlas(directory,tex['materialMap'],'u1') if (
        'materialMap' in tex and {'metalness','roughness'} <= set(manifest['maps'])) else None
    result['specularMask'] = 'metalness>0.5 and roughness<0.35' if mat is not None else 'unavailable'
    views={}
    for f in store.FACES:
        entry=manifest['faces'][f]
        if entry.get('depthEncoding') != 'linear-axis-0-near-1-far;nan-background' or entry.get('imageRowOrder')!='top-to-bottom':
            raise ValueError('Unsupported guide depth encoding for '+f)
        depths = np.memmap(store.contained(directory,entry['depth']), mode='r', dtype='<f2', shape=(r,r))
        views[f]=View(f,r,entry,depths,colors,normals,mat)
    for a,b in combinations(store.FACES,2):
        if a[1] == b[1]:
            continue
        store.check_interrupted()
        def measure():
            return pair_report(views[a],views[b],lengths)
        if cache and face_keys is not None:
            key=store.key(VERSION,manifest['parentAsset'],a,face_keys[a],b,face_keys[b])
            def write(stage):
                (stage/'pair.json').write_text(json.dumps(measure(),allow_nan=False),encoding='utf-8')
            folder=cache('cross-view',key,write)
            summary=json.loads((folder/'pair.json').read_bytes())
        else:
            summary=measure()
        result['pairs'][a+'/'+b]=summary
    measured={key:pair for key,pair in result['pairs'].items() if pair['status']=='measured'}
    result['measuredPairs']=len(measured)
    result['status']='measured' if measured else 'insufficient-overlap'
    if measured:
        result['worstPair']=max(measured,key=lambda k:measured[k]['p95LinearRGBMAE'])
    return result
