"""Prepare a Hi3D geometry-only mesh for Box baking in the input proxy's frame.

Transfers the source's sampled base color onto the refined mesh for a diagnostic
Box baseline.  Generated six-face colors and PBR are still meant to be rebaked.
This is NOT a replacement for H3/SuperMat or a guarantee of mesh quality.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree
import trimesh


def key_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(mesh, role):
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) < 10:
        raise ValueError(f'{role} must be a nonempty triangle mesh')
    if not np.isfinite(mesh.vertices).all():
        raise ValueError(f'{role} has non-finite vertices')
    if np.min(mesh.extents) <= 0:
        raise ValueError(f'{role} contains a zero-length axis')


def sample_source_basecolor(source):
    visual = source.visual
    if visual.kind == 'vertex' and len(visual.vertex_colors) == len(source.vertices):
        return np.asarray(visual.vertex_colors, dtype=np.uint8)
    if visual.kind != 'texture' or visual.uv is None:
        raise ValueError('Source GLB has no UV or vertex colors to transfer')
    material = getattr(visual, 'material', None)
    base = getattr(material, 'baseColorTexture', None)
    if base is None:
        base = getattr(material, 'image', None)
    if base is None:
        raise ValueError('Source GLB has no baseColorTexture')
    image = np.asarray(base.convert('RGBA'), dtype=np.float32)
    h,w = image.shape[:2]
    uvs = np.asarray(visual.uv, dtype=np.float64)
    coords = np.stack([(1-np.clip(uvs[:, 1],0,1)) * (h-1),
                       np.clip(uvs[:,0],0,1) * (w-1)])
    sampled = np.stack([map_coordinates(image[..., i],coords,order=1,mode='nearest')
                        for i in range(4)], axis=-1)
    factor = getattr(material, 'baseColorFactor', None)
    if factor is not None:
        factor = np.asarray(factor, dtype=np.float32)
        if factor.shape == (4,):
            sampled *= factor[None,:]/(255. if np.max(factor)>1 else 1.)
    return np.clip(np.rint(sampled), 0, 255).astype(np.uint8)


def prepare(original, refined, smooth=0, max_nearest_p95=0.05):
    source = trimesh.load(original, force='mesh', process=False)
    hi = trimesh.load(refined, force='mesh', process=False)
    validate(source, 'source')
    validate(hi, 'Hi3D')
    if smooth:
        trimesh.smoothing.filter_taubin(hi, lamb=.5, nu=.53, iterations=int(smooth))
    # Input and output are already aligned in orientation by the inference
    # pipeline. Restore one isotropic scale; never independently stretch axes.
    center_src = source.bounds.mean(axis=0)
    center_hi = hi.bounds.mean(axis=0)
    ratio = float(source.extents[1] / hi.extents[1])
    hi.vertices = (hi.vertices - center_hi) * ratio + center_src
    extent_ratio = hi.extents / source.extents
    if not (0.75 < extent_ratio[0] < 1.25 and 0.75 < extent_ratio[2] < 1.25):
        raise ValueError(f'Hi3D shape differs excessively: width/depth ratios {extent_ratio}')
    nearest = cKDTree(source.vertices)
    distances, indices = nearest.query(hi.vertices, k=1, workers=-1)
    p95 = float(np.quantile(distances, .95))
    if p95 > float(max_nearest_p95) * float(source.extents[1]):
        raise ValueError(f'Hi3D differs excessively from source geometry: p95={p95}')
    source_colors = sample_source_basecolor(source)
    hi.visual = trimesh.visual.ColorVisuals(mesh=hi,vertex_colors=source_colors[indices])
    report = {
      'source':str(original), 'hi3d_input':str(refined),
      'source_sha256':key_file(original), 'hi3d_sha256':key_file(refined),
      'smooth_iterations':int(smooth), 'scale':ratio,
      'source_extents':np.round(source.extents,6).tolist(),
      'refined_extents':np.round(hi.extents,6).tolist(),
      'extent_ratio':np.round(extent_ratio,6).tolist(),
      'nearest_dist_p50_p95':np.round(np.quantile(distances,[.5,.95]),6).tolist(),
      'vertices':int(len(hi.vertices)), 'faces':int(len(hi.faces)),
      'has_vertex_color':hi.visual.kind == 'vertex'
    }
    return hi, report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--hi3d',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--smooth',type=int,choices=[0,15,40,80],default=0)
    parser.add_argument('--max-nearest-p95',type=float,default=.05)
    parser.add_argument('--report',type=Path)
    parser.add_argument('--input-manifest',type=Path)
    parser.add_argument('--canonical-meta',type=Path)
    args=parser.parse_args()
    if bool(args.input_manifest) != bool(args.canonical_meta):
        parser.error('--input-manifest and --canonical-meta must be supplied together')
    result,report=prepare(args.source,args.hi3d,args.smooth,args.max_nearest_p95)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    result.export(args.output)
    report['output_sha256']=key_file(args.output)
    if args.input_manifest:
        m=json.loads(args.input_manifest.read_text())
        meta={
           'transform':m['transform'], 'inverseTransform':m['inverseTransform'],
           'tightBounds':{'min':result.bounds[0].tolist(), 'max':result.bounds[1].tolist()},
           'rotationDeg':m.get('provenance',{}).get('rotationDegreesXYZ',[0,0,0]),
           'sourceSha256':m.get('provenance',{}).get('originalGlbSha256',report['source_sha256']),
           'glbSha256':report['output_sha256'],
           'refinement':{'backend':'hi3d-v1', 'resolution':512, 'mesh_sha256':report['hi3d_sha256'],
                         'smoothing':report['smooth_iterations'], 'coarseSha256':report['source_sha256']},
        }
        args.canonical_meta.parent.mkdir(parents=True,exist_ok=True)
        args.canonical_meta.write_text(json.dumps(meta,indent=2,allow_nan=False)+'\n')
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(report,ensure_ascii=False),flush=True)
if __name__=='__main__':
    main()
