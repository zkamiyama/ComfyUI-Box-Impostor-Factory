"""Prepare a Hi3D geometry-only mesh for Box baking in the input proxy's frame.

Transfers the source's sampled base color onto the refined mesh for a diagnostic
Box baseline.  Generated six-face colors and PBR are still meant to be rebaked.
This is NOT a replacement for H3/SuperMat or a guarantee of mesh quality.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree
import trimesh

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from box_nodes import depth_mesh


def key_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(mesh, role):
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) < 10:
        raise ValueError(f'{role} must be a nonempty triangle mesh')
    if not np.isfinite(mesh.vertices).all():
        raise ValueError(f'{role} has non-finite vertices')
    if np.min(mesh.extents) <= 0:
        raise ValueError(f'{role} contains a zero-length axis')


COLOR_TRANSFER_VERSION = 'linear-rgba-f32-v2'
LINEAR_COLOR_ATTRIBUTE = '_BOX_LINEAR_RGBA'


def _linear_rgba(values):
    values = np.asarray(values)
    if values.dtype.kind in 'ui':
        if values.dtype.kind != 'u' or values.dtype.itemsize not in (1, 2):
            raise ValueError('Integer vertex colors must be normalized uint8/uint16.')
        values = values.astype('f4') / np.iinfo(values.dtype).max
    else:
        values = values.astype('f4')
    if values.ndim != 2 or values.shape[1] not in (3, 4):
        raise ValueError('Expected per-vertex RGB or RGBA colors.')
    if values.shape[1] == 3:
        values = np.column_stack((values, np.ones(len(values), 'f4')))
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise ValueError('Expected finite linear RGBA in [0,1].')
    return values.astype('f4', copy=False)


def srgb_to_linear(rgb):
    rgb = np.asarray(rgb, dtype='f4')
    return np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4).astype('f4')


def _source_color_metadata(path, mesh):
    """Recover GLB float colors/factors before Trimesh's ColorVisuals quantization."""
    data = Path(path).read_bytes()
    if data[:4] != b'glTF':
        return None, None
    doc, binary = depth_mesh._decode(data)
    primitives = list(depth_mesh._primitives(doc, binary))
    if len(primitives) != 1:
        if any('COLOR_0' in p[0]['attributes'] and
               doc['accessors'][p[0]['attributes']['COLOR_0']]['componentType'] == 5126
               for p in primitives):
            raise ValueError('Cannot preserve FLOAT32 source colors across merged GLB primitives.')
        return None, None
    primitive, positions, _, transform = primitives[0]
    material = doc.get('materials', [])
    factor = None
    if 'material' in primitive:
        factor = material[primitive['material']].get('pbrMetallicRoughness', {}).get(
            'baseColorFactor', [1,1,1,1])
    index = primitive['attributes'].get('COLOR_0')
    if index is None:
        return None, factor
    world = positions @ transform[:3,:3].T + transform[:3,3]
    if world.shape != mesh.vertices.shape or not np.allclose(world, mesh.vertices, rtol=0, atol=1e-6):
        raise ValueError('Source COLOR_0 and loaded vertex ordering do not agree.')
    spec = doc['accessors'][index]
    if spec.get('sparse') or spec['type'] not in ('VEC3','VEC4'):
        raise ValueError('Sparse or non-vector source colors are unsupported.')
    dtype = np.dtype({5121:'u1',5123:'<u2',5126:'<f4'}[spec['componentType']])
    view = doc['bufferViews'][spec['bufferView']]
    components = 3 if spec['type'] == 'VEC3' else 4
    if view.get('buffer',0) != 0:
        raise ValueError('Source colors must be in the embedded GLB buffer.')
    colors = np.ndarray((spec['count'],components), dtype, buffer=binary,
                        offset=view.get('byteOffset',0)+spec.get('byteOffset',0),
                        strides=(view.get('byteStride',components*dtype.itemsize),dtype.itemsize))
    if dtype.kind != 'f' and not spec.get('normalized'):
        raise ValueError('Integer glTF COLOR_0 must be normalized.')
    return _linear_rgba(colors), factor


def sample_source_basecolor(source, *, vertex_rgba=None, base_color_factor=None):
    """Return straight linear FLOAT32 RGBA; decode textures before interpolation."""
    visual = source.visual
    material = getattr(visual, 'material', None)
    base = getattr(material, 'baseColorTexture', None)
    if base is None:
        base = getattr(material, 'image', None)
    if vertex_rgba is None and visual.kind == 'vertex':
        vertex_rgba = _linear_rgba(visual.vertex_colors)
    elif vertex_rgba is not None:
        vertex_rgba = _linear_rgba(vertex_rgba)
    if base is not None and visual.kind == 'texture' and visual.uv is not None:
        image = np.asarray(base.convert('RGBA'), dtype='f4') / 255.
        image[...,:3] = srgb_to_linear(image[...,:3])
        h,w = image.shape[:2]
        uvs = np.asarray(visual.uv, dtype=np.float64)
        coords = np.stack([(1-np.clip(uvs[:,1],0,1))*(h-1),
                           np.clip(uvs[:,0],0,1)*(w-1)])
        sampled = np.stack([map_coordinates(image[...,i],coords,order=1,mode='nearest')
                            for i in range(4)], axis=-1)
        if vertex_rgba is not None:
            sampled *= vertex_rgba
    elif vertex_rgba is not None and len(vertex_rgba) == len(source.vertices):
        sampled = vertex_rgba.copy()
    else:
        raise ValueError('Source GLB has no sampleable base texture or vertex colors.')
    factor = base_color_factor if base_color_factor is not None else getattr(material, 'baseColorFactor', None)
    if factor is not None:
        factor = np.asarray(factor, dtype='f4')
        if factor.shape != (4,) or not np.isfinite(factor).all():
            raise ValueError('Invalid baseColorFactor.')
        if np.max(factor) > 1:
            factor = factor / 255.  # Trimesh stores PBRMaterial factors as uint8.
        sampled *= factor[None,:]  # Factors and alpha are linear.
    return np.clip(sampled,0,1).astype('f4')


def export_prepared_glb(mesh, output=None):
    """Export the precise custom float attribute as standard glTF COLOR_0.

    Trimesh's ColorVisuals preview is uint8. Its custom-attribute path preserves
    FLOAT32; redirecting COLOR_0 only changes JSON, not any geometry/normal bytes.
    No vertex_normals property is evaluated unless its cache already exists.
    """
    colors = mesh.vertex_attributes.get(LINEAR_COLOR_ATTRIBUTE)
    if colors is None:
        raise ValueError('Mesh was not prepared with precise linear vertex colors.')
    colors = _linear_rgba(colors)
    if colors.shape != (len(mesh.vertices),4):
        raise ValueError('Prepared color count differs from the mesh.')
    include_normals = 'vertex_normals' in mesh._cache
    data = mesh.export(file_type='glb', include_normals=include_normals)
    doc, binary = depth_mesh._decode(data)
    for obj in doc['meshes']:
        for primitive in obj['primitives']:
            attributes = primitive['attributes']
            index = attributes.pop(LINEAR_COLOR_ATTRIBUTE)
            spec = doc['accessors'][index]
            if spec['componentType'] != 5126 or spec['type'] != 'VEC4' or spec['count'] != len(mesh.vertices):
                raise ValueError('Exporter did not retain FLOAT32 RGBA.')
            if spec.get('normalized'):
                raise ValueError('FLOAT32 glTF colors must not be normalized integers.')
            attributes['COLOR_0'] = index
    encoded = depth_mesh._encode(doc, bytearray(binary))
    if output is not None:
        Path(output).write_bytes(encoded)
    return encoded


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
    raw_vertex_rgba, raw_factor = _source_color_metadata(original, source)
    source_colors = sample_source_basecolor(source, vertex_rgba=raw_vertex_rgba,
                                           base_color_factor=raw_factor)
    linear_colors = source_colors[indices].astype('f4')
    # Preview-only quantized colors; export_prepared_glb uses the precise attribute.
    hi.visual = trimesh.visual.ColorVisuals(mesh=hi,vertex_colors=np.rint(linear_colors*255).astype('u1'))
    hi.vertex_attributes[LINEAR_COLOR_ATTRIBUTE] = linear_colors
    report = {
      'source':str(original), 'hi3d_input':str(refined),
      'source_sha256':key_file(original), 'hi3d_sha256':key_file(refined),
      'smooth_iterations':int(smooth), 'scale':ratio,
      'source_extents':np.round(source.extents,6).tolist(),
      'refined_extents':np.round(hi.extents,6).tolist(),
      'extent_ratio':np.round(extent_ratio,6).tolist(),
      'nearest_dist_p50_p95':np.round(np.quantile(distances,[.5,.95]),6).tolist(),
      'vertices':int(len(hi.vertices)), 'faces':int(len(hi.faces)),
      'has_vertex_color':hi.visual.kind == 'vertex',
      'color_transfer_version':COLOR_TRANSFER_VERSION,
      'color_space':'linear RGBA; texture RGB decoded before interpolation; vertex RGB already linear',
      'color_attribute':'FLOAT32 COLOR_0',
      'normal_policy':'retain existing cached normals only; never compute or add normals'
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
    export_prepared_glb(result,args.output)
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
                         'smoothing':report['smooth_iterations'], 'coarseSha256':report['source_sha256'],
                          'colorTransferVersion':COLOR_TRANSFER_VERSION},
        }
        args.canonical_meta.parent.mkdir(parents=True,exist_ok=True)
        args.canonical_meta.write_text(json.dumps(meta,indent=2,allow_nan=False)+'\n')
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(report,ensure_ascii=False),flush=True)
if __name__=='__main__':
    main()
