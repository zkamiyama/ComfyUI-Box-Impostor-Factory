"""Make jointly visible material texels agree with an accepted capture view."""
import json
import shutil
import warnings

import numpy as np
from scipy.ndimage import map_coordinates

from . import multiview, store


VERSION = 'visible-material-reprojection-v1'
CHANNELS = ('colorMap', 'normalMap', 'materialMap', 'emissiveMap')


def _reproject(directory, manifest, anchor_face, depth_tolerance_texels):
    r = manifest['resolution']
    descriptors = manifest['textures']
    atlases = {name: np.array(multiview.atlas(directory, descriptors[name],
               '<f2' if descriptors[name]['dtype'] == 'float16' else 'u1'))
               for name in CHANNELS if name in descriptors}
    views = {}
    for fid in store.FACES:
        face = manifest['faces'][fid]
        if face.get('depthEncoding') != 'linear-axis-0-near-1-far;nan-background' or face.get('imageRowOrder') != 'top-to-bottom':
            raise ValueError('Material reprojection requires normalized axis depth guides.')
        depths = np.memmap(store.contained(directory, face['depth']), mode='r', dtype='<f2', shape=(r, r))
        views[fid] = multiview.View(fid, r, face, depths, atlases['colorMap'], atlases['normalMap'], None)
    anchor = views[anchor_face]
    col, row = manifest['faces'][anchor_face]['tile']
    samples = {}
    for name, atlas in atlases.items():
        tile = atlas[row*r:(row+1)*r, col*r:(col+1)*r][::-1].astype(np.float32)
        if atlas.dtype == np.uint8:
            tile /= 255
        samples[name] = tile[..., :3] / np.maximum(anchor.color[..., 3:4].astype(np.float32), 1e-3)
    anchor_depth = anchor.depth.astype(np.float32)
    anchor_covered = (np.isfinite(anchor_depth) & (anchor.color[..., 3] > .99)).astype(np.float32)
    anchor_depth = np.nan_to_num(anchor_depth, nan=0.)
    yy, xx = np.indices((r, r), dtype=np.float32)
    u, v = (xx+.5)/r-.5, .5-(yy+.5)/r
    tolerance = depth_tolerance_texels/r + .0005  # Includes half-float depth quantization.
    reports = {}
    for fid, target in views.items():
        if abs(float(target.side @ anchor.side)) > .5:
            continue
        store.check_interrupted()
        depth = target.depth.astype(np.float32)
        alpha = target.color[..., 3].astype(np.float32)
        valid = np.isfinite(depth) & (alpha > .99)
        relative = (u[..., None]*target.right + v[..., None]*target.up +
                    (.5-np.nan_to_num(depth, nan=0.))[..., None]*target.side)
        sx = (.5+relative @ anchor.right)*r-.5
        sy = (.5-relative @ anchor.up)*r-.5
        coordinates = np.stack([sy.ravel(), sx.ravel()])
        projected_depth = map_coordinates(anchor_depth, coordinates, order=1, mode='constant', cval=0.).reshape(r, r)
        covered = map_coordinates(anchor_covered, coordinates, order=1, mode='constant', cval=0.).reshape(r, r)
        error = np.abs((.5-relative @ anchor.side)-projected_depth)
        visible = valid & (covered > .9999) & (error <= tolerance)
        reports[fid] = {'coveredPixels': int(valid.sum()), 'reprojectedPixels': int(visible.sum())}
        col, row = manifest['faces'][fid]['tile']
        # Grazing capture normals are deliberately not rejected: the Box shader can still blend those texels.
        for name, atlas in atlases.items():
            tile = atlas[row*r:(row+1)*r, col*r:(col+1)*r][::-1]
            projected = np.stack([map_coordinates(samples[name][..., c], coordinates, order=1,
                                  mode='nearest').reshape(r, r) for c in range(3)], axis=-1)
            if name == 'normalMap':
                normal = 2*projected-1
                normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-4)
                projected = normal*.5+.5
            packed = projected*alpha[..., None]
            if atlas.dtype == np.uint8:
                packed = np.rint(packed*255)
            tile[..., :3][visible] = packed[visible].astype(atlas.dtype)
    for name, atlas in atlases.items():
        descriptor = descriptors[name]
        suffix = '.rgba16f.bin' if descriptor['dtype'] == 'float16' else '.rgba8.bin'
        relative = 'maps/reprojected_'+name.removesuffix('Map')+suffix
        path = store.contained(directory, relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        atlas.tofile(path)
        descriptor.update(path=relative, sha256=store.file_hash(path), byteLength=path.stat().st_size)
    return {'method': VERSION, 'anchor': anchor_face, 'depthToleranceTexels': depth_tolerance_texels,
            'normalizedDepthTolerance': tolerance, 'minimumInterpolatedCoverage': .9999,
            'channels': list(atlases), 'faces': reports,
            'preserved': ['geometry', 'intervalMap', 'rangeMap', 'coverage', 'anchor', 'oppositeFace']}


def reconcile_views(asset, anchor_face='+Z', depth_tolerance_texels=2.0):
    if anchor_face not in store.FACES:
        raise ValueError('Unknown material anchor face.')
    if not np.isfinite(depth_tolerance_texels) or depth_tolerance_texels < 0:
        raise ValueError('Depth tolerance must be a finite nonnegative texel count.')
    directory = store.asset_dir(asset)
    manifest = store.validate(directory, asset)
    if manifest['faces'][anchor_face].get('edit', {}).get('status') != 'accepted':
        warnings.warn('Anchor face was not accepted; view consistency was skipped.', stacklevel=2)
        return asset
    asset_id = store.key(VERSION, multiview.VERSION, asset, anchor_face, depth_tolerance_texels)

    def write(stage):
        shutil.copytree(directory, stage, ignore=shutil.ignore_patterns('previews'))
        result = store.validate(stage, asset, check_bytes=False)
        report = _reproject(stage, result, anchor_face, depth_tolerance_texels)
        result.update(asset_id=asset_id, parentAsset=asset)
        result.setdefault('provenance', {})['viewConsistency'] = report
        qa = store.contained(stage, result.get('qa', 'qa/report.json'))
        evidence = json.loads(qa.read_bytes()) if qa.is_file() else {}
        evidence['viewConsistency'] = report
        evidence['crossView'] = multiview.evaluate_pairs(stage, result)
        qa.parent.mkdir(parents=True, exist_ok=True)
        qa.write_text(json.dumps(evidence, indent=2, allow_nan=False), encoding='utf-8')
        result['qa'] = qa.relative_to(stage).as_posix()
        (stage/'manifest.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')

    return store.publish(asset_id, write)
