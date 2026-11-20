#!/usr/bin/env python3
"""Create a Box normal-RGB-only depth variant or face-axis appearance diagnostic.

Sigma zero reproduces zip_render_faces.normal_from_depth. Positive sigma filters
only a temporary nearest-filled depth array before differentiation. Sharp jumps
are always detected in the unfiltered filled array using the existing 3/R rule.
--face-axis uses each constant face.side instead, without any depth gradients.
These face-axis normals diagnose appearance; they are not surface normals.
All original depth, alpha, color, interval, range, proxy and camera bytes remain.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import zipfile

import numpy as np
from scipy import ndimage

FACES = {'px': '+X', 'nx': '-X', 'py': '+Y', 'ny': '-Y', 'pz': '+Z', 'nz': '-Z'}
METHOD = 'expected-depth-normal-rgb-only-v1'
FACE_AXIS_METHOD = 'face-axis-normal-rgb-appearance-diagnostic-v1'
GEOMETRY_KEYS = ('schemaVersion', 'layout', 'proxy', 'proxySha256', 'resolution', 'tightBounds',
                 'bakeBox', 'transform', 'inverseTransform', 'faceOrder', 'faces')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def safe(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Asset path escapes its directory: ' + str(relative))
    return path


def files(root):
    return {path.relative_to(root).as_posix(): sha(path) for path in sorted(root.rglob('*')) if path.is_file()}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def alpha_sha(path):
    alpha = np.fromfile(path, dtype='<u2').reshape(-1, 4)[:, 3].copy()
    return hashlib.sha256(alpha.tobytes()).hexdigest()


def depth_normals(depth, alpha, face, span, resolution, sigma):
    """Existing finite-difference basis/signs, with optional temporary-array smoothing."""
    valid = np.isfinite(depth)
    side, up = np.asarray(face['side'], dtype='f4'), np.asarray(face['up'], dtype='f4')
    right = np.cross(up, side)
    jump = np.zeros(depth.shape, dtype=bool)
    work_difference = np.zeros(depth.shape, dtype='f4')
    if valid.any():
        nearest = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
        filled = depth[tuple(nearest)]
        original_dr, original_dc = np.gradient(filled)
        work = ndimage.gaussian_filter(filled, sigma, mode='reflect') if sigma > 0 else filled
        dr, dc = np.gradient(work) if sigma > 0 else (original_dr, original_dc)
        left, right_edge, bottom, top = face['frustum'][:4]
        step_c = right * ((right_edge - left) / resolution) - side * (dc[..., None] * span)
        step_r = -up * ((top - bottom) / resolution) - side * (dr[..., None] * span)
        normal = np.cross(step_r, step_c)
        normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-8)
        jump = (np.abs(original_dr) > 3 / resolution) | (np.abs(original_dc) > 3 / resolution)
        normal[jump] = side
        work_difference = (work - filled) * span
    else:
        normal = np.broadcast_to(side, (*depth.shape, 3)).copy()
    encoded = np.asarray((.5 + .5 * normal) * alpha[..., None], dtype='f4')
    foreground = valid & (alpha > 0)
    stats = {'foregroundPixels': int(foreground.sum()),
             'sharpJumpFallbackPixels': int((jump & foreground).sum()),
             'temporarySmoothDepthRmsCanonical': float(np.sqrt(np.mean(work_difference[foreground] ** 2))) if foreground.any() else 0.,
             'temporarySmoothDepthMaxCanonical': float(np.max(np.abs(work_difference[foreground]))) if foreground.any() else 0.}
    return encoded, stats


def angular_stats(encoded, alpha, valid):
    normal = 2 * encoded / np.maximum(alpha[..., None], 1e-8) - 1
    normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-8)
    mask = valid & (alpha > .5)
    interior = ndimage.binary_erosion(mask, iterations=3)
    angles = []
    for axis in (0, 1):
        first, second = [slice(None)] * 2, [slice(None)] * 2
        first[axis], second[axis] = slice(0, -1), slice(1, None)
        first, second = tuple(first), tuple(second)
        included = interior[first] & interior[second]
        dots = np.sum(normal[first] * normal[second], axis=-1)
        angles.append(np.degrees(np.arccos(np.clip(dots[included], -1, 1))))
    angles = np.concatenate(angles)
    weight = ndimage.gaussian_filter(mask.astype('f4'), 2.)
    smooth = np.stack([ndimage.gaussian_filter(np.where(mask, normal[..., axis], 0), 2.) /
                       np.maximum(weight, 1e-8) for axis in range(3)], axis=-1)
    smooth /= np.maximum(np.linalg.norm(smooth, axis=-1, keepdims=True), 1e-8)
    residual = np.degrees(np.arccos(np.clip(np.sum(normal * smooth, axis=-1)[interior], -1, 1)))
    quantiles = lambda values: {str(q): float(np.percentile(values, q)) for q in (50, 90, 95, 99)} if len(values) else None
    return {'interiorPixelsEroded3': int(interior.sum()), 'adjacentPairCount': int(len(angles)),
            'adjacentAngleDegrees': quantiles(angles), 'sigma2ResidualAngleDegrees': quantiles(residual)}


def run(args):
    baseline = args.baseline_asset.resolve(strict=True)
    template = baseline / 'manifest.json' if baseline.is_dir() else baseline
    source_root, captures, output = template.parent, args.face_captures.resolve(strict=True), args.output.resolve()
    if template.name != 'manifest.json' or output.is_relative_to(source_root) or output.is_relative_to(captures):
        raise ValueError('Use a native manifest and an output parent outside the input directories')
    if not math.isfinite(args.smooth_sigma) or args.smooth_sigma < 0:
        raise ValueError('Smooth sigma must be finite and nonnegative')
    if args.face_axis and args.smooth_sigma != 0:
        raise ValueError('--face-axis cannot be combined with a nonzero --smooth-sigma')
    method = FACE_AXIS_METHOD if args.face_axis else METHOD
    source, status = json.loads(template.read_text()), json.loads((captures / 'status.json').read_text())
    capture_hash = sha(captures / 'status.json')
    if source['provenance']['metricCapture']['captureStatusSha256'] != capture_hash:
        raise ValueError('Depth captures are not the baseline Box original captures')
    resolution = int(source['resolution'])
    if resolution != 512 or status['resolution'] != resolution or set(status['faces']) != set(FACES) or not status['completed']:
        raise ValueError('Require six complete original Box512 depth faces')
    descriptor = source['textures']['normalMap']
    contract = {'width': 3 * resolution, 'height': 2 * resolution, 'channels': 4, 'dtype': 'float16',
                'endian': 'little', 'rowOrder': 'increasingShaderV', 'colorSpace': 'data', 'alpha': 'coverage'}
    if any(descriptor.get(key) != value for key, value in contract.items()):
        raise ValueError('Unsupported native normal atlas contract')
    for value in [*source['textures'].values(), source.get('baseColor', {})]:
        if value and sha(safe(source_root, value['path'])) != value['sha256']:
            raise ValueError('Source texture digest changed: ' + value['path'])
    if sha(safe(source_root, source['proxy'])) != source['proxySha256']:
        raise ValueError('Source proxy digest changed')
    source_hashes = files(source_root)
    original = np.fromfile(safe(source_root, descriptor['path']), dtype='<u2').reshape(2 * resolution, 3 * resolution, 4)
    color_bits = np.fromfile(safe(source_root, source['textures']['colorMap']['path']), dtype='<u2').reshape(original.shape)
    if not np.array_equal(original[..., 3], color_bits[..., 3]):
        raise ValueError('Original normalMap/colorMap alpha bits differ; the official shader decodes normals with color alpha')
    updated = original.copy()
    depth_hashes, face_stats = {}, {}
    spans = np.subtract(source['bakeBox']['max'], source['bakeBox']['min'])
    for slug, fid in FACES.items():
        path = captures / slug / 'depth.npy'
        depth_hashes[slug] = sha(path)
        if depth_hashes[slug] != status['faces'][slug]['depthSha256']:
            raise ValueError('Source float32 depth digest changed: ' + slug)
        depth = np.load(path, allow_pickle=False).astype('f4')
        if depth.shape != (resolution, resolution):
            raise ValueError('Unexpected depth shape: ' + slug)
        face = source['faces'][fid]
        tx, ty = face['tile']
        region = np.s_[ty * resolution:(ty + 1) * resolution, tx * resolution:(tx + 1) * resolution]
        before = original[region][::-1].view('<f2').astype('f4')
        alpha = before[..., 3]
        if not np.isfinite(before).all() or np.any((alpha < 0) | (alpha > 1)):
            raise ValueError('Invalid native normal atlas values: ' + fid)
        active = alpha > 0
        if np.any(active & ~np.isfinite(depth)):
            raise ValueError('Original normal coverage lacks original depth support: ' + fid)
        span = float(abs(np.asarray(face['side']) @ spans))
        if args.face_axis:
            side = np.asarray(face['side'], dtype='f4')
            if not np.isfinite(side).all() or not np.isclose(np.linalg.norm(side), 1.):
                raise ValueError('Face-axis diagnostic requires a finite unit face.side: ' + fid)
            encoded = np.asarray((.5 + .5 * side) * alpha[..., None], dtype='f4')
            details = {'foregroundPixels': int(active.sum()), 'sharpJumpFallbackPixels': 0,
                       'temporarySmoothDepthRmsCanonical': 0., 'temporarySmoothDepthMaxCanonical': 0.,
                       'depthGradientsUsed': False, 'constantWorldNormal': side.tolist()}
        else:
            encoded, details = depth_normals(depth, alpha, face, span, resolution, args.smooth_sigma)
            details['depthGradientsUsed'] = True
        if not np.isfinite(encoded).all() or np.any(encoded < 0) or np.any(encoded > alpha[..., None] + 1e-6):
            raise ValueError('Non-finite or invalid encoded normal: ' + fid)
        tile = updated[region][::-1]
        tile[..., :3][active] = encoded.astype('<f2').view('<u2')[active]
        after = tile.view('<f2').astype('f4')
        changed = np.any(tile[..., :3] != original[region][::-1][..., :3], axis=-1)
        face_stats[fid] = {**details, 'sourceDepth': str(path), 'sourceDepthSha256': depth_hashes[slug],
                           'axisSpan': span, 'changedNormalRgbPixels': int(changed.sum()),
                           'before': angular_stats(before[..., :3], alpha, np.isfinite(depth)),
                           'after': angular_stats(after[..., :3], alpha, np.isfinite(depth))}
    if not np.array_equal(original[..., 3], updated[..., 3]):
        raise RuntimeError('Normal atlas alpha bits changed')
    identity = {'method': method, 'faceAxis': args.face_axis, 'sourceFiles': source_hashes, 'captureStatusSha256': capture_hash,
                'sourceDepthHashes': depth_hashes, 'smoothSigmaPixels': args.smooth_sigma, 'scriptSha256': sha(__file__)}
    asset_id = object_sha(identity)
    root, archive, receipt_path = output / asset_id, output / (asset_id + '.box.zip'), output / (asset_id + '.json')
    if any(path.exists() for path in (root, archive, receipt_path)):
        raise FileExistsError('Immutable output already exists: ' + str(root))
    output.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_root, root)
    try:
        if files(root) != source_hashes:
            raise RuntimeError('Source changed while copying')
        updated.tofile(safe(root, descriptor['path']))
        normal_sha = sha(safe(root, descriptor['path']))
        manifest = copy.deepcopy(source)
        mode = 'face-axis-normal-appearance-diagnostic' if args.face_axis else 'expected-depth-normal-only'
        manifest.update(asset_id=asset_id, parentAsset=source['asset_id'], mode=mode)
        manifest['textures']['normalMap']['sha256'] = normal_sha
        geometry_before = object_sha({key: source[key] for key in GEOMETRY_KEYS})
        geometry_after = object_sha({key: manifest[key] for key in GEOMETRY_KEYS})
        if geometry_before != geometry_after:
            raise RuntimeError('Geometry or camera metadata changed')
        rgba_alpha = {name: {'source': alpha_sha(safe(source_root, name)), 'output': alpha_sha(safe(root, name))}
                      for name in source_hashes if name.endswith('.rgba16f.bin')}
        if any(value['source'] != value['output'] for value in rgba_alpha.values()):
            raise RuntimeError('An original atlas alpha payload changed')
        preserved = {name: digest for name, digest in source_hashes.items() if name not in ('manifest.json', descriptor['path'])}
        actual = files(root)
        if any(actual.get(name) != digest for name, digest in preserved.items()):
            raise RuntimeError('An original file other than normal RGB/manifest changed')
        color_path = source['textures']['colorMap']['path']
        qa = {'method': method, 'mode': mode, 'faceAxis': args.face_axis, 'sourceAssetId': source['asset_id'], 'assetId': asset_id,
              'sourceManifestSha256': sha(template), 'captureStatusSha256': capture_hash,
              'normalPolicyOrigin': 'scripts/zipsplat_experiment/zip_render_faces.py::normal_from_depth',
              'smoothSigmaPixels': args.smooth_sigma,
              'temporaryDepthPolicy': 'Nearest finite foreground fills missing depth, then optional scipy gaussian_filter sigma in atlas pixels, reflect boundary; source depth never written',
              'sharpJumpPolicy': 'Original unsmoothed nearest-filled depth gradient: abs(dr)>3/R or abs(dc)>3/R; use face.side normal there, independent of smoothing',
              'normalFormula': 'normalize(cross(-up*H/R-side*span*dr, cross(up,side)*W/R-side*span*dc))',
              'encoding': '(0.5+0.5*worldNormal)*original normalMap alpha; preserve original uint16 alpha bits',
              'sourceNormalAndColorAlphaBitsIdentical': True,
              'officialShaderNormalDecode': 'normalMap.rgb/max(colorMap.a,0.001)*2-1',
              'faces': face_stats, 'normalAtlasAlphaBitDifferences': 0, 'rgbaAlphaChecks': rgba_alpha,
              'sourceNormalSha256': descriptor['sha256'], 'outputNormalSha256': normal_sha,
              'sourceColorSha256': sha(safe(source_root, color_path)), 'outputColorSha256': sha(safe(root, color_path)),
              'geometryMetadataSourceSha256': geometry_before, 'geometryMetadataOutputSha256': geometry_after,
              'preservedSourceFiles': preserved, 'depthAlphaColorIntervalRangeProxyCameraPreserved': True,
              'diagnosticMethod': 'Complete per-face foreground; alpha>0.5; erode mask by 3 pixels; adjacent horizontal/vertical normal-angle percentiles and sigma2 masked-normal residual; no crops or semantic regions',
              'limitations': ['Normals affect official Box lighting and face-color weights, so RGB appearance can change with identical colorMap.',
                             'Lower normal roughness is not proof of better shape accuracy.',
                             'Expected GS depth is not first-hit surface depth; derivative and nearest-fill artifacts remain possible.',
                             'Temporary filtering is shading-only; actual depth and silhouette geometry remain unchanged.']}
        if args.face_axis:
            qa.update(normalPolicyOrigin='manifest.faces[*].side; appearance diagnostic, not physical surface normals',
                      temporaryDepthPolicy='No depth filling, filtering or differentiation; depth is read only for original-asset integrity and coverage checks',
                      sharpJumpPolicy='Not applicable; constant face.side is used at every foreground pixel',
                      normalFormula='worldNormal=face.side; encodedRGB=(0.5+0.5*face.side)*originalAlpha',
                      limitations=qa['limitations'][:2] + [
                          'Constant face-axis normals are an appearance diagnostic and do not represent physical surface shape.',
                          'Suppressing local normal variation changes lighting and face-color selection; it may flatten shading or expose face transitions.',
                          'Depth, silhouette geometry, colorMap and every original alpha remain unchanged.'])
        qa_relative = 'qa/' + ('face-axis-normals-' if args.face_axis else 'depth-normals-') + asset_id[:16] + '.json'
        manifest['qa'] = qa_relative
        provenance_key = 'faceAxisNormals' if args.face_axis else 'expectedDepthNormals'
        manifest.setdefault('provenance', {})[provenance_key] = {
            'method': method, 'faceAxis': args.face_axis, 'sourceAsset': source['asset_id'], 'qa': qa_relative,
            'smoothSigmaPixels': args.smooth_sigma, 'normalRGBOnly': True, 'scriptSha256': sha(__file__)}
        write_json(safe(root, qa_relative), qa)
        (root / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
        if files(source_root) != source_hashes or sha(captures / 'status.json') != capture_hash:
            raise RuntimeError('A source asset or capture status changed during processing')
        if any(sha(captures / slug / 'depth.npy') != value for slug, value in depth_hashes.items()):
            raise RuntimeError('A source depth changed during processing')
        with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zipped:
            for path in sorted(root.rglob('*')):
                if path.is_file():
                    zipped.write(path, path.relative_to(root).as_posix())
        with zipfile.ZipFile(archive) as zipped:
            if zipped.testzip() is not None:
                raise RuntimeError('Box ZIP integrity check failed')
        receipt = {'assetId': asset_id, 'assetPath': str(root), 'archive': str(archive), 'archiveSha256': sha(archive),
                   'identity': identity, 'qa': qa, 'outputFiles': files(root)}
        write_json(receipt_path, receipt)
    except Exception:
        (root / 'INCOMPLETE').write_text('Normal-only build failed; do not use this asset.\n')
        raise
    return {'assetId': asset_id, 'assetPath': str(root), 'archive': str(archive), 'archiveSha256': sha(archive),
            'receiptPath': str(receipt_path), 'receiptSha256': sha(receipt_path), 'smoothSigmaPixels': args.smooth_sigma,
            'method': method, 'faceAxis': args.face_axis, 'normalAlphaBitsPreserved': True, 'colorMapBytesPreserved': True,
            'changedNormalRgbPixels': sum(value['changedNormalRgbPixels'] for value in face_stats.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-asset', type=Path, required=True)
    parser.add_argument('--face-captures', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smooth-sigma', type=float, default=0.)
    parser.add_argument('--face-axis', action='store_true',
                        help='Appearance diagnostic: constant face.side normals, no depth gradients; incompatible with nonzero smoothing')
    print(json.dumps(run(parser.parse_args()), indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
