#!/usr/bin/env python3
"""Select foreground Gaussians, fix a canonical frame, and render metric Box faces."""
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

FACES = {'px': '+X', 'nx': '-X', 'py': '+Y', 'ny': '-Y', 'pz': '+Z', 'nz': '-Z'}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def select_foreground(args):
    packed = np.load(args.gaussians, allow_pickle=False)
    cameras = json.loads(args.cameras.read_text())
    masks = json.loads(args.masks.read_text())['views']
    mask_by_id = {record['id']: record for record in masks}
    means = packed[:, :3]
    visible = np.zeros(len(means), dtype='i4')
    inside = np.zeros_like(visible)
    records = []
    for index, view in enumerate(cameras['views']):
        record = mask_by_id[view['id']]
        path = Path(record['mask_path'])
        path = path if path.is_absolute() else args.masks.parent / path
        digest = sha(path)
        if record.get('mask_sha256') not in (None, digest):
            raise ValueError('Mask digest changed: ' + str(path))
        width, height, fx, fy, cx, cy = cameras['original_intrinsics'][index]
        with Image.open(path) as image:
            mask = np.asarray(image.convert('L')) > 127
        if mask.shape != (round(height), round(width)):
            raise ValueError('Mask must have original image dimensions: ' + str(path))
        if args.mask_dilate:
            mask = ndimage.maximum_filter(mask, size=2 * args.mask_dilate + 1)
        c2w = np.asarray(cameras['c2w'][index])
        camera_points = (means - c2w[:3, 3]) @ c2w[:3, :3]
        z = camera_points[:, 2]
        positive = z > 1e-6
        u = camera_points[:, 0] / np.maximum(z, 1e-6) * fx + cx
        v = camera_points[:, 1] / np.maximum(z, 1e-6) * fy + cy
        observed = positive & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        indices = np.flatnonzero(observed)
        hit = mask[np.floor(v[indices]).astype(int), np.floor(u[indices]).astype(int)]
        visible[indices] += 1
        inside[indices[hit]] += 1
        records.append({'id': view['id'], 'mask': str(path.resolve()), 'sha256': digest,
                        'projectedGaussians': len(indices), 'foregroundProjected': int(hit.sum())})
    accepted = (inside >= args.min_views) & (inside >= args.mask_fraction * np.maximum(visible, 1))
    accepted &= np.isfinite(packed).all(axis=1) & (packed[:, 10] >= args.min_opacity)
    scale_filter = None
    if args.max_sigma_height_fraction is not None:
        if args.transform is None or args.template is None:
            raise ValueError('--max-sigma-height-fraction requires --transform and --template')
        transform = np.asarray(json.loads(args.transform.read_text())['canonical_from_scene'])
        frame = json.loads(args.template.read_text())
        uniform_scale = float(np.linalg.norm(transform[:3, 0]))
        height = frame['bakeBox']['max'][1] - frame['bakeBox']['min'][1]
        sigma_fraction = packed[:, 3:6].max(1) * uniform_scale / height
        before_count = int(accepted.sum())
        accepted &= sigma_fraction <= args.max_sigma_height_fraction
        scale_filter = {'maxSigmaHeightFraction': args.max_sigma_height_fraction,
                        'referenceHeight': height, 'canonicalUniformScale': uniform_scale,
                        'maxSigmaSceneUnits': args.max_sigma_height_fraction * height / uniform_scale,
                        'transformSha256': sha(args.transform), 'frameSha256': sha(args.template),
                        'acceptedBeforeScaleLimit': before_count,
                        'removedByScaleLimit': before_count - int(accepted.sum()),
                        'acceptedSigmaFractionQuantiles': dict(zip(('50', '95', '99', '100'),
                            map(float, np.percentile(sigma_fraction[accepted], [50, 95, 99, 100])))) if accepted.any() else None}
    args.output.mkdir(parents=True, exist_ok=False)
    np.save(args.output / 'selection.npy', accepted, allow_pickle=False)
    np.save(args.output / 'gaussians.npy', packed[accepted], allow_pickle=False)
    report = {'method': 'Gaussian-center multiview foreground-mask intersection',
              'inputSha256': sha(args.gaussians), 'camerasSha256': sha(args.cameras),
              'masksManifestSha256': sha(args.masks), 'settings': {'minViews': args.min_views,
              'maskFraction': args.mask_fraction, 'minOpacity': args.min_opacity,
              'maskDilatePixels': args.mask_dilate}, 'inputCount': len(packed),
              'acceptedCount': int(accepted.sum()), 'views': records,
              'scaleFilter': scale_filter,
              'acceptedForegroundVotes': {str(i): int((accepted & (inside == i)).sum()) for i in range(len(records) + 1)},
              'acceptedInFrameViews': {str(i): int((accepted & (visible == i)).sum()) for i in range(len(records) + 1)},
              'selectedSha256': sha(args.output / 'gaussians.npy'),
              'selectionSha256': sha(args.output / 'selection.npy'),
              'allMeanBounds': [means.min(0).tolist(), means.max(0).tolist()],
              'selectedMeanBounds': [means[accepted].min(0).tolist(), means[accepted].max(0).tolist()] if accepted.any() else None,
              'limitation': 'Center projection is a visual-hull filter, not visibility-aware segmentation of splat extents.'}
    write_json(args.output / 'selection.json', report)
    print(json.dumps(report, indent=2), flush=True)


def alignment(args):
    packed = np.load(args.gaussians, allow_pickle=False)
    cameras = json.loads(args.cameras.read_text())
    template = json.loads(args.template.read_text())
    first = np.asarray(cameras['c2w'][0], dtype=float)
    rotation = np.diag([1., -1., -1.]) @ first[:3, :3].T
    points = packed[:, :3] @ rotation.T
    low, high = np.quantile(points, [args.quantile, 1 - args.quantile], axis=0)
    target = template['tightBounds']
    target_low, target_high = np.asarray(target['min']), np.asarray(target['max'])
    scale = (target_high[1] - target_low[1]) / (high[1] - low[1])
    center = (low + high) / 2
    target_center = (target_low + target_high) / 2
    matrix = np.eye(4)
    matrix[:3, :3] = scale * rotation
    matrix[:3, 3] = target_center - scale * center
    canonical_low = (low - center) * scale + target_center
    canonical_high = (high - center) * scale + target_center
    padding = np.max(canonical_high - canonical_low) * .02
    bounds = {'min': (canonical_low - padding).tolist(), 'max': (canonical_high + padding).tolist()}
    canonical_points = (points - center) * scale + target_center
    outside = ((canonical_points < np.asarray(bounds['min'])) | (canonical_points > np.asarray(bounds['max']))).any(1)
    report = {'canonical_from_scene': matrix.tolist(),
              'method': 'First estimated input camera defines front/up; robust foreground height fits template height',
              'gaussiansSha256': sha(args.gaussians), 'camerasSha256': sha(args.cameras),
              'templateSha256': sha(args.template), 'quantile': args.quantile,
              'scale': float(scale), 'sceneRotatedRobustBounds': [low.tolist(), high.tolist()],
              'sceneRotatedAllBounds': [points.min(0).tolist(), points.max(0).tolist()],
              'targetTightBounds': target, 'derivedBakeBox': bounds,
              'selectedMeanCount': len(points), 'selectedMeansOutsideBakeBox': int(outside.sum()),
              'instruction': 'Reuse this exact matrix and common frame for every refinement branch.',
              'limitation': 'Normalization fixes gauge and framing, not calibrated geometric alignment to the original object.'}
    write_json(args.output, report)
    if args.frame_dir:
        args.frame_dir.mkdir(parents=True, exist_ok=False)
        frame = {key: template[key] for key in ('schemaVersion', 'asset_id', 'layout', 'proxy',
                  'proxySha256', 'transform', 'inverseTransform', 'faceOrder', 'resolution')}
        frame['tightBounds'] = {'min': canonical_low.tolist(), 'max': canonical_high.tolist()}
        frame['bakeBox'] = bounds
        frame['faces'] = {fid: make_face(bounds, template['faces'][fid]) for fid in FACES.values()}
        frame['provenance'] = {'canonicalFrame': report, 'proxyIsMetadataOnly': True}
        proxy_source = (args.template.parent / template['proxy']).resolve()
        proxy_target = (args.frame_dir / template['proxy']).resolve()
        if not proxy_source.is_relative_to(args.template.parent.resolve()) or not proxy_target.is_relative_to(args.frame_dir.resolve()):
            raise ValueError('Proxy path escapes frame directory')
        proxy_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(proxy_source, proxy_target)
        write_json(args.frame_dir / 'manifest.json', frame)
    print(json.dumps(report, indent=2), flush=True)


def make_face(bounds, reference_face):
    low, high = np.asarray(bounds['min']), np.asarray(bounds['max'])
    size, center = high - low, (high + low) / 2
    side, up = np.asarray(reference_face['side']), np.asarray(reference_face['up'])
    right = np.cross(up, side)
    half_depth = float(np.abs(side) @ size) / 2
    half_width, half_height = float(np.abs(right) @ size) / 2, float(np.abs(up) @ size) / 2
    margin = float(size.max()) * .1
    near, far = margin / 2, margin * 1.5 + 2 * half_depth
    c2w = np.eye(4)
    c2w[:3, :3] = np.column_stack((right, up, side))
    c2w[:3, 3] = center + side * (half_depth + margin)
    projection = np.eye(4)
    projection[0, 0], projection[1, 1] = 1 / half_width, 1 / half_height
    projection[2, 2], projection[2, 3] = -2 / (far - near), -(far + near) / (far - near)
    return {key: reference_face[key] for key in ('up', 'side', 'tile')} | {
        'viewMatrix': np.linalg.inv(c2w).flatten(order='F').tolist(),
        'projectionMatrix': projection.flatten(order='F').tolist(),
        'frustum': [-half_width, half_width, -half_height, half_height, near, far]}


def camera_for_face(manifest, face_id, canonical_from_scene, resolution):
    transform = np.asarray(canonical_from_scene, dtype=float)
    scale = np.linalg.norm(transform[:3, 0])
    rotation = transform[:3, :3] / scale
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or np.linalg.det(rotation) < .999:
        raise ValueError('canonical_from_scene must be a proper uniform similarity transform')
    face = manifest['faces'][face_id]
    c2w_gl = np.linalg.inv(np.asarray(face['viewMatrix']).reshape(4, 4, order='F'))
    raw = np.linalg.inv(transform) @ c2w_gl
    raw[:3, :3] *= scale
    raw = raw @ np.diag([1., -1., -1., 1.])
    left, right, bottom, top, near, far = face['frustum']
    K = np.array([[resolution / (right - left) * scale, 0, -left * resolution / (right - left)],
                  [0, resolution / (top - bottom) * scale, top * resolution / (top - bottom)],
                  [0, 0, 1.]])
    side = np.asarray(face['side'], dtype=float)
    bounds = manifest['bakeBox']
    span = np.asarray(bounds['max']) - np.asarray(bounds['min'])
    center = (np.asarray(bounds['max']) + np.asarray(bounds['min'])) / 2
    axis_span = float(np.abs(side) @ span)
    box_near = float((c2w_gl[:3, 3] - center) @ side - axis_span / 2)
    return {'K': K, 'c2w': raw, 'scale': scale, 'axis_span': axis_span, 'box_near': box_near,
            'near': near / scale, 'far': far / scale, 'canonical_c2w': c2w_gl}


def linear(srgb):
    return np.where(srgb <= .04045, srgb / 12.92, ((np.maximum(srgb, 0) + .055) / 1.055) ** 2.4)


def srgb(linear_values):
    return np.where(linear_values <= .0031308, 12.92 * linear_values,
                    1.055 * np.maximum(linear_values, 0) ** (1 / 2.4) - .055)


def reduce_captures(rgb_premul_srgb, alpha, depth, resolution, factor, threshold, depth_tolerance):
    valid = (alpha > .0005) & np.isfinite(depth) & (depth >= 0) & (depth <= 1)
    rgb = np.divide(rgb_premul_srgb, np.maximum(alpha[..., None], 1e-8))
    premul = linear(np.clip(rgb, 0, 1)) * alpha[..., None]
    shape = (resolution, factor, resolution, factor)
    block_depth = np.where(valid, depth, np.inf).reshape(shape)
    closest = block_depth.min((1, 3))
    chosen = valid & (depth <= np.repeat(np.repeat(closest, factor, 0), factor, 1) + depth_tolerance)
    weights = np.where(chosen, alpha, 0)
    coverage = weights.reshape(shape).mean((1, 3))
    weighted_z = np.where(chosen, depth * alpha, 0).reshape(shape).mean((1, 3))
    z = np.divide(weighted_z, coverage, out=np.full_like(coverage, np.nan), where=coverage > threshold)
    color = (premul * chosen[..., None]).reshape(resolution, factor, resolution, factor, 3).mean((1, 3))
    final_valid = np.isfinite(z)
    coverage[~final_valid] = 0
    color[~final_valid] = 0
    return np.asarray(z, dtype='f4'), np.asarray(coverage, dtype='f4'), np.asarray(color, dtype='f4'), {
        'rawValidPixels': int(valid.sum()), 'depthClusterRejectedPixels': int((valid & ~chosen).sum()),
        'finalValidPixels': int(final_valid.sum())}


def normal_from_depth(depth, alpha, face, span, resolution):
    valid = np.isfinite(depth)
    side = np.asarray(face['side'], dtype='f4')
    up = np.asarray(face['up'], dtype='f4')
    right = np.cross(up, side)
    if valid.any():
        nearest = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
        filled = depth[tuple(nearest)]
        dr, dc = np.gradient(filled)
        left, right_edge, bottom, top = face['frustum'][:4]
        step_c = right * ((right_edge - left) / resolution) - side * (dc[..., None] * span)
        step_r = -up * ((top - bottom) / resolution) - side * (dr[..., None] * span)
        normal = np.cross(step_r, step_c)
        normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-8)
        discontinuity = (np.abs(dr) > 3 / resolution) | (np.abs(dc) > 3 / resolution)
        normal[discontinuity] = side
    else:
        normal = np.broadcast_to(side, (*depth.shape, 3)).copy()
    return np.asarray((.5 + .5 * normal) * alpha[..., None], dtype='f4')


def render(args):
    import torch
    from zipsplat import Camera, Pose
    from zipsplat.gaussians import Gaussians

    began = time.monotonic()
    manifest = json.loads(args.template.read_text())
    transform = json.loads(args.transform.read_text())['canonical_from_scene']
    packed = np.load(args.gaussians, allow_pickle=False)
    gaussians = Gaussians(torch.as_tensor(packed, device=args.device, dtype=torch.float32))
    resolution, factor = args.resolution, args.supersample
    raw_size = resolution * factor
    args.output.mkdir(parents=True, exist_ok=False)
    record = {'kind': 'ZipSplat-six-orthographic-expected-depth-v1', 'resolution': resolution,
              'splatSHA256': sha(args.gaussians), 'transformSha256': sha(args.transform),
              'templateSha256': sha(args.template), 'canonical_from_scene': transform,
              'gaussians': len(packed), 'faces': {}, 'device': args.device,
              'supersample': factor, 'alphaThreshold': args.alpha_threshold,
              'depthClusterTolerance': args.depth_tolerance,
              'projection': 'gsplat camera_model=ortho, exact rectangular physical frusta',
              'colorPolicy': 'Model RGB treated as sRGB; unpremultiply, linearize, then premultiply before area reduction',
              'normalPolicy': 'Expected-depth finite differences; face normal at >3/R depth jumps',
              'depthPolicy': 'ED camera-axis mean, canonical scale, normalize from near Box side; NaN invalid',
              'foregroundPolicy': 'Input Gaussian field must already be selected using multiview masks; opacity alone is not segmentation'}
    torch.cuda.reset_peak_memory_stats(args.device)
    for slug, fid in FACES.items():
        start = time.monotonic()
        plan = camera_for_face(manifest, fid, transform, raw_size)
        camera = Camera.from_K(torch.tensor(plan['K'], dtype=torch.float32), raw_size, raw_size)
        pose = Pose.from_4x4mat(torch.tensor(plan['c2w'], dtype=torch.float32))
        rendered, info = gaussians.render(camera, pose, mode='RGB+ED', camera_model='ortho',
                                          near_plane=plan['near'], far_plane=plan['far'])
        rgba_depth = rendered[0].detach().float().cpu().numpy().transpose(1, 2, 0)
        alpha = info['alphas'][0, 0].detach().float().cpu().numpy()
        raw_depth = (rgba_depth[..., 3] * plan['scale'] - plan['box_near']) / plan['axis_span']
        visible = alpha > .0005
        depth, coverage, color, reduction = reduce_captures(rgba_depth[..., :3], alpha, raw_depth,
            resolution, factor, args.alpha_threshold, args.depth_tolerance)
        normal = normal_from_depth(depth, coverage, manifest['faces'][fid], plan['axis_span'], resolution)
        directory = args.output / slug
        directory.mkdir()
        values = {'depth': depth, 'alpha': coverage, 'premul-color': color, 'premul-normal': normal}
        hashes = {}
        for name, value in values.items():
            path = directory / (name + '.npy')
            np.save(path, value, allow_pickle=False)
            hashes[name + 'Sha256'] = sha(path)
        rgb = srgb(color / np.maximum(coverage[..., None], 1e-8))
        preview = np.concatenate((np.clip(rgb, 0, 1), coverage[..., None]), axis=2)
        Image.fromarray(np.rint(preview * 255).astype('uint8'), 'RGBA').save(directory / 'color-preview.png')
        Image.fromarray(np.rint(np.nan_to_num(1 - depth) * 255).astype('uint8')).save(directory / 'depth-preview.png')
        record['faces'][slug] = {**hashes, **reduction, 'axisSpan': plan['axis_span'],
                                'rawOpacityPixels': int(visible.sum()),
                                'rawDepthBeforeBoxPixels': int((visible & (raw_depth < 0)).sum()),
                                'rawDepthBehindBoxPixels': int((visible & (raw_depth > 1)).sum()),
                                'rawCameraK': plan['K'].tolist(), 'rawC2W': plan['c2w'].tolist(),
                                'depthMin': float(np.nanmin(depth)) if np.isfinite(depth).any() else None,
                                'depthMax': float(np.nanmax(depth)) if np.isfinite(depth).any() else None,
                                'seconds': time.monotonic() - start}
        write_json(args.output / 'status.json', record)
        print(fid, record['faces'][slug]['finalValidPixels'], flush=True)
    record.update(totalSeconds=time.monotonic() - began,
                  peakAllocatedMiB=torch.cuda.max_memory_allocated(args.device) / 2 ** 20)
    write_json(args.output / 'status.json', record)
    print(json.dumps({'output': str(args.output), 'seconds': record['totalSeconds'],
                      'peakAllocatedMiB': record['peakAllocatedMiB']}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    select = commands.add_parser('select')
    select.add_argument('--gaussians', type=Path, required=True)
    select.add_argument('--cameras', type=Path, required=True)
    select.add_argument('--masks', type=Path, required=True, help='JSON views with matching id and mask_path')
    select.add_argument('--output', type=Path, required=True)
    select.add_argument('--min-views', type=int, default=1)
    select.add_argument('--mask-fraction', type=float, default=.5)
    select.add_argument('--min-opacity', type=float, default=.01)
    select.add_argument('--mask-dilate', type=int, default=3)
    select.add_argument('--max-sigma-height-fraction', type=float,
                        help='Optional maximum Gaussian sigma divided by the fixed canonical reference height')
    select.add_argument('--transform', type=Path, help='Existing branch canonical transform for the sigma limit')
    select.add_argument('--template', type=Path, help='Existing common frame whose bakeBox defines reference height')
    select.set_defaults(run=select_foreground)
    align = commands.add_parser('align')
    align.add_argument('--gaussians', type=Path, required=True)
    align.add_argument('--cameras', type=Path, required=True)
    align.add_argument('--template', type=Path, required=True)
    align.add_argument('--output', type=Path, required=True)
    align.add_argument('--frame-dir', type=Path, help='Create a shared Box frame with bounds derived from selected Gaussians')
    align.add_argument('--quantile', type=float, default=.005)
    align.set_defaults(run=alignment)
    draw = commands.add_parser('render')
    draw.add_argument('--gaussians', type=Path, required=True)
    draw.add_argument('--template', type=Path, required=True)
    draw.add_argument('--transform', type=Path, required=True)
    draw.add_argument('--output', type=Path, required=True)
    draw.add_argument('--resolution', type=int, default=512)
    draw.add_argument('--supersample', type=int, default=1)
    draw.add_argument('--alpha-threshold', type=float, default=.5)
    draw.add_argument('--depth-tolerance', type=float, default=.01)
    draw.add_argument('--device', default='cuda')
    draw.set_defaults(run=render)
    args = parser.parse_args()
    args.run(args)


if __name__ == '__main__':
    main()
