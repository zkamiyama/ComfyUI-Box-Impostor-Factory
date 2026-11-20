#!/usr/bin/env python3
"""Replay unchanged Inspector orthographic camera records against a Gaussian field."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from zip_render_faces import linear, sha, srgb, write_json


def replay_camera(matrix_world, projection_matrix, canonical_from_scene, width, height):
    c2w_gl = np.asarray(matrix_world, dtype=float).reshape(4, 4, order='F')
    projection = np.asarray(projection_matrix, dtype=float).reshape(4, 4, order='F')
    transform = np.asarray(canonical_from_scene, dtype=float)
    if not np.allclose(projection[3], [0, 0, 0, 1], atol=1e-8):
        raise ValueError('Recorded camera must be orthographic')
    if not np.allclose([projection[0, 1], projection[0, 2], projection[1, 0], projection[1, 2]], 0, atol=1e-8):
        raise ValueError('Skewed orthographic camera is not supported by gsplat')
    scale = np.linalg.norm(transform[:3, 0])
    rotation = transform[:3, :3] / scale
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or np.linalg.det(rotation) < .999:
        raise ValueError('Canonical transform must be a proper uniform similarity')
    c2w = np.linalg.inv(transform) @ c2w_gl
    c2w[:3, :3] *= scale
    c2w = c2w @ np.diag([1., -1., -1., 1.])
    K = np.array([[width * .5 * projection[0, 0] * scale, 0, width * (.5 + .5 * projection[0, 3])],
                  [0, height * .5 * projection[1, 1] * scale, height * (.5 - .5 * projection[1, 3])],
                  [0, 0, 1.]])
    span = -2 / projection[2, 2]
    near_plus_far = -projection[2, 3] * span
    near, far = (near_plus_far - span) / (2 * scale), (near_plus_far + span) / (2 * scale)
    return {'K': K, 'c2w': c2w, 'scale': scale, 'near': near, 'far': far}


def resolve_output(rgb, alpha, factor):
    if factor == 1:
        return np.clip(rgb / np.maximum(alpha[..., None], 1e-8), 0, 1), np.clip(alpha, 0, 1)
    h, w = alpha.shape
    straight = np.clip(rgb / np.maximum(alpha[..., None], 1e-8), 0, 1)
    weighted_linear = linear(straight) * alpha[..., None]
    output_alpha = alpha.reshape(h // factor, factor, w // factor, factor).mean((1, 3))
    weighted_linear = weighted_linear.reshape(h // factor, factor, w // factor, factor, 3).mean((1, 3))
    output_rgb = srgb(weighted_linear / np.maximum(output_alpha[..., None], 1e-8))
    return np.clip(output_rgb, 0, 1), np.clip(output_alpha, 0, 1)


def alpha_metrics(box_alpha, gs_alpha):
    box, gs = box_alpha.astype(float) / 255, gs_alpha.astype(float) / 255
    bmask, gmask = box > .5, gs > .5
    union, overlap = bmask | gmask, bmask & gmask
    difference = np.abs(box - gs)
    soft_union = np.maximum(box, gs).sum()
    active = (box > 0) | (gs > 0)
    return {'alphaMAEFull': float(difference.mean()), 'alphaMaxAbs': float(difference.max()),
            'alphaMAEActive': float(difference[active].mean()) if active.any() else None,
            'silhouetteIoU': float(overlap.sum() / union.sum()) if union.any() else None,
            'softAlphaIoU': float(np.minimum(box, gs).sum() / soft_union) if soft_union else None,
            'boxForegroundPixels': int(bmask.sum()), 'gsForegroundPixels': int(gmask.sum()),
            'foregroundUnionPixels': int(union.sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gaussians', type=Path, required=True)
    parser.add_argument('--transform', type=Path, required=True)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--supersample', type=int, default=1)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--save-depth', action='store_true')
    args = parser.parse_args()
    capture = json.loads(args.capture.read_text())
    transform = json.loads(args.transform.read_text())['canonical_from_scene']
    frames = capture['frames'][:args.limit] if args.limit else capture['frames']
    width, height = capture['camera']['pixels']
    factor = args.supersample
    identity = {'gaussiansSha256': sha(args.gaussians), 'transformSha256': sha(args.transform),
                'captureSha256': sha(args.capture), 'scriptSha256': sha(__file__),
                'supersample': factor, 'frames': len(frames), 'saveDepth': args.save_depth}
    args.output.mkdir(parents=True, exist_ok=True)
    status_path = args.output / 'replay.json'
    if status_path.exists():
        report = json.loads(status_path.read_text())
        if report['identity'] != identity:
            raise ValueError('Replay output belongs to different inputs/settings')
    else:
        report = {'identity': identity, 'cameraPixels': [width, height], 'framesExpected': len(frames),
                  'completed': False, 'frames': [], 'method': 'gsplat RGB+ED with exact recorded Inspector orthographic cameras',
                  'coordinatePolicy': 'Camera matrices converted through branch canonical_from_scene; Gaussian SH remains in original scene frame',
                  'colorComparison': 'Qualitative only: GS predicts image RGB; native Inspector Box applies its own material lighting and ACES.',
                  'alphaComparison': 'Same camera/projection, different volume/surface and antialiasing models; alpha equality is not expected.',
                  'sourceCapture': str(args.capture.resolve())}
    existing = {record['index']: record for record in report['frames']}
    for record in existing.values():
        if sha(record['path']) != record['sha256']:
            raise ValueError('Existing replay frame digest changed')
    if report['completed']:
        print(json.dumps({'status': 'already-complete', 'path': str(status_path)}, indent=2))
        return
    import torch
    from zipsplat import Camera, Pose
    from zipsplat.gaussians import Gaussians

    started = time.monotonic()
    packed = np.load(args.gaussians, allow_pickle=False)
    gaussians = Gaussians(torch.as_tensor(packed, device=args.device, dtype=torch.float32))
    torch.cuda.reset_peak_memory_stats(args.device)
    for frame in frames:
        index = frame['index']
        if index in existing:
            continue
        frame_camera = frame['camera']
        plan = replay_camera(frame_camera['matrixWorld'], frame_camera['projectionMatrix'], transform,
                             width * factor, height * factor)
        camera = Camera.from_K(torch.tensor(plan['K'], dtype=torch.float32), width * factor, height * factor)
        pose = Pose.from_4x4mat(torch.tensor(plan['c2w'], dtype=torch.float32))
        rendered, info = gaussians.render(camera, pose, mode='RGB+ED', camera_model='ortho',
                                          near_plane=plan['near'], far_plane=plan['far'])
        values = rendered[0].detach().float().cpu().numpy().transpose(1, 2, 0)
        alpha = info['alphas'][0, 0].detach().float().cpu().numpy()
        rgb, output_alpha = resolve_output(values[..., :3], alpha, factor)
        rgba = np.rint(np.concatenate((rgb, output_alpha[..., None]), 2) * 255).astype('uint8')
        path = args.output / f'frame-{index:04d}.png'
        Image.fromarray(rgba, 'RGBA').save(path)
        box_path = Path(frame['path'])
        if not box_path.is_absolute():
            box_path = args.capture.parent / box_path
        if sha(box_path) != frame['sha256']:
            raise ValueError('Source Box capture changed: ' + str(box_path))
        with Image.open(box_path) as image:
            box_alpha = np.asarray(image.convert('RGBA'))[..., 3]
        if box_alpha.shape != output_alpha.shape:
            raise ValueError('Source and replay image sizes differ')
        record = {'index': index, 'path': str(path.resolve()), 'sha256': sha(path),
                  'requestedYawDegrees': frame['requestedYawDegrees'],
                  'sourceFrameSha256': frame['sha256'], 'sourceCamera': frame_camera,
                  'rawCamera': {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in plan.items()},
                  'alpha': alpha_metrics(box_alpha, rgba[..., 3])}
        if args.save_depth:
            depth = values[..., 3] * plan['scale']
            weighted = np.where(alpha > 0, depth * alpha, 0).reshape(height, factor, width, factor).mean((1, 3))
            output_depth = np.divide(weighted, output_alpha, out=np.full_like(output_alpha, np.nan), where=output_alpha > .01)
            depth_path = args.output / f'depth-{index:04d}.npy'
            np.save(depth_path, output_depth.astype('f4'), allow_pickle=False)
            record['depth'] = {'path': str(depth_path.resolve()), 'sha256': sha(depth_path),
                               'encoding': 'Expected camera-axis Z in canonical units; NaN where alpha<=.01'}
        report['frames'].append(record)
        report['frames'].sort(key=lambda value: value['index'])
        report['secondsThisSession'] = time.monotonic() - started
        write_json(status_path, report)
        if index % 12 == 0:
            print('FRAME', index, 'seconds', round(time.monotonic() - started, 2), flush=True)
    report['completed'] = len(report['frames']) == len(frames)
    summary = {}
    for name in ('alphaMAEFull', 'alphaMaxAbs', 'alphaMAEActive', 'silhouetteIoU', 'softAlphaIoU'):
        values = [record['alpha'][name] for record in report['frames'] if record['alpha'][name] is not None]
        summary[name] = {'mean': float(np.mean(values)), 'min': float(min(values)), 'max': float(max(values))} if values else None
    report['alphaSummary'] = summary
    report['peakAllocatedMiB'] = torch.cuda.max_memory_allocated(args.device) / 2 ** 20
    write_json(status_path, report)
    print(json.dumps({'completed': report['completed'], 'frames': len(report['frames']), 'alphaSummary': summary}, indent=2), flush=True)


if __name__ == '__main__':
    main()
