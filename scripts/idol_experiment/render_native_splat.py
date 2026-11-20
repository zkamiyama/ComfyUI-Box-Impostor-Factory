#!/usr/bin/env python3
"""Capture a saved ComfyUI SPLAT with fixed Box cameras, without Gaussian filtering.

faces writes 512px metric captures consumed unchanged by build_metric_box.py.
turntable writes full 512px RGB frames plus coverage PNGs and exact camera records.
export only writes the rotated native SPLAT through ComfyUI's official PLY writer.
Requires the existing native ComfyUI renderer and experimental_gaussian_depth.py.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

FACES = {'px': '+X', 'nx': '-X', 'py': '+Y', 'ny': '-Y', 'pz': '+Z', 'nz': '-Z'}
ARRAYS = ('depth', 'alpha', 'premul-color', 'premul-normal')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def srgb_to_linear(value):
    return np.where(value <= .04045, value / 12.92, ((np.maximum(value, 0) + .055) / 1.055) ** 2.4)


def linear_to_srgb(value):
    return np.where(value <= .0031308, value * 12.92, 1.055 * np.maximum(value, 0) ** (1 / 2.4) - .055)


def save_png(path, value):
    Image.fromarray(np.uint8(np.clip(np.rint(value * 255), 0, 255))).save(path)


def coverage_stats(alpha, threshold=.5):
    mask = alpha > threshold
    yy, xx = np.nonzero(mask)
    edge = np.zeros_like(mask)
    edge[[0, -1], :] = True
    edge[:, [0, -1]] = True
    return {'threshold': threshold, 'pixels': int(mask.sum()),
            'boundsXYXY': [int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())] if len(xx) else None,
            'edgePixels': int((mask & edge).sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['faces', 'turntable', 'export'])
    parser.add_argument('--splat', type=Path, required=True, help='Native batch-one SPLAT NPZ, or Gaussian PLY')
    parser.add_argument('--template', type=Path, required=True, help='Common native Box manifest')
    parser.add_argument('--output', type=Path, required=True, help='New capture directory')
    parser.add_argument('--comfy-root', type=Path, required=True)
    parser.add_argument('--factory-root', type=Path, required=True)
    parser.add_argument('--metric-renderer', type=Path, help='Existing experimental_gaussian_depth.py; required for faces')
    parser.add_argument('--export-ply', type=Path, help='Save transformed SPLAT through native _gaussian_ply_bytes')
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--rotate-y', type=float, default=90., help='Same native SPLAT transform as previous TripoSplat')
    parser.add_argument('--frames', type=int, default=240)
    parser.add_argument('--fps', type=int, default=24)
    parser.add_argument('--turntable-padding', type=float, default=1., help='Common view-plane multiplier; never an individual object fit')
    args = parser.parse_args()
    if args.resolution < 64 or args.resolution > 2048 or args.frames < 1 or args.fps < 1:
        parser.error('Resolution must be 64..2048; frames and fps must be positive')
    if args.turntable_padding < 1 or not math.isfinite(args.turntable_padding):
        parser.error('Turntable padding must be finite and at least 1')
    if args.mode == 'faces' and args.metric_renderer is None:
        parser.error('faces requires --metric-renderer')
    if args.mode == 'export' and args.export_ply is None:
        parser.error('export requires --export-ply')
    for name in ('splat', 'template', 'comfy_root', 'factory_root'):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    if args.metric_renderer is not None:
        args.metric_renderer = args.metric_renderer.resolve(strict=True)
    args.output = args.output.resolve()
    if args.output.exists():
        parser.error('Output already exists; use a new directory')
    if args.export_ply is not None:
        args.export_ply = args.export_ply.resolve()
        if args.export_ply.exists():
            parser.error('Export PLY already exists; use a new path')

    # Load external GPU dependencies only after CLI validation, so --help is portable.
    sys.path[:0] = [str(args.comfy_root), str(args.factory_root)]
    import torch
    from box_nodes.splat import camera_plan, atlas_pixels
    from comfy_api.latest import Types
    from comfy_extras import nodes_gaussian_splat as native
    from comfy_extras.nodes_camera import _rotate_camera_info

    if args.splat.suffix.lower() == '.npz':
        with np.load(args.splat, allow_pickle=False) as arrays:
            tensors = [torch.from_numpy(arrays[key].copy()) for key in
                       ('positions', 'scales', 'rotations', 'opacities', 'sh')]
        splat = Types.SPLAT(*tensors)
    elif args.splat.suffix.lower() == '.ply':
        # Official inverse of native SaveSplat's storage convention, with all SH coefficients retained.
        tensors = [torch.from_numpy(np.array(array, copy=True))[None] for array in
                   native._parse_ply_gaussian(args.splat.read_bytes())]
        tensors[3] = tensors[3].reshape(1, -1, 1)
        splat = Types.SPLAT(*tensors)
    else:
        parser.error('Only native SPLAT .npz and Gaussian .ply are supported')
    if splat.positions.ndim != 3 or splat.positions.shape[0] != 1 or splat.positions.shape[1] == 0:
        raise ValueError('Expected a nonempty batch-one SPLAT')
    splat = native.TransformSplat.execute(splat, 0, 0, 0, 0, args.rotate_y, 0, 1, 1, 1).result[0]
    metric = None
    if args.mode == 'faces':
        spec = importlib.util.spec_from_file_location('idol_metric_renderer', args.metric_renderer)
        metric = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(metric)

    manifest = json.loads(args.template.read_text())
    manifest['resolution'] = args.resolution
    args.output.mkdir(parents=True)
    began = time.monotonic()
    record = {'kind': 'native-splat-' + args.mode + '-v1', 'completed': False,
              'sourceSplat': str(args.splat), 'splatSha256': sha(args.splat),
              'template': str(args.template), 'templateSha256': sha(args.template),
              'resolution': args.resolution, 'splatRotationNativeY': args.rotate_y,
              'gaussianCount': int(splat.positions.shape[1]),
              'nativeRenderer': str(Path(native.__file__).resolve()),
              'nativeRendererSha256': sha(native.__file__), 'scriptSha256': sha(__file__),
              'renderSettings': {'splatScale': 1., 'sharpen': 1., 'headlightShading': 0.,
                                 'opacityThreshold': 0., 'background': '#808080'},
              'gaussianFiltering': 'none', 'errors': []}

    if args.export_ply is not None:
        count = native._real_len(splat, 0)
        args.export_ply.parent.mkdir(parents=True, exist_ok=True)
        values = [getattr(splat, name)[0, :count] for name in
                  ('positions', 'scales', 'rotations', 'opacities', 'sh')]
        with args.export_ply.open('xb') as stream:
            stream.write(native._gaussian_ply_bytes(*values))
        record['exportPLY'] = {'path': str(args.export_ply), 'sha256': sha(args.export_ply),
                               'gaussianCount': count, 'bytes': args.export_ply.stat().st_size,
                               'writer': 'comfy_extras.nodes_gaussian_splat._gaussian_ply_bytes',
                               'coordinates': 'native SPLAT after TransformSplat rotate_y, no further axis conversion'}
    if args.mode == 'export':
        record['completed'] = True
        record['totalSeconds'] = round(time.monotonic() - began, 3)
        write_json(args.output / 'status.json', record)
        print('COMPLETE ' + str(args.output / 'status.json'), flush=True)
        return

    def render(camera, size, style):
        renderer = metric.RenderSplat if style == 'metric_depth' else native.RenderSplat
        color, alpha = renderer.execute(splat=splat, width=int(size[0]), height=int(size[1]),
                                       frames=1, splat_scale=1., sharpen=1., headlight_shading=0.,
                                       opacity_threshold=0., background='#808080', render_style=style,
                                       camera_info=camera).result
        return (color[0].detach().float().cpu().numpy(), alpha[0].detach().float().cpu().numpy())

    if args.mode == 'faces':
        record.update(metricRenderer=str(args.metric_renderer), metricRendererSha256=sha(args.metric_renderer),
                      depthSemantics='front-weighted expected camera-axis depth, supported where native alpha > 0.5',
                      calibration='absolute camera depth normalized by common bakeBox axis span',
                      resampling='existing camera_plan plus atlas_pixels at Baker texel centers', faces={})
        for slug, fid in FACES.items():
            started = time.monotonic()
            directory = args.output / slug
            directory.mkdir()
            face = manifest['faces'][fid]
            camera, size, content = camera_plan(manifest, fid, 'canonical_box')
            delta = np.array([camera['position'][axis] - camera['target'][axis] for axis in 'xyz'])
            distance = float(np.linalg.norm(delta))
            span = np.subtract(manifest['bakeBox']['max'], manifest['bakeBox']['min'])
            axis_span = float(abs(np.asarray(face['side']) @ span))
            if not math.isfinite(axis_span) or axis_span <= 0:
                raise ValueError('Invalid bakeBox axis span: ' + fid)
            metric_rgb, alpha = render(camera, size, 'metric_depth')
            color, alpha_c = render(camera, size, 'color')
            normal, alpha_n = render(camera, size, 'normal')
            alpha_error = max(float(np.abs(alpha - alpha_c).max()), float(np.abs(alpha - alpha_n).max()))
            if alpha_error > 3e-5:
                raise ValueError('Native and metric renderers produce different coverage: ' + fid)
            depth = (metric_rgb[..., 0] - (distance - axis_span / 2)) / axis_span
            supported = alpha > .5
            valid = supported & np.isfinite(depth) & (depth >= 0) & (depth <= 1)
            weight = np.where(valid, alpha, 0)
            coverage = atlas_pixels(weight, content, args.resolution).astype('f4')
            numerator = atlas_pixels(np.where(valid, depth * alpha, 0), content, args.resolution)
            z = np.divide(numerator, coverage, out=np.full(coverage.shape, np.nan, dtype='f4'), where=coverage > .5)
            valid_output = np.isfinite(z)
            background_linear = srgb_to_linear(128 / 255)
            premul = np.maximum(0, srgb_to_linear(np.clip(color, 0, 1)) - (1 - alpha[..., None]) * background_linear)
            rgb = atlas_pixels(np.where(valid[..., None], premul, 0), content, args.resolution).astype('f4')
            rgb = np.minimum(rgb, coverage[..., None])
            raw = np.stack((2 * normal[..., 0] - 1, 1 - 2 * normal[..., 1], 1 - 2 * normal[..., 2]), axis=-1)
            side, up = np.asarray(face['side']), np.asarray(face['up'])
            world = raw[..., 0, None] * np.cross(up, side) - raw[..., 1, None] * up - raw[..., 2, None] * side
            world /= np.maximum(np.linalg.norm(world, axis=-1, keepdims=True), 1e-6)
            premul_normal = (.5 + .5 * world) * weight[..., None]
            normals = atlas_pixels(premul_normal, content, args.resolution).astype('f4')
            rgb[~valid_output], normals[~valid_output], coverage[~valid_output] = 0, 0, 0
            arrays = dict(zip(ARRAYS, (z, coverage, rgb, normals)))
            entry = {'face': fid, 'camera': camera, 'renderPixels': list(size), 'contentPixels': content.tolist(),
                     'axisSpan': axis_span, 'distance': distance, 'alphaDifferenceMax': alpha_error,
                     'outOfBoundsAlphaPixels': int((supported & ~valid).sum()),
                     'nativeCoverage': coverage_stats(alpha), 'atlasCoverage': coverage_stats(coverage),
                     'depthMin': float(z[valid_output].min()) if valid_output.any() else None,
                     'depthMax': float(z[valid_output].max()) if valid_output.any() else None}
            for name, value in arrays.items():
                path = directory / (name + '.npy')
                np.save(path, value.astype('f4'), allow_pickle=False)
                entry[name + 'Sha256'] = sha(path)
            straight = np.divide(rgb, coverage[..., None], out=np.zeros_like(rgb), where=coverage[..., None] > 0)
            save_png(directory / 'color-preview.png', np.dstack((linear_to_srgb(straight), coverage)))
            save_png(directory / 'depth-preview.png', np.nan_to_num(1 - z))
            entry['seconds'] = round(time.monotonic() - started, 3)
            record['faces'][slug] = entry
            write_json(args.output / 'status.json', record)
            print(json.dumps({'face': fid, 'pixels': entry['atlasCoverage']['pixels'],
                              'outOfBoundsAlphaPixels': entry['outOfBoundsAlphaPixels'],
                              'seconds': entry['seconds']}), flush=True)
    else:
        face = manifest['faces']['+Z']
        camera, _, _ = camera_plan(manifest, '+Z', 'canonical_box')
        physical = [abs(face['frustum'][1] - face['frustum'][0]), abs(face['frustum'][3] - face['frustum'][2])]
        delta = np.array([camera['position'][axis] - camera['target'][axis] for axis in 'xyz'])
        distance = float(np.linalg.norm(delta))
        camera['fov'] = 2 * math.degrees(math.atan(max(physical) * args.turntable_padding / (2 * distance)))
        record.update(camera={'projection': 'orthographic', 'pixels': [args.resolution] * 2,
                              'baseCamera': camera, 'turntablePadding': args.turntable_padding},
                      framesExpected=args.frames, fps=args.fps, frames=[],
                      imageEncoding='native sRGB composite over #808080; separate 8-bit native coverage PNG')
        (args.output / 'alpha').mkdir()
        for index in range(args.frames):
            yaw = index * 360. / args.frames
            frame_camera = _rotate_camera_info(copy.deepcopy(camera), (0., 1., 0.), yaw)
            color, alpha = render(frame_camera, (args.resolution, args.resolution), 'color')
            image_path = args.output / f'frame-{index:04d}.png'
            alpha_path = args.output / 'alpha' / f'frame-{index:04d}.png'
            save_png(image_path, color)
            save_png(alpha_path, alpha)
            record['frames'].append({'index': index, 'requestedYawDegrees': yaw, 'camera': frame_camera,
                                     'path': str(image_path), 'sha256': sha(image_path),
                                     'alphaPath': str(alpha_path), 'alphaSha256': sha(alpha_path),
                                     'coverage': coverage_stats(alpha)})
            if index % 12 == 0 or index + 1 == args.frames:
                write_json(args.output / 'status.json', record)
                print(json.dumps({'frame': index + 1, 'frames': args.frames,
                                  'seconds': round(time.monotonic() - began, 3)}), flush=True)
    record['completed'] = True
    record['totalSeconds'] = round(time.monotonic() - began, 3)
    write_json(args.output / 'status.json', record)
    print('COMPLETE ' + str(args.output / 'status.json'), flush=True)


if __name__ == '__main__':
    main()
