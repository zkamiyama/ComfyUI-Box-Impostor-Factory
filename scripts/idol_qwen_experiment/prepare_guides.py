#!/usr/bin/env python3
"""Render full-body square Qwen guides in the existing Box face cameras."""
import argparse
import copy
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

FACES = {'front': '+Z', 'right': '+X', 'back': '-Z', 'left': '-X', 'top': '+Y', 'bottom': '-Y'}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save_png(path, values):
    Image.fromarray(np.uint8(np.clip(np.rint(values * 255), 0, 255))).save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--splat', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--comfy-root', type=Path, required=True)
    parser.add_argument('--factory-root', type=Path, required=True)
    parser.add_argument('--metric-renderer', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resolution', type=int, default=1024)
    parser.add_argument('--views', nargs='+', choices=list(FACES), default=['front'])
    parser.add_argument('--rotate-y', type=float, default=0.)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output directory')
    if args.resolution not in (512, 1024, 2048):
        parser.error('Supported square guide resolutions: 512, 1024, 2048')
    sys.path[:0] = [str(args.comfy_root), str(args.factory_root)]
    import torch
    from box_nodes.splat import camera_plan
    from comfy_api.latest import Types
    from comfy_extras import nodes_gaussian_splat as native

    spec = importlib.util.spec_from_file_location('idol_qwen_metric', args.metric_renderer)
    metric = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metric)
    with np.load(args.splat, allow_pickle=False) as arrays:
        splat = Types.SPLAT(*[torch.from_numpy(arrays[key].copy()) for key in
                             ('positions', 'scales', 'rotations', 'opacities', 'sh')])
    splat = native.TransformSplat.execute(splat, 0, 0, 0, 0, args.rotate_y, 0, 1, 1, 1).result[0]
    manifest = json.loads(args.manifest.read_text())
    manifest['resolution'] = args.resolution
    args.output.mkdir(parents=True)
    records = {}

    for view in args.views:
        fid = FACES[view]
        camera, rectangular_size, rectangular_content = camera_plan(manifest, fid, 'canonical_box')
        camera = copy.deepcopy(camera)
        frustum = manifest['faces'][fid]['frustum']
        physical = np.array([frustum[1] - frustum[0], frustum[3] - frustum[2]], dtype=float)
        distance = float(np.linalg.norm([camera['position'][k] - camera['target'][k] for k in 'xyz']))
        # Keep the exact orthographic face basis, while fitting its physical rectangle to a square.
        square_span = float(physical.max())
        camera['fov'] = 2 * math.degrees(math.atan(square_span / (2 * distance)))
        size = (args.resolution, args.resolution)
        content = physical / square_span * args.resolution

        def draw(style):
            renderer = metric.RenderSplat if style == 'metric_depth' else native.RenderSplat
            values = renderer.execute(splat=splat, width=size[0], height=size[1], frames=1,
                                      splat_scale=1., sharpen=1., headlight_shading=0.,
                                      opacity_threshold=0., background='#808080', render_style=style,
                                      camera_info=camera).result
            return [x[0].detach().float().cpu().numpy() for x in values]

        rgb, alpha = draw('color')
        depth_rgb, depth_alpha = draw('metric_depth')
        alpha_error = float(np.abs(alpha - depth_alpha).max())
        if alpha_error > 3e-5:
            raise ValueError('RGB and metric-depth coverage differ')
        spans = np.subtract(manifest['bakeBox']['max'], manifest['bakeBox']['min'])
        axis_span = float(abs(np.asarray(manifest['faces'][fid]['side']) @ spans))
        z = (depth_rgb[..., 0] - (distance - axis_span / 2)) / axis_span
        supported = alpha > .5
        valid = supported & np.isfinite(z) & (z >= 0) & (z <= 1)
        if np.count_nonzero(supported & ~valid):
            raise ValueError('The square guide clips depth outside the existing bake box')
        directory = args.output / view
        directory.mkdir()
        save_png(directory / 'rgb.png', rgb)
        save_png(directory / 'alpha.png', alpha)
        np.save(directory / 'alpha.npy', alpha.astype('f4'), allow_pickle=False)
        np.save(directory / 'depth.npy', np.where(valid, z, np.nan).astype('f4'), allow_pickle=False)
        depth_preview = np.where(valid, np.clip(1 - z, 0, 1), 0)
        save_png(directory / 'depth.png', depth_preview)
        with Image.open(directory / 'depth.png') as im:
            im.resize((512, 512), Image.Resampling.BILINEAR).convert('RGB').save(directory / 'depth512.png')
        yy, xx = np.nonzero(supported)
        edge_pixels = int(supported[0].sum() + supported[-1].sum() + supported[1:-1, 0].sum()
                          + supported[1:-1, -1].sum())
        record = {
            'kind': 'idol-qwen-square-guide-v1', 'view': view, 'face': fid,
            'sourceSplat': str(args.splat), 'splatSha256': sha(args.splat),
            'sourceManifest': str(args.manifest), 'manifestSha256': sha(args.manifest),
            'camera': camera, 'size': list(size), 'contentPixels': content.tolist(),
            'physicalSize': physical.tolist(), 'squarePhysicalSpan': square_span,
            'nativeRectangularPixels': list(rectangular_size),
            'nativeRectangularContent': rectangular_content.tolist(),
            'rotateYDegrees': args.rotate_y, 'alphaDifferenceMax': alpha_error,
            'fullBody': True, 'foregroundBoundsXYXY': [int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())],
            'foregroundPixels': int(supported.sum()), 'edgePixels': edge_pixels,
            'depthEncoding': 'metric camera-axis / existing bakeBox; near-white, black background',
            'imageEncoding': 'native sRGB composite over #808080',
            'nativeRendererSha256': sha(native.__file__), 'metricRendererSha256': sha(args.metric_renderer),
            'files': {name: {'path': str(directory / name), 'sha256': sha(directory / name)} for name in
                      ['rgb.png', 'alpha.png', 'alpha.npy', 'depth.npy', 'depth.png', 'depth512.png']}}
        if edge_pixels:
            raise ValueError('Full-body guide touches the image edge')
        (directory / 'guide.json').write_text(json.dumps(record, indent=2) + '\n')
        records[view] = record
        print(json.dumps({'view': view, 'directory': str(directory), 'edgePixels': edge_pixels}), flush=True)
    (args.output / 'guides.json').write_text(json.dumps(records, indent=2) + '\n')


if __name__ == '__main__':
    main()
