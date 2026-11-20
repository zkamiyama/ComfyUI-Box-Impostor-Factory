#!/usr/bin/env python3
"""Derive common-resolution metric captures from an existing native Box asset."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from build_metric_box import FACES, safe_path, sha
from zip_render_faces import reduce_captures, srgb, write_json


def run(args):
    manifest_path = args.asset / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    source_r, resolution = manifest['resolution'], args.resolution
    if source_r % resolution or source_r < resolution:
        raise ValueError('Baseline reduction requires an integer downsampling factor')
    factor = source_r // resolution
    maps = {}
    for name in ('colorMap', 'normalMap'):
        descriptor = manifest['textures'][name]
        path = safe_path(args.asset, descriptor['path'])
        if sha(path) != descriptor['sha256']:
            raise ValueError('Baseline map digest changed: ' + name)
        maps[name] = np.fromfile(path, dtype='<f2').astype('f4').reshape(2 * source_r, 3 * source_r, 4)
    args.output.mkdir(parents=True, exist_ok=False)
    record = {'kind': 'Existing-Box-resolution-normalized-derivative-v1', 'resolution': resolution,
              'sourceResolution': source_r, 'sourceAssetId': manifest['asset_id'],
              'sourceAssetPath': str(args.asset.resolve()), 'templateSha256': sha(manifest_path),
              'sourceMapSha256': {name: manifest['textures'][name]['sha256'] for name in maps},
              'supersample': factor, 'alphaThreshold': args.alpha_threshold,
              'depthClusterTolerance': args.depth_tolerance,
              'depthPolicy': 'Reduce recorded expected-depth guides; no new 3D reconstruction',
              'colorPolicy': 'Area averaging premultiplied linear RGB inside the same foreground depth cluster',
              'normalPolicy': 'Area average original premultiplied encoded normals over the selected depth cluster',
              'limitation': 'New resolution and rebuilt paired intervals can change silhouette; prior 240-frame alpha equality is not assumed.',
              'faces': {}}
    for slug, fid in FACES.items():
        face = manifest['faces'][fid]
        if face['depthEncoding'] != 'linear-axis-0-near-1-far;nan-background' or face['imageRowOrder'] != 'top-to-bottom':
            raise ValueError('Unsupported baseline depth contract: ' + fid)
        depth_path = safe_path(args.asset, face['depth'])
        if face.get('depthSha256') and sha(depth_path) != face['depthSha256']:
            raise ValueError('Baseline depth digest changed: ' + fid)
        depth = np.fromfile(depth_path, dtype='<f2').astype('f4').reshape(source_r, source_r)
        x, y = face['tile']
        color = maps['colorMap'][y * source_r:(y + 1) * source_r, x * source_r:(x + 1) * source_r][::-1].copy()
        normal = maps['normalMap'][y * source_r:(y + 1) * source_r, x * source_r:(x + 1) * source_r][::-1].copy()
        alpha = color[..., 3]
        # Normalize the boundary convention to the common capture reducer; no gamma-space resize occurs.
        rgb_srgb = srgb(np.clip(color[..., :3] / np.maximum(alpha[..., None], 1e-8), 0, 1))
        z, coverage, premul, reduction = reduce_captures(rgb_srgb * alpha[..., None], alpha, depth,
            resolution, factor, args.alpha_threshold, args.depth_tolerance)
        valid = (alpha > .0005) & np.isfinite(depth) & (depth >= 0) & (depth <= 1)
        shape = (resolution, factor, resolution, factor)
        closest = np.where(valid, depth, np.inf).reshape(shape).min((1, 3))
        chosen = valid & (depth <= np.repeat(np.repeat(closest, factor, 0), factor, 1) + args.depth_tolerance)
        reduced_normal = (normal[..., :3] * chosen[..., None]).reshape(resolution, factor, resolution, factor, 3).mean((1, 3))
        reduced_normal[~np.isfinite(z)] = 0
        values = {'depth': z, 'alpha': coverage, 'premul-color': premul, 'premul-normal': reduced_normal.astype('f4')}
        directory = args.output / slug
        directory.mkdir()
        hashes = {}
        for name, value in values.items():
            path = directory / (name + '.npy')
            np.save(path, value, allow_pickle=False)
            hashes[name + 'Sha256'] = sha(path)
        preview = np.concatenate((np.clip(srgb(premul / np.maximum(coverage[..., None], 1e-8)), 0, 1), coverage[..., None]), 2)
        Image.fromarray(np.rint(preview * 255).astype('uint8'), 'RGBA').save(directory / 'color-preview.png')
        record['faces'][slug] = {**hashes, **reduction, 'sourceDepthSha256': sha(depth_path)}
        write_json(args.output / 'status.json', record)
    print(json.dumps({'output': str(args.output.resolve()), 'sourceAsset': manifest['asset_id'],
                      'sourceResolution': source_r, 'resolution': resolution,
                      'faces': {slug: face['finalValidPixels'] for slug, face in record['faces'].items()}}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--alpha-threshold', type=float, default=.5)
    parser.add_argument('--depth-tolerance', type=float, default=.01)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
