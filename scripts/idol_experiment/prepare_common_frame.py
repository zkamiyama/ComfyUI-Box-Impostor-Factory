#!/usr/bin/env python3
"""Give all registered native SPLATs one Box frame without scaling or filtering."""
import argparse
import copy
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--template', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--comfy-root', type=Path, required=True)
    parser.add_argument('--face-helper-dir', type=Path, required=True)
    parser.add_argument('--padding-height-fraction', type=float, default=.03)
    args = parser.parse_args()
    if args.output.exists() or not 0 < args.padding_height_fraction < .2:
        parser.error('Use a new output directory and padding between 0 and 0.2')
    sys.path[:0] = [str(args.comfy_root.resolve()), str(args.face_helper_dir.resolve())]
    import torch
    from comfy_extras.nodes_gaussian_splat import _euler_to_quat, _quat_to_mat
    from zip_render_faces import make_face

    source = json.loads(args.template.read_text())
    definitions = json.loads(args.models.read_text())['models']
    records, lows, highs = {}, [], []
    for item in definitions:
        path = Path(item['splat']).resolve(strict=True)
        with np.load(path, allow_pickle=False) as data:
            points = torch.from_numpy(data['positions'][0].copy())
        quaternion = _euler_to_quat(0, item['rotate_y'], 0).to(dtype=points.dtype)
        rotation = _quat_to_mat(quaternion[None])[0]
        points = (points @ rotation.T).numpy()
        if not len(points) or not np.isfinite(points).all():
            raise ValueError('SPLAT positions must be nonempty and finite')
        low, high = points.min(0).astype(float), points.max(0).astype(float)
        lows.append(low)
        highs.append(high)
        records[item['id']] = {'splat': str(path), 'splatSha256': sha(path),
            'rotateYDegrees': item['rotate_y'], 'gaussians': len(points),
            'allMeanBounds': [low.tolist(), high.tolist()]}
    low, high = np.min(lows, axis=0), np.max(highs, axis=0)
    padding = float(high[1] - low[1]) * args.padding_height_fraction
    previous = source['bakeBox']
    half_extent = np.maximum(np.maximum(np.abs(low), np.abs(high)) + padding,
                             np.maximum(np.abs(previous['min']), np.abs(previous['max'])))
    bounds = {'min': (-half_extent).tolist(), 'max': half_extent.tolist()}
    report = {'method': 'Origin-centered union of all registered Gaussian centers plus common height padding',
        'models': records, 'paddingHeightFraction': args.padding_height_fraction,
        'paddingCanonicalUnits': padding, 'bakeBox': bounds, 'previousBakeBox': previous,
        'gaussianFiltering': 'none', 'scaleChanges': False, 'translationChanges': False,
        'templateSha256': sha(args.template), 'scriptSha256': sha(__file__),
        'limitations': ['Gaussian tails can exceed center bounds; verify rendered depth and edge coverage.',
                        'Rigid Y registration aligns display direction, not pose or shape to ground truth.',
                        'The retained proxy is metadata only and is not used to generate geometry.']}
    frame = {key: copy.deepcopy(source[key]) for key in ('schemaVersion', 'asset_id', 'layout',
        'proxy', 'proxySha256', 'transform', 'inverseTransform', 'faceOrder')}
    frame.update(resolution=512, tightBounds={'min': low.tolist(), 'max': high.tolist()}, bakeBox=bounds)
    frame['faces'] = {key: make_face(bounds, value) for key, value in source['faces'].items()}
    frame['provenance'] = {'commonFrame': report, 'proxyIsMetadataOnly': True}
    args.output.mkdir(parents=True)
    proxy = (args.template.parent / source['proxy']).resolve(strict=True)
    target = (args.output / source['proxy']).resolve()
    if not proxy.is_relative_to(args.template.parent.resolve()) or not target.is_relative_to(args.output.resolve()):
        raise ValueError('Proxy path escapes frame directory')
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(proxy, target)
    assert sha(target) == source['proxySha256']
    for filename, content in [('manifest.json', frame), ('bounds.json', report)]:
        (args.output / filename).write_text(json.dumps(content, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
