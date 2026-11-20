#!/usr/bin/env python3
"""Pack six orthographic metric captures into the project's native Box schema."""
import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

FACES = {'px': '+X', 'nx': '-X', 'py': '+Y', 'ny': '-Y', 'pz': '+Z', 'nz': '-Z'}
METHOD = 'metric-six-view-box-v2'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def shader_grid(values, slug):
    if slug in ('px', 'nz'):
        return values[::-1, ::-1]
    return values if slug == 'py' else values[::-1]


def safe_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Asset path escapes output: ' + relative)
    return path


def write_map(root, name, values):
    values = np.asarray(values, dtype='<f2')
    if not np.isfinite(values).all():
        raise ValueError('Non-finite atlas: ' + name)
    path = root / 'maps' / (name.removesuffix('Map') + '.rgba16f.bin')
    values.tofile(path)
    h, w, channels = values.shape
    filtering = 'nearest' if name == 'rangeMap' else 'linear'
    return {'path': str(path.relative_to(root)), 'width': w, 'height': h, 'channels': channels,
            'dtype': 'float16', 'endian': 'little', 'rowOrder': 'increasingShaderV',
            'rowBytes': w * 8, 'byteLength': values.nbytes, 'sha256': sha(path),
            'minFilter': filtering, 'magFilter': filtering, 'wrap': 'clamp-to-edge',
            'mipmaps': False, 'colorSpace': 'linear-srgb' if name == 'colorMap' else 'data',
            'alpha': {'colorMap': 'premultiplied-coverage', 'normalMap': 'coverage',
                      'intervalMap': 'gap-end', 'rangeMap': 'constant-one'}[name]}


def srgb(linear):
    return np.where(linear <= .0031308, 12.92 * linear, 1.055 * np.maximum(linear, 0) ** (1 / 2.4) - .055)


def pair_intervals(data, resolution):
    interval = np.zeros((resolution, 3 * resolution, 4), dtype='f4')
    interval[..., 3] = 1
    stats = {}
    for axis, (plus, minus) in enumerate((('px', 'nx'), ('py', 'ny'), ('pz', 'nz'))):
        high = 1 - shader_grid(data[plus]['depth'], plus)
        low = shader_grid(data[minus]['depth'], minus)
        ap = shader_grid(data[plus]['mask'], plus)
        am = shader_grid(data[minus]['mask'], minus)
        both, posonly, negonly = ap & am, ap & ~am, am & ~ap
        crossed = both & (high <= low)
        width = 2 / resolution
        hi, lo = np.zeros_like(high), np.ones_like(low)
        # Opposite renders supply the outer endpoints; missing/crossed pairs use a recorded thin slab.
        center = np.clip((high + low) / 2, width / 2, 1 - width / 2)
        narrow = both & (high - low < width)
        regular = both & ~narrow
        hi[regular], lo[regular] = high[regular], low[regular]
        hi[narrow], lo[narrow] = center[narrow] + width / 2, center[narrow] - width / 2
        hi[posonly] = np.maximum(high[posonly], width)
        lo[posonly] = np.maximum(0, high[posonly] - width)
        lo[negonly] = np.minimum(low[negonly], 1 - width)
        hi[negonly] = np.minimum(1, low[negonly] + width)
        mask = ap | am
        valid = mask & (hi > lo)
        kernel = np.ones((3, 3), dtype='f4')
        kernel[1, 1] = 0
        count = ndimage.convolve(valid.astype('f4'), kernel, mode='constant', cval=0)
        neighbor_hi = ndimage.convolve(np.where(valid, hi, 0), kernel, mode='constant', cval=0)
        neighbor_lo = ndimage.convolve(np.where(valid, lo, 0), kernel, mode='constant', cval=0)
        padding = ~valid & (count > 0)
        hi[padding] = neighbor_lo[padding] / count[padding] - 3 / resolution
        lo[padding] = neighbor_hi[padding] / count[padding] + 3 / resolution
        tile = interval[:, axis * resolution:(axis + 1) * resolution]
        tile[..., 0] = np.where(valid | padding, hi, 0)
        tile[..., 1] = np.where(valid | padding, 1 - lo, 0)
        stats['XYZ'[axis]] = {'both': int(both.sum()), 'plusOnly': int(posonly.sum()),
                              'minusOnly': int(negonly.sum()), 'crossed': int(crossed.sum()),
                              'minimumThicknessApplied': int(narrow.sum()), 'filled': int(valid.sum()),
                              'edgePad': int(padding.sum()), 'noCoverage': int((~mask).sum()),
                              'crossedDepthMax': float(np.max((low - high)[crossed])) if crossed.any() else 0}
    return interval, stats


def range_map(interval, resolution):
    cells = (resolution + 7) // 8
    centers = np.minimum(np.arange(cells) * 8 + 4, resolution - 1)
    result = np.zeros((cells, 3 * cells, 4), dtype='f4')
    result[..., 3] = 1
    for axis in range(3):
        tile = interval[:, axis * resolution:(axis + 1) * resolution]
        lower = ndimage.minimum_filter(1 - tile[..., 1], size=10, mode='nearest')
        upper = ndimage.maximum_filter(tile[..., 0], size=10, mode='nearest')
        result[:, axis * cells:(axis + 1) * cells, 0] = lower[np.ix_(centers, centers)]
        result[:, axis * cells:(axis + 1) * cells, 1] = upper[np.ix_(centers, centers)]
    return result


def build(template_path, captures, output, alpha_threshold=.5):
    source = json.loads(template_path.read_text())
    status_path = captures / 'status.json'
    status = json.loads(status_path.read_text())
    resolution = int(status['resolution'])
    if status.get('templateSha256') not in (None, sha(template_path)):
        raise ValueError('Capture and output template cameras differ')
    if set(status['faces']) != set(FACES):
        raise ValueError('Six complete metric faces are required')
    identity = {'method': METHOD, 'templateSha256': sha(template_path),
                'captureStatusSha256': sha(status_path), 'resolution': resolution,
                'alphaThreshold': alpha_threshold, 'scriptSha256': sha(__file__)}
    asset_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    root = output / asset_id
    root.mkdir(parents=True, exist_ok=False)
    for directory in ('maps', 'guides', 'geometry', 'qa'):
        (root / directory).mkdir()
    try:
        manifest = {k: copy.deepcopy(source[k]) for k in ('schemaVersion', 'layout', 'proxy', 'proxySha256',
                    'tightBounds', 'bakeBox', 'transform', 'inverseTransform', 'faceOrder', 'faces')}
        manifest.update(asset_id=asset_id, parentAsset=source['asset_id'], mode='metric-six-view',
                        resolution=resolution, maps=[])
        proxy = safe_path(template_path.parent, source['proxy'])
        destination = safe_path(root, source['proxy'])
        destination.parent.mkdir(exist_ok=True, parents=True)
        shutil.copyfile(proxy, destination)
        if sha(destination) != source['proxySha256']:
            raise ValueError('Template proxy digest changed')
        color = np.zeros((2 * resolution, 3 * resolution, 4), dtype='f4')
        normal = np.zeros_like(color)
        data = {}
        for slug, fid in FACES.items():
            directory = captures / slug
            values = {name: np.load(directory / (name + '.npy'), allow_pickle=False).astype('f4')
                      for name in ('depth', 'alpha', 'premul-color', 'premul-normal')}
            expected = {'depth': (resolution, resolution), 'alpha': (resolution, resolution),
                        'premul-color': (resolution, resolution, 3), 'premul-normal': (resolution, resolution, 3)}
            for name, value in values.items():
                if value.shape != expected[name] or (name != 'depth' and not np.isfinite(value).all()):
                    raise ValueError(f'Invalid {slug}/{name}')
                recorded = status['faces'][slug].get(name + 'Sha256')
                if recorded is not None and sha(directory / (name + '.npy')) != recorded:
                    raise ValueError(f'Capture digest changed: {slug}/{name}')
            d, a = values['depth'], values['alpha']
            valid = (a > alpha_threshold) & np.isfinite(d) & (d >= 0) & (d <= 1)
            a = np.where(valid, np.clip(a, 0, 1), 0)
            d = np.where(valid, d, np.nan)
            c = np.where(valid[..., None], values['premul-color'], 0)
            n = np.where(valid[..., None], values['premul-normal'], 0)
            if np.any(c < -1e-5) or np.any(c > a[..., None] + 1e-4):
                raise ValueError(f'RGB must be linear and premultiplied: {slug}')
            face = manifest['faces'][fid]
            face = {key: face[key] for key in ('up', 'side', 'tile', 'viewMatrix', 'projectionMatrix', 'frustum')}
            face.update(guide=f'guides/{slug}.png', depth=f'guides/{slug}.depth16f.bin',
                        depthEncoding='linear-axis-0-near-1-far;nan-background', imageRowOrder='top-to-bottom',
                        imageColorSpace='sRGB', foregroundPixels=int(valid.sum()))
            manifest['faces'][fid] = face
            x, y = face['tile']
            color[y * resolution:(y + 1) * resolution, x * resolution:(x + 1) * resolution] = np.concatenate((c, a[..., None]), 2)[::-1]
            normal[y * resolution:(y + 1) * resolution, x * resolution:(x + 1) * resolution] = np.concatenate((n, a[..., None]), 2)[::-1]
            data[slug] = {'depth': d, 'mask': valid}
            depth_file = root / face['depth']
            np.asarray(d, dtype='<f2').tofile(depth_file)
            face['depthSha256'] = sha(depth_file)
            rgb = srgb(c / np.maximum(a[..., None], 1e-8))
            pixels = np.concatenate((np.clip(rgb, 0, 1), a[..., None]), 2)
            guide = root / face['guide']
            Image.fromarray(np.rint(pixels * 255).astype('uint8'), 'RGBA').save(guide)
            face['guideSha256'] = sha(guide)
        interval, stats = pair_intervals(data, resolution)
        manifest['textures'] = {name: write_map(root, name, values) for name, values in
                                (('colorMap', color), ('normalMap', normal), ('intervalMap', interval),
                                 ('rangeMap', range_map(interval, resolution)))}
        manifest['baseColor'] = copy.deepcopy(manifest['textures']['colorMap'])
        provenance = {'method': METHOD, 'captureStatus': str(status_path.resolve()),
                      'captureStatusSha256': sha(status_path), 'source': status,
                      'templateAsset': source['asset_id'], 'templateProxyUsedForGeometry': False,
                      'templateProxyRetainedForSchemaOnly': True,
                      'boundsPolicy': 'Use recorded common canonical Box and camera frame',
                      'depthPolicy': 'Expected projection Z; not first-hit depth or ground-truth surface',
                      'pairPolicy': 'Opposite expected-depth endpoints; minimum 2/R thickness, symmetric for crossings',
                      'interiorGaps': 'None inferred from six expected-depth captures',
                      'emptyTexelPolicy': '8-neighbor inside-out interval padding, 3/R separation',
                      'alphaThreshold': alpha_threshold, 'colorSpace': 'premultiplied linear sRGB',
                      'identity': identity}
        manifest['provenance'] = {'metricCapture': provenance}
        manifest['qa'] = 'qa/metric-box.json'
        qa = {'assetId': asset_id, 'resolution': resolution, 'axes': stats,
              'faces': {slug: int(values['mask'].sum()) for slug, values in data.items()},
              'mapSha256': {name: value['sha256'] for name, value in manifest['textures'].items()},
              'retainedProxyIsNotAReconstruction': True}
        (root / manifest['qa']).write_text(json.dumps(qa, indent=2) + '\n')
        (root / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    except Exception:
        # Preserve failed new output for diagnosis; never replace an existing asset.
        (root / 'INCOMPLETE').write_text('Build failed; inspect log before use.\n')
        raise
    result = {'assetId': asset_id, 'assetPath': str(root.resolve()), 'qa': qa}
    (output / (asset_id + '.json')).write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--template', type=Path, required=True, help='Common-frame manifest.json')
    parser.add_argument('--captures', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--alpha-threshold', type=float, default=.5)
    args = parser.parse_args()
    print(json.dumps(build(args.template, args.captures, args.output, args.alpha_threshold), indent=2), flush=True)


if __name__ == '__main__':
    main()
