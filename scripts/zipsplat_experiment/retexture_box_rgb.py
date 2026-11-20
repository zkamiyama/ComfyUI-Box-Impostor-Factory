#!/usr/bin/env python3
"""Mix six native GS RGB captures into a Box while preserving every geometry byte."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

FACES = {'px': '+X', 'nx': '-X', 'py': '+Y', 'ny': '-Y', 'pz': '+Z', 'nz': '-Z'}
METHOD = 'native-six-face-gs-rgb-only-v1'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def safe_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Asset path escapes its directory: ' + str(relative))
    return path


def file_hashes(root):
    result = {}
    for path in sorted(root.rglob('*')):
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            result[relative] = sha(safe_path(root, relative))
    return result


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(data, indent=2, allow_nan=False) + '\n')


def alpha_digest(path):
    words = np.fromfile(path, dtype='<u2')
    if len(words) % 4:
        raise ValueError('Invalid RGBA16F byte length: ' + str(path))
    # Hash the uint16 payload directly, including signed-zero and NaN payload bits.
    return hashlib.sha256(words.reshape(-1, 4)[:, 3].copy().tobytes()).hexdigest()


def check_descriptor(root, descriptor):
    path = safe_path(root, descriptor['path'])
    if sha(path) != descriptor['sha256']:
        raise ValueError('Source digest changed: ' + descriptor['path'])
    return path


def blend_tile(base_bits, gs_alpha, gs_premul, alpha_low, alpha_high, strength, qa_threshold):
    """Inputs are top-to-bottom; return uint16 RGB/A without recoding untouched pixels."""
    base = base_bits.view('<f2').astype('f4')
    base_alpha = base[..., 3]
    if not np.isfinite(base).all() or np.any(base_alpha < 0) or np.any(base_alpha > 1):
        raise ValueError('Invalid source color/coverage values')
    if not np.isfinite(gs_alpha).all() or np.any(gs_alpha < 0) or np.any(gs_alpha > 1):
        raise ValueError('Invalid GS support alpha')
    if not np.isfinite(gs_premul).all() or np.any(gs_premul < -1e-5) or np.any(gs_premul > gs_alpha[..., None] + 1e-4):
        raise ValueError('GS RGB must be finite, linear and premultiplied by support alpha')
    t = np.clip((gs_alpha - alpha_low) / (alpha_high - alpha_low), 0, 1)
    weight = strength * t * t * (3 - 2 * t)
    active = (weight > 0) & (base_alpha > 0)
    candidate = np.clip(gs_premul / np.maximum(gs_alpha[..., None], 1e-8), 0, 1)
    candidate *= base_alpha[..., None]
    mixed = base[..., :3] * (1 - weight[..., None]) + candidate * weight[..., None]
    result = base_bits.copy()
    result[..., :3][active] = mixed.astype('<f2').view('<u2')[active]
    if not np.array_equal(result[..., 3], base_bits[..., 3]):
        raise RuntimeError('Color atlas alpha bits changed')
    changed = np.any(result[..., :3] != base_bits[..., :3], axis=-1)
    if np.any(changed & ~active):
        raise RuntimeError('RGB changed outside supported source coverage')
    base_mask, support_mask = base_alpha > qa_threshold, gs_alpha > qa_threshold
    intersection, union = base_mask & support_mask, base_mask | support_mask
    count, support_count = int(base_mask.sum()), int(support_mask.sum())
    alpha_sum = float(base_alpha.sum())
    difference = np.abs(result.view('<f2').astype('f4')[..., :3] - base[..., :3])
    stats = {
        'baseForegroundPixels': count, 'gsSupportPixels': support_count,
        'intersectionPixels': int(intersection.sum()), 'unionPixels': int(union.sum()),
        'supportMaskIoU': float(intersection.sum() / union.sum()) if union.any() else None,
        'baseForegroundCoverage': float(intersection.sum() / count) if count else None,
        'gsSupportOutsideBasePixels': int((support_mask & ~base_mask).sum()),
        'unsupportedBaseForegroundPixels': int((base_mask & ~support_mask).sum()),
        'alphaWeightedBlend': float((base_alpha * weight).sum() / alpha_sum) if alpha_sum else None,
        'blendActivePixels': int(active.sum()), 'changedRgbPixels': int(changed.sum()),
        'lowSupportFallbackPixels': int(((gs_alpha <= alpha_low) & (base_alpha > 0)).sum()),
        'changedLowSupportPixels': int((changed & (gs_alpha <= alpha_low)).sum()),
        'changedZeroBaseAlphaPixels': int((changed & (base_alpha == 0)).sum()),
        'maxPremultipliedRgbDifference': float(difference.max()),
        'alphaBitDifferences': int(np.count_nonzero(result[..., 3] != base_bits[..., 3])),
    }
    return result, stats


def run(args):
    template = args.template.resolve()
    source_root, captures, output = template.parent, args.captures.resolve(), args.output.resolve()
    if template.name != 'manifest.json':
        raise ValueError('--template must be a native asset manifest.json')
    if output.is_relative_to(source_root) or output.is_relative_to(captures):
        raise ValueError('Output must be outside both input directories')
    if not 0 <= args.alpha_low < args.alpha_high <= 1 or not 0 <= args.strength <= 1:
        raise ValueError('Require 0 <= alpha-low < alpha-high <= 1 and 0 <= strength <= 1')
    if not 0 <= args.qa_alpha_threshold <= 1:
        raise ValueError('QA alpha threshold must be in [0, 1]')
    source = json.loads(template.read_text())
    status_path = captures / 'status.json'
    status = json.loads(status_path.read_text())
    if status.get('kind') != 'ZipSplat-six-orthographic-expected-depth-v1':
        raise ValueError('Expected zip_render_faces.py render captures')
    if status.get('templateSha256') != sha(template):
        raise ValueError('Captures must use this exact original Box native frame')
    resolution = int(source['resolution'])
    if status['resolution'] != resolution or set(status['faces']) != set(FACES):
        raise ValueError('Require all six faces at the original Box resolution')
    descriptor = source['textures']['colorMap']
    expected = {'width': 3 * resolution, 'height': 2 * resolution, 'channels': 4,
                'dtype': 'float16', 'endian': 'little', 'rowOrder': 'increasingShaderV',
                'colorSpace': 'linear-srgb', 'alpha': 'premultiplied-coverage'}
    if any(descriptor.get(key) != value for key, value in expected.items()):
        raise ValueError('Unsupported native color atlas contract')
    color_path = check_descriptor(source_root, descriptor)
    for value in source['textures'].values():
        check_descriptor(source_root, value)
    if 'baseColor' in source:
        check_descriptor(source_root, source['baseColor'])
    if sha(safe_path(source_root, source['proxy'])) != source['proxySha256']:
        raise ValueError('Source proxy digest changed')
    tiles = [tuple(source['faces'][fid]['tile']) for fid in FACES.values()]
    if set(tiles) != {(x, y) for x in range(3) for y in range(2)}:
        raise ValueError('Native color atlas must contain six distinct 3 by 2 tiles')
    source_hashes, capture_hashes = file_hashes(source_root), file_hashes(captures)
    parameters = {'alphaLow': args.alpha_low, 'alphaHigh': args.alpha_high,
                  'strength': args.strength, 'qaAlphaThreshold': args.qa_alpha_threshold}
    identity = {'method': METHOD, 'sourceFiles': source_hashes, 'captureFiles': capture_hashes,
                'parameters': parameters, 'scriptSha256': sha(__file__)}
    asset_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    root, receipt_path = output / asset_id, output / (asset_id + '.json')
    if root.exists() or receipt_path.exists():
        raise FileExistsError('RGB-only output already exists: ' + str(root))
    original_bits = np.fromfile(color_path, dtype='<u2').reshape(2 * resolution, 3 * resolution, 4)
    updated_bits = original_bits.copy()
    face_stats = {}
    for slug, fid in FACES.items():
        arrays = {}
        for name in ('alpha', 'premul-color'):
            path = captures / slug / (name + '.npy')
            if status['faces'][slug].get(name + 'Sha256') != sha(path):
                raise ValueError('Capture digest changed: ' + slug + '/' + name)
            arrays[name] = np.load(path, allow_pickle=False).astype('f4')
        if arrays['alpha'].shape != (resolution, resolution) or arrays['premul-color'].shape != (resolution, resolution, 3):
            raise ValueError('Capture dimensions changed: ' + slug)
        x, y = source['faces'][fid]['tile']
        region = np.s_[y * resolution:(y + 1) * resolution, x * resolution:(x + 1) * resolution]
        # Native capture rows descend; the atlas stores increasing shader V for every face.
        tile, stats = blend_tile(original_bits[region][::-1], arrays['alpha'], arrays['premul-color'],
                                 args.alpha_low, args.alpha_high, args.strength, args.qa_alpha_threshold)
        updated_bits[region] = tile[::-1]
        face_stats[fid] = {**stats, 'captureSlug': slug, 'tile': [x, y]}
    if not np.array_equal(original_bits[..., 3], updated_bits[..., 3]):
        raise RuntimeError('Global color atlas alpha bits changed')
    rgba_paths = {name for name in source_hashes if name.endswith('.rgba16f.bin')}
    rgba_paths.update(value['path'] for value in [*source['textures'].values(), source.get('baseColor', {})]
                      if value.get('dtype') == 'float16' and value.get('channels') == 4)
    source_alpha_hashes = {name: alpha_digest(safe_path(source_root, name)) for name in sorted(rgba_paths)}
    output.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_root, root)
    try:
        if file_hashes(root) != source_hashes:
            raise RuntimeError('Input changed while copying the source asset')
        updated_bits.tofile(safe_path(root, descriptor['path']))
        color_sha = sha(safe_path(root, descriptor['path']))
        alpha_checks = {name: {'sourceSha256': digest,
                              'outputSha256': alpha_digest(safe_path(root, name))}
                        for name, digest in source_alpha_hashes.items()}
        if any(check['sourceSha256'] != check['outputSha256'] for check in alpha_checks.values()):
            raise RuntimeError('RGBA16F alpha payload changed')
        qa = {
            'method': METHOD, 'assetId': asset_id, 'sourceAssetId': source['asset_id'],
            'sourceManifestSha256': sha(template), 'captureStatusSha256': sha(status_path),
            'parameters': parameters, 'faces': face_stats,
            'blendFormula': 'w=strength*smoothstep(alphaLow,alphaHigh,GSalpha); RGBout=(1-w)*RGBbase+w*(GSpremul/GSalpha)*baseAlpha',
            'colorSpace': 'premultiplied linear sRGB; no gamma conversion during mixing',
            'registration': 'Exact native face cameras and original resolution; no affine or per-face warp',
            'rgbaAtlasAlphaChecks': alpha_checks, 'geometryDepthNormalIntervalRangeProxyPreserved': True,
            'sourceColorSha256': descriptor['sha256'], 'outputColorSha256': color_sha,
            'qaPolicy': 'Report support IoU and coverage; no quality threshold admission test',
            'limitations': ['GS alpha describes opacity, not semantic foreground or geometric correspondence.',
                            'RGB support may cover a different surface when the reconstructions disagree.',
                            'Original guide PNGs are retained; the viewer uses the updated color atlas.'],
        }
        manifest = copy.deepcopy(source)
        manifest.update(asset_id=asset_id, parentAsset=source['asset_id'], mode='gs-rgb-only')
        for value in [*manifest['textures'].values(), manifest.get('baseColor', {})]:
            if value.get('path') == descriptor['path']:
                value['sha256'] = color_sha
        qa_relative = 'qa/rgb-retexture-' + asset_id[:16] + '.json'
        manifest['qa'] = qa_relative
        manifest.setdefault('provenance', {})['gsRgbRetexture'] = {
            'method': METHOD, 'sourceAsset': source['asset_id'], 'sourceManifestSha256': sha(template),
            'captureStatusSha256': sha(status_path), 'parameters': parameters,
            'unchangedGeometryAndAlpha': True, 'qa': qa_relative, 'scriptSha256': sha(__file__),
        }
        write_json(safe_path(root, qa_relative), qa)
        (root / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
        after = file_hashes(root)
        preserved = {name: digest for name, digest in source_hashes.items()
                     if name not in ('manifest.json', descriptor['path'])}
        if any(after.get(name) != digest for name, digest in preserved.items()):
            raise RuntimeError('An original file other than manifest/color RGB changed')
        if file_hashes(source_root) != source_hashes or file_hashes(captures) != capture_hashes:
            raise RuntimeError('An input changed during retexturing')
        receipt = {'assetId': asset_id, 'assetPath': str(root), 'resolution': resolution,
                   'sourceAssetPath': str(source_root), 'capturesPath': str(captures),
                   'identity': identity, 'outputFiles': after, 'preservedSourceFiles': preserved,
                   'qa': qa, 'receiptHashPolicy': 'Receipt SHA256 is printed after writing; it is not self-embedded.'}
        write_json(receipt_path, receipt)
    except Exception:
        (root / 'INCOMPLETE').write_text('RGB-only build failed; inspect the error before use.\n')
        raise
    return {'assetId': asset_id, 'assetPath': str(root), 'receiptPath': str(receipt_path),
            'receiptSha256': sha(receipt_path), 'changedRgbPixels': sum(item['changedRgbPixels'] for item in face_stats.values()),
            'allRgbaAlphaBitsPreserved': True, 'allOtherSourceFilesPreserved': True, 'faces': face_stats}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--template', type=Path, required=True, help='Original A512 native asset manifest.json')
    parser.add_argument('--captures', type=Path, required=True, help='zip_render_faces.py render output using that exact template')
    parser.add_argument('--output', type=Path, required=True, help='Parent directory for a new immutable asset')
    parser.add_argument('--alpha-low', type=float, default=.5, help='Keep original RGB exactly at or below this GS alpha')
    parser.add_argument('--alpha-high', type=float, default=.95, help='Reach full configured blend strength at this GS alpha')
    parser.add_argument('--strength', type=float, default=1., help='Maximum linear RGB blend weight')
    parser.add_argument('--qa-alpha-threshold', type=float, default=.5, help='Binary alpha threshold used only for coverage/IoU reporting')
    print(json.dumps(run(parser.parse_args()), indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
