#!/usr/bin/env python3
"""Register full-body Qwen RGB to one square guide and replace only Box +Z RGB.

The complete generated foreground receives one small affine transform. Exact
orthographic camera coordinates map it to 512px face texel centers. Original
atlas alpha bits, other faces, geometry files, and camera metadata are preserved.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
import zipfile

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.optimize import differential_evolution
from scipy.spatial.transform import Rotation

METHOD = 'qwen-fullbody-affine-front-rgb-only-v1'
GEOMETRY_KEYS = ('schemaVersion', 'layout', 'proxy', 'proxySha256', 'resolution', 'tightBounds',
                 'bakeBox', 'transform', 'inverseTransform', 'faceOrder', 'faces')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def safe(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Asset path escapes directory: ' + str(relative))
    return path


def file_hashes(root):
    return {path.relative_to(root).as_posix(): sha(path) for path in sorted(root.rglob('*')) if path.is_file()}


def alpha_sha(path):
    bits = np.fromfile(path, dtype='<u2').reshape(-1, 4)
    return hashlib.sha256(bits[:, 3].copy().tobytes()).hexdigest()


def read_alpha(path):
    if path.suffix.lower() == '.npy':
        alpha = np.load(path, allow_pickle=False).astype('f4')
    else:
        with Image.open(path) as image:
            alpha = np.asarray(image.convert('L'), dtype='f4') / 255
    if alpha.ndim != 2 or not np.isfinite(alpha).all() or np.any((alpha < 0) | (alpha > 1)):
        raise ValueError('Expected finite 2D alpha in [0,1]: ' + str(path))
    return alpha


def linear(rgb):
    return np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)


def affine(parameters, size):
    sx, sy, angle, dy, dx = map(float, parameters)
    center = (size - 1) / 2
    co, si = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    return np.array([[sx * co, -sy * si, center + dx - sx * co * center + sy * si * center],
                     [sx * si, sy * co, center + dy - sx * si * center - sy * co * center]], dtype='f8')


def inverse_affine(matrix):
    return np.linalg.inv(np.vstack((matrix, [0, 0, 1])))[:2]


def warp(values, matrix, shape):
    inverse = inverse_affine(matrix)
    return ndimage.affine_transform(values, inverse[:, :2][::-1, ::-1], inverse[:, 2][::-1],
                                    output_shape=shape, order=1, mode='constant', cval=0, prefilter=False)


def iou(a, b):
    return float(np.count_nonzero(a & b) / max(1, np.count_nonzero(a | b)))


def register(qmask, target, size, iou_reference):
    """Same bounded five-parameter full-subject affine family as the earlier Tripo pilot."""
    started = time.monotonic()
    size = min(size, target.shape[0])
    small = lambda array: np.asarray(Image.fromarray(array).resize((size, size), Image.Resampling.BOX)) > .5
    moving, fixed = small(qmask), small(target)
    if not moving.any() or not fixed.any():
        raise ValueError('Whole-body foreground masks must be nonempty')
    bounds = [(.96, 1.06), (.96, 1.06), (-3., 3.), (-size / 64, size / 64), (-size / 64, size / 64)]
    def score(parameters):
        return iou(warp(moving.astype('f4'), affine(parameters, size), moving.shape) > .5, fixed)
    initial = np.array([1., 1., 0., 0., 0.])
    opt = differential_evolution(lambda parameters: -score(parameters), bounds=bounds,
                                 maxiter=48, popsize=7, seed=20261120, tol=.0004,
                                 polish=False, x0=initial, workers=1)
    optimized = np.asarray(opt.x, dtype='f8')
    optimized[3:] *= target.shape[0] / size
    candidates = [('identity', initial), ('optimized-fullbody-affine', optimized)]
    results = []
    for name, parameters in candidates:
        matrix = affine(parameters, target.shape[0])
        registered = warp(qmask, matrix, target.shape)
        results.append((iou(registered > .5, target > .5), name, parameters, matrix, registered))
    best = max(results, key=lambda result: result[0])
    score_after, name, parameters, matrix, registered = best
    report = {'method': 'bounded-whole-foreground-affine; XY scale, rotation, translation; no shear',
              'selection': name, 'parametersSX_SY_angleDY_DX': parameters.tolist(),
              'matrixInputToOutput': matrix.tolist(), 'originalIoU': results[0][0],
              'registeredIoU': score_after, 'iouReference': iou_reference,
              'referencePassed': bool(score_after >= iou_reference),
              'policy': 'Diagnostic reference only; lower IoU does not block an immutable comparison variant',
              'optimizationPixels': [size, size], 'fullResolution': list(target.shape[::-1]),
              'scaleBounds': [.96, 1.06], 'rotationBoundsDegrees': [-3, 3],
              'translationBoundsPixels': [-target.shape[0] / 64, target.shape[0] / 64],
              'seed': 20261120, 'optimizerEvaluations': int(opt.nfev),
              'seconds': round(time.monotonic() - started, 3),
              'subjectRegion': 'entire foreground; no facial crops, keypoints, local warps, or inpainting'}
    return matrix, registered, report


def face_to_square(manifest, guide):
    """Map top-down face index (j,i) to native RenderSplat square pixel centers (x,y)."""
    face, camera = manifest['faces']['+Z'], guide['camera']
    size = guide['size']
    if guide.get('face') != '+Z' or size[0] != size[1] or camera['cameraType'] != 'orthographic':
        raise ValueError('Require the actual square orthographic +Z guide camera')
    if not guide.get('fullBody') or guide.get('edgePixels') != 0:
        raise ValueError('Require a complete full-body square guide without edge clipping')
    quaternion = camera['quaternion']
    basis = Rotation.from_quat([quaternion[key] for key in 'xyzw']).as_matrix()
    view_inverse = np.linalg.inv(np.asarray(face['viewMatrix']).reshape(4, 4, order='F'))
    if not np.allclose(basis, view_inverse[:3, :3], atol=1e-6):
        raise ValueError('Square guide orientation differs from the native +Z face')
    eye = np.array([camera['position'][key] for key in 'xyz'])
    target = np.array([camera['target'][key] for key in 'xyz'])
    distance = float(np.linalg.norm(eye - target))
    span = 2 * distance * math.tan(math.radians(camera['fov']) / 2) / camera.get('zoom', 1.)
    if not math.isfinite(span) or span <= 0:
        raise ValueError('Invalid square guide view span')
    if not math.isclose(span, guide['squarePhysicalSpan'], rel_tol=1e-6):
        raise ValueError('Recorded camera and square physical span disagree')
    left, right, bottom, top = face['frustum'][:4]
    physical = np.array([right - left, top - bottom])
    if not np.allclose(physical, guide['physicalSize'], atol=1e-8):
        raise ValueError('Square guide and baseline Box frustums differ')
    if not np.allclose(physical / span * size[0], guide['contentPixels'], atol=1e-6):
        raise ValueError('Square guide content dimensions disagree with its camera')
    resolution = manifest['resolution']
    # Face world points at texel centers. Z=0 is safe because the two cameras have the same basis.
    index_to_face = np.array([[physical[0] / resolution, 0, left + physical[0] / (2 * resolution)],
                              [0, -physical[1] / resolution, top - physical[1] / (2 * resolution)],
                              [0, 0, 0], [0, 0, 1]])
    world = view_inverse @ index_to_face
    world[:3, 2] -= eye
    projection = np.stack((basis[:, 0], -basis[:, 1])) * (size[0] / span)
    matrix = projection @ world[:3]
    matrix[:, 2] += size[0] / 2
    return matrix


def atlas_rgb(rgb, qmask, guide_alpha, registration, mapping, resolution):
    """Joint inverse sampling avoids an intermediate rounded crop or repeated RGB warp."""
    inverse = inverse_affine(registration)
    premul = linear(rgb) * qmask[..., None]
    yy, xx = np.mgrid[:resolution, :resolution].astype('f8')
    nx, ny = (max(1, math.ceil(float(np.linalg.norm(mapping[:, axis])) - 1e-9)) for axis in (0, 1))
    if nx > 16 or ny > 16:
        raise ValueError('Unexpectedly large square-to-face sampling footprint')
    alpha_sum, support_sum = np.zeros((resolution, resolution)), np.zeros((resolution, resolution))
    color_sum = np.zeros((resolution, resolution, 3))
    sample = lambda array, x, y: ndimage.map_coordinates(array, [y, x], order=1, mode='constant', cval=0, prefilter=False)
    for oy in (np.arange(ny) + .5) / ny - .5:
        for ox in (np.arange(nx) + .5) / nx - .5:
            gx = mapping[0, 0] * (xx + ox) + mapping[0, 1] * (yy + oy) + mapping[0, 2]
            gy = mapping[1, 0] * (xx + ox) + mapping[1, 1] * (yy + oy) + mapping[1, 2]
            qx = inverse[0, 0] * gx + inverse[0, 1] * gy + inverse[0, 2]
            qy = inverse[1, 0] * gx + inverse[1, 1] * gy + inverse[1, 2]
            area = sample(guide_alpha, gx, gy)
            support_sum += area * sample(qmask, qx, qy)
            alpha_sum += area
            for channel in range(3):
                color_sum[..., channel] += area * sample(premul[..., channel], qx, qy)
    candidate = np.clip(color_sum / np.maximum(support_sum[..., None], 1e-8), 0, 1)
    support = np.clip(support_sum / np.maximum(alpha_sum, 1e-8), 0, 1)
    return candidate, support, np.clip(alpha_sum / (nx * ny), 0, 1), [nx, ny]


def run(args):
    baseline = args.baseline_asset.resolve(strict=True)
    template = baseline / 'manifest.json' if baseline.is_dir() else baseline
    source_root = template.parent
    output = args.output.resolve()
    if template.name != 'manifest.json' or output.is_relative_to(source_root):
        raise ValueError('Use a native manifest.json and a separate output parent')
    inputs = {name: getattr(args, name).resolve(strict=True) for name in
              ('qwen_rgb', 'square_alpha', 'generated_mask', 'guide_meta')}
    input_hashes = {key: sha(path) for key, path in inputs.items()}
    source = json.loads(template.read_text())
    guide = json.loads(inputs['guide_meta'].read_text())
    capture = source['provenance']['metricCapture']['source']
    for baseline_key, guide_key in [('splatSha256', 'splatSha256'), ('templateSha256', 'manifestSha256'),
                                    ('splatRotationNativeY', 'rotateYDegrees'),
                                    ('nativeRendererSha256', 'nativeRendererSha256')]:
        if capture[baseline_key] != guide[guide_key]:
            raise ValueError('Guide differs from the baseline reconstruction: ' + guide_key)
    for key in ('position', 'target', 'quaternion', 'zoom'):
        if capture['faces']['pz']['camera'][key] != guide['camera'][key]:
            raise ValueError('Guide and baseline face camera differ: ' + key)
    resolution = int(source['resolution'])
    if resolution != 512:
        raise ValueError('This pilot requires Box512')
    descriptor = source['textures']['colorMap']
    contract = {'width': 3 * resolution, 'height': 2 * resolution, 'channels': 4,
                'dtype': 'float16', 'endian': 'little', 'rowOrder': 'increasingShaderV',
                'colorSpace': 'linear-srgb', 'alpha': 'premultiplied-coverage'}
    if any(descriptor.get(key) != value for key, value in contract.items()):
        raise ValueError('Unsupported native color atlas contract')
    for value in [*source['textures'].values(), source.get('baseColor', {})]:
        if value and sha(safe(source_root, value['path'])) != value['sha256']:
            raise ValueError('Source texture digest changed: ' + value['path'])
    if sha(safe(source_root, source['proxy'])) != source['proxySha256']:
        raise ValueError('Source proxy digest changed')
    recorded_alpha = guide['files'].get(inputs['square_alpha'].name)
    if recorded_alpha is None or recorded_alpha['sha256'] != input_hashes['square_alpha']:
        raise ValueError('Square alpha does not match the recorded guide')
    with Image.open(inputs['qwen_rgb']) as image:
        rgb = np.asarray(image.convert('RGB'), dtype='f4') / 255
    qmask, guide_alpha = read_alpha(inputs['generated_mask']), read_alpha(inputs['square_alpha'])
    shape = tuple(guide['size'][::-1])
    if rgb.shape != (*shape, 3) or qmask.shape != shape or guide_alpha.shape != shape:
        raise ValueError('Qwen RGB, generated mask and square guide must have identical pixel dimensions')
    if not 0 <= args.iou_reference <= 1 or not 64 <= args.registration_size <= 1024:
        raise ValueError('Invalid diagnostic IoU or registration size')
    source_hashes = file_hashes(source_root)
    mapping = face_to_square(source, guide)
    registration, registered_mask, reg = register(qmask, guide_alpha, args.registration_size, args.iou_reference)
    candidate, support, native_alpha, footprint = atlas_rgb(rgb, qmask, guide_alpha, registration, mapping, resolution)
    original = np.fromfile(safe(source_root, descriptor['path']), dtype='<u2').reshape(2 * resolution, 3 * resolution, 4)
    updated = original.copy()
    tx, ty = source['faces']['+Z']['tile']
    region = np.s_[ty * resolution:(ty + 1) * resolution, tx * resolution:(tx + 1) * resolution]
    before = original[region][::-1].view('<f2').astype('f4')
    if not np.isfinite(before).all() or np.any((before[..., 3] < 0) | (before[..., 3] > 1)):
        raise ValueError('Invalid original front RGB/coverage')
    active = (native_alpha > .02) & (before[..., 3] > 0) & (support > 0)
    mixed = before[..., :3] * (1 - support[..., None]) + candidate * before[..., 3, None] * support[..., None]
    tile = updated[region][::-1]
    tile[..., :3][active] = mixed.astype('<f2').view('<u2')[active]
    if not np.array_equal(original[..., 3], updated[..., 3]):
        raise RuntimeError('Color atlas alpha bits changed')
    changed = np.any(updated[..., :3] != original[..., :3], axis=-1)
    outside = changed.copy()
    outside[region] = False
    if outside.any():
        raise RuntimeError('A face other than +Z changed')
    geometry_before = {key: source[key] for key in GEOMETRY_KEYS}
    geometry_sha = object_sha(geometry_before)
    identity = {'method': METHOD, 'sourceFiles': source_hashes, 'inputHashes': input_hashes,
                'registrationMatrix': registration.tolist(), 'mapping': mapping.tolist(),
                'iouReference': args.iou_reference, 'scriptSha256': sha(__file__)}
    asset_id = object_sha(identity)
    root, receipt_path, archive = output / asset_id, output / (asset_id + '.json'), output / (asset_id + '.box.zip')
    if any(path.exists() for path in (root, receipt_path, archive)):
        raise FileExistsError('Immutable output already exists: ' + str(root))
    output.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_root, root)
    try:
        if file_hashes(root) != source_hashes:
            raise RuntimeError('Source asset changed during copy')
        updated.tofile(safe(root, descriptor['path']))
        color_sha = sha(safe(root, descriptor['path']))
        manifest = copy.deepcopy(source)
        manifest.update(asset_id=asset_id, parentAsset=source['asset_id'], mode='qwen-front-rgb-only')
        for value in [*manifest['textures'].values(), manifest.get('baseColor', {})]:
            if value.get('path') == descriptor['path']:
                value['sha256'] = color_sha
        geometry_after = {key: manifest[key] for key in GEOMETRY_KEYS}
        if object_sha(geometry_after) != geometry_sha:
            raise RuntimeError('Geometry or camera metadata changed')
        rgba_paths = [name for name in source_hashes if name.endswith('.rgba16f.bin')]
        alpha_checks = {name: {'source': alpha_sha(safe(source_root, name)),
                               'output': alpha_sha(safe(root, name))} for name in rgba_paths}
        if any(value['source'] != value['output'] for value in alpha_checks.values()):
            raise RuntimeError('An atlas alpha bit payload changed')
        preserved = {name: digest for name, digest in source_hashes.items() if name not in ('manifest.json', descriptor['path'])}
        actual = file_hashes(root)
        if any(actual.get(name) != digest for name, digest in preserved.items()):
            raise RuntimeError('An original file other than manifest/color RGB changed')
        qa_relative = 'qa/qwen-front-' + asset_id[:16] + '.json'
        mask_relative = 'qa/qwen-front-' + asset_id[:16] + '-registered-mask.png'
        Image.fromarray(np.uint8(np.clip(np.rint(registered_mask * 255), 0, 255))).save(safe(root, mask_relative))
        qa = {'method': METHOD, 'sourceAssetId': source['asset_id'], 'assetId': asset_id,
              'status': 'front-only-color-pilot' if reg['referencePassed'] else 'front-only-color-pilot-below-iou-reference',
              'inputs': {key: {'path': str(path), 'sha256': input_hashes[key]} for key, path in inputs.items()},
              'registration': reg, 'mappingFaceTexelXYToSquareXY': mapping.tolist(),
              'registeredMask': {'path': mask_relative, 'sha256': sha(safe(root, mask_relative))},
              'guideMatchesBaselineSplatFrameRotationRenderer': True,
              'squareGuideSize': guide['size'], 'squareContentPixels': guide['contentPixels'],
              'sampling': 'continuous texel-center mapping; one inverse affine; linear-premultiplied mask/guide-alpha weighted subpixel integration',
              'subsamplesXY': footprint, 'face': '+Z', 'resolution': resolution,
              'blendFormula': 'w=integral(guideAlpha*Qmask)/integral(guideAlpha); RGBout=(1-w)*RGBold+w*Qlinear*oldAlpha',
              'mappedGuideVsOriginalBoxAlphaIoU': iou(native_alpha > .5, before[..., 3] > .5),
              'qwenSupportVsOriginalBoxAlphaIoU': iou((support * native_alpha) > .5, before[..., 3] > .5),
              'changedRgbPixels': int(changed.sum()), 'blendActivePixels': int(active.sum()),
              'fallbackForegroundPixels': int(((before[..., 3] > .5) & ((support < .5) | (native_alpha <= .02))).sum()),
              'sourceColorSha256': descriptor['sha256'], 'outputColorSha256': color_sha,
              'colorAlphaBitDifferences': int(np.count_nonzero(original[..., 3] != updated[..., 3])),
              'rgbaAlphaChecks': alpha_checks, 'otherFiveFacesBitIdentical': True,
              'geometryMetadataSourceSha256': geometry_sha, 'geometryMetadataOutputSha256': object_sha(geometry_after),
              'geometryDepthNormalIntervalRangeProxyCameraPreserved': True,
              'preservedSourceFiles': preserved,
              'limitations': ['Only +Z RGB is refined; this is not consistent six-view texturing.',
                             'Whole-body mask registration does not establish internal texture correspondence.',
                             'Original guide PNGs remain unchanged; rendering uses the updated color atlas.']}
        manifest['qa'] = qa_relative
        manifest.setdefault('provenance', {})['qwenFrontRGB'] = {
            'method': METHOD, 'sourceAsset': source['asset_id'], 'qa': qa_relative,
            'geometryAndAlphaPreserved': True, 'registeredWholeBodyIoU': reg['registeredIoU'],
            'iouReferencePassed': reg['referencePassed'], 'scriptSha256': sha(__file__)}
        write_json(safe(root, qa_relative), qa)
        (root / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
        if file_hashes(source_root) != source_hashes or any(sha(path) != input_hashes[key] for key, path in inputs.items()):
            raise RuntimeError('An input changed during composition')
        with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zipped:
            for path in sorted(root.rglob('*')):
                if path.is_file():
                    zipped.write(path, path.relative_to(root).as_posix())
        with zipfile.ZipFile(archive) as zipped:
            if zipped.testzip() is not None:
                raise RuntimeError('Output Box ZIP integrity check failed')
        receipt = {'assetId': asset_id, 'assetPath': str(root), 'archive': str(archive), 'archiveSha256': sha(archive),
                   'identity': identity, 'qa': qa, 'outputFiles': file_hashes(root)}
        write_json(receipt_path, receipt)
    except Exception:
        (root / 'INCOMPLETE').write_text('Composition failed; do not use this asset.\n')
        raise
    return {'assetId': asset_id, 'assetPath': str(root), 'archive': str(archive), 'archiveSha256': sha(archive),
            'receiptPath': str(receipt_path), 'receiptSha256': sha(receipt_path),
            'registeredIoU': reg['registeredIoU'], 'iouReferencePassed': reg['referencePassed'],
            'changedRgbPixels': int(changed.sum()), 'geometryAndAlphaPreserved': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-asset', type=Path, required=True, help='Native asset directory or manifest.json')
    parser.add_argument('--qwen-rgb', type=Path, required=True)
    parser.add_argument('--square-alpha', type=Path, required=True, help='Exact guide alpha.npy or recorded alpha PNG')
    parser.add_argument('--generated-mask', type=Path, required=True, help='Full-body Qwen/RMBG foreground mask')
    parser.add_argument('--guide-meta', type=Path, required=True, help='Actual square +Z camera guide.json')
    parser.add_argument('--output', type=Path, required=True, help='Parent directory for immutable variant, receipt, ZIP')
    parser.add_argument('--registration-size', type=int, default=256)
    parser.add_argument('--iou-reference', type=float, default=.88, help='Diagnostic reference; lower IoU still produces a comparison variant')
    print(json.dumps(run(parser.parse_args()), indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
