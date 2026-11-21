#!/usr/bin/env python3
"""Compose explicitly selected whole-subject Qwen candidates into native Box512 RGB.

Selection JSON: {"faces":[{"face":"+Z","result":"/path/result.json",
"guide":"/path/guide.json","reviewNote":"visual review notes"}]}.
All faces start from one immutable baseline. No crops or local warps are used.
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


METHOD = 'qwen-whole-subject-selected-faces-rgb-only-v1'
FACES = ('+X', '-X', '+Y', '-Y', '+Z', '-Z')


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



def resolve_record(record, parent, name):
    path = Path(record['path'])
    path = (path if path.is_absolute() else parent / path).resolve(strict=True)
    if not path.is_file() or sha(path) != record['sha256']:
        raise ValueError(name + ': file SHA256 does not match its record')
    return path


def manifest_path(path):
    path = Path(path).resolve(strict=True)
    path = path / 'manifest.json' if path.is_dir() else path
    if path.name != 'manifest.json':
        raise ValueError('Expected a native manifest.json or its asset directory')
    return path


def structural_manifest(manifest):
    """All metadata except declared output identity, QA, and color-file digests."""
    result = copy.deepcopy(manifest)
    color_path = result['textures']['colorMap']['path']
    for key in ('asset_id', 'parentAsset', 'mode', 'qa'):
        result.pop(key, None)
    for value in [*result['textures'].values(), result.get('baseColor', {})]:
        if value.get('path') == color_path:
            value['sha256'] = '<RGB-only-file-digest>'
    return result


def validate_asset(root, manifest):
    resolution = int(manifest['resolution'])
    if resolution != 512 or set(manifest['faces']) != set(FACES):
        raise ValueError('Require the native six-face Box512 asset')
    descriptor = manifest['textures']['colorMap']
    contract = {'width': 3 * resolution, 'height': 2 * resolution, 'channels': 4,
                'dtype': 'float16', 'endian': 'little', 'rowOrder': 'increasingShaderV',
                'colorSpace': 'linear-srgb', 'alpha': 'premultiplied-coverage'}
    if any(descriptor.get(key) != value for key, value in contract.items()):
        raise ValueError('Unsupported native color atlas contract')
    for value in [*manifest['textures'].values(), manifest.get('baseColor', {})]:
        if value and sha(safe(root, value['path'])) != value['sha256']:
            raise ValueError('Native texture digest changed: ' + value['path'])
    if sha(safe(root, manifest['proxy'])) != manifest['proxySha256']:
        raise ValueError('Native proxy digest changed')
    tiles = [tuple(manifest['faces'][face]['tile']) for face in FACES]
    if len(set(tiles)) != 6 or set(tiles) != {(x, y) for x in range(3) for y in range(2)}:
        raise ValueError('Native face tiles must partition the 3x2 atlas')
    return resolution, descriptor


def face_to_square(manifest, template_sha, guide):
    """The producer records top-down, zero-based pixel-center coordinates."""
    face_name = guide['face']
    face = manifest['faces'][face_name]
    if guide.get('kind') != 'native-mesh-square-guide-v1':
        raise ValueError('Require a native mesh square guide')
    if guide['baselineManifest']['sha256'] != template_sha:
        raise ValueError('Guide was not produced from this exact baseline manifest')
    if guide['nativeResolution'] != manifest['resolution'] or guide['tile'] != face['tile']:
        raise ValueError('Guide face resolution or atlas tile differs from baseline')
    if guide['size'] != [1024, 1024]:
        raise ValueError('This experiment requires full square 1024px guides')
    camera = guide['camera']
    for key in ('viewMatrix', 'projectionMatrix', 'frustum', 'up', 'side'):
        if camera.get(key) != face[key]:
            raise ValueError('Guide native camera differs from baseline: ' + key)
    matrix = np.asarray(guide['nativeToSquarePixel'], dtype='f8')
    inverse = np.asarray(guide['squareToNativePixel'], dtype='f8')
    if matrix.shape != (2, 3) or inverse.shape != (2, 3) or not np.isfinite([matrix, inverse]).all():
        raise ValueError('Invalid guide pixel-center affine')
    if not np.allclose(inverse_affine(matrix), inverse, rtol=0, atol=1e-10):
        raise ValueError('Guide affine and inverse disagree')
    left, right, bottom, top = face['frustum'][:4]
    physical = np.asarray([right - left, top - bottom], dtype='f8')
    span = float(guide['squarePhysicalSpan'])
    if not np.isfinite(physical).all() or np.min(physical) <= 0 or span <= 0 or not math.isfinite(span):
        raise ValueError('Invalid physical guide dimensions')
    if not np.allclose(physical, guide['physicalSize'], rtol=1e-10, atol=1e-10):
        raise ValueError('Guide physical size differs from the native camera')
    n, r = guide['size'][0], manifest['resolution']
    scales = physical / span * n / r
    expected = np.array([[scales[0], 0, n / 2 - scales[0] * r / 2 + scales[0] / 2 - .5],
                         [0, scales[1], n / 2 - scales[1] * r / 2 + scales[1] / 2 - .5]])
    if not np.allclose(matrix, expected, rtol=1e-10, atol=1e-9):
        raise ValueError('Guide affine is not the declared centered, aspect-preserving letterbox')
    corners = matrix @ np.array([[-.5, r - .5, -.5, r - .5],
                                 [-.5, -.5, r - .5, r - .5], [1, 1, 1, 1]])
    if corners.min() < -.500000001 or corners.max() > n - .499999999:
        raise ValueError('Guide mapping crops the native face rectangle')
    return matrix


def atlas_paths(root, manifest, hashes):
    paths = {name for name in hashes if name.endswith('.rgba16f.bin')}
    for value in manifest['textures'].values():
        if value.get('channels') == 4 and value.get('dtype') == 'float16':
            paths.add(value['path'])
    for name in paths:
        if safe(root, name).stat().st_size % 8:
            raise ValueError('RGBA16F atlas byte count is not a multiple of eight: ' + name)
    return sorted(paths)


def compare_assets(before_path, after_path, selected_faces):
    """Independent byte-level audit, also used by the Inspector comparison wrapper."""
    before_path, after_path = manifest_path(before_path), manifest_path(after_path)
    before, after = [json.loads(path.read_text()) for path in (before_path, after_path)]
    resolution, descriptor = validate_asset(before_path.parent, before)
    validate_asset(after_path.parent, after)
    if structural_manifest(before) != structural_manifest(after):
        raise ValueError('A structural/camera/texture-descriptor manifest field changed')
    if not selected_faces or len(set(selected_faces)) != len(selected_faces) or not set(selected_faces) <= set(FACES):
        raise ValueError('Expected unique explicitly selected native faces')
    old_files, new_files = file_hashes(before_path.parent), file_hashes(after_path.parent)
    color_name = descriptor['path']
    preserved = {name: digest for name, digest in old_files.items() if name not in ('manifest.json', color_name)}
    if any(new_files.get(name) != digest for name, digest in preserved.items()):
        raise ValueError('A file other than manifest/color RGB changed or disappeared')
    alpha_checks = {}
    for name in atlas_paths(before_path.parent, before, old_files):
        pair = {'source': alpha_sha(safe(before_path.parent, name)),
                'output': alpha_sha(safe(after_path.parent, name))}
        if pair['source'] != pair['output']:
            raise ValueError('An atlas alpha uint16 bit payload changed: ' + name)
        alpha_checks[name] = pair
    shape = (2 * resolution, 3 * resolution, 4)
    original = np.fromfile(safe(before_path.parent, color_name), dtype='<u2').reshape(shape)
    updated = np.fromfile(safe(after_path.parent, color_name), dtype='<u2').reshape(shape)
    changed = np.any(original[..., :3] != updated[..., :3], axis=-1)
    per_face = {}
    for face_name in FACES:
        tx, ty = before['faces'][face_name]['tile']
        region = np.s_[ty * resolution:(ty + 1) * resolution, tx * resolution:(tx + 1) * resolution]
        count = int(changed[region].sum())
        if face_name not in selected_faces and not np.array_equal(original[region], updated[region]):
            raise ValueError('An unselected face changed: ' + face_name)
        per_face[face_name] = count
    if not changed.any():
        raise ValueError('Composition changed no RGB pixels')
    return {'selectedFaces': list(selected_faces), 'unchangedFaces': [f for f in FACES if f not in selected_faces],
            'changedRgbPixels': int(changed.sum()), 'changedRgbPixelsByFace': per_face,
            'colorAlphaBitDifferences': int(np.count_nonzero(original[..., 3] != updated[..., 3])),
            'rgbaAlphaChecks': alpha_checks, 'unselectedFacesBitIdentical': True,
            'geometryMetadataSourceSha256': object_sha(structural_manifest(before)),
            'geometryMetadataOutputSha256': object_sha(structural_manifest(after)),
            'geometryDepthNormalIntervalRangeProxyCameraPreserved': True,
            'preservedSourceFiles': preserved,
            'sourceManifestSha256': old_files['manifest.json'], 'outputManifestSha256': new_files['manifest.json']}


def run(args):
    template = manifest_path(args.baseline_asset)
    source_root, output = template.parent, args.output.resolve()
    selection_path = args.selection.resolve(strict=True)
    if output.is_relative_to(source_root):
        raise ValueError('Output must be outside the immutable baseline asset')
    if not 0 <= args.iou_reference <= 1 or not 64 <= args.registration_size <= 1024:
        raise ValueError('Invalid diagnostic IoU or registration size')
    source = json.loads(template.read_text())
    source_hashes = file_hashes(source_root)
    template_sha = source_hashes['manifest.json']
    resolution, descriptor = validate_asset(source_root, source)
    selection = json.loads(selection_path.read_text())
    items = selection['faces']
    selected_faces = [item['face'] for item in items]
    if not items or len(set(selected_faces)) != len(items) or not set(selected_faces) <= set(FACES):
        raise ValueError('Selection must explicitly name one to six unique native faces')
    original = np.fromfile(safe(source_root, descriptor['path']), dtype='<u2').reshape(2 * resolution, 3 * resolution, 4)
    updated = original.copy()
    inputs = {str(selection_path): sha(selection_path)}
    reports, masks = [], {}
    for item in items:
        name = item['face']
        def selection_file(key):
            path = Path(item[key])
            return (path if path.is_absolute() else selection_path.parent / path).resolve(strict=True)
        guide_path, result_path = selection_file('guide'), selection_file('result')
        guide, result = [json.loads(path.read_text()) for path in (guide_path, result_path)]
        if guide.get('face') != name or result.get('face') != name:
            raise ValueError('Selected face, guide and Qwen result must name the same face')
        if result.get('completed') is not True or result.get('faceSlug') != guide['slug']:
            raise ValueError('Selected Qwen result is incomplete or has a different face slug')
        for result_key, guide_key in [('target-color.png', 'rgb'), ('target-depth.png', 'depth512')]:
            if result['inputsSHA256'][result_key] != guide['files'][guide_key]['sha256']:
                raise ValueError('Qwen result used a different guide input: ' + result_key)
        inputs.update({str(guide_path): sha(guide_path), str(result_path): sha(result_path)})
        mapping = face_to_square(source, template_sha, guide)
        paths = {key: resolve_record(record, guide_path.parent, 'guide ' + key)
                 for key, record in guide['files'].items()}
        rgb_path = resolve_record(result['images']['generated'], result_path.parent, 'generated RGB')
        mask_path = resolve_record(result['images']['mask'], result_path.parent, 'generated mask')
        for path in [*paths.values(), rgb_path, mask_path]:
            inputs[str(path)] = sha(path)
        with Image.open(rgb_path) as image:
            rgb = np.asarray(image.convert('RGB'), dtype='f4') / 255
        qmask, guide_alpha = read_alpha(mask_path), read_alpha(paths['alpha'])
        shape = tuple(guide['size'][::-1])
        if rgb.shape != (*shape, 3) or qmask.shape != shape or guide_alpha.shape != shape:
            raise ValueError('RGB/mask/guide dimensions disagree for ' + name)
        gm = guide_alpha > .5
        edge = int(gm[0].sum() + gm[-1].sum() + gm[1:-1, 0].sum() + gm[1:-1, -1].sum())
        if not gm.any() or edge:
            raise ValueError('The full-subject native guide is empty or clipped: ' + name)
        registration, registered_mask, reg = register(qmask, guide_alpha, args.registration_size, args.iou_reference)
        candidate, support, native_alpha, footprint = atlas_rgb(rgb, qmask, guide_alpha, registration, mapping, resolution)
        tx, ty = source['faces'][name]['tile']
        region = np.s_[ty * resolution:(ty + 1) * resolution, tx * resolution:(tx + 1) * resolution]
        before = original[region][::-1].view('<f2').astype('f4')
        expected_structure = {'proxySha256': source['proxySha256'], 'depthSha256': source['faces'][name]['depthSha256'],
                              'nativeAlphaUint16Sha256': hashlib.sha256(original[region][::-1][..., 3].copy().tobytes()).hexdigest()}
        expected_structure.update({key + 'Sha256': source['textures'][key]['sha256']
                                   for key in ('normalMap', 'intervalMap', 'rangeMap')})
        if guide['structure'] != expected_structure:
            raise ValueError('Guide structural hashes differ from the baseline: ' + name)
        if sha(safe(source_root, source['faces'][name]['depth'])) != expected_structure['depthSha256']:
            raise ValueError('Native depth file digest changed: ' + name)
        if not np.isfinite(before).all() or np.any((before[..., 3] < 0) | (before[..., 3] > 1)):
            raise ValueError('Nonfinite original color/coverage for ' + name)
        active = (native_alpha > .02) & (before[..., 3] > 0) & (support > 0)
        mixed = before[..., :3] * (1 - support[..., None]) + candidate * before[..., 3, None] * support[..., None]
        tile = updated[region][::-1]
        tile[..., :3][active] = mixed.astype('<f2').view('<u2')[active]
        changed = int(np.any(updated[region][..., :3] != original[region][..., :3], axis=-1).sum())
        if changed == 0:
            raise ValueError('Selected candidate changes no RGB pixels: ' + name)
        masks[name] = registered_mask
        reports.append({'face': name, 'slug': guide['slug'], 'selectedResult': str(result_path),
                        'selectedResultSha256': inputs[str(result_path)], 'guide': str(guide_path),
                        'guideSha256': inputs[str(guide_path)], 'selectionReviewNote': item.get('reviewNote', ''),
                        'visualReviewStatus': item.get('visualReviewStatus', 'not-recorded'),
                        'registration': reg, 'mappingFaceTexelXYToSquareXY': mapping.tolist(),
                        'guideMatchesExactBaselineManifestAndCamera': True,
                        'guideForegroundPixels': int(gm.sum()), 'guideEdgePixels': edge,
                        'squareGuideSize': guide['size'], 'subsamplesXY': footprint,
                        'mappedGuideVsOriginalBoxAlphaIoU': iou(native_alpha > .5, before[..., 3] > .5),
                        'qwenSupportVsOriginalBoxAlphaIoU': iou((support * native_alpha) > .5, before[..., 3] > .5),
                        'changedRgbPixels': changed, 'blendActivePixels': int(active.sum()),
                        'fallbackForegroundPixels': int(((before[..., 3] > .5) & ((support < .5) | (native_alpha <= .02))).sum())})
    if not np.array_equal(original[..., 3], updated[..., 3]):
        raise RuntimeError('Color alpha uint16 bits changed')
    identity = {'method': METHOD, 'sourceFiles': source_hashes, 'inputHashes': inputs,
                'selectedFaces': selected_faces,
                'registration': {r['face']: r['registration']['matrixInputToOutput'] for r in reports},
                'mapping': {r['face']: r['mappingFaceTexelXYToSquareXY'] for r in reports},
                'iouReference': args.iou_reference, 'scriptSha256': sha(__file__)}
    asset_id = object_sha(identity)
    root, receipt_path, archive = output / asset_id, output / (asset_id + '.json'), output / (asset_id + '.box.zip')
    if any(path.exists() for path in (root, receipt_path, archive)):
        raise FileExistsError('Immutable output already exists: ' + str(root))
    output.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_root, root)
    try:
        if file_hashes(root) != source_hashes:
            raise RuntimeError('Source changed during copy')
        updated.tofile(safe(root, descriptor['path']))
        manifest = copy.deepcopy(source)
        manifest.update(asset_id=asset_id, parentAsset=source['asset_id'], mode='qwen-selected-faces-rgb-only')
        for value in [*manifest['textures'].values(), manifest.get('baseColor', {})]:
            if value.get('path') == descriptor['path']:
                value['sha256'] = sha(safe(root, descriptor['path']))
        qa_relative = 'qa/qwen-selected-faces-' + asset_id[:16] + '.json'
        manifest['qa'] = qa_relative
        (root / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
        checks = compare_assets(template, root, selected_faces)
        for report in reports:
            relative = 'qa/qwen-' + report['slug'] + '-' + asset_id[:16] + '-registered-mask.png'
            path = safe(root, relative)
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.uint8(np.clip(np.rint(masks[report['face']] * 255), 0, 255))).save(path)
            report['registeredMask'] = {'path': relative, 'sha256': sha(path)}
        qa = {'method': METHOD, 'status': 'rgb-only-comparison-variant',
              'sourceAssetId': source['asset_id'], 'assetId': asset_id, **checks,
              'selection': {'path': str(selection_path), 'sha256': inputs[str(selection_path)]},
              'inputs': {path: {'sha256': digest} for path, digest in inputs.items()},
              'faces': reports, 'resolution': resolution, 'allFacesBeginFromSameBaseline': True,
              'sampling': 'continuous pixel-center mapping; one inverse whole-subject affine; linear-premultiplied mask/guide-alpha weighted subpixel integration',
              'blendFormula': 'w=integral(guideAlpha*Qmask)/integral(guideAlpha); RGBout=(1-w)*RGBold+w*Qlinear*oldAlpha',
              'limitations': ['Mask IoU measures silhouette overlap; it does not prove internal texture/anatomical correspondence.',
                              'The selected candidates require full-body visual review, including gaps, hands, shoes and cross-face seams.',
                              'RGB changes do not improve the baseline 3D geometry, depth or normals.',
                              'Original source guide/preview files remain unchanged; Inspector renders the updated color atlas.',
                              'Six independently generated faces are not a guarantee of multiview consistency.']}
        write_json(safe(root, qa_relative), qa)
        if file_hashes(source_root) != source_hashes or any(sha(path) != digest for path, digest in inputs.items()):
            raise RuntimeError('A source/input file changed during composition')
        with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
            for path in sorted(root.rglob('*')):
                if path.is_file():
                    bundle.write(path, path.relative_to(root).as_posix())
        with zipfile.ZipFile(archive) as bundle:
            if bundle.testzip() is not None:
                raise RuntimeError('Archive CRC verification failed')
        receipt = {'assetId': asset_id, 'assetPath': str(root), 'archive': str(archive), 'archiveSha256': sha(archive),
                   'identity': identity, 'qa': qa, 'outputFiles': file_hashes(root)}
        write_json(receipt_path, receipt)
    except Exception:
        (root / 'INCOMPLETE').write_text('Composition failed; do not use this asset.\n')
        raise
    return {'assetId': asset_id, 'assetPath': str(root), 'archive': str(archive), 'archiveSha256': sha(archive),
            'receiptPath': str(receipt_path), 'receiptSha256': sha(receipt_path),
            'selectedFaces': selected_faces,
            'registeredIoUByFace': {r['face']: r['registration']['registeredIoU'] for r in reports},
            'changedRgbPixels': checks['changedRgbPixels'], 'geometryAndAlphaPreserved': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-asset', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True, help='Explicit per-face result.json + guide.json selection')
    parser.add_argument('--output', type=Path, required=True, help='Parent for immutable asset, receipt and ZIP')
    parser.add_argument('--registration-size', type=int, default=256)
    parser.add_argument('--iou-reference', type=float, default=.88, help='Diagnostic reference; does not substitute for visual review')
    print(json.dumps(run(parser.parse_args()), indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()

