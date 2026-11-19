#!/usr/bin/env python3
"""Evaluate saved Box orbit captures without modifying models or Inspector HTML.

Front references come from the actual source atlas, independently of every
rendering variant. Motion comparisons reproject captured depth and exclude
occlusion; their residuals measure consistency, not ground-truth facial accuracy.
PNG depth containers hold raw little-endian Float32 bytes in their RGBA channels.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_erosion, map_coordinates

METHOD = 'independent-atlas-and-depth-reprojected-orbit-evaluation-v1'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def resolve(root, value):
    path = Path(value)
    return path if path.is_absolute() else root / path


def matrix(values):
    result = np.asarray(values, dtype='f8').reshape(4, 4).T
    if not np.isfinite(result).all():
        raise ValueError('Non-finite camera matrix.')
    return result


def srgb_to_linear(rgb):
    return np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)


def linear_to_srgb(rgb):
    rgb = np.maximum(rgb, 0)
    return np.where(rgb <= .0031308, rgb * 12.92, 1.055 * rgb ** (1 / 2.4) - .055)


def rgba_linear(rgba):
    result = np.asarray(rgba, dtype='f4') / 255
    result[..., :3] = srgb_to_linear(result[..., :3])
    return result


def rgba_png(linear):
    data = np.asarray(linear).copy()
    data[..., :3] = linear_to_srgb(data[..., :3])
    return Image.fromarray(np.rint(np.clip(data, 0, 1) * 255).astype('u1'))


def read_rgba(path, expected_hash=None):
    if expected_hash is not None and sha(path) != expected_hash:
        raise ValueError('Capture PNG checksum mismatch: ' + str(path))
    with Image.open(path) as image:
        if image.mode != 'RGBA':
            raise ValueError('Expected an exact RGBA PNG: ' + str(path))
        return np.asarray(image).copy()


def read_depth(path, expected_raw_hash, width, height):
    pixels = read_rgba(path)
    if pixels.shape != (height, width, 4):
        raise ValueError('Depth container dimensions differ from the capture.')
    raw = np.ascontiguousarray(pixels).tobytes()
    if hashlib.sha256(raw).hexdigest() != expected_raw_hash:
        raise ValueError('Packed Float32 depth checksum mismatch: ' + str(path))
    return np.frombuffer(raw, dtype='<f4').reshape(height, width)


def pixel_grid(roi):
    x0, y0, x1, y1 = roi
    y, x = np.mgrid[y0:y1, x0:x1]
    return x.astype('f8'), y.astype('f8')


def unproject(frame, x, y, depth, width, height):
    clip = np.stack((2 * (x + .5) / width - 1, 1 - 2 * (y + .5) / height,
                     2 * depth - 1, np.ones_like(x)), axis=-1)
    world = clip @ np.linalg.inv(matrix(frame['projectionMatrix']) @ matrix(frame['viewMatrix'])).T
    return world[..., :3] / world[..., 3:4]


def project(frame, world, width, height):
    homogeneous = np.concatenate((world, np.ones((*world.shape[:-1], 1))), axis=-1)
    clip = homogeneous @ (matrix(frame['projectionMatrix']) @ matrix(frame['viewMatrix'])).T
    ndc = clip[..., :3] / clip[..., 3:4]
    return ((ndc[..., 0] + 1) * width / 2 - .5,
            (1 - ndc[..., 1]) * height / 2 - .5, (ndc[..., 2] + 1) / 2, clip[..., 3])


def sample(data, x, y, *, premultiplied=False):
    values = np.asarray(data, dtype='f4').copy()
    if premultiplied:
        values[..., :3] *= values[..., 3:4]
    out = np.stack([map_coordinates(values[..., c], [y, x], order=1,
                                    mode='grid-constant', cval=0, prefilter=False)
                    for c in range(values.shape[-1])], axis=-1)
    if premultiplied:
        np.divide(out[..., :3], out[..., 3:4], out=out[..., :3], where=out[..., 3:4] > 0)
    return out


def percentile(values):
    values = np.asarray(values)
    if not values.size:
        return None
    return {'mean': float(values.mean()), 'p50': float(np.percentile(values, 50)),
            'p95': float(np.percentile(values, 95)), 'max': float(values.max())}


def error_metrics(a, b, mask):
    if not mask.any():
        return {'pixels': 0, 'linearRGBAbsoluteError': None, 'sRGB8AbsoluteError': None}
    linear = np.abs(a[..., :3] - b[..., :3])[mask]
    encoded = np.abs(linear_to_srgb(a[..., :3]) - linear_to_srgb(b[..., :3]))[mask] * 255
    return {'pixels': int(mask.sum()), 'linearRGBAbsoluteError': percentile(linear),
            'sRGB8AbsoluteError': percentile(encoded),
            'pixelsWithinOneSRGB8CodeValueAllChannels': float(np.mean(np.max(encoded, axis=1) <= 1))}


def foreground(rgba, erode=2):
    return binary_erosion(rgba[..., 3] >= .99, iterations=erode, border_value=0) if erode else rgba[..., 3] >= .99


def front_world(frame, width, height):
    view, projection = matrix(frame['viewMatrix']), matrix(frame['projectionMatrix'])
    if (not np.allclose(view[:3, :3], np.eye(3), rtol=0, atol=1e-10)
            or not np.allclose(projection[3], [0, 0, 0, 1], rtol=0, atol=1e-12)):
        raise ValueError('Independent front reference requires the recorded +Z orthographic camera.')
    x, y = pixel_grid([0, 0, width, height])
    return unproject(frame, x, y, np.full_like(x, .5), width, height)


def native_atlas_reference(sweep, root, frame, width, height):
    source = sweep['source']
    manifest_path = resolve(root, source['manifestPath'])
    if sha(manifest_path) != source['manifestSHA256']:
        raise ValueError('Source manifest changed.')
    manifest = json.loads(manifest_path.read_text())
    desc = manifest['textures']['colorMap']
    supplied = source.get('textures', {}).get('colorMap', {})
    texture_path = resolve(manifest_path.parent, supplied.get('absolutePath', supplied.get('path', desc['path'])))
    if sha(texture_path) != desc['sha256']:
        raise ValueError('Source color atlas changed.')
    r = manifest['resolution']
    if desc['dtype'] != 'float16' or (desc['width'], desc['height']) != (3 * r, 2 * r):
        raise ValueError('Expected the native linear RGBA16F six-face Box atlas.')
    atlas = np.fromfile(texture_path, dtype='<f2').reshape(2 * r, 3 * r, 4)
    # +Z is face4, tile(1,1); file rows increase native V. Do not use a variant render.
    premultiplied = atlas[r:2 * r, r:2 * r][::-1].astype('f4')
    if not np.isfinite(premultiplied).all():
        raise ValueError('Non-finite +Z color atlas samples.')
    world = front_world(frame, width, height)
    low, high = np.asarray(manifest['bakeBox']['min']), np.asarray(manifest['bakeBox']['max'])
    x = (world[..., 0] - low[0]) / (high[0] - low[0]) * r - .5
    y = (high[1] - world[..., 1]) / (high[1] - low[1]) * r - .5
    sampled = sample(premultiplied, x, y)
    np.divide(sampled[..., :3], sampled[..., 3:4], out=sampled[..., :3], where=sampled[..., 3:4] > 0)
    return sampled, {
        'kind': 'independently projected actual native +Z atlas', 'assetId': manifest.get('asset_id'),
        'manifest': str(manifest_path), 'manifestSHA256': sha(manifest_path),
        'texture': str(texture_path), 'textureSHA256': sha(texture_path), 'tile': [1, 1],
        'nativeResolution': r, 'bakeBox': manifest['bakeBox'],
        'sampling': 'one bilinear pass in premultiplied linear RGBA; then unpremultiply',
        'sourcePixels': 'original native atlas, increasing-V storage corrected to top-down',
        'displayTransform': 'linear RGB to sRGB only; no tone mapping of independent reference',
        'notOriginalGeneratedPNG': True,
        'limitation': 'H atlas contains its recorded alignment/reprojection of frozen F sources; not byte-identical original RGB',
    }, manifest


def original_png_reference(paths, frame, width, height):
    image_path, manifest_path, qa_path = map(Path, paths)
    manifest = json.loads(manifest_path.read_text())
    face = manifest['faces']['+Z']
    qa = json.loads(qa_path.read_text())['faces']['+Z']
    if qa['registration'] != {'scale': 1., 'centerShiftYX': [0., 0.]}:
        raise ValueError('Original PNG secondary track currently requires recorded identity registration.')
    canvas = face['edit']['provenance']['canvas']
    image = read_rgba(image_path)
    if image.shape[:2] != (canvas['size'], canvas['size']):
        raise ValueError('Original physical source canvas size differs.')
    world = front_world(frame, width, height)
    x, y, z, clip_w = project(face, world, canvas['size'], canvas['size'])
    # project() uses the entire native capture tile. Place that tile into its
    # physical-aspect rectangle on the actual generated square source canvas.
    gx, gy, gw, gh = canvas['guideRect']
    sx = gx + (x + .5) * gw / canvas['size'] - .5
    sy = gy + (y + .5) * gh / canvas['size'] - .5
    sampled = sample(rgba_linear(image), sx, sy, premultiplied=True)
    return sampled, {'kind': 'original generated full-body physical PNG, independently projected',
                     'source': str(image_path), 'sourceSHA256': sha(image_path),
                     'manifest': str(manifest_path), 'manifestSHA256': sha(manifest_path),
                     'qa': str(qa_path), 'qaSHA256': sha(qa_path), 'canvas': canvas,
                     'registration': qa['registration'], 'sourceDimensions': [image.shape[1], image.shape[0]],
                     'notNativeAlignedAtlas': True}


def correspondence(a, b, roi, width, height, tolerance_world):
    x, y = pixel_grid(roi)
    x0, y0, x1, y1 = roi
    cut = (slice(y0, y1), slice(x0, x1))
    depth = a['_depth'][cut]
    world = unproject(a, x, y, depth, width, height)
    tx, ty, expected, clip_w = project(b, world, width, height)
    other_depth = map_coordinates(b['_depth'], [ty, tx], order=0, mode='constant', cval=np.nan, prefilter=False)
    a_rgba, b_rgba = rgba_linear(a['_rgba']), rgba_linear(b['_rgba'])
    from_color = a_rgba[cut]
    to_color = sample(b_rgba, tx, ty, premultiplied=True)
    source_mask = foreground(a_rgba)[cut] & np.isfinite(depth) & (depth >= 0) & (depth < 1)
    target_mask = map_coordinates(foreground(b_rgba).astype('u1'), [ty, tx], order=0,
                                  mode='constant', cval=0, prefilter=False).astype(bool)
    residual_world = np.abs(other_depth - expected) * (b['far'] - b['near'])
    valid = (source_mask & target_mask & np.isfinite(expected) & np.isfinite(other_depth)
             & (expected >= 0) & (expected < 1) & (clip_w > 0)
             & (residual_world <= tolerance_world))
    return {'a': from_color, 'b': to_color, 'valid': valid, 'sourceMask': source_mask,
            'x': x, 'y': y, 'tx': tx, 'ty': ty, 'world': world, 'depthResidual': residual_world}


def motion_metrics(a, b, roi, width, height, tolerance_world, frozen):
    c = correspondence(a, b, roi, width, height, tolerance_world)
    valid = c['valid']
    result = {'fromAngle': a['angleDegrees'], 'toAngle': b['angleDegrees'], 'roi': roi,
              'sourceInteriorPixels': int(c['sourceMask'].sum()), 'mutuallyVisiblePixels': int(valid.sum()),
              'mutuallyVisibleFractionOfSourceInterior': float(valid.sum() / max(1, c['sourceMask'].sum())),
              'colorAfterDepthReprojection': error_metrics(c['a'], c['b'], valid),
              'depthReprojectionResidualWorld': percentile(c['depthResidual'][valid]),
              'projectedMotionPixels': percentile(np.hypot(c['tx'] - c['x'], c['ty'] - c['y'])[valid])}
    frozen_from = sample(frozen, c['x'], c['y'], premultiplied=True)
    frozen_to = sample(frozen, c['tx'], c['ty'], premultiplied=True)
    common = valid & (frozen_from[..., 3] >= .99) & (frozen_to[..., 3] >= .99)
    result['frozenFrontPixelControl'] = {
        'description': 'repeat the same front image at every angle; apply the same measured geometry reprojection',
        'commonOpaquePixels': int(common.sum()),
        'actualVariant': error_metrics(c['a'], c['b'], common),
        'frozenFrontPixels': error_metrics(frozen_from, frozen_to, common),
        'notPhysicalBillboardRender': True,
    }
    if valid.any():
        # A constant world-Z plane is an explicitly labeled planar-motion control.
        plane_z = float(np.median(c['world'][..., 2][valid]))
        near = unproject(a, c['x'], c['y'], np.zeros_like(c['x']), width, height)
        far = unproject(a, c['x'], c['y'], np.ones_like(c['x']), width, height)
        ray = far - near
        fraction = np.divide(plane_z - near[..., 2], ray[..., 2],
                             out=np.full_like(c['x'], np.nan), where=abs(ray[..., 2]) > 1e-12)
        plane_world = near + ray * fraction[..., None]
        px, py, _, _ = project(b, plane_world, width, height)
        finite = valid & np.isfinite(px) & np.isfinite(py)
        result['motionBeyondConstantZPlane'] = {
            'planeWorldZ': plane_z, 'pixels': int(finite.sum()),
            'motionResidualAgainstConstantZPlanePixels': percentile(np.hypot(c['tx'] - px, c['ty'] - py)[finite]),
            'interpretation': 'motion implied by captured depth relative to this constant world-Z plane; does not exclude every oriented plane or establish measured ground truth',
        }
        candidates = np.flatnonzero(valid)
        ordering = candidates[np.argsort(c['world'][..., 2].ravel()[candidates])]
        chosen = ordering[np.linspace(0, len(ordering) - 1, min(5, len(ordering))).astype(int)]
        result['depthCorrespondenceExamples'] = [{
            'sourcePixelXY': [float(c['x'].ravel()[i]), float(c['y'].ravel()[i])],
            'targetPixelXY': [float(c['tx'].ravel()[i]), float(c['ty'].ravel()[i])],
            'capturedWorldXYZ': c['world'].reshape(-1, 3)[i].tolist(),
            'sourceLinearRGBA': c['a'].reshape(-1, 4)[i].tolist(),
            'sampledTargetLinearRGBA': c['b'].reshape(-1, 4)[i].tolist(),
        } for i in chosen]
    return result


def curvature(previous, current, following, roi, width, height, tolerance):
    left = correspondence(current, previous, roi, width, height, tolerance)
    right = correspondence(current, following, roi, width, height, tolerance)
    valid = left['valid'] & right['valid']
    dl = current['angleDegrees'] - previous['angleDegrees']
    dr = following['angleDegrees'] - current['angleDegrees']
    if dl <= 0 or dr <= 0:
        raise ValueError('Curvature samples must have increasing distinct camera angles.')
    derivative_left = (left['a'][..., :3] - left['b'][..., :3]) / dl
    derivative_right = (right['b'][..., :3] - right['a'][..., :3]) / dr
    jump = np.abs(derivative_right - derivative_left)
    return {'anglesDegrees': [previous['angleDegrees'], current['angleDegrees'], following['angleDegrees']],
            'roi': roi, 'mutuallyVisibleInAllThreePixels': int(valid.sum()),
            'linearRGBDerivativeJumpPerDegree': percentile(jump[valid]),
            'linearRGBSecondDerivativePerDegreeSquared': percentile((jump / ((dl + dr) / 2))[valid]),
            'interpretation': 'change after geometric motion compensation; can include view-dependent color differences; not a ghosting score'}


def check_geometry(frames, width, height):
    groups = {}
    for frame in frames:
        groups.setdefault((frame['sequenceIndex'], frame['angleDegrees']), []).append(frame)
    result = []
    for (sequence, angle), group in groups.items():
        base = next((f for f in group if f['variant'] == 'native'), group[0])
        for other in group:
            if other is base:
                continue
            same_camera = all(np.allclose(matrix(base[k]), matrix(other[k]), rtol=0, atol=1e-12)
                              for k in ('viewMatrix', 'projectionMatrix'))
            if not same_camera:
                raise ValueError('Variant cameras differ at the same sweep sample.')
            opaque = (base['_rgba'][..., 3] > 127) & (other['_rgba'][..., 3] > 127)
            difference = abs(base['_depth'] - other['_depth']) * (base['far'] - base['near'])
            result.append({'sequenceIndex': sequence, 'angleDegrees': angle, 'baseline': base['variant'],
                           'variant': other['variant'], 'alphaBytesEqual': bool(np.array_equal(base['_rgba'][..., 3], other['_rgba'][..., 3])),
                           'depthFloat32BytesEqual': base['_depth'].tobytes() == other['_depth'].tobytes(),
                           'depthWorldDifferenceOnOpaqueIntersection': percentile(difference[opaque])})
    return result


def repeats(frames):
    groups = {}
    for f in frames:
        groups.setdefault((f['variant'], f['angleDegrees']), []).append(f)
    result = []
    for (variant, angle), group in groups.items():
        first = group[0]
        for other in group[1:]:
            same = all(np.allclose(matrix(first[k]), matrix(other[k]), rtol=0, atol=1e-12)
                       for k in ('viewMatrix', 'projectionMatrix'))
            delta = abs(first['_rgba'].astype('i2') - other['_rgba'].astype('i2'))
            result.append({'variant': variant, 'angleDegrees': angle, 'sequenceIndices': [first['sequenceIndex'], other['sequenceIndex']],
                           'cameraMatricesEqualWithin1e12': same, 'rgba8AbsoluteDifference': percentile(delta),
                           'RGBABytesEqual': bool(np.array_equal(first['_rgba'], other['_rgba'])),
                           'depthBytesEqual': first['_depth'].tobytes() == other['_depth'].tobytes()})
    return result


def image_assets(frames, output, references, display_variants=None, display_max_yaw=None):
    variants = list(dict.fromkeys(f['variant'] for f in frames))
    unique = {variant: {} for variant in variants}
    for f in sorted(frames, key=lambda f: f['sequenceIndex']):
        unique[f['variant']].setdefault(f['angleDegrees'], f)
    maximum = max(map(len, unique.values()))
    variants = [v for v in variants if len(unique[v]) == maximum]
    if display_variants:
        if any(v not in variants for v in display_variants):
            raise ValueError('Requested display variant lacks the complete angle set.')
        variants = list(display_variants)
    angles = sorted(set.intersection(*(set(unique[v]) for v in variants)))
    if display_max_yaw is not None:
        angles = [a for a in angles if abs(a) <= display_max_yaw]
    if not angles:
        return []
    rois = np.array([f['headROI'] for f in frames], dtype=int)
    roi = [int(rois[:, 0].min()), int(rois[:, 1].min()), int(rois[:, 2].max()), int(rois[:, 3].max())]
    scale = max(1, min(3, 300 // max(1, roi[2] - roi[0])))
    size = ((roi[2] - roi[0]) * scale, (roi[3] - roi[1]) * scale)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 15)
    except OSError:
        font = ImageFont.load_default()
    background = (43, 46, 51)

    def tile(frame):
        image = Image.fromarray(frame['_rgba']).crop(roi)
        panel = Image.new('RGBA', image.size, (*background, 255))
        panel.alpha_composite(image)
        return panel.convert('RGB').resize(size, Image.Resampling.NEAREST)

    def label(variant):
        return 'native weights / unlit' if variant == 'native' else variant

    assets = []
    display_angles = [angles[i] for i in np.unique(np.linspace(0, len(angles) - 1, min(7, len(angles))).astype(int))]
    board = Image.new('RGB', (len(display_angles) * (size[0] + 12) + 12,
                              len(variants) * (size[1] + 40) + 55), background)
    draw = ImageDraw.Draw(board)
    draw.text((12, 10), 'Actual saved camera orbit | fixed head crop | no optical-flow alignment', font=font, fill='white')
    for row, variant in enumerate(variants):
        for col, angle in enumerate(display_angles):
            x, y = 12 + col * (size[0] + 12), 45 + row * (size[1] + 40)
            draw.text((x, y), f'{label(variant)} / {angle:+g} deg', font=font, fill='white')
            board.paste(tile(unique[variant][angle]), (x, y + 24))
    path = output / 'parallax-contact-sheet.webp'
    board.save(path, lossless=True, method=6)
    assets.append({'path': str(path), 'sha256': sha(path), 'kind': 'fixed-crop contact sheet',
                   'anglesDegrees': display_angles, 'variants': variants, 'fixedCropXYXY': roi, 'nearestEnlargement': scale})
    animation = []
    playback = angles + angles[-2:0:-1] if len(angles) > 1 else angles
    for angle in playback:
        canvas = Image.new('RGB', (len(variants) * (size[0] + 12) + 12, size[1] + 70), background)
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 8), f'Actual orbit: {angle:+g} deg | same fixed crop and display scale', font=font, fill='white')
        for col, variant in enumerate(variants):
            x = 12 + col * (size[0] + 12)
            draw.text((x, 30), label(variant), font=font, fill='white')
            canvas.paste(tile(unique[variant][angle]), (x, 52))
        animation.append(canvas)
    path = output / 'parallax-orbit.webp'
    animation[0].save(path, save_all=True, append_images=animation[1:], duration=180, loop=0, lossless=True, method=4)
    assets.append({'path': str(path), 'sha256': sha(path), 'kind': 'animation of actual saved frames',
                   'playbackAnglesDegrees': playback, 'durationMsPerFrame': 180,
                   'fixedCropXYXY': roi, 'nearestEnlargement': scale,
                   'reversePlayback': 'reuses saved forward frames; does not replace the independent repeat-camera check'})
    return assets


def evaluate(sweep_path, output, original_paths=None, tolerance_texels=2., display_variants=None, display_max_yaw=None):
    sweep_path, output = Path(sweep_path).resolve(), Path(output).resolve()
    root = sweep_path.parent
    sweep = json.loads(sweep_path.read_text())
    if sweep.get('schemaVersion') != 1 or sweep.get('outputColorSpace') != 'srgb':
        raise ValueError('Unsupported orbit capture schema or output color space.')
    if 'WebGL' not in sweep.get('depthEncoding', '') or 'Float32' not in sweep['depthEncoding']:
        raise ValueError('Expected declared WebGL Float32 depth encoding.')
    width, height = sweep['width'], sweep['height']
    frames, depth_cache, input_files = [], {}, {}
    for entry in sweep['frames']:
        f = dict(entry)
        path = resolve(root, f['rgbaPNG'])
        f['_rgba'] = read_rgba(path, f['rgbaSHA256'])
        if f['_rgba'].shape != (height, width, 4):
            raise ValueError('Capture dimensions differ.')
        if hashlib.sha256(np.ascontiguousarray(f['_rgba'][..., 3]).tobytes()).hexdigest() != f['alphaSHA256']:
            raise ValueError('Alpha checksum mismatch.')
        depth_path = resolve(root, f['depthPNG'])
        key = (str(depth_path), f['depthRawSHA256'])
        if key not in depth_cache:
            depth_cache[key] = read_depth(depth_path, f['depthRawSHA256'], width, height)
        f['_depth'] = depth_cache[key]
        opaque = f['_rgba'][..., 3] > 127
        if not np.isfinite(f['_depth'][opaque]).all():
            raise ValueError('Non-finite depth under visible rendered pixels.')
        if not f['far'] > f['near'] > 0:
            raise ValueError('Invalid camera depth range.')
        roi = f['headROI']
        if len(roi) != 4 or not (0 <= roi[0] < roi[2] <= width and 0 <= roi[1] < roi[3] <= height):
            raise ValueError('Invalid head ROI.')
        for p in (path, depth_path):
            input_files[str(p)] = sha(p)
        frames.append(f)
    fronts = [f for f in frames if abs(f['angleDegrees']) < 1e-9]
    if not fronts:
        raise ValueError('The sweep must contain an actual front camera capture.')
    if any(not np.allclose(matrix(f[key]), matrix(fronts[0][key]), rtol=0, atol=1e-12)
           for f in fronts for key in ('projectionMatrix', 'viewMatrix')):
        raise ValueError('Front camera captures do not share one camera.')
    reference, reference_proof, manifest = native_atlas_reference(sweep, root, fronts[0], width, height)
    references = {'nativeAtlas': (reference, reference_proof)}
    if original_paths:
        references['originalGeneratedPNG'] = original_png_reference(original_paths, fronts[0], width, height)
    output.mkdir(parents=True, exist_ok=True)
    reference_artifacts = []
    for name, (rgba, proof) in references.items():
        path = output / ('front-reference-' + name + '.png')
        rgba_png(rgba).save(path)
        reference_artifacts.append({'path': str(path), 'sha256': sha(path), 'reference': name})
    front_scores = []
    seen_front = set()
    for f in fronts:
        if f['variant'] in seen_front:
            continue
        seen_front.add(f['variant'])
        for name, (ref, proof) in references.items():
            actual = rgba_linear(f['_rgba'])
            for region, roi in [('body', [0, 0, width, height]), ('head', f['headROI'])]:
                x0, y0, x1, y1 = roi
                cut = (slice(y0, y1), slice(x0, x1))
                rm, cm = ref[..., 3] >= .99, actual[..., 3] >= .99
                mask = (foreground(ref) & foreground(actual))[cut]
                front_scores.append({'variant': f['variant'], 'reference': name, 'region': region, 'roi': roi,
                    'referenceOpaquePixels': int(rm[cut].sum()), 'candidateOpaquePixels': int(cm[cut].sum()),
                    'opaqueIntersectionPixels': int((rm & cm)[cut].sum()),
                    'erodedIntersectionPixels': int(mask.sum()), 'erosionRadiusPixels': 2,
                    'silhouetteOpaqueIoU': float((rm & cm)[cut].sum() / max(1, (rm | cm)[cut].sum())),
                    **error_metrics(actual[cut], ref[cut], mask)})
    span = np.subtract(manifest['bakeBox']['max'], manifest['bakeBox']['min'])
    tolerance = float(max(span) / manifest['resolution'] * tolerance_texels)
    variants = list(dict.fromkeys(f['variant'] for f in frames))
    motions, curves = [], []
    for variant in variants:
        unique = {}
        for f in sorted([x for x in frames if x['variant'] == variant], key=lambda f: f['sequenceIndex']):
            unique.setdefault(f['angleDegrees'], f)
        ordered = [unique[a] for a in sorted(unique)]
        zero = next((f for f in ordered if abs(f['angleDegrees']) < 1e-9), None)
        if zero is None:
            continue
        frozen = rgba_linear(zero['_rgba'])
        for a, b in zip(ordered, ordered[1:]):
            motions.append({'variant': variant, 'region': 'head',
                            **motion_metrics(a, b, a['headROI'], width, height, tolerance, frozen)})
        for previous, current, following in zip(ordered, ordered[1:], ordered[2:]):
            curves.append({'variant': variant, 'region': 'head',
                           **curvature(previous, current, following, current['headROI'], width, height, tolerance)})
    assets = reference_artifacts + image_assets(frames, output, references, display_variants, display_max_yaw)
    result = {
        'method': METHOD, 'sweep': str(sweep_path), 'sweepSHA256': sha(sweep_path),
        'captureStatus': sweep.get('status'), 'captureError': sweep.get('error'),
        'evaluator': str(Path(__file__).resolve()), 'evaluatorSHA256': sha(__file__),
        'inputFrameFileSHA256': input_files, 'source': sweep['source'],
        'display': {k: sweep.get(k) for k in ('outputColorSpace', 'toneMapping', 'exposure', 'depthEncoding')},
        'references': {name: proof for name, (rgba, proof) in references.items()},
        'frontFidelity': front_scores, 'geometryAcrossVariants': check_geometry(frames, width, height),
        'originalNativeGeometryControl': sweep.get('nativeGeometryControl'),
        'depthReprojectedMotion': motions, 'motionCompensatedCurvature': curves,
        'repeatedCameraStability': repeats(frames),
        'visibilityPolicy': {'opaqueAlphaAtLeast': .99, 'boundaryErosionPixels': 2,
                             'targetDepthSampling': 'nearest; foreground colors bilinear in premultiplied linear RGBA',
                             'worldDepthTolerance': tolerance, 'toleranceNativeLongestSpanTexels': tolerance_texels,
                             'excludesOcclusionAndDisocclusion': True},
        'assets': assets, 'originalInspectorHTMLModified': False,
        'limitations': [
            'Native-atlas fidelity and original generated-PNG fidelity are distinct tracks.',
            'Reference colors receive only linear-to-sRGB encoding; a different captured tone curve remains a real reported display difference.',
            'Depth is the renderer own reconstructed geometry, not measured truth about the person.',
            'Reprojection metrics cover mutually visible opaque interiors; excluded coverage is reported.',
            'Low temporal residual can hide blur or consistent errors; consistency alone does not establish natural appearance.',
            'Curvature after motion compensation can include legitimate view-dependent appearance; it is not a ghosting score.',
            'Frozen-front pixels are an explicit screen-image control, not a rendered physical billboard.',
            'Montages and animation use actual saved images with a single fixed crop; they do not replace an interactive Inspector deliverable.',
        ],
    }
    (output / 'evaluation.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sweep', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--original-source', type=Path)
    ap.add_argument('--original-manifest', type=Path)
    ap.add_argument('--original-qa', type=Path)
    ap.add_argument('--depth-tolerance-texels', type=float, default=2.)
    ap.add_argument('--display-variants', help='Optional comma-separated montage columns; metrics still include all variants')
    ap.add_argument('--display-max-yaw', type=float, help='Optional absolute yaw limit for the main animation/contact sheet')
    args = ap.parse_args()
    original = [args.original_source, args.original_manifest, args.original_qa]
    if any(original) and not all(original):
        ap.error('Supply all three original-source/manifest/qa arguments or none.')
    if not np.isfinite(args.depth_tolerance_texels) or args.depth_tolerance_texels <= 0:
        ap.error('Depth tolerance must be positive and finite.')
    selected = args.display_variants.split(',') if args.display_variants else None
    result = evaluate(args.sweep, args.output, original if all(original) else None, args.depth_tolerance_texels,
                      selected, args.display_max_yaw)
    print(json.dumps({'report': str(args.output / 'evaluation.json'),
        'captureStatus': result['captureStatus'],
        'frontHeadFidelity': [r for r in result['frontFidelity'] if r['region'] == 'head'],
        'completeNativeGeometryControlAndVariantEquality': result['captureStatus'] == 'complete'
            and bool(result['originalNativeGeometryControl'])
            and result['originalNativeGeometryControl'].get('alphaEqual') is True
            and result['originalNativeGeometryControl'].get('depthBitEqual') is True
            and all(r['alphaBytesEqual'] and r['depthFloat32BytesEqual'] for r in result['geometryAcrossVariants']),
        'motionPairs': len(result['depthReprojectedMotion']),
        'repeatCameraComparisons': len(result['repeatedCameraStability']), 'assets': result['assets']}, indent=2))


if __name__ == '__main__':
    main()
