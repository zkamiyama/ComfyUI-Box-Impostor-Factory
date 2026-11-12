"""Foreground-aware registration and byte-preserving replacement of one Box face."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import tempfile

import numpy as np
from PIL import Image
from scipy import ndimage

from . import store, multiview

SLUGS = dict(zip(store.FACES, ('px', 'nx', 'py', 'ny', 'pz', 'nz')))
MASK_MODES = ['rgba', 'foreground', 'load_image', 'opaque']


def array(value):
    if hasattr(value, 'detach'):
        value = value.detach().to(device='cpu', dtype=__import__('torch').float32).numpy()
    return np.asarray(value, dtype=np.float32)


def rgba(image, mask_mode, mask=None):
    image = array(image)
    if image.ndim != 4 or image.shape[0] != 1 or image.shape[-1] not in (3, 4):
        raise ValueError('Connect a single ComfyUI RGB or RGBA image for each face.')
    image = image[0]
    h, w = image.shape[:2]
    if mask_mode == 'rgba' and image.shape[-1] == 4:
        alpha = image[..., 3]
    elif mask_mode == 'opaque':
        alpha = np.ones((h, w), dtype=np.float32)
    elif mask_mode in ('load_image', 'foreground') and mask is not None:
        alpha = array(mask)
        if alpha.ndim == 3 and alpha.shape[0] == 1:
            alpha = alpha[0]
        if alpha.shape != (h, w):
            raise ValueError('Mask size differs from the image. For images without alpha select opaque or supply a foreground mask.')
        if mask_mode == 'load_image':
            alpha = 1 - alpha
    else:
        raise ValueError('Provide RGBA, connect the selected mask, or explicitly select opaque.')
    result = np.concatenate((image[..., :3], alpha[..., None]), axis=2).astype('<f4', copy=False)
    if not np.isfinite(result).all() or np.any(result < 0) or np.any(result > 1):
        raise ValueError('Expected finite straight sRGB and coverage in [0,1].')
    return result


def cache_dir(kind, digest, create):
    base = store.root() / '.stages' / kind
    directory = base / digest
    with store.lock(digest):
        if not directory.is_dir():
            base.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=base) as temp:
                stage = Path(temp) / 'data'
                stage.mkdir()
                create(stage)
                os.rename(stage, directory)
    return directory


def png(path, a):
    Image.fromarray(np.rint(np.clip(a, 0, 1) * 255).astype(np.uint8)).save(path)


def save_view(data):
    digest = store.key('face-v1', data.shape, hashlib.sha256(data.tobytes()).hexdigest())
    def write(stage):
        np.save(stage / 'source.npy', data, allow_pickle=False)
        png(stage / 'source.png', data)
    directory = cache_dir('faces', digest, write)
    return {'sha256': digest, 'path': str(directory)}


def resize(data, shape):
    # Filter premultiplied color to avoid importing undefined colors from transparent pixels.
    return np.stack([np.asarray(Image.fromarray(channel.astype(np.float32)).resize((shape[1], shape[0]), Image.Resampling.BILINEAR)) for channel in np.moveaxis(data, -1, 0)], axis=-1)


def bounds(mask):
    yy, xx = np.nonzero(mask)
    if yy.size == 0:
        return None
    lo, hi = np.array([yy.min(), xx.min()]), np.array([yy.max(), xx.max()])
    return (lo + hi) / 2, hi - lo + 1


def scores(got, want):
    union = np.count_nonzero(got | want)
    iou = float(np.count_nonzero(got & want) / union) if union else 0.0
    if not got.any() or not want.any():
        return {'iou': iou, 'boundaryP95Px': None}
    edge_a = got ^ ndimage.binary_erosion(got)
    edge_b = want ^ ndimage.binary_erosion(want)
    distances = np.concatenate((ndimage.distance_transform_edt(~edge_b)[edge_a], ndimage.distance_transform_edt(~edge_a)[edge_b]))
    return {'iou': iou, 'boundaryP95Px': float(np.percentile(distances, 95))}


def align_one(source, guide, min_iou, max_shift, max_scale_change):
    h, w = guide.shape
    source = source.copy()
    rgb = source[..., :3]
    rgb[:] = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
    rgb *= source[..., 3:4]
    moving = resize(source, (h, w))
    fixed = guide >= .5
    mask = moving[..., 3] >= .5
    before = scores(mask, fixed)
    initial, target = bounds(mask), bounds(fixed)
    report = {'before': before, 'status': 'fallback', 'sourceSize': [source.shape[1], source.shape[0]], 'tileSize': [w, h]}
    if initial is None or target is None:
        return None, {**report, 'reason': 'empty-foreground'}
    center, extent = initial
    target_center, target_extent = target
    # Least-squares uniform scale for the two bounding-box dimensions.
    scale = float(np.dot(extent, target_extent) / np.dot(extent, extent))
    displacement = target_center - center
    report['registration'] = {'scale': scale, 'centerShiftYX': displacement.tolist()}
    if np.max(np.abs(displacement) / [h, w]) > max_shift or abs(scale - 1) > max_scale_change:
        return None, {**report, 'reason': 'transform-limit'}
    if scale != 1 or np.any(displacement):
        offset = center - target_center / scale
        aligned = np.stack([ndimage.affine_transform(channel, np.eye(2) / scale, offset=offset, output_shape=(h, w), order=1, mode='constant', cval=0., prefilter=False) for channel in np.moveaxis(moving, -1, 0)], axis=-1)
    else:
        aligned = moving
    after = scores(aligned[..., 3] >= .5, fixed)
    report['after'] = after
    if after['iou'] < min_iou:
        return None, {**report, 'reason': 'silhouette-iou'}
    # Store straight linear RGB and the admission weight. Baker coverage is applied only at compose time.
    np.divide(aligned[..., :3], aligned[..., 3:4], out=aligned[..., :3], where=aligned[..., 3:4] > 0)
    return aligned, {**report, 'status': 'accepted', 'reason': 'silhouette-passed'}


def align(faces, guide_id, min_iou, max_shift, max_scale_change):
    directory = store.asset_dir(guide_id)
    manifest = store.validate(directory, guide_id, check_bytes=False)
    aligned = {}
    for face_id, source in faces.items():
        if face_id not in SLUGS:
            raise ValueError('Unknown face: ' + face_id)
        guide_path = store.contained(directory, manifest['faces'][face_id]['guide'])
        canvas = source.get('provenance', {}).get('canvas')
        if canvas is not None and canvas.get('faceId') != face_id:
            raise ValueError('Generation canvas belongs to a different face.')
        measurement = store.key('face-measure-v2', source['sha256'], face_id, store.file_hash(guide_path), canvas)
        def measure(stage):
            data = np.load(Path(source['path']) / 'source.npy', allow_pickle=False)
            original_size = [data.shape[1], data.shape[0]]
            if canvas is not None:
                if data.shape[0] != data.shape[1]:
                    report = {'status': 'fallback', 'reason': 'generation-canvas-aspect', 'originalSize': original_size, 'guideCanvas': canvas}
                    (stage / 'report.json').write_text(json.dumps(report), encoding='utf-8')
                    return
                from .canvas import undo_guide_canvas
                data = undo_guide_canvas(data, canvas)
            with Image.open(guide_path) as guide_png:
                if 'A' not in guide_png.getbands():
                    raise ValueError('Face guide must include capture coverage as alpha.')
                guide = np.asarray(guide_png.getchannel('A'), dtype=np.float32) / 255
            # Measure once up to the widest limits offered by the node widgets.
            result, report = align_one(data, guide, 0., 1., 1.)
            report['originalSize'] = original_size
            report['guideCanvas'] = canvas
            if result is not None:
                np.save(stage / 'aligned.npy', result, allow_pickle=False)
            (stage / 'report.json').write_text(json.dumps(report, allow_nan=False), encoding='utf-8')
        measured = cache_dir('measure', measurement, measure)
        digest = store.key('admission-v1', measurement, min_iou, max_shift, max_scale_change)
        def decide(stage):
            report = json.loads((measured / 'report.json').read_bytes())
            report['thresholds'] = {'minIoU': min_iou, 'maxShiftFraction': max_shift, 'maxScaleChange': max_scale_change}
            if report['status'] == 'accepted':
                reg = report['registration']
                w, h = report['tileSize']
                dy, dx = reg['centerShiftYX']
                if max(abs(dx)/w, abs(dy)/h) > max_shift or abs(reg['scale'] - 1) > max_scale_change:
                    report.update(status='fallback', reason='transform-limit')
                elif report['after']['iou'] < min_iou:
                    report.update(status='fallback', reason='silhouette-iou')
            (stage / 'report.json').write_text(json.dumps(report, allow_nan=False), encoding='utf-8')
        output = cache_dir('align', digest, decide)
        aligned[face_id] = {'key': digest, 'source': source, 'directory': str(output), 'measured': str(measured)}
    return {'baseline': guide_id, 'faces': aligned}


def tile_view(atlas, face_id):
    r = atlas.shape[1] // 3
    idx = store.FACES.index(face_id)
    row, col = divmod(idx, 3)
    return atlas[row*r:(row+1)*r, col*r:(col+1)*r]


def replace_tile(atlas, face_id, candidate):
    tile = tile_view(atlas, face_id)
    candidate = candidate[::-1]  # top-origin PNG to increasing shader V
    w = candidate[..., 3:4]
    blend = tile[..., :3].astype(np.float32) * (1 - w) + candidate[..., :3] * tile[..., 3:4].astype(np.float32) * w
    changed = (w[..., 0] > 0) & (tile[..., 3] > 0)
    tile[..., :3][changed] = blend[changed].astype('<f2')


def compose(baseline, aligned):
    if aligned['baseline'] != baseline:
        raise ValueError('The guides and bake must use the same proxy, resolution, and backend.')
    if not aligned['faces']:
        return baseline
    provenance = {fid: item['source'].get('provenance', {}) for fid, item in aligned['faces'].items()}
    models = {item.get('backend', 'manual_faces') for item in provenance.values()}
    mode = next(iter(models)) if len(models) == 1 else 'manual_faces'
    asset_id = store.key('compose-v3', multiview.VERSION, baseline, mode, provenance, {fid: item['key'] for fid, item in aligned['faces'].items()})
    source = store.asset_dir(baseline)
    def write(stage):
        shutil.copytree(source, stage, ignore=shutil.ignore_patterns('previews'))
        manifest = store.validate(stage, baseline, check_bytes=False)
        base = manifest['baseColor']
        shape = (base['height'], base['width'], 4)
        backup = np.memmap(store.contained(stage, base['path']), dtype='<f2', mode='r', shape=shape)
        current = manifest['textures']['colorMap']
        atlas = np.fromfile(store.contained(stage, current['path']), dtype='<f2').reshape(shape)
        prior_qa = manifest.get('qa')
        reports = json.loads(store.contained(stage, prior_qa).read_bytes()).get('faces', {}) if prior_qa else {}
        for face_id in store.FACES:
            item = aligned['faces'].get(face_id)
            if item is None:
                reports.setdefault(face_id, {'status': 'baseline', 'reason': 'not-supplied'})
                continue
            folder = Path(item['directory'])
            report = json.loads((folder / 'report.json').read_bytes())
            rel = f'views/{SLUGS[face_id]}'
            view = stage / rel
            if view.is_dir():
                shutil.rmtree(view)
            view.mkdir(parents=True, exist_ok=True)
            for name in ('source.png', 'source.npy'):
                shutil.copyfile(Path(item['source']['path']) / name, view / name)
            tile_view(atlas, face_id)[:] = tile_view(backup, face_id)
            if report['status'] == 'accepted':
                measured = Path(item['measured']) / 'aligned.npy'
                candidate = np.load(measured, allow_pickle=False)
                replace_tile(atlas, face_id, candidate)
                shutil.copyfile(measured, view / 'aligned.npy')
            reports[face_id] = report
            manifest['faces'][face_id]['edit'] = {'status': report['status'], 'reason': report['reason'], 'source': provenance[face_id].get('backend', 'manual_faces'), 'alignmentKey': item['key'], 'provenance': provenance[face_id], 'view': rel + '/source.png'}
            ref = item['source'].get('reference')
            if ref:
                if not store.ID.fullmatch(ref['sha256']):
                    raise ValueError('Invalid reference digest.')
                if manifest.get('reference', {}).get('sha256') == ref['sha256']:
                    reference_path = manifest['reference']['image']
                else:
                    ref_rel = 'reference/' + ref['sha256']
                    ref_dir = store.contained(stage, ref_rel)
                    ref_dir.mkdir(parents=True, exist_ok=True)
                    for filename in ('source.npy', 'source.png'):
                        if not (ref_dir / filename).exists():
                            shutil.copyfile(Path(ref['path']) / filename, ref_dir / filename)
                    reference_path = ref_rel + '/source.png'
                manifest['faces'][face_id]['edit']['reference'] = reference_path
        data_path = stage / 'maps' / 'color_edited.rgba16f.bin'
        atlas.tofile(data_path)
        manifest['textures']['colorMap'] = {**base, 'path': 'maps/' + data_path.name, 'sha256': store.file_hash(data_path)}
        active_modes = {entry['edit']['source'] for entry in manifest['faces'].values() if entry.get('edit', {}).get('status') == 'accepted'}
        effective_mode = next(iter(active_modes)) if len(active_modes) == 1 else mode if not active_modes else 'manual_faces'
        manifest.update(asset_id=asset_id, parentAsset=baseline, geometryAsset=manifest.get('geometryAsset', manifest.get('parentAsset', baseline)), mode=effective_mode, qa='qa/report.json')
        (stage / 'qa').mkdir(exist_ok=True)
        face_keys = {fid: manifest['faces'][fid].get('edit', {}).get('alignmentKey', current['sha256']) if report['status'] == 'accepted' else 'baseline'
                     for fid, report in reports.items()}
        cross_view = multiview.evaluate_pairs(stage, manifest, face_keys=face_keys, cache=cache_dir)
        (stage / 'qa/report.json').write_text(json.dumps({'faces': reports, 'crossView': cross_view}, indent=2, allow_nan=False), encoding='utf-8')
        (stage / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding='utf-8')
    return store.publish(asset_id, write)
