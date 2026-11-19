"""Reproject frozen face samples between known parallel orthographic cameras.

This opt-in adapter never estimates registration from the new silhouette. It
retains source PNG/NPY bytes and records the additional aligned-image sampling.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from . import faces, store

VERSION = 'parallel-orthographic-frozen-rgb-v1'
DEPTH_ENCODING = 'linear-axis-0-near-1-far;nan-background'


def _matrix(values, label):
    matrix = np.asarray(values, dtype='f8')
    if matrix.size != 16 or not np.isfinite(matrix).all():
        raise ValueError('Invalid camera matrix: '+label)
    return matrix.reshape(4, 4).T  # Three.js column-major storage


def camera_pixel_affine(original, revised, source_resolution, target_resolution):
    """Return target top-origin pixel centers -> source pixel centers."""
    if source_resolution < 1 or target_resolution < 1:
        raise ValueError('Capture resolution must be positive.')
    for key in ('side', 'up', 'tile', 'imageRowOrder', 'depthEncoding'):
        if original.get(key) != revised.get(key):
            raise ValueError('Capture convention changed: '+key)
    if original.get('imageRowOrder') != 'top-to-bottom':
        raise ValueError('Only top-to-bottom source samples are supported.')
    if original.get('depthEncoding') != DEPTH_ENCODING:
        raise ValueError('Unsupported native depth convention.')
    cameras = []
    for label, face in (('source', original), ('target', revised)):
        projection = _matrix(face['projectionMatrix'], label+' projection')
        view = _matrix(face['viewMatrix'], label+' view')
        if not np.allclose(projection[3], [0, 0, 0, 1], rtol=0, atol=1e-12):
            raise ValueError('Camera must be orthographic: '+label)
        if not np.allclose(view[3], [0, 0, 0, 1], rtol=0, atol=1e-12):
            raise ValueError('Camera view must be affine: '+label)
        rotation = view[:3, :3]
        if not np.allclose(rotation @ rotation.T, np.eye(3), rtol=0, atol=1e-10):
            raise ValueError('Camera view axes must be orthonormal: '+label)
        cameras.append((projection, view))
    if not np.allclose(cameras[0][1][:3, :3], cameras[1][1][:3, :3],
                       rtol=0, atol=1e-10):
        raise ValueError('Camera direction changed; a depth-independent RGB warp is invalid.')
    clip = cameras[0][0] @ cameras[0][1] @ np.linalg.inv(cameras[1][0] @ cameras[1][1])
    if (not np.allclose(clip[3], [0, 0, 0, 1], rtol=0, atol=1e-10)
            or not np.allclose(clip[:2, 2], 0, rtol=0, atol=1e-10)):
        raise ValueError('Camera mapping depends on depth or perspective.')
    old_r, new_r = float(source_resolution), float(target_resolution)
    pixel_to_clip = np.array([[2/new_r, 0, 1/new_r-1],
                              [0, -2/new_r, 1-1/new_r], [0, 0, 1]])
    clip_to_pixel = np.array([[old_r/2, 0, (old_r-1)/2],
                              [0, -old_r/2, (old_r-1)/2], [0, 0, 1]])
    plane = clip[np.ix_([0, 1, 3], [0, 1, 3])]
    affine = clip_to_pixel @ plane @ pixel_to_clip
    if (not np.allclose(affine[[0, 1], [1, 0]], 0, rtol=0, atol=1e-9)
            or np.any(np.diag(affine)[:2] <= 0)):
        raise ValueError('Camera mapping must preserve the face axes and handedness.')
    return affine, {
        'targetToSourcePixelXY': affine.tolist(),
        'sourceToTargetPixelXY': np.linalg.inv(affine).tolist(),
        'sourceClipFromTargetClip': clip.tolist(),
        'depthIndependent': True, 'orthographic': True,
        'sourceResolution': source_resolution, 'targetResolution': target_resolution,
        'pixelCenters': 'integer indices; top-to-bottom rows',
    }


def reproject_samples(straight_linear_rgba, affine, target_resolution):
    """One bilinear pass in premultiplied linear RGBA; return straight RGBA."""
    data = np.asarray(straight_linear_rgba, dtype='f4')
    if (data.ndim != 3 or data.shape[-1] != 4 or data.shape[0] != data.shape[1]
            or not np.isfinite(data).all() or np.any(data < 0) or np.any(data > 1)):
        raise ValueError('Expected square finite straight linear RGBA in [0,1].')
    if data.shape[0] == target_resolution and np.allclose(affine, np.eye(3), rtol=0, atol=1e-10):
        return data.copy(), False
    premultiplied = data.copy()
    premultiplied[..., :3] *= premultiplied[..., 3:4]
    matrix_yx = affine[:2, :2][::-1, ::-1]
    offset_yx = affine[:2, 2][::-1]
    warped = np.stack([
        ndimage.affine_transform(channel, matrix_yx, offset_yx,
                                 output_shape=(target_resolution, target_resolution),
                                 order=1, mode='grid-constant', cval=0, prefilter=False)
        for channel in np.moveaxis(premultiplied, -1, 0)
    ], axis=-1)
    np.divide(warped[..., :3], warped[..., 3:4], out=warped[..., :3],
              where=warped[..., 3:4] > 0)
    warped[..., :3][warped[..., 3] == 0] = 0
    return np.clip(warped, 0, 1).astype('<f4'), True


def affine_limits(affine, source_resolution, target_resolution, thresholds):
    forward = np.linalg.inv(affine)
    scales = np.diag(forward)[:2] * source_resolution / target_resolution
    source_center = np.array([(source_resolution-1)/2]*2+[1.])
    target_center = np.array([(target_resolution-1)/2]*2)
    shifts = (forward @ source_center)[:2] - target_center
    fractions = shifts / target_resolution
    passed = (np.max(abs(fractions)) <= thresholds['maxShiftFraction'] + 1e-12
              and np.max(abs(scales-1)) <= thresholds['maxScaleChange'] + 1e-12)
    return {'passed': bool(passed), 'scaleXY': scales.tolist(),
            'centerShiftXY': shifts.tolist(), 'centerShiftFractionXY': fractions.tolist()}


def rebase_views(source_dir, original, revised_dir, revised, work_dir):
    """Build faces.compose inputs, admitting all six only with the original QA limits."""
    source_dir, revised_dir, work = Path(source_dir), Path(revised_dir), Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    for key in ('transform', 'inverseTransform', 'faceOrder', 'coordinateSystem'):
        if original.get(key) != revised.get(key):
            raise ValueError('Source and target must share the canonical world frame: '+key)
    if original.get('coordinateSystem') != 'right-handed-y-up':
        raise ValueError('Unsupported canonical coordinate system.')
    previous = json.loads(store.contained(source_dir, original['qa']).read_bytes())['faces']
    r_old, r_new = original['resolution'], revised['resolution']
    evidence = {
        'method': VERSION, 'sourceAsset': original['asset_id'], 'targetBaseline': revised['asset_id'],
        'sourceTightBounds': original['tightBounds'], 'targetTightBounds': revised['tightBounds'],
        'sourceBakeBox': original['bakeBox'], 'targetBakeBox': revised['bakeBox'],
        'sourceRGBPolicy': 'unchanged source PNG/NPY; one additional aligned-image interpolation',
        'registrationOptimized': False, 'faces': {},
    }
    items = {}
    for fid, old_info in original['faces'].items():
        edit = old_info.get('edit', {})
        old_qa = previous[fid]
        if edit.get('status') != 'accepted' or old_qa.get('status') != 'accepted':
            raise ValueError('Every frozen source view must already be accepted: '+fid)
        view = store.contained(source_dir, edit['view']).parent
        hashes = {name: store.file_hash(view/name) for name in ('source.png', 'source.npy', 'aligned.npy')}
        if hashes['aligned.npy'] != edit.get('alignedSha256'):
            raise ValueError('Frozen source aligned hash mismatch: '+fid)
        old = np.load(view/'aligned.npy', allow_pickle=False)
        if old.shape != (r_old, r_old, 4):
            raise ValueError('Frozen source aligned shape mismatch: '+fid)
        affine, mapping = camera_pixel_affine(old_info, revised['faces'][fid], r_old, r_new)
        candidate, resampled = reproject_samples(old, affine, r_new)
        thresholds = dict(old_qa['thresholds'])
        if set(thresholds) != {'minIoU', 'maxShiftFraction', 'maxScaleChange'}:
            raise ValueError('Original admission thresholds are incomplete: '+fid)
        transform_score = affine_limits(affine, r_old, r_new, thresholds)
        with Image.open(store.contained(revised_dir, revised['faces'][fid]['guide'])) as im:
            target_mask = np.asarray(im.getchannel('A')) >= 128
        if target_mask.shape != (r_new, r_new):
            raise ValueError('New guide dimensions do not match its capture: '+fid)
        candidate_mask = candidate[..., 3] >= .5
        score = faces.scores(candidate_mask, target_mask)
        passed = bool(target_mask.any() and candidate_mask.any()
                      and transform_score['passed'] and score['iou'] >= thresholds['minIoU'])
        status = 'accepted' if passed else 'fallback'
        report = {
            'status': status, 'reason': 'known-camera-affine-passed' if passed else 'known-camera-affine-qa',
            'before': old_qa['after'], 'after': score, 'thresholds': thresholds,
            'sourceSize': old_qa.get('sourceSize'), 'tileSize': [r_new, r_new],
            'sampling': VERSION, 'sourceRegistration': old_qa.get('registration'),
            'sourceAlignmentKey': edit['alignmentKey'], 'mapping': mapping,
            'cameraTransformLimits': transform_score,
            'registrationOptimized': False, 'alignedResampled': resampled,
            'interpolation': 'premultiplied linear bilinear; straight RGBA restored before compose',
            'newGeometryUncoveredPixels': int(np.count_nonzero(target_mask & ~candidate_mask)),
        }
        folder = work/faces.SLUGS[fid]
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder/'aligned.npy', candidate, allow_pickle=False)
        (folder/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        evidence['faces'][fid] = {
            **report, 'sourceFiles': hashes, 'alignedSha256': store.file_hash(folder/'aligned.npy'),
            'sameAlignedBytes': hashes['aligned.npy'] == store.file_hash(folder/'aligned.npy'),
        }
        item_source = {'path': str(view), 'provenance': {
            **edit.get('provenance', {}), 'cameraRebase': {
                'method': VERSION, 'sourceAsset': original['asset_id'],
                'sourceAlignmentKey': edit['alignmentKey'], 'mapping': mapping,
                'alignedResampled': resampled, 'registrationOptimized': False}}}
        if original.get('reference'):
            ref = original['reference']
            item_source['reference'] = {**ref, 'path': str(store.contained(source_dir, ref['image']).parent)}
        items[fid] = {
            'key': store.key(VERSION, original['asset_id'], revised['asset_id'], fid, hashes, store.file_hash(folder/'aligned.npy'), report),
            'source': item_source, 'directory': str(folder), 'measured': str(folder)}
    evidence['passed'] = all(v['status'] == 'accepted' for v in evidence['faces'].values())
    (work/'camera-rebase-proof.json').write_text(json.dumps(evidence, indent=2, allow_nan=False)+'\n')
    if not evidence['passed']:
        bad = [fid for fid, item in evidence['faces'].items() if item['status'] != 'accepted']
        raise ValueError('Known-camera affine QA failed for '+', '.join(bad)+'; no comparison published.')
    return {'baseline': revised['asset_id'], 'faces': items}, evidence
