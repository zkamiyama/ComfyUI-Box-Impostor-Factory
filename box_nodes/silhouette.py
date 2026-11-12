"""Inspect a supplied face against the original Baker camera before registration."""
import json
from pathlib import PurePosixPath

import numpy as np
from PIL import Image

from . import faces, store
from .canvas import undo_guide_canvas


def stored_source(directory, edit):
    """Native assets keep full-precision RGBA beside the view declared in the manifest."""
    if not edit or not edit.get('view'):
        return None
    view = edit['view']
    png = store.contained(directory, view)
    if PurePosixPath(view).name == 'source.png':
        raw = store.contained(directory, str(PurePosixPath(view).with_suffix('.npy')))
        if raw.is_file():
            return raw
    return png if png.is_file() else None


def source_coverage(path):
    if path.suffix == '.npy':
        raw = np.load(path, mmap_mode='r', allow_pickle=False)
        if raw.ndim != 3 or raw.shape[-1] != 4 or raw.dtype.kind != 'f':
            raise ValueError('Saved face samples must be a floating-point RGBA image.')
        alpha = raw[..., 3]
        if not np.all(np.isfinite(alpha)) or np.any(alpha < 0) or np.any(alpha > 1):
            raise ValueError('Invalid saved foreground coverage.')
        return alpha
    with Image.open(path) as image:
        if 'A' not in image.getbands():
            raise ValueError('Saved face image has no foreground alpha channel.')
        return np.asarray(image.getchannel('A'), dtype=np.float32) / 255


def measure(guide, coverage, edit, face_id, guide_hash):
    alpha = guide[..., 3].astype(np.float32) / 255
    base_rgb = np.rint(guide[..., :3] * alpha[..., None] + 48 * (1 - alpha[..., None])).astype(np.uint8)
    extra = np.zeros(alpha.shape, dtype=bool)
    missing = np.zeros_like(extra)
    report = {'status': 'unavailable', 'reason': None, 'metrics': None, 'coordinateSpace': None}
    if not edit:
        report.update(status='no-source', reason='no-face-supplied')
    elif coverage is None:
        report['reason'] = 'source-file-missing'
    else:
        provenance = edit.get('provenance', {})
        canvas = provenance.get('canvas')
        recorded = provenance.get('guideSha256')
        if recorded and recorded != guide_hash:
            report['reason'] = 'different-guide-digest'
        elif canvas is not None:
            try:
                if not isinstance(canvas, dict) or canvas.get('faceId') != face_id:
                    raise ValueError('Canvas belongs to another face or is malformed.')
                if canvas.get('imageRowOrder', 'top-to-bottom') != 'top-to-bottom':
                    raise ValueError('Unsupported camera canvas row order.')
                coverage = undo_guide_canvas(coverage, canvas)
                report['coordinateSpace'] = 'reversed-guide-letterbox'
            except (ValueError, KeyError, TypeError) as exc:
                report.update(reason='invalid-canvas', detail=str(exc))
        elif edit.get('source') in ('qwen_enhanced', 'h3_enhanced'):
            report['reason'] = 'generated-face-without-camera-canvas'
        else:
            report['coordinateSpace'] = 'supplied-full-camera-frame'
    if report['coordinateSpace']:
        candidate = faces.resize(coverage[..., None], alpha.shape)[..., 0] >= .5
        proxy = alpha >= .5
        extra = candidate & ~proxy
        missing = proxy & ~candidate
        metrics = faces.scores(candidate, proxy)
        if not (candidate | proxy).any():
            metrics['iou'] = None
        metrics.update(extraPixels=int(extra.sum()), missingPixels=int(missing.sum()),
                       candidatePixels=int(candidate.sum()), guidePixels=int(proxy.sum()))
        base_rgb[extra] = (255, 94, 32)
        base_rgb[missing] = (0, 190, 255)
        report.update(status='compared', metrics=metrics)
    return base_rgb, extra, missing, report


class InspectFaceSilhouette:
    CATEGORY = 'Box Impostor'
    FUNCTION = 'inspect'
    RETURN_TYPES = ('IMAGE', 'MASK', 'MASK', 'STRING')
    RETURN_NAMES = ('overlay', 'extra_foreground', 'missing_foreground', 'qa_json')
    DESCRIPTION = ('Compare a stored face with the Baker guide before registration. Orange marks candidate-only '
                   'pixels and cyan marks guide-only pixels. Read qa_json for unavailable comparisons.')

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'asset': ('BOX_IMPOSTOR',), 'face_id': (list(store.FACES),)}}

    def inspect(self, asset, face_id):
        from .nodes import tensor
        if face_id not in store.FACES:
            raise ValueError('Unknown Box face.')
        directory = store.asset_dir(asset)
        manifest = store.validate(directory, asset, check_bytes=False)
        face = manifest['faces'][face_id]
        if face.get('imageRowOrder', 'top-to-bottom') != 'top-to-bottom':
            raise ValueError('Unsupported guide row order.')
        guide_path = store.contained(directory, face['guide'])
        guide_hash = store.file_hash(guide_path)
        if face.get('guideSha256') and face['guideSha256'] != guide_hash:
            raise ValueError('Saved guide digest mismatch.')
        edit = face.get('edit')
        source = stored_source(directory, edit)
        source_hash = store.file_hash(source) if source else None
        digest = store.key('silhouette-before-registration-v1', face_id, guide_hash, edit, source_hash)

        def write(stage):
            with Image.open(guide_path) as image:
                if 'A' not in image.getbands():
                    raise ValueError('Baker guide must contain foreground coverage.')
                guide = np.asarray(image.convert('RGBA'))
            coverage = source_coverage(source) if source else None
            overlay, extra, missing, report = measure(guide, coverage, edit, face_id, guide_hash)
            Image.fromarray(overlay).save(stage / 'overlay.png')
            Image.fromarray(extra.astype(np.uint8) * 255).save(stage / 'extra.png')
            Image.fromarray(missing.astype(np.uint8) * 255).save(stage / 'missing.png')
            report.update(faceId=face_id, stage='before-registration', threshold=0.5,
                          guideSha256=guide_hash, guideSize=list(guide.shape[1::-1]),
                          sourceSha256=source_hash, sourceFormat=source.suffix if source else None,
                          sourceSize=list(coverage.shape[::-1]) if coverage is not None else None,
                          admission=edit.get('status') if edit else None,
                          admissionReason=edit.get('reason') if edit else None,
                          overlayColors={'extra': '#ff5e20', 'missing': '#00beff'},
                          files={'overlay': 'overlay.png', 'extra': 'extra.png', 'missing': 'missing.png'})
            (stage / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')

        stage = faces.cache_dir('silhouettes', digest, write)
        qa = json.loads((stage / 'report.json').read_bytes())
        qa.update(assetId=asset, reportPath=str(stage / 'report.json'))
        outputs = []
        for name in ('overlay', 'extra', 'missing'):
            with Image.open(stage / qa['files'][name]) as image:
                outputs.append(tensor((np.asarray(image, dtype=np.float32) / 255)[None]))
        return (*outputs, json.dumps(qa, allow_nan=False))
