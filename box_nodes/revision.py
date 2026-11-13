"""Rebake corrected geometry and compare orthographic guides in GLB source coordinates."""
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from . import store


DEPTH_ENCODING = 'linear-axis-0-near-1-far;nan-background'


def matrix(values):
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (16,) or not np.isfinite(result).all():
        raise ValueError('Missing or invalid camera matrix.')
    return result.reshape(4, 4, order='F')


class Guide:
    def __init__(self, root, manifest, face_id):
        face = manifest['faces'][face_id]
        if face.get('imageRowOrder') != 'top-to-bottom':
            raise ValueError('Unsupported or missing guide row order.')
        transform = matrix(manifest['transform'])
        projection = matrix(face.get('projectionMatrix'))
        self.clip = projection @ matrix(face.get('viewMatrix')) @ transform
        if not np.allclose(self.clip[3], [0, 0, 0, 1], rtol=0, atol=1e-8):
            raise ValueError('The guide has no affine orthographic projection.')
        self.source = np.linalg.inv(self.clip)
        side = np.asarray(face['side'], dtype=np.float64)
        bounds = manifest['bakeBox']
        low, high = np.asarray(bounds['min']), np.asarray(bounds['max'])
        thickness = np.abs(side) @ (high - low)
        normal = side @ transform[:3, :3]
        length = np.linalg.norm(normal)
        if not (thickness > 0 and length > 0):
            raise ValueError('Invalid depth axis or capture bounds.')
        self.normal = normal / length
        # Encoded depth d maps to dot(sourcePoint, outward unit normal).
        self.depth_origin = (side @ ((low + high) / 2 - transform[:3, 3]) + thickness / 2) / length
        self.depth_span = thickness / length
        with Image.open(store.contained(root, face['guide'])) as image:
            if 'A' not in image.getbands() or image.size != (manifest['resolution'],) * 2:
                raise ValueError('Invalid guide coverage or size.')
            self.mask = np.asarray(image.getchannel('A')) >= 128
        self.depth = None
        if face.get('depth') and face.get('depthEncoding') == DEPTH_ENCODING:
            self.depth = np.memmap(store.contained(root, face['depth']), '<f2', mode='r', shape=self.mask.shape)


def compare_grid(query, target, warp):
    """Visit query pixel centers in strips. Ignore out-of-capture pixels when scoring shared coverage."""
    h, w = query.mask.shape
    th, tw = target.mask.shape
    xx = (np.arange(w)[None, :] + .5) * (2 / w) - 1
    shared = intersection = union = query_only = target_only = outside_foreground = 0
    depth_count = 0
    depth_sum = depth_max = 0.
    for y in range(0, h, 64):
        store.check_interrupted()
        stop = min(y + 64, h)
        yy = 1 - (np.arange(y, stop)[:, None] + .5) * (2 / h)
        u = (warp[0, 0] * xx + warp[0, 1] * yy + warp[0, 3] + 1) * (tw / 2)
        v = (1 - warp[1, 0] * xx - warp[1, 1] * yy - warp[1, 3]) * (th / 2)
        inside = (u >= 0) & (v >= 0) & (u < tw) & (v < th)
        col = np.clip(np.floor(u).astype(int), 0, tw - 1)
        row = np.clip(np.floor(v).astype(int), 0, th - 1)
        qmask = query.mask[y:stop]
        tmask = target.mask[row, col]
        both = inside & qmask & tmask
        shared += int(inside.sum())
        intersection += int(both.sum())
        union += int((inside & (qmask | tmask)).sum())
        query_only += int((inside & qmask & ~tmask).sum())
        target_only += int((inside & tmask & ~qmask).sum())
        outside_foreground += int((~inside & qmask).sum())
        if query.depth is not None and target.depth is not None:
            q = query.depth[y:stop][both].astype(np.float64)
            t = target.depth[row[both], col[both]].astype(np.float64)
            finite = np.isfinite(q) & np.isfinite(t)
            if finite.any():
                delta = np.abs((query.depth_origin - target.depth_origin) -
                               q[finite] * query.depth_span + t[finite] * target.depth_span)
                depth_count += len(delta)
                depth_sum += float(delta.sum())
                depth_max = max(depth_max, float(delta.max()))
    return {'sharedFramePixels': shared, 'foregroundOutsideOtherFrame': outside_foreground,
            'queryOnlyOnSharedFrame': query_only, 'otherOnlyOnSharedFrame': target_only,
            'intersectionPixels': intersection, 'unionPixels': union,
            'sharedFrameIoU': intersection / union if union else None,
            'depthSamples': depth_count, 'meanAbsDepthSourceUnits': depth_sum / depth_count if depth_count else None,
            'maxAbsDepthSourceUnits': depth_max if depth_count else None}


def compare_face(old_root, old, new_root, new, face_id):
    try:
        a, b = Guide(old_root, old, face_id), Guide(new_root, new, face_id)
        forward, backward = b.clip @ a.source, a.clip @ b.source
        if (not np.allclose(a.normal, b.normal, atol=1e-7, rtol=0) or
                any(not np.allclose(m[:2, 2], 0, atol=1e-7, rtol=0) for m in (forward, backward))):
            return {'status': 'uncomparable', 'reason': 'different-source-view-directions'}
    except (ValueError, KeyError, np.linalg.LinAlgError) as exc:
        return {'status': 'uncomparable', 'reason': str(exc)}
    return {'status': 'compared', 'sampling': 'nearest-pixel-center',
            'originalToRevised': compare_grid(a, b, forward),
            'revisedToOriginal': compare_grid(b, a, backward)}


class RebakeProxyRevision:
    CATEGORY = 'Box Impostor'
    FUNCTION = 'revise'
    RETURN_TYPES = ('BOX_IMPOSTOR', 'STRING')
    RETURN_NAMES = ('revised_asset', 'comparison_json')
    DESCRIPTION = ('Rebake a corrected proxy in the original GLB source coordinates. Rebuild the six guides '
                   'and all Baker maps. Compare parallel camera rays in source units.')

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'asset': ('BOX_IMPOSTOR',), 'corrected_proxy': ('BOX_PROXY',),
                             'backend': (['auto', 'webgpu', 'webgl2'],)}}

    def revise(self, asset, corrected_proxy, backend='auto'):
        if backend not in ('auto', 'webgpu', 'webgl2'):
            raise ValueError('Unknown Box backend.')
        old_root = store.asset_dir(asset)
        old = store.validate(old_root, asset)
        new_hash = store.file_hash(Path(corrected_proxy['path']))
        if new_hash != corrected_proxy['sha256']:
            raise ValueError('Corrected proxy file digest mismatch.')
        if old['proxySha256'] == new_hash:
            return asset, json.dumps({'status': 'unchanged-proxy', 'assetId': asset, 'proxySha256': new_hash})
        # A fresh GLB bake inherits the geometry bake selection, not channels inferred from RGB.
        original_maps = old.get('provenance', {}).get('requestedMaterialMaps',
                        old.get('faceMaterialBase', {}).get('maps', old['maps']))
        baked_id = store.baseline(corrected_proxy, old['resolution'], backend, tuple(original_maps))
        baked_root = store.asset_dir(baked_id)
        revision_id = store.key('proxy-revision-v1', asset, old['proxySha256'], baked_id, new_hash)

        def write(stage):
            new = store.validate(baked_root, baked_id)
            if new['proxySha256'] != new_hash or any(f.get('edit') for f in new['faces'].values()):
                raise ValueError('Revised geometry requires a clean bake of the corrected proxy.')
            shutil.copytree(baked_root, stage, ignore=shutil.ignore_patterns('previews'))
            report = {'status': 'rebaked', 'sourceCoordinates': 'assumed-shared-input-glb',
                      'originalAsset': asset, 'revisedAsset': revision_id, 'bakedGeometryAsset': baked_id,
                      'originalProxySha256': old['proxySha256'], 'correctedProxySha256': new_hash,
                      'guideComparisons': {f: compare_face(old_root, old, stage, new, f) for f in store.FACES}}
            new.update(asset_id=revision_id, parentAsset=asset, geometryAsset=baked_id, qa='qa/geometry-revision.json')
            new.setdefault('provenance', {})['geometryRevision'] = {
                k: v for k, v in report.items() if k in ('originalAsset', 'bakedGeometryAsset', 'originalProxySha256', 'correctedProxySha256')}
            if corrected_proxy.get('depthRefinement'):
                new['provenance']['depthRefinement'] = corrected_proxy['depthRefinement']
            (stage / 'qa').mkdir(exist_ok=True)
            (stage / new['qa']).write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
            (stage / 'manifest.json').write_text(json.dumps(new, indent=2, allow_nan=False), encoding='utf-8')

        store.publish(revision_id, write)
        report = (store.asset_dir(revision_id) / 'qa' / 'geometry-revision.json').read_text(encoding='utf-8')
        return revision_id, report
