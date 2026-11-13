"""Refine first-hit depths against accepted RGB, deform the proxy, and rebake with three-impostor."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from . import faces, store
from .revision import DEPTH_ENCODING, RebakeProxyRevision, matrix


DEPTH_SOURCES = ['geometry', 'rgb_guided']


def bound_paired_depth(delta, front, opposite, face_id):
    # Opposite cameras mirror X for the X/Z views, and Y for the Y views.
    back = np.flip(opposite, axis=0 if face_id[1] == 'Y' else 1)
    separation = np.maximum(0., 1. - front - back)
    limit = np.where(np.isfinite(separation), .45 * separation, 0.)
    return np.clip(delta, -limit, limit)


def vertex_displacements(points, views, bounds, resolution):
    """Project visible vertices into the same orthographic cameras used for capture."""
    low, high = np.array(bounds['min']), np.array(bounds['max'])
    span = high - low
    position = np.asarray(points, dtype=np.float64)
    displacement = np.zeros_like(position)
    weights = np.zeros_like(position)
    centered = (position - .5 * (low + high)) / span
    for clip, side, original, change in views:
        projected = position @ clip[:2, :3].T + clip[:2, 3]
        x = (projected[:, 0] + 1) * resolution / 2 - .5
        y = (1 - projected[:, 1]) * resolution / 2 - .5
        coords = np.stack((y, x))
        z = ndimage.map_coordinates(original, coords, order=0, mode='constant', cval=np.nan)
        offset = ndimage.map_coordinates(change, coords, order=1, mode='constant', cval=0.)
        axial = .5 - centered @ side
        thickness = abs(side @ span)
        pitch = np.max(span[side == 0]) / resolution
        # Other layers along the same ray must not move with the visible surface.
        tolerance = .75 * pitch / thickness + .001
        weight = np.where(np.isfinite(z), np.maximum(0., 1. - np.abs(axial - z) / tolerance), 0.)
        displacement -= side * (offset * weight * thickness)[:, None]
        weights += np.abs(side) * weight[:, None]
    displacement /= np.maximum(1., weights)
    result = np.clip(position + displacement, low, high) - position
    # Preserve the original AABB and projection framing on each axis.
    extrema = np.isclose(position, low, rtol=0, atol=1e-7) | np.isclose(position, high, rtol=0, atol=1e-7)
    result[extrema] = 0.
    return result


def refine_box_depth(asset, backend='auto'):
    from . import depth_mesh, depth_refine
    from .face_material import accepted_view
    directory = store.asset_dir(asset)
    original = store.validate(directory, asset)
    supplied = {fid: entry['edit'] for fid, entry in original['faces'].items() if entry.get('edit')}
    accepted = {fid: edit for fid, edit in supplied.items() if edit.get('status') == 'accepted'}
    if not accepted:
        raise ValueError('RGB-guided depth requires an accepted face image.')
    r = original['resolution']
    # The bakeBox includes the baker margin; the tight bounds anchor mesh extrema.
    def build(stage):
        views = []
        for fid in accepted:
            store.check_interrupted()
            entry = original['faces'][fid]
            if entry.get('depthEncoding') != DEPTH_ENCODING or entry.get('imageRowOrder') != 'top-to-bottom':
                raise ValueError(f'{fid} has no supported geometry depth capture.')
            z = np.fromfile(store.contained(directory, entry['depth']), dtype='<f2').reshape(r, r).astype('f4')
            rgba, _ = accepted_view(directory, original, fid)
            linear = np.clip(rgba[..., :3], 0, 1)
            rgb = np.where(linear <= .0031308, linear * 12.92, 1.055 * linear ** (1 / 2.4) - .055)
            with Image.open(store.contained(directory, entry['guide'])) as guide:
                coverage = np.asarray(guide.getchannel('A'), dtype='f4') / 255
            refined = depth_refine.guided_depth(z, rgb, np.minimum(coverage, rgba[..., 3]))
            delta = np.zeros_like(z)
            valid = np.isfinite(z) & np.isfinite(refined)
            delta[valid] = refined[valid] - z[valid]
            other = original['faces'][('+' if fid[0] == '-' else '-') + fid[1]]
            if other.get('depthEncoding') != DEPTH_ENCODING:
                raise ValueError(f'{fid} requires a matching opposite depth capture.')
            opposite = np.fromfile(store.contained(directory, other['depth']), dtype='<f2').reshape(r, r).astype('f4')
            delta = bound_paired_depth(delta, z, opposite, fid)
            clip = matrix(entry['projectionMatrix']) @ matrix(entry['viewMatrix'])
            views.append((clip, np.asarray(entry['side']), z, delta))

        tight = original['tightBounds']
        def shift(points):
            offset = vertex_displacements(points, views, original['bakeBox'], r)
            for bound in ('min', 'max'):
                offset[np.isclose(points, tight[bound], atol=1e-7, rtol=0)] = 0.
            return np.clip(points + offset, tight['min'], tight['max']) - points

        data = store.contained(directory, original['proxy']).read_bytes()
        mesh, stats = depth_mesh.deform_glb(data, shift)
        path = stage / 'proxy.glb'
        path.write_bytes(mesh)
        digest = hashlib.sha256(mesh).hexdigest()
        metadata = {'glbSha256': digest,
                    'sourceSha256': original.get('provenance', {}).get('originalGlbSha256', original['proxySha256']),
                    'transform': original['transform'], 'inverseTransform': original['inverseTransform'],
                    'tightBounds': original['tightBounds'], 'refinement': stats}
        (stage / 'canonical.json').write_text(json.dumps(metadata, allow_nan=False))

    key = store.key('guided-depth-mesh-v1', original['proxySha256'], r, original['bakeBox'],
                    {fid: [edit['alignmentKey'], original['faces'][fid].get('depthSha256')]
                     for fid, edit in accepted.items()},
                    [store.file_hash(Path(f)) for f in (__file__, depth_refine.__file__, depth_mesh.__file__)])
    stage = faces.cache_dir('depth-refinement', key, build)
    meta = json.loads((stage / 'canonical.json').read_bytes())
    if meta['glbSha256'] == original['proxySha256']:
        return asset
    proxy = {'sha256': meta['glbSha256'], 'stageKey': key, 'path': str(stage / 'proxy.glb'),
             'metadata': str(stage / 'canonical.json'),
             'depthRefinement': {'method': 'rgb_guided', 'sourceAsset': asset, 'faces': list(accepted),
                                 'mesh': meta['refinement']},
             'rotationDegreesXYZ': original.get('provenance', {}).get('rotationDegreesXYZ', [0., 0., 0.])}
    ref = original.get('reference')
    if ref:
        folder = store.contained(directory, ref['image']).parent
        if (folder / 'source.npy').is_file():
            proxy['reference'] = {**ref, 'path': str(folder)}
            if ref.get('camera'):
                proxy['camera'] = ref['camera']
                proxy['croppedImage'] = str(store.contained(directory, ref['camera']['image']))
    if original.get('provenance', {}).get('meshGeneration'):
        proxy['provenance'] = original['provenance']['meshGeneration']
    revised, _ = RebakeProxyRevision().revise(asset, proxy, backend)
    qa = json.loads(store.contained(directory, original['qa']).read_bytes())['faces']
    items = {}
    # Repeat the original admission test, face by face. Old material estimates belonged to the old geometry.
    for fid, edit in supplied.items():
        folder = store.contained(directory, edit['view']).parent
        source = faces.save_view(np.load(folder / 'source.npy', allow_pickle=False))
        source['provenance'] = {**edit.get('provenance', {}),
                                'depthRefinementSourceGuideSha256': original['faces'][fid].get('guideSha256')}
        if edit.get('reference'):
            source['reference'] = {'path': str(store.contained(directory, edit['reference']).parent),
                                   'sha256': source['provenance']['referenceSha256']}
        t = qa[fid]['thresholds']
        items.update(faces.align({fid: source}, revised, t['minIoU'], t['maxShiftFraction'], t['maxScaleChange'])['faces'])
    return faces.compose(revised, {'baseline': revised, 'faces': items})
