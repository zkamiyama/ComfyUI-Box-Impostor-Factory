"""Estimate inputs and face-local material overlays for an accepted generated Box view."""
import hashlib
import json
import shutil

import numpy as np

from . import faces, multiview, store


SURFACE = ('roughness', 'metalness', 'ao')
PACKINGS = {'RM': (0, 1), 'RMA': (0, 1, 2), 'ORM': (1, 2, 0)}


def accepted_view(directory, manifest, face_id):
    if face_id not in store.FACES:
        raise ValueError('Unknown Box face: ' + str(face_id))
    edit = manifest['faces'][face_id].get('edit', {})
    if edit.get('status') != 'accepted':
        raise ValueError(f'{face_id} needs an accepted generated face before material estimation.')
    path = store.contained(directory, f'views/{faces.SLUGS[face_id]}/aligned.npy')
    if not path.is_file():
        raise ValueError(f'{face_id} has no archived aligned generated face.')
    digest = store.file_hash(path)
    if edit.get('alignedSha256', digest) != digest:
        raise ValueError(f'Aligned generated face digest mismatch for {face_id}.')
    data = np.load(path, allow_pickle=False, mmap_mode='r')
    r = manifest['resolution']
    if (data.shape != (r, r, 4) or data.dtype != np.float32 or not np.isfinite(data).all()
            or data.min() < -1e-6 or data.max() > 1.000001):
        raise ValueError(f'{face_id} has invalid aligned face samples.')
    return data, digest


class BILoadGeneratedFace:
    CATEGORY = 'Box Impostor'
    FUNCTION = 'load'
    RETURN_TYPES = ('IMAGE', 'MASK')
    RETURN_NAMES = ('generated_face_srgb', 'foreground_coverage')
    DESCRIPTION = 'Read the accepted appearance before albedo estimation, registered to the Box tile. Mask is 1 on foreground.'

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'asset': ('BOX_IMPOSTOR',), 'face_id': (list(store.FACES),)}}

    def load(self, asset, face_id):
        import torch
        directory = store.asset_dir(asset)
        manifest = store.validate(directory, asset, check_bytes=False)
        rgba, _ = accepted_view(directory, manifest, face_id)
        rgb = np.clip(rgba[..., :3], 0, 1)
        srgb = np.where(rgb <= .0031308, rgb * 12.92, 1.055 * rgb ** (1 / 2.4) - .055)
        alpha = np.clip(rgba[..., 3:4], 0, 1)
        image = srgb * alpha + .5 * (1 - alpha)
        return torch.from_numpy(image[None]), torch.from_numpy(alpha[None, ..., 0].copy())


def _image(value, name):
    data = faces.array(value)
    if (data.ndim != 4 or data.shape[0] != 1 or data.shape[-1] != 3 or not data.shape[1]
            or data.shape[1] != data.shape[2] or not np.isfinite(data).all() or data.min() < 0 or data.max() > 1):
        raise ValueError(f'{name} must be one registered square RGB image with finite values in [0,1].')
    return np.ascontiguousarray(data[0], dtype='<f4')


def _at_tile_resolution(data, weight):
    if data.shape[:2] == weight.shape:
        return data
    # Interpolate valid samples only so a model's background cannot stain the silhouette.
    source_weight = faces.resize(weight[..., None], data.shape[:2])
    weighted = faces.resize(data * source_weight, weight.shape)
    area = faces.resize(source_weight, weight.shape)
    return np.divide(weighted, area, out=np.zeros_like(weighted), where=area > 0)


def _atlas(directory, descriptor, mode='r'):
    dtype = {'float16': '<f2', 'uint8': 'u1'}[descriptor['dtype']]
    return np.memmap(store.contained(directory, descriptor['path']), mode=mode, dtype=dtype,
                     shape=(descriptor['height'], descriptor['width'], 4))


def _writable(directory, manifest, name, base, filename):
    """Copy the current atlas when first derived. Later edits mutate only a private staging copy."""
    current = manifest['textures'].get(name)
    relative = 'maps/' + filename
    if base and relative == base['path']:
        relative = 'maps/derived_' + filename
    target = store.contained(directory, relative)
    if current:
        if relative != current['path']:
            shutil.copyfile(store.contained(directory, current['path']), target)
        result = {**current, 'path': relative}
    else:
        color = manifest['textures']['colorMap']
        result = {**color, 'path': relative, 'dtype': 'uint8', 'colorSpace': 'data', 'alpha': 'coverage',
                  'byteLength': color['byteLength'] // 2, 'rowBytes': color['rowBytes'] // 2}
        with target.open('wb') as stream:
            stream.truncate(result['byteLength'])
    return _atlas(directory, result, mode='r+'), result


def _baseline_tile(base, face_id, coverage, base_maps):
    encoded = np.rint(np.clip(coverage, 0, 1) * 255).astype('u1')
    if base is None:
        result = np.zeros((*encoded.shape, 4), dtype='u1')
        result[..., 3] = encoded
    else:
        result = faces.tile_view(base, face_id).copy()
    for index, channel in enumerate(SURFACE):
        if channel not in base_maps:
            result[..., index] = 0 if channel == 'metalness' else encoded
    return result


def _surface_overlay(directory, manifest, changed):
    """Restore newly edited tiles from the baked surface and activate the required global shader slots."""
    textures = manifest['textures']
    previous = set(manifest['maps'])
    base_info = manifest.setdefault('faceMaterialBase', {'maps': manifest['maps'][:], 'materialMap': textures.get('materialMap')})
    base_desc, base_maps = base_info['materialMap'], base_info['maps']
    required = set(base_maps)
    for face in manifest['faces'].values():
        required.update(face.get('materialEdit', {}).get('surface', {}).get('channels', ()))
    if not any('surface' in face.get('materialEdit', {}) for face in manifest['faces'].values()):
        manifest['maps'] = base_maps
        current = textures.pop('materialMap', None)
        if base_desc:
            textures['materialMap'] = base_desc
        if current and current != base_desc:
            store.contained(directory, current['path']).unlink()
        del manifest['faceMaterialBase']
        return

    color = _atlas(directory, manifest['baseColor'])
    base = _atlas(directory, base_desc) if base_desc else None
    fresh = 'materialMap' not in textures
    out, desc = _writable(directory, manifest, 'materialMap', base_desc, 'face_surface.rgba8.bin')
    for face_id in store.FACES:
        coverage = faces.tile_view(color, face_id)[..., 3].astype('f4')
        tile = faces.tile_view(out, face_id)
        original = _baseline_tile(base, face_id, coverage, base_maps)
        if fresh or face_id in changed:
            tile[:] = original
        else:
            for index, channel in enumerate(SURFACE):
                if (channel in previous) != (channel in required):
                    tile[..., index] = original[..., index]
        patch = changed.get(face_id)
        if patch is None:
            continue
        packed, indices, weight = patch
        weight = weight[::-1]
        for index, source_channel in enumerate(indices):
            if index == 1 and manifest['faces'][face_id]['materialEdit']['surface'].get('binaryMetalness'):
                # Select one class at the admitted silhouette. The atlas stores
                # coverage-premultiplied data, so a metal texel's G equals A.
                selected = weight >= 0.5
                metal = packed[::-1, :, source_channel] >= 0.5
                tile[..., index][selected] = np.where(metal[selected], tile[..., 3][selected], 0)
                continue
            target = packed[::-1, :, source_channel] * coverage * 255
            tile[..., index] = np.rint(original[..., index] * (1 - weight) + target * weight).clip(0, 255).astype('u1')
    out.flush()
    textures['materialMap'] = {**desc, 'sha256': store.file_hash(store.contained(directory, desc['path']))}
    manifest['maps'] = [p for p in (*SURFACE, 'emissive') if p in required]


def clear_replaced_materials(directory, manifest, face_ids):
    changed = {}
    for face_id in face_ids:
        old = manifest['faces'][face_id].pop('materialEdit', {})
        if 'surface' in old:
            changed[face_id] = None
    if changed:
        _surface_overlay(directory, manifest, changed)


def qa_face_keys(manifest):
    # Material can change the measured colour or the specular exclusion mask. Global slot activation
    # matters even on faces whose pixels were not selected, e.g. on the first inferred RM map.
    specular = {'metalness', 'roughness'} <= set(manifest['maps'])
    return {fid: store.key('appearance-qa-v1', entry.get('edit', {}).get('alignmentKey'),
                           entry.get('materialEdit'), specular) for fid, entry in manifest['faces'].items()}


def apply_face_materials(asset, estimates):
    """Apply a face-keyed batch in one content-addressed publication; maps use registered Box pixels."""
    directory = store.asset_dir(asset)
    manifest = store.validate(directory, asset)
    requests, records = {}, {}
    for face_id, request in estimates.items():
        candidate, aligned_hash = accepted_view(directory, manifest, face_id)
        packing = request.get('packing', 'RM')
        if packing not in PACKINGS:
            raise ValueError('Unknown material packing: ' + str(packing))
        images = {name: _image(request[name], name) for name in ('albedo', 'rma') if request.get(name) is not None}
        if not images:
            raise ValueError(f'Supply an albedo or packed material image for {face_id}.')
        provider = request.get('provider', 'manual')
        recipe = request.get('recipe', {})
        if not isinstance(provider, str) or not isinstance(recipe, dict):
            raise ValueError('Material provider must be text and recipe must be a JSON object.')
        details = {'provider': provider, 'recipe': recipe, 'alignedSha256': aligned_hash,
                   'alignmentKey': manifest['faces'][face_id]['edit']['alignmentKey']}
        record = dict(manifest['faces'][face_id].get('materialEdit', {}))
        if 'albedo' in images:
            record['albedo'] = {**details, 'inputSha256': hashlib.sha256(images['albedo']).hexdigest(), 'encoding': 'srgb'}
        if 'rma' in images:
            record['surface'] = {**details, 'inputSha256': hashlib.sha256(images['rma']).hexdigest(),
                                 'packing': packing, 'channels': list(SURFACE[:len(PACKINGS[packing])])}
            if request.get('binaryMetalness'):
                record['surface']['binaryMetalness'] = True
        records[face_id] = record
        requests[face_id] = (images, PACKINGS[packing], np.clip(candidate[..., 3], 0, 1))
    if all(manifest['faces'][fid].get('materialEdit') == rec for fid, rec in records.items()):
        return asset
    asset_id = store.key('face-material-v1', multiview.VERSION, asset, records)

    def write(stage):
        shutil.copytree(directory, stage, ignore=shutil.ignore_patterns('previews'))
        result = store.validate(stage, asset, check_bytes=False)
        surface = {}
        color = color_desc = baseline = None
        for fid, (images, indices, weight) in requests.items():
            store.check_interrupted()
            result['faces'][fid]['materialEdit'] = records[fid]
            if 'albedo' in images:
                if color is None:
                    baseline = _atlas(stage, result['baseColor'])
                    color, color_desc = _writable(stage, result, 'colorMap', result['baseColor'], 'face_albedo.rgba16f.bin')
                rgb = images['albedo']
                linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
                faces.tile_view(color, fid)[:] = faces.tile_view(baseline, fid)
                faces.replace_tile(color, fid, np.dstack((_at_tile_resolution(linear, weight), weight)))
            if 'rma' in images:
                surface[fid] = (_at_tile_resolution(images['rma'], weight), indices, weight)
        if color is not None:
            color.flush()
            result['textures']['colorMap'] = {**color_desc, 'sha256': store.file_hash(store.contained(stage, color_desc['path']))}
        if surface:
            _surface_overlay(stage, result, surface)
        result.update(asset_id=asset_id, parentAsset=asset)
        qa = store.contained(stage, result.get('qa', 'qa/report.json'))
        report = json.loads(qa.read_bytes()) if qa.is_file() else {}
        report['crossView'] = multiview.evaluate_pairs(stage, result, face_keys=qa_face_keys(result), cache=faces.cache_dir)
        qa.parent.mkdir(parents=True, exist_ok=True)
        qa.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
        result['qa'] = qa.relative_to(stage).as_posix()
        (stage / 'manifest.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')

    return store.publish(asset_id, write)


class BIApplyFaceMaterial:
    CATEGORY = 'Box Impostor'
    FUNCTION = 'apply'
    RETURN_TYPES = ('BOX_IMPOSTOR',)
    DESCRIPTION = 'Apply aligned, unlit sRGB albedo and/or linear material channels. RM preserves baked ambient occlusion.'

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'asset': ('BOX_IMPOSTOR',), 'face_id': (list(store.FACES),), 'packing': (list(PACKINGS),)},
                'optional': {'albedo': ('IMAGE',), 'rma': ('IMAGE',), 'provider': ('STRING', {'default': 'manual'})}}

    def apply(self, asset, face_id, packing='RM', albedo=None, rma=None, provider='manual'):
        return (apply_face_materials(asset, {face_id: dict(packing=packing, albedo=albedo, rma=rma, provider=provider)}),)


class BIApplyFaceScalarMaps:
    CATEGORY = 'Box Impostor'
    FUNCTION = 'apply'
    RETURN_TYPES = ('BOX_IMPOSTOR',)
    DESCRIPTION = ('Apply separate linear roughness and metallic images to one registered face. '
                   'Metalness is thresholded before filtering. The baked ambient occlusion is retained.')

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'asset': ('BOX_IMPOSTOR',), 'face_id': (list(store.FACES),),
                             'roughness': ('IMAGE',), 'metallic': ('IMAGE',)},
                'optional': {'albedo': ('IMAGE',), 'provider': ('STRING', {'default': 'manual'}),
                             'metal_threshold': ('FLOAT', {'default': .5, 'min': 0., 'max': 1., 'step': .01})}}

    def apply(self, asset, face_id, roughness, metallic, albedo=None, provider='manual', metal_threshold=.5):
        if not 0 <= metal_threshold <= 1:
            raise ValueError('Metalness threshold must be in [0, 1].')
        rough = _image(roughness, 'roughness').mean(axis=-1)
        metal = _image(metallic, 'metallic').mean(axis=-1)
        if rough.shape != metal.shape:
            raise ValueError('Roughness and metallic maps must use the same image grid.')
        rm = np.zeros((1, *rough.shape, 3), dtype=np.float32)
        rm[0, ..., 0] = rough
        rm[0, ..., 1] = metal >= metal_threshold
        request = {'packing': 'RM', 'rma': rm, 'binaryMetalness': True,
                   'albedo': albedo, 'provider': provider, 'recipe': {'metalnessThreshold': metal_threshold}}
        return (apply_face_materials(asset, {face_id: request}),)
