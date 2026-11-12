"""Persist native TripoSplat decodes without re-sampling an unchanged reference."""
import json
from pathlib import Path

import numpy as np

from . import faces, generation, store

FIELDS = ('positions', 'scales', 'rotations', 'opacities', 'sh')
LOADERS = {'CLIPVisionLoader': ('clip_vision', 'clip_name'), 'VAELoader': ('vae', 'vae_name'),
           'UNETLoader': ('diffusion_models', 'unet_name')}
OPERATORS = set(LOADERS) | {'TripoSplatConditioning', 'KSampler', 'VAEDecodeTripoSplat',
                            'TripoSplatPreprocessImage', 'PrimitiveInt', 'BIPrepareReference'}


def decoder_request(reference, prompt, unique_id):
    """Only skip inputs for deterministic native graphs with provable reference/weight identities."""
    pins = (prompt or {}).get(str(unique_id), {}).get('inputs', {})
    output = pins.get('splat')
    ref = pins.get('reference')
    if not isinstance(output, list) or not isinstance(ref, list):
        return None
    if ref[1] != 2 or prompt.get(str(output[0]), {}).get('class_type') != 'VAEDecodeTripoSplat':
        return None
    pending, seen = [str(output[0])], set()
    references, files = set(), set()
    while pending:
        node_id = pending.pop()
        if node_id in seen:
            continue
        seen.add(node_id)
        n = prompt[node_id]
        kind = n['class_type']
        if kind not in OPERATORS:
            return None
        if kind == 'BIPrepareReference':
            references.add(node_id)
            continue
        if kind in LOADERS:
            folder, field = LOADERS[kind]
            name = n['inputs'].get(field)
            if not isinstance(name, str):
                return None
            files.add(folder + '/' + name)
        pending.extend(str(v[0]) for v in n['inputs'].values() if isinstance(v, list) and len(v) == 2)
    if references != {str(ref[0])}:
        return None
    graph, weights, settings = generation.lineage(prompt, unique_id, sockets=('splat',))
    if not files or files != set(weights):
        return None
    from comfy import model_management
    comfy_root = Path(model_management.__file__).resolve().parents[1]
    # Account for local edits to the decoder as well as official ComfyUI upgrades.
    sources = [comfy_root/'comfy_extras'/'nodes_triposplat.py', comfy_root/'comfy_version.py',
               *(comfy_root/'comfy'/'ldm'/'triposplat').glob('*.py')]
    code = {p.name: store.file_hash(p) for p in sources if p.is_file()}
    meta = {'graphSha256': graph, 'referenceSha256': reference['sha256'], 'weights': weights,
            'settings': settings, 'implementation': code}
    return store.key('decoded-triposplat-v1', meta), meta


def save_decode(stage, splat, metadata):
    if splat.positions.ndim != 3 or splat.positions.shape[0] != 1:
        raise ValueError('BI CachedTripoSplat supports one native decode per reference.')
    arrays = {name: np.ascontiguousarray(faces.array(getattr(splat, name)), dtype='<f4') for name in FIELDS}
    # TripoSplat's decoder emits a single dense (unpadded) float32 item.
    counts = getattr(splat, 'counts', None)
    if counts is not None and int(counts[0].item()) != arrays['positions'].shape[1]:
        raise ValueError('Native TripoSplat returned a padded batch.')
    n = arrays['positions'].shape[1]
    shapes = [(1, n, 3), (1, n, 3), (1, n, 4), (1, n, 1)]
    if n == 0 or any(arrays[k].shape != shape for k, shape in zip(FIELDS, shapes)):
        raise ValueError('Invalid decoded Gaussian attribute shapes.')
    sh = arrays['sh']
    if sh.ndim != 4 or sh.shape[:2] != (1, n) or sh.shape[2] < 1 or sh.shape[-1] != 3:
        raise ValueError('Invalid spherical harmonics in native decode.')
    if not all(np.isfinite(v).all() for v in arrays.values()):
        raise ValueError('The native decoder returned non-finite Gaussian attributes.')
    path = stage / 'splat.npz'
    np.savez(path, **arrays)
    (stage/'result.json').write_text(json.dumps({**metadata, 'sha256': store.file_hash(path), 'gaussians': n},
                                                indent=2, allow_nan=False), encoding='utf-8')


def load_decode(stage):
    path = stage / 'splat.npz'
    meta = json.loads((stage/'result.json').read_bytes())
    if store.file_hash(path) != meta['sha256']:
        raise ValueError('Cached TripoSplat tensor archive digest mismatch.')
    with np.load(path, allow_pickle=False) as archive:
        return [archive[name] for name in FIELDS]


def restore_splat(stage):
    from comfy_api.latest import Types
    import torch
    return Types.SPLAT(*(torch.from_numpy(array) for array in load_decode(stage)))


class CachedTripoSplat:
    CATEGORY = 'Box Impostor'
    FUNCTION = 'resolve'
    RETURN_TYPES = ('SPLAT',)
    DESCRIPTION = ('Cache the native decoder and skip its sampler when the reference, graph and weights match. '
                   'Other SPLAT producers pass through.')

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'reference': ('BOX_REFERENCE',), 'splat': ('SPLAT', {'lazy': True})},
                'hidden': {'prompt': 'PROMPT', 'unique_id': 'UNIQUE_ID'}}

    def check_lazy_status(self, reference, splat=None, prompt=None, unique_id=None):
        if splat is not None:
            return []
        req = decoder_request(reference, prompt, unique_id)
        if req and (store.root()/'.stages'/'triposplat'/req[0]/'result.json').is_file():
            return []
        return ['splat']

    def resolve(self, reference, splat=None, prompt=None, unique_id=None):
        req = decoder_request(reference, prompt, unique_id)
        if req is None:
            if splat is None:
                raise ValueError('Connect a SPLAT output to the cache.')
            return (splat,)

        def write(stage):
            if splat is None:
                raise ValueError('The cached decode disappeared; rerun with the native decoder connected.')
            save_decode(stage, splat, req[1])

        folder = faces.cache_dir('triposplat', req[0], write)
        return (restore_splat(folder),)
