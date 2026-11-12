"""Persistent pre-inference caching for the two local ComfyUI adapter graphs."""
import json
import shutil
from functools import wraps
from pathlib import Path

from . import faces, store

MODES = ('qwen_enhanced', 'h3_enhanced')
# The connector filenames used by the shipped local workflows. File bytes are part of the request.
MODEL_INPUTS = {'unet_name': 'diffusion_models', 'clip_name': 'text_encoders', 'vae_name': 'vae', 'lora_name': 'loras', 'ckpt_name': 'checkpoints', 'bg_removal_name': 'background_removal', 'moge_name': 'geometry_estimation', 'clip_vision_name': 'clip_vision'}


def fingerprint_file(path):
    path = Path(path)
    st = path.stat()
    digest = store.key('file-sha-v1', str(path.resolve()), st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
    def write(stage):
        (stage / 'sha256').write_text(store.file_hash(path), encoding='ascii')
    return (faces.cache_dir('file-sha', digest, write) / 'sha256').read_text(encoding='ascii')


def migrate_stage(kind, marker):
    """Adopt a committed cache entry from the exact old graph once, under the normal stage lock."""
    def decorate(request):
        @wraps(request)
        def lookup(*args, **kwargs):
            current = request(*args, **kwargs)
            if current is None:
                return None
            def digest(value):
                return value[0] if isinstance(value, tuple) else value
            base = store.root() / '.stages' / kind
            key = digest(current)
            if base.is_dir() and not (base / key / marker).is_file():
                previous = base / digest(request(*args, **kwargs, legacy=True))
                if (previous / marker).is_file():
                    faces.cache_dir(kind, key, lambda stage: shutil.copytree(previous, stage, dirs_exist_ok=True))
            return current
        return lookup
    return decorate


def structural_graph(bindings, nodes):
    """Number reachable nodes by sorted socket traversal, keeping fan-out and output slots."""
    indices = {}
    ordered = []
    def encode(value):
        if not (isinstance(value, list) and len(value) == 2 and isinstance(value[0], (str, int))
                and type(value[1]) is int and str(value[0]) in nodes):
            return {'literal': value}
        key = str(value[0])
        if key not in indices:
            indices[key] = len(ordered)
            raw = nodes[key]
            normalized = {'type': raw['type']}
            ordered.append(normalized)
            if 'inputs' in raw:
                normalized['inputs'] = {k: encode(raw['inputs'][k]) for k in sorted(raw['inputs'])}
        return {'node': indices[key], 'output': value[1]}
    roots = {k: encode(bindings[k]) for k in sorted(bindings)}
    return roots, ordered, indices


def lineage(prompt, node_id, sockets=('image', 'mask'), *, legacy=False):
    """Trace the actual input edges, terminating at already hashed reference and guide nodes."""
    if not isinstance(prompt, dict) or str(node_id) not in prompt:
        raise ValueError('ComfyUI prompt graph is required to cache model output safely.')
    nodes = {}
    model_files = {}
    sampler_settings = {}
    def literal(value):
        if isinstance(value, list) and len(value) == 2:
            entry = prompt.get(str(value[0]), {})
            if entry.get('class_type', '').startswith('Primitive'):
                return entry.get('inputs', {}).get('value', value)
        return value
    image_inputs = {}
    def visit(value):
        if not isinstance(value, list) or len(value) != 2 or not isinstance(value[0], (str, int)) or type(value[1]) is not int:
            return
        node = str(value[0])
        if node in nodes or node not in prompt:
            return
        d = prompt[node]
        kind = d['class_type']
        if kind in ('BIMakeSixGuides', 'BIPrepareReference'):
            nodes[node] = {'type': kind}
            return
        pins = d.get('inputs', {})
        nodes[node] = {'type': kind, 'inputs': pins}
        for field, pin in pins.items():
            if not isinstance(pin, str):
                visit(pin)
                continue
            group = MODEL_INPUTS.get(field)
            if kind == 'CLIPVisionLoader' and field == 'clip_name':
                group = 'clip_vision'
            if kind == 'LoadMoGeModel' and pin.endswith(('.safetensors', '.pt', '.pth')):
                group = 'geometry_estimation'
            if group:
                import folder_paths
                path = folder_paths.get_full_path(group, pin)
                if path:
                    model_files[group + '/' + pin] = fingerprint_file(path)
            if kind in ('LoadImage', 'LoadImageMask') and field == 'image':
                import folder_paths
                image_inputs[pin] = fingerprint_file(folder_paths.get_annotated_filepath(pin))
        if kind in ('KSampler', 'KSamplerAdvanced'):
            parameters = ('seed', 'noise_seed', 'steps', 'cfg', 'denoise', 'sampler_name', 'scheduler')
            sampler_settings[node] = {k: literal(v) for k, v in pins.items() if k in parameters}
        if kind in ('TextEncodeQwenImage21', 'MiniMaxH3ImageConditioning'):
            sampler_settings[node] = {'conditioning': kind, **{k: literal(v) for k, v in pins.items() if k in ('prompt', 'negative_prompt', 'width', 'height', 'resolution')}}
    pins = prompt[str(node_id)]['inputs']
    bindings = {name: pins.get(name) for name in sockets}
    for value in bindings.values():
        visit(value)
    # Two outputs of the same upstream node may contain different images, masks or GLBs.
    if legacy:
        return store.key(bindings, nodes, model_files, image_inputs), model_files, list(sampler_settings.values())
    roots, ordered, indices = structural_graph(bindings, nodes)
    settings = []
    for key in indices:
        if key in sampler_settings:
            item = sampler_settings[key].copy()
            for pin, value in item.items():
                if isinstance(value, list) and len(value) == 2 and str(value[0]) in indices:
                    item[pin] = {'node': indices[str(value[0])], 'output': value[1]}
            settings.append(item)
    return store.key('graph-v2', roots, ordered, model_files, image_inputs), model_files, settings


@migrate_stage('generated', 'result.json')
def generation_request(guides, reference, face_id, backend, recipe, seed, mask_mode, prompt, unique_id, canvas=None, *, legacy=False):
    if face_id not in store.FACES or backend not in MODES:
        raise ValueError('Unknown face or model backend.')
    recipe_data = json.loads(recipe or '{}')
    if not isinstance(recipe_data, dict):
        raise ValueError('Recipe must be a JSON object.')
    base = store.asset_dir(guides)
    manifest = json.loads((base / 'manifest.json').read_bytes())
    guide_hash = store.file_hash(store.contained(base, manifest['faces'][face_id]['guide']))
    graph_hash, weights, actual = lineage(prompt, unique_id, legacy=legacy)
    source = {'backend': backend, 'faceId': face_id, 'recipe': recipe_data, 'recipeSeed': seed, 'samplers': actual, 'graphSha256': graph_hash, 'weights': weights, 'referenceSha256': reference['sha256'], 'guideSha256': guide_hash, 'maskMode': mask_mode}
    if canvas is not None:
        if canvas.get('faceId') != face_id:
            raise ValueError('Generated image and canvas refer to different faces.')
        source['canvas'] = canvas
    digest = store.key('generated-face-v2', source)
    return digest, source


class CachedGeneratedFace:
    CATEGORY = 'Box Impostor'
    FUNCTION = 'resolve'
    RETURN_TYPES = ('BOX_FACE',)

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'guides': ('BOX_GUIDES',), 'reference': ('BOX_REFERENCE',), 'face_id': (list(store.FACES),), 'backend': (list(MODES),), 'recipe': ('STRING', {'default': '{}', 'multiline': True}), 'seed': ('INT', {'default': 0, 'min': 0, 'max': 0xffffffffffffffff}), 'mask_mode': (faces.MASK_MODES,), 'image': ('IMAGE', {'lazy': True})}, 'optional': {'mask': ('MASK', {'lazy': True}), 'canvas': ('BOX_CANVAS',)}, 'hidden': {'prompt': 'PROMPT', 'unique_id': 'UNIQUE_ID'}}

    def check_lazy_status(self, guides, reference, face_id, backend, recipe, seed, mask_mode, image=None, mask=None, prompt=None, unique_id=None, canvas=None):
        digest, _ = generation_request(guides, reference, face_id, backend, recipe, seed, mask_mode, prompt, unique_id, canvas)
        if (store.root() / '.stages' / 'generated' / digest / 'result.json').is_file():
            return []
        missing = ['image'] if image is None else []
        if mask_mode in ('foreground', 'load_image') and mask is None:
            if 'mask' not in prompt[str(unique_id)]['inputs']:
                raise ValueError('The selected generation mask mode needs a connected mask.')
            missing.append('mask')
        return missing

    def resolve(self, guides, reference, face_id, backend, recipe, seed, mask_mode, image=None, mask=None, prompt=None, unique_id=None, canvas=None):
        digest, source = generation_request(guides, reference, face_id, backend, recipe, seed, mask_mode, prompt, unique_id, canvas)
        def write(stage):
            if image is None:
                raise ValueError('Missing generated image after cache miss.')
            item = faces.save_view(faces.rgba(image, mask_mode, mask))
            item['provenance'] = source
            # Internal paths are reconstructed relative to the live cache, supporting directory moves.
            (stage / 'result.json').write_text(json.dumps({'sha256': item['sha256'], 'provenance': source}), encoding='utf-8')
        stage = faces.cache_dir('generated', digest, write)
        item = json.loads((stage / 'result.json').read_bytes())
        item['provenance'] = source
        item['path'] = str(store.root() / '.stages' / 'faces' / item['sha256'])
        if 'path' in reference:
            item['reference'] = reference
        if not (Path(item['path']) / 'source.npy').is_file():
            raise ValueError('Generated image cache payload is missing.')
        return (item,)
