"""ComfyUI entry points; assets travel between nodes as immutable identifiers."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np
from PIL import Image

from . import faces, store
from .generation import CachedGeneratedFace, lineage

CATEGORY = 'Box Impostor'
BACKENDS = ['auto', 'webgpu', 'webgl2']
RESOLUTION = ('INT', {'default': 256, 'min': 8, 'max': 2048, 'step': 8})
NAME = re.compile(r'[A-Za-z0-9_-]{1,100}\Z')


def inputs(**fields):
    return {'required': fields}


def input_path(relative):
    import folder_paths
    return store.contained(folder_paths.get_input_directory(), relative)


def tensor(a):
    import torch
    return torch.from_numpy(np.ascontiguousarray(a).copy())


def export_root():
    return store.root().parent / 'box_impostor_exports'


class PrepareReference:
    CATEGORY = CATEGORY
    FUNCTION = 'prepare'
    RETURN_TYPES = ('IMAGE', 'MASK', 'BOX_REFERENCE')
    RETURN_NAMES = ('rgb', 'coverage', 'reference')

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'image': ('IMAGE',), 'mask_mode': (['opaque', 'rgba', 'foreground', 'load_image'],)}, 'optional': {'mask': ('MASK', {'lazy': True})}, 'hidden': {'prompt': 'PROMPT', 'unique_id': 'UNIQUE_ID'}}

    def request(self, image, mask_mode, prompt, unique_id):
        if prompt is None or str(unique_id) not in prompt or mask_mode not in ('foreground', 'load_image'):
            return None
        if 'mask' not in prompt[str(unique_id)].get('inputs', {}):
            raise ValueError('Connect a mask or choose an explicit opaque/RGBA reference.')
        pixels = faces.array(image)
        graph, _, _ = lineage(prompt, unique_id, sockets=('mask',))
        return store.key('prepared-reference-v1', pixels.shape, hashlib.sha256(pixels.tobytes()).hexdigest(), mask_mode, graph)

    def check_lazy_status(self, image, mask_mode, mask=None, prompt=None, unique_id=None):
        digest = self.request(image, mask_mode, prompt, unique_id)
        if digest is not None and mask is None and not (store.root() / '.stages' / 'prepared' / digest / 'result.json').is_file():
            return ['mask']
        return []

    def prepare(self, image, mask_mode, mask=None, prompt=None, unique_id=None):
        digest = self.request(image, mask_mode, prompt, unique_id)
        if digest is None:
            rgba = faces.rgba(image, mask_mode, mask)
            item = faces.save_view(rgba)
        else:
            def write(stage):
                view = faces.save_view(faces.rgba(image, mask_mode, mask))
                (stage / 'result.json').write_text(json.dumps({'sha256': view['sha256']}), encoding='utf-8')
            stage = faces.cache_dir('prepared', digest, write)
            item = json.loads((stage / 'result.json').read_bytes())
            item['path'] = str(store.root() / '.stages' / 'faces' / item['sha256'])
            rgba = np.load(Path(item['path']) / 'source.npy', allow_pickle=False)
        h, w = rgba.shape[:2]
        meta = {**item, 'width': w, 'height': h, 'maskMode': mask_mode, 'sourceToReference': [1, 0, 0, 0, 1, 0, 0, 0, 1]}
        return tensor(rgba[None, ..., :3]), tensor(rgba[None, ..., 3]), meta


class CanonicalizeProxy:
    CATEGORY = CATEGORY
    FUNCTION = 'canonicalize'
    RETURN_TYPES = ('BOX_PROXY',)

    @classmethod
    def INPUT_TYPES(cls):
        angle = ('FLOAT', {'default': 0., 'min': -360., 'max': 360., 'step': .1})
        return {'required': {'filename': ('STRING', {'default': ''})}, 'optional': {'model_3d': ('FILE_3D_GLB', {'lazy': True}), 'reference': ('BOX_REFERENCE',), 'rotate_x': angle, 'rotate_y': angle, 'rotate_z': angle}, 'hidden': {'prompt': 'PROMPT', 'unique_id': 'UNIQUE_ID'}}

    @classmethod
    def IS_CHANGED(cls, filename, model_3d=None, **kwargs):
        if model_3d is not None:
            return hashlib.sha256(model_3d.get_bytes()).hexdigest() if hasattr(model_3d, 'get_bytes') else float('nan')
        return store.file_hash(input_path(filename)) if filename else ''

    def cached_request(self, reference, prompt, unique_id):
        lineage_key, weights, _ = lineage(prompt, unique_id, sockets=('model_3d',))
        digest = store.key('generated-glb-v1', reference['sha256'], lineage_key)
        return digest, {'backend': 'native_mesh', 'graphSha256': lineage_key, 'weights': weights}

    def check_lazy_status(self, filename='', model_3d=None, reference=None, prompt=None, unique_id=None, **pose):
        linked = prompt is not None and 'model_3d' in prompt[str(unique_id)]['inputs']
        if filename and linked:
            raise ValueError('Choose one source: filename or connected GLB.')
        if filename or model_3d is not None or not linked:
            return []
        if reference is not None:
            digest, _ = self.cached_request(reference, prompt, unique_id)
            if (store.root() / '.stages' / 'native-mesh' / digest / 'source.glb').is_file():
                return []
        return ['model_3d']

    def canonicalize(self, filename='', model_3d=None, reference=None, rotate_x=0., rotate_y=0., rotate_z=0., prompt=None, unique_id=None):
        rotation = tuple(float(x) for x in (rotate_x, rotate_y, rotate_z))
        if not all(math.isfinite(v) and -360. <= v <= 360. for v in rotation):
            raise ValueError('Rotation angles must be finite degrees in [-360,360].')
        if filename and model_3d is not None:
            raise ValueError('Choose one source: filename or connected GLB.')
        def native_bytes():
            if model_3d is None or getattr(model_3d, 'format', '').lower() != 'glb':
                raise ValueError('Supply a GLB inside ComfyUI/input or connect a native FILE_3D_GLB.')
            return model_3d.get_bytes()
        provenance = None
        if reference is not None and not filename:
            request_key, provenance = self.cached_request(reference, prompt, unique_id)
            def write_raw(stage):
                (stage / 'source.glb').write_bytes(native_bytes())
            raw_dir = faces.cache_dir('native-mesh', request_key, write_raw)
            raw = (raw_dir / 'source.glb').read_bytes()
        elif filename:
            src = input_path(filename)
            if src.suffix.lower() != '.glb':
                raise ValueError('Box input must be a self-contained .glb.')
            raw = src.read_bytes()
        else:
            raw = native_bytes()
        digest = store.key('canonical-v2', store.worker_version(), hashlib.sha256(raw).hexdigest(), rotation)
        def write(stage):
            raw_path = stage / 'source.glb'
            raw_path.write_bytes(raw)
            options = [v for axis, val in zip('xyz', rotation) if val for v in ('--rotate-' + axis, str(val))]
            store.run_worker('canonicalize', '--input', raw_path, '--output', stage / 'canonical', *options)
        directory = faces.cache_dir('proxies', digest, write) / 'canonical'
        path = directory / 'canonical.glb'
        metadata = directory / 'canonical.json'
        glb_hash = store.file_hash(path)
        if json.loads(metadata.read_bytes())['glbSha256'] != glb_hash:
            raise ValueError('Cached canonical GLB digest mismatch.')
        result = {'stageKey': digest, 'sha256': glb_hash, 'path': str(path), 'metadata': str(metadata), 'rotationDegreesXYZ': rotation}
        if provenance:
            result.update(stageKey=store.key(digest, request_key), reference=reference, provenance=provenance)
        return (result,)


class MakeSixGuides:
    CATEGORY = CATEGORY
    FUNCTION = 'guides'
    RETURN_TYPES = ('IMAGE',) * 6 + ('BOX_GUIDES',)
    RETURN_NAMES = tuple(faces.SLUGS.values()) + ('guides',)

    @classmethod
    def INPUT_TYPES(cls):
        return inputs(proxy=('BOX_PROXY',), resolution=RESOLUTION, backend=(BACKENDS,))

    def guides(self, proxy, resolution, backend):
        asset = store.baseline(proxy, resolution, backend)
        directory = store.asset_dir(asset)
        manifest = store.validate(directory, asset, check_bytes=False)
        result = []
        for face_id in store.FACES:
            with Image.open(store.contained(directory, manifest['faces'][face_id]['guide'])) as image:
                result.append(tensor(np.asarray(image.convert('RGBA'), dtype=np.float32)[None] / 255))
        return (*result, asset)


class FaceImage:
    CATEGORY = CATEGORY
    FUNCTION = 'capture'
    RETURN_TYPES = ('BOX_FACE',)

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'image': ('IMAGE',), 'mask_mode': (faces.MASK_MODES,)}, 'optional': {'mask': ('MASK',), 'provenance': ('STRING', {'default': '{}', 'multiline': True})}}

    def capture(self, image, mask_mode, mask=None, provenance='{}'):
        details = json.loads(provenance or '{}')
        if not isinstance(details, dict) or details.get('backend', 'manual_faces') not in ('manual_faces', 'qwen_enhanced', 'h3_enhanced'):
            raise ValueError('Invalid face provenance.')
        return ({**faces.save_view(faces.rgba(image, mask_mode, mask)), 'provenance': details},)


class FaceInputs:
    CATEGORY = CATEGORY
    FUNCTION = 'collect'
    RETURN_TYPES = ('BOX_FACES',)

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {}, 'optional': {slug: ('BOX_FACE',) for slug in faces.SLUGS.values()}}

    def collect(self, **kwargs):
        return ({fid: kwargs[slug] for fid, slug in faces.SLUGS.items() if kwargs.get(slug) is not None},)


class AlignAndValidateFaces:
    CATEGORY = CATEGORY
    FUNCTION = 'align'
    RETURN_TYPES = ('BOX_ALIGNED_FACES',)

    @classmethod
    def INPUT_TYPES(cls):
        return inputs(faces=('BOX_FACES',), guides=('BOX_GUIDES',), min_iou=('FLOAT', {'default': .85, 'min': 0., 'max': 1.}), max_shift=('FLOAT', {'default': .1, 'min': 0., 'max': 1.}), max_scale_change=('FLOAT', {'default': .15, 'min': 0., 'max': 1.}))

    def align(self, faces, guides, min_iou, max_shift, max_scale_change):
        from .faces import align
        if not all(0 <= v <= 1 for v in (min_iou, max_shift, max_scale_change)):
            raise ValueError('QA thresholds must be in [0,1].')
        return (align(faces, guides, min_iou, max_shift, max_scale_change),)


class BakeBox:
    CATEGORY = CATEGORY
    FUNCTION = 'bake'
    RETURN_TYPES = ('BOX_IMPOSTOR',)

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'proxy': ('BOX_PROXY',), 'resolution': RESOLUTION, 'backend': (BACKENDS,)}, 'optional': {'faces': ('BOX_ALIGNED_FACES',)}}

    def bake(self, proxy, resolution, backend, faces=None):
        asset = store.baseline(proxy, resolution, backend)
        if faces:
            from .faces import compose
            asset = compose(asset, faces)
        return (asset,)


class PreviewBox:
    CATEGORY = CATEGORY
    OUTPUT_NODE = True
    FUNCTION = 'preview'
    RETURN_TYPES = ('STRING', 'IMAGE')
    RETURN_NAMES = ('viewer_url', 'thumbnail')

    @classmethod
    def INPUT_TYPES(cls):
        return inputs(asset=('BOX_IMPOSTOR',))

    def preview(self, asset):
        directory = store.asset_dir(asset)
        manifest = store.validate(directory, asset, check_bytes=False)
        image_path = store.thumbnail(directory, manifest)
        with Image.open(image_path) as image:
            preview = tensor(np.asarray(image.convert('RGB'), np.float32)[None] / 255)
        url = f'/box-impostor/web/viewer.html?manifest=/box-impostor/assets/{asset}/manifest.json'
        return {'ui': {'box_impostor': [url]}, 'result': (url, preview)}


class SaveBox:
    CATEGORY = CATEGORY
    OUTPUT_NODE = True
    FUNCTION = 'save'
    RETURN_TYPES = ('STRING',)
    RETURN_NAMES = ('manifest_path',)

    @classmethod
    def INPUT_TYPES(cls):
        return inputs(asset=('BOX_IMPOSTOR',), name=('STRING', {'default': ''}))

    def save(self, asset, name=''):
        name = name or asset
        if not NAME.fullmatch(name):
            raise ValueError('Use an export name containing only letters, digits, _ or -.')
        source = store.asset_dir(asset)
        store.validate(source, asset)
        base = export_root()
        base.mkdir(parents=True, exist_ok=True)
        dest = store.contained(base, name)
        with store.lock(store.key('export', name)):
            if dest.exists():
                try:
                    store.validate(dest, asset)
                    hero = source / 'previews' / 'hero.png'
                    if hero.is_file() and not (dest / 'previews' / 'hero.png').exists():
                        store.copy_thumbnail(hero, dest)
                except (ValueError, KeyError, OSError) as exc:
                    raise ValueError('That export name already exists with different or incomplete content.') from exc
            else:
                with tempfile.TemporaryDirectory(dir=base) as temp:
                    staging = Path(temp) / 'asset'
                    shutil.copytree(source, staging)
                    store.validate(staging, asset)
                    os.rename(staging, dest)
        return (str(dest / 'manifest.json'),)


class LoadBox:
    CATEGORY = CATEGORY
    FUNCTION = 'load'
    RETURN_TYPES = ('BOX_IMPOSTOR',)

    @classmethod
    def INPUT_TYPES(cls):
        return inputs(source=('STRING', {'default': ''}))

    @classmethod
    def IS_CHANGED(cls, source):
        # Files under input/ or exports can be replaced between queue submissions.
        return float('nan')

    def load(self, source):
        if store.ID.fullmatch(source) and store.asset_dir(source).is_dir():
            store.validate(store.asset_dir(source), source)
            return (source,)
        if NAME.fullmatch(source) and store.contained(export_root(), source).is_dir():
            directory = store.contained(export_root(), source)
        else:
            path = input_path(source)
            if path.name != 'manifest.json':
                raise ValueError('Enter a saved asset ID, export name, or input-relative manifest.json.')
            directory = path.parent
        manifest = store.validate(directory)
        asset_id = manifest['asset_id']
        if not store.ID.fullmatch(asset_id):
            asset_id = store.key('import-v1', store.file_hash(directory / 'manifest.json'))
        def copy(stage):
            copy_asset(directory, stage, manifest)
            manifest['asset_id'] = asset_id
            (stage / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding='utf-8')
        # A foreign manifest is not trusted to claim the identity of a different published asset.
        dest = store.asset_dir(asset_id)
        if dest.exists():
            current = store.validate(dest, asset_id)
            expected = {**manifest, 'asset_id': asset_id}
            if current != expected:
                raise ValueError('An asset with this ID already exists with a different manifest.')
        return (store.publish(asset_id, copy),)


def copy_asset(directory, output, manifest):
    # Never follow links or walk neighbouring directories when importing a portable package.
    extra_dirs = {'geometry', 'guides', 'maps', 'views', 'reference', 'previews', 'qa'}
    declared = {manifest['proxy'], manifest['baseColor']['path'], *(tex['path'] for tex in manifest['textures'].values())}
    for face in manifest['faces'].values():
        declared.add(face['guide'])
        if face.get('depth'):
            declared.add(face['depth'])
    output.mkdir()
    for rel in declared:
        src = store.contained(directory, rel)
        dest = store.contained(output, rel)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
    for name in extra_dirs:
        src = directory / name
        if src.is_symlink():
            raise ValueError('Symlinks are not supported in imported assets.')
        if not src.is_dir():
            continue
        for path in src.rglob('*'):
            if path.is_symlink():
                raise ValueError('Symlinks are not supported in imported assets.')
            if path.is_file():
                target = store.contained(output, path.relative_to(directory).as_posix())
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(path, target)


NODE_CLASS_MAPPINGS = {'BICachedGeneratedFace': CachedGeneratedFace, 'BIPrepareReference': PrepareReference, 'BICanonicalizeProxy': CanonicalizeProxy, 'BIMakeSixGuides': MakeSixGuides, 'BIFaceImage': FaceImage, 'BIFaceInputs': FaceInputs, 'BIAlignAndValidateFaces': AlignAndValidateFaces, 'BIBakeBox': BakeBox, 'BIPreviewBox': PreviewBox, 'BISaveBox': SaveBox, 'BILoadBox': LoadBox}
NODE_DISPLAY_NAME_MAPPINGS = {name: 'BI ' + name[2:] for name in NODE_CLASS_MAPPINGS}
