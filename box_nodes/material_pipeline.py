"""Estimate materials on the original face pixels, then register them into the Box atlas."""
import json
from contextlib import ExitStack
from functools import lru_cache
from importlib.util import find_spec
from pathlib import Path

import numpy as np
from scipy import ndimage

from . import canvas, faces, generation, store
from . import face_material, material_estimate, material_iid

# Base checkpoints from the two reference workflows, without a view-changing LoRA.
NATIVE_MODELS = {
    'qwen': ('qwen_image_2.1_int8_convrot.safetensors', 'qwen3vl_8b_int8_convrot.safetensors',
             'qwen_image_2.1_vae_bf16.safetensors', 'qwen_image'),
    'h3': ('minimax_h3_image_pruned_int8_convrot.safetensors', 'qwen3vl_32b_minimax_h3_int8_convrot.safetensors',
           'minimax_h3_image_vae_fp16.safetensors', 'minimax'),
}
SOURCES = ['keep', 'qwen', 'h3', 'marigold']
MODEL_FOLDERS = ('diffusion_models', 'text_encoders', 'vae')


def material_availability(source):
    """Check installed components without opening weights or starting inference."""
    if source not in SOURCES:
        raise ValueError('Unknown material estimator.')
    missing = []
    if source in NATIVE_MODELS:
        import folder_paths
        for folder, filename in zip(MODEL_FOLDERS, NATIVE_MODELS[source][:3]):
            if folder_paths.get_full_path(folder, filename) is None:
                missing.append(f'{folder}/{filename}')
    if source == 'marigold':
        for package in ('diffusers', 'transformers', 'accelerate'):
            if find_spec(package) is None:
                missing.append(f'Python package {package}')
        try:
            checkpoint = material_iid.checkpoint_path()
        except FileNotFoundError:
            missing.append(f'diffusers/{material_iid.CHECKPOINT}/model_index.json')
        else:
            variant = material_iid.weight_variant(checkpoint)
            suffix = f'.{variant}' if variant else ''
            required = ['scheduler/scheduler_config.json', 'tokenizer/vocab.json', 'tokenizer/merges.txt']
            for folder, stem in (('text_encoder', 'model'), ('unet', 'diffusion_pytorch_model'),
                                 ('vae', 'diffusion_pytorch_model')):
                required.extend((f'{folder}/config.json', f'{folder}/{stem}{suffix}.safetensors'))
            missing.extend(f'diffusers/{material_iid.CHECKPOINT}/{file}' for file in required
                           if not (checkpoint / file).is_file())
    return {'source': source, 'available': not missing, 'missing': missing}


def model_identity(source):
    if source == 'marigold':
        return material_iid.checkpoint_identity()
    import folder_paths
    return {name: generation.fingerprint_file(folder_paths.get_full_path_or_raise(group, name))
            for group, name in zip(MODEL_FOLDERS, NATIVE_MODELS[source][:3])}


def load_native(source):
    from nodes import CLIPLoader, UNETLoader, VAELoader
    model, clip, vae, clip_type = NATIVE_MODELS[source]
    return (UNETLoader().load_unet(model, 'default')[0],
            CLIPLoader().load_clip(clip, clip_type, 'default')[0],
            VAELoader().load_vae(vae)[0])


def register_estimates(albedo, rma, original, report):
    """Replay the accepted crop and registration on both model outputs, including soft edges."""
    def resize_to(array, shape):
        return array if array.shape[:2] == tuple(shape) else faces.resize(array, shape)

    def crop(array):
        return canvas.undo_guide_canvas(array, report['guideCanvas']) if report.get('guideCanvas') else array

    albedo = crop(faces.array(albedo)[0, ..., :3])
    rma = crop(faces.array(rma)[0, ..., :3])
    if albedo.shape != rma.shape:
        raise ValueError('Base color and RMA must use the same pixel grid.')
    source_alpha = crop(original[..., 3:4])
    width, height = report['tileSize']
    target = (height, width)
    # Only coverage comes from the source image. Filter premultiplied maps
    # straight from their native grid to retain fine detail and avoid edge bleed.
    alpha = resize_to(source_alpha, albedo.shape[:2])
    linear = np.where(albedo <= .04045, albedo / 12.92, ((albedo + .055) / 1.055) ** 2.4)
    tile = np.concatenate((resize_to(linear * alpha, target),
                           resize_to(rma * alpha, target),
                           resize_to(alpha, target)), axis=-1)
    reg = report['registration']
    scale = reg['scale']
    shift = np.array(reg['centerShiftYX'])
    if scale != 1 or np.any(shift):
        # Reuse the center measured from the accepted image, even if its coverage
        # crossed a half-pixel boundary on the higher-resolution working grid.
        center, _ = faces.bounds(resize_to(source_alpha, target)[..., 0] >= .5)
        offset = center - (center + shift) / scale
        tile = np.stack([ndimage.affine_transform(channel, np.eye(2) / scale, offset=offset,
                                                  order=1, mode='constant', cval=0., prefilter=False)
                         for channel in np.moveaxis(tile, -1, 0)], axis=-1)
    np.divide(tile[..., :6], tile[..., 6:], out=tile[..., :6], where=tile[..., 6:] > 0)
    rgb = np.clip(tile[..., :3], 0., 1.)
    srgb = np.where(rgb <= .0031308, 12.92 * rgb, 1.055 * rgb ** (1. / 2.4) - .055)
    return srgb[None, ...], np.clip(tile[None, ..., 3:6], 0., 1.)


class GenerateFaceMaterials:
    CATEGORY = 'Box Impostor'
    FUNCTION = 'generate'
    RETURN_TYPES = ('BOX_IMPOSTOR',)
    DESCRIPTION = ('Choose a material estimator independently of the face image generator. '
                   'Keep preserves the incoming Box. Qwen and H3 use prompt-based image editing. '
                   'Marigold IID Appearance requires its own checkpoint and preserves baked AO.')

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'asset': ('BOX_IMPOSTOR',),
            'source': (SOURCES,),
            'face': (['all', *store.FACES],),
            'material_seed': ('INT', {'default': 0, 'min': 0, 'max': (1 << 53) - 1}),
            'edit_resolution': ('INT', {'default': 2048, 'min': 32, 'max': 4096, 'step': 32,
                                        'tooltip': 'Target size for the Qwen/H3 edit. Marigold uses its trained resolution.'}),
        }}

    @classmethod
    def IS_CHANGED(cls, source='keep', **kwargs):
        return model_identity(source) if source in SOURCES[1:] else source

    def generate(self, asset, source='keep', face='all', material_seed=0, edit_resolution=2048):
        if source not in SOURCES or face not in ('all', *store.FACES):
            raise ValueError('Unknown material estimator or Box face.')
        if source == 'keep':
            return (asset,)
        directory = store.asset_dir(asset)
        manifest = store.validate(directory, asset, check_bytes=False)
        selected = [fid for fid in store.FACES if face in ('all', fid)
                    and manifest['faces'][fid].get('edit', {}).get('status') == 'accepted']
        if not selected:
            raise ValueError(f'No accepted face image for {face}. Generate and align a face before estimating its material.')
        report = json.loads(store.contained(directory, manifest['qa']).read_bytes())['faces']
        backend_module = material_iid if source == 'marigold' else material_estimate
        implementation = store.key(*(store.file_hash(Path(filename)) for filename in
                                     (__file__, backend_module.__file__, canvas.__file__, faces.__file__)))
        recipe = {'provider': source, 'model': model_identity(source), 'seed': material_seed,
                  'implementation': implementation}
        if source != 'marigold':
            recipe['editResolution'] = edit_resolution
        else:
            recipe['steps'] = 4
        # Cache lazy loaders only for the lifetime of this execution.
        with ExitStack() as stack:
            native = lru_cache(maxsize=1)(load_native)
            iid = lru_cache(maxsize=1)(lambda: stack.enter_context(material_iid.loaded_estimator()))
            estimates = {}
            for fid in selected:
                store.check_interrupted()
                edit = manifest['faces'][fid]['edit']
                face_recipe = {**recipe, 'alignmentKey': edit['alignmentKey']}
                key = store.key('registered-material-v1', fid, face_recipe)

                def write(stage):
                    import torch
                    view_dir = store.contained(directory, edit['view']).parent
                    original = np.load(view_dir / 'source.npy', allow_pickle=False)
                    alpha = original[..., 3:4]
                    rgb = original[..., :3] * alpha + .5 * (1 - alpha)
                    image = torch.from_numpy(rgb[None, ...])
                    if source == 'marigold':
                        outputs = iid()(image, seed=material_seed, steps=4)
                    else:
                        outputs = material_estimate.MaterialEstimate().estimate(
                            image, *native(source), source + '_enhanced', material_seed, resolution=edit_resolution)
                    base, rma = register_estimates(*outputs, original, report[fid])
                    np.save(stage / 'albedo.npy', base, allow_pickle=False)
                    np.save(stage / 'rma.npy', rma, allow_pickle=False)

                cached = faces.cache_dir('estimated-material', key, write)
                estimates[fid] = {'provider': source, 'recipe': face_recipe, 'packing': 'RM' if source == 'marigold' else 'RMA',
                                  'albedo': np.load(cached / 'albedo.npy', mmap_mode='r', allow_pickle=False),
                                  'rma': np.load(cached / 'rma.npy', mmap_mode='r', allow_pickle=False)}
        return (face_material.apply_face_materials(asset, estimates),)
