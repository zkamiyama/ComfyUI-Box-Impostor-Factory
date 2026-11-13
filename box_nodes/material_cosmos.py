"""Estimate face material with Materia's Cosmos inverse renderer."""
import inspect
from pathlib import Path

from . import generation, store

CHECKPOINT = 'Diffusion_Renderer_Inverse_Cosmos_7B/model.pt'
TOKENIZER_DIR = 'Cosmos-1.0-Tokenizer-CV8x8x8/vae'
VAE_FILES = ('config.json', 'diffusion_pytorch_model.safetensors')
NODES = ('MateriaLoadModel', 'MateriaInverseRenderer')
MAX_INFERENCE = 512
STEPS = 15


def missing_components():
    import folder_paths
    from nodes import NODE_CLASS_MAPPINGS

    missing = []
    if not folder_paths.get_full_path('diffusion_models', CHECKPOINT):
        missing.append('diffusion_models/' + CHECKPOINT)
    vae_dir = Path(folder_paths.models_dir) / 'vae' / TOKENIZER_DIR
    missing.extend(f'vae/{TOKENIZER_DIR}/{name}' for name in VAE_FILES if not (vae_dir / name).is_file())
    if any(node not in NODE_CLASS_MAPPINGS for node in NODES):
        missing.append('comfyui-materia (Load Materia Model and Inverse Rendering nodes)')
    return missing


def checkpoint_identity():
    import folder_paths
    from nodes import NODE_CLASS_MAPPINGS

    missing = missing_components()
    if missing:
        raise FileNotFoundError('Cosmos requires ' + ', '.join(missing))
    vae_dir = Path(folder_paths.models_dir) / 'vae' / TOKENIZER_DIR
    identity = {CHECKPOINT: generation.fingerprint_file(folder_paths.get_full_path_or_raise('diffusion_models', CHECKPOINT))}
    identity.update({name: generation.fingerprint_file(vae_dir / name) for name in VAE_FILES})
    wrapper = Path(inspect.getfile(NODE_CLASS_MAPPINGS['MateriaInverseRenderer'])).parent
    for name in ('nodes.py', 'CleanVAE.py', 'CleanGeneralDIT.py', 'model_diffusion_renderer.py',
                 'diffusion_renderer_config.py', 'diffusion_renderer_pipeline.py'):
        identity[name] = store.file_hash(wrapper / name)
    return identity


def load():
    import comfy.model_management as mm
    from nodes import NODE_CLASS_MAPPINGS

    pipe, = NODE_CLASS_MAPPINGS['MateriaLoadModel']().load_pipeline(CHECKPOINT)
    device = mm.get_torch_device()
    weights = mm.module_size(pipe.pre_loaded_model_instance) + mm.module_size(pipe.vae_instance.model)
    if device.type != 'cpu':
        mm.free_memory(weights + mm.minimum_inference_memory(), device)
    return pipe


def dimensions(h, w, longest):
    height = max(1, round(longest * h / max(h, w)))
    width = max(1, round(longest * w / max(h, w)))
    return height, width, (height + 15) // 16 * 16, (width + 15) // 16 * 16


def estimate(image, pipeline, resolution, seed):
    import torch
    import torch.nn.functional as F
    from nodes import NODE_CLASS_MAPPINGS

    h, w = image.shape[1:3]
    longest = min(resolution, MAX_INFERENCE)
    height, width, padded_h, padded_w = dimensions(h, w, longest)
    top, left = (padded_h - height) // 2, (padded_w - width) // 2
    target_h = max(1, round(resolution * h / max(h, w)))
    target_w = max(1, round(resolution * w / max(h, w)))

    def resize(x, shape):
        if x.shape[1:3] == shape:
            return x
        return F.interpolate(x.movedim(-1, 1), size=shape, mode='bilinear', align_corners=False,
                             antialias=True).movedim(1, -1)

    with torch.inference_mode():
        resized = resize(image, (height, width))
        padded = image.new_full((1, padded_h, padded_w, 3), .5)
        padded[:, top:top+height, left:left+width] = resized
        # On failure the public node can leave both the DiT and the tokenizer resident.
        try:
            albedo, metal, rough, _, _ = NODE_CLASS_MAPPINGS['MateriaInverseRenderer']().run_inverse_pass(
                pipeline, padded, guidance=0.0, seed=seed, num_steps=STEPS)
        except BaseException:
            pipeline.to_cpu()
            raise
        def registered(x):
            if x.shape != (1, padded_h, padded_w, 3) or not x.isfinite().all() or x.amin() < 0 or x.amax() > 1:
                raise ValueError('Materia returned a map with invalid dimensions or values.')
            return resize(x[:, top:top+height, left:left+width], (target_h, target_w))
        rough_scalar = registered(rough).mean(dim=-1)
        metal_scalar = registered(metal).mean(dim=-1) >= .5
        return registered(albedo), torch.stack((rough_scalar, metal_scalar, torch.ones_like(rough_scalar)), dim=-1)
