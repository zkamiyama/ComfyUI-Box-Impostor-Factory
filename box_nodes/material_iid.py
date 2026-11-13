"""Marigold IID Appearance inference on individually generated Box faces."""
from contextlib import contextmanager
from functools import partial
from pathlib import Path

from . import store
from .generation import fingerprint_file


CHECKPOINT = 'marigold-iid-appearance-v1-1'


def checkpoint_path():
    import folder_paths

    for root in folder_paths.get_folder_paths('diffusers'):
        checkpoint = Path(root) / CHECKPOINT
        if (checkpoint / 'model_index.json').is_file():
            return checkpoint
    raise FileNotFoundError(
        f'Install the local Diffusers checkpoint prs-eth/{CHECKPOINT} in '
        f'ComfyUI/models/diffusers/{CHECKPOINT}/ (including model_index.json, '
        'scheduler, tokenizer, text_encoder, unet, and vae).'
    )


def weight_variant(checkpoint):
    return 'fp16' if (checkpoint / 'unet/diffusion_pytorch_model.fp16.safetensors').is_file() else None


def checkpoint_identity():
    checkpoint = checkpoint_path()
    files = []
    for path in sorted(checkpoint.rglob('*')):
        relative = path.relative_to(checkpoint)
        if '.cache' in relative.parts or path.suffix not in ('.safetensors', '.json', '.txt', '.model'):
            continue
        if path.is_file():
            files.append((str(relative), fingerprint_file(path)))
    return store.key(CHECKPOINT, files)


def _estimate(pipe, image, seed=0, steps=4):
    import torch
    from comfy import model_management as mm

    if image.ndim != 4 or image.shape[-1] not in (3, 4) or min(image.shape[:3]) == 0:
        raise ValueError('Expected a nonempty batch of ComfyUI RGB or RGBA images.')
    if not 1 <= steps <= 4:
        raise ValueError('Marigold IID Appearance supports 1 to 4 denoising steps here.')
    shape = (*image.shape[:3], 3)
    albedo = torch.empty(shape, dtype=torch.float32, device='cpu')
    rma = torch.empty_like(albedo)
    with torch.inference_mode():
        for index, face in enumerate(image):
            mm.throw_exception_if_processing_interrupted()
            rgb = face[..., :3]
            if face.shape[-1] == 4:
                rgb = torch.lerp(torch.full_like(rgb, 0.5), rgb, face[..., 3:])
            # Keep per-face results reproducible when the same face is rerun alone.
            result = pipe(image=rgb.movedim(-1, 0).unsqueeze(0), num_inference_steps=steps,
                          generator=torch.Generator(device='cpu').manual_seed(seed),
                          match_input_resolution=True, output_type='pt').prediction
            if result.shape != (2, 3, *face.shape[:2]) or not torch.isfinite(result).all():
                raise ValueError('Marigold returned invalid albedo or material predictions.')
            raw = result.detach().movedim(1, -1).to(device='cpu', dtype=torch.float32)
            albedo[index].copy_(raw[0])
            rma[index, ..., :2].copy_(raw[1, ..., :2])
            rma[index, ..., 2].fill_(1)
            del result, raw
    return albedo, rma


@contextmanager
def loaded_estimator():
    """Own the pipeline for one graph execution, including all uncached faces."""
    import torch
    from comfy import model_management as mm

    checkpoint = checkpoint_path()
    try:
        from diffusers import MarigoldIntrinsicsPipeline
    except ImportError as exc:
        raise ImportError('Install diffusers with MarigoldIntrinsicsPipeline, transformers and '
                          'accelerate in the ComfyUI Python environment.') from exc

    device = mm.get_torch_device()
    dtype = torch.float16 if mm.should_use_fp16(device) else torch.float32
    # An fp16-only snapshot also works on CPU: load its files into float32.
    pipe = MarigoldIntrinsicsPipeline.from_pretrained(
        str(checkpoint), torch_dtype=dtype, variant=weight_variant(checkpoint), local_files_only=True, use_safetensors=True,
    )
    try:
        props = pipe.target_properties or {}
        if (props.get('target_names') != ['albedo', 'material']
                or props.get('albedo', {}).get('prediction_space') != 'srgb'
                or props.get('material', {}).get('sub_target_names') != ['roughness', 'metallicity', None]
                or any(props.get(name, {}).get('prediction_space') != 'linear'
                       for name in ('roughness', 'metallicity'))):
            raise ValueError(f'Expected the albedo and linear material targets from prs-eth/{CHECKPOINT}.')
        if device.type != 'cpu':
            weights = sum(mm.module_size(module) for module in (pipe.unet, pipe.vae, pipe.text_encoder))
            mm.free_memory(weights + mm.minimum_inference_memory(), device)
        pipe.to(device)
        pipe.set_progress_bar_config(disable=True)
        yield partial(_estimate, pipe)
    finally:
        # Diffusers is outside Comfy's patcher registry. Release its VRAM even on cancellation.
        if device.type != 'cpu':
            pipe.to('cpu')
            pipe.empty_text_embedding = None
        del pipe
        mm.soft_empty_cache()


class EstimateFaceMaterialIID:
    CATEGORY = 'Box Impostor'
    FUNCTION = 'estimate'
    RETURN_TYPES = ('IMAGE', 'IMAGE')
    RETURN_NAMES = ('albedo_srgb', 'rma')
    DESCRIPTION = ('Estimates each face independently with Marigold IID Appearance. '
                   'Its default processing resolution is 768 px. Outputs match the input canvas. '
                   'Use RM packing to retain baked AO; blue is a neutral placeholder.')

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'image': ('IMAGE',),
            'seed': ('INT', {'default': 0, 'min': 0, 'max': 0xffffffffffffffff}),
            'steps': ('INT', {'default': 4, 'min': 1, 'max': 4}),
        }}

    @classmethod
    def IS_CHANGED(cls, image, seed, steps):
        try:
            return checkpoint_identity()
        except FileNotFoundError:
            return 'missing-marigold-iid-appearance'

    def estimate(self, image, seed=0, steps=4):
        with loaded_estimator() as estimate:
            return estimate(image, seed=seed, steps=steps)
