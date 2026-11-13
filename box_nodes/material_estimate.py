"""Image-conditioned material estimation for one accepted Box face."""

import math


BACKENDS = ('qwen_enhanced', 'h3_enhanced')
_MAX_SEED = (1 << 64) - 1
_BASE_COLOR = (
    'Produce a flat, unlit PBR base-color (albedo) map. Preserve the original pigments, '
    'material boundaries, labels and markings. Remove shadows, highlights, reflections '
    'and ambient occlusion. Keep colors as sRGB. Output only the aligned texture.'
)
_RMA = (
    'Produce a channel-packed PBR data texture. Each pixel independently stores '
    'roughness in red (0 smooth, 1 rough), metalness in green (0 dielectric, 1 metal), '
    'and ambient occlusion in blue (1 unoccluded, lower only in crevices). Use flat '
    'numeric values, not shaded colors. Keep material boundaries aligned to the source. '
    'Do not include the original colors, captions, legends, panels or swatches.'
)


def _reference_image(image, name):
    if image.ndim != 4 or image.shape[0] != 1 or image.shape[-1] not in (3, 4):
        raise ValueError(f'{name} must be one RGB or RGBA face image')
    if min(image.shape[1:3]) < 1:
        raise ValueError(f'{name} has empty dimensions')
    rgb = image[..., :3]
    if image.shape[-1] == 4:
        alpha = image[..., 3:4]
        rgb = rgb * alpha + 1 - alpha
    return rgb


class MaterialEstimate:
    CATEGORY = 'Box Impostor'
    FUNCTION = 'estimate'
    RETURN_TYPES = ('IMAGE', 'IMAGE')
    RETURN_NAMES = ('base_color', 'rma')
    DESCRIPTION = ('Estimate aligned base color and roughness/metalness/AO from one generated face '
                   'with a Qwen-Image-2.1 or MiniMax-H3-Image editing model. Connect its base '
                   'MODEL, CLIP and VAE. RMA channels store scalar data in R/G/B order.')

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'image': ('IMAGE',),
                'model': ('MODEL',),
                'clip': ('CLIP',),
                'vae': ('VAE',),
                'backend': (list(BACKENDS),),
                'seed': ('INT', {'default': 0, 'min': 0, 'max': _MAX_SEED}),
            },
            'optional': {
                'guide': ('IMAGE',),
                'steps': ('INT', {'default': 0, 'min': 0, 'max': 100, 'advanced': True,
                                  'tooltip': '0 selects 24 steps for Qwen or 6 for H3.'}),
                'cfg': ('FLOAT', {'default': -1.0, 'min': -1.0, 'max': 30.0, 'step': 0.1, 'advanced': True,
                                  'tooltip': '-1 selects guidance 3 for Qwen or 1 for H3.'}),
                'resolution': ('INT', {'default': 2048, 'min': 0, 'max': 4096, 'step': 32,
                                      'tooltip': 'Target area at the face aspect ratio; 0 uses its original size rounded to multiples of 32.'}),
            },
        }

    def estimate(self, image, model, clip, vae, backend, seed, guide=None, resolution=2048,
                 steps=0, cfg=-1.0):
        if backend not in BACKENDS:
            raise ValueError(f'Unknown material estimation backend: {backend}')
        if not 0 <= resolution <= 4096:
            raise ValueError('Material resolution must be between 0 and 4096')
        source = _reference_image(image, 'image')
        height, width = source.shape[1:3]
        scale = resolution / math.sqrt(width * height) if resolution else 1
        width = max(32, round(width * scale / 32) * 32)
        height = max(32, round(height * scale / 32) * 32)
        references = {'image_1': source}
        qwen = backend == 'qwen_enhanced'
        if not 0 <= steps <= 100 or not (cfg == -1 or 0 <= cfg <= 30):
            raise ValueError('Invalid material sampling steps or CFG')
        sampling_steps = steps or (24 if qwen else 6)
        guidance = cfg if cfg >= 0 else (3.0 if qwen else 1.0)
        source_label, guide_label = ('<image1>', '<image2>') if qwen else ('<Picture 1>', '<Picture 2>')
        instruction = (f'Edit {source_label}. Match its camera, framing, silhouette and position '
                       'of every feature exactly. ')
        if guide is not None:
            references['image_2'] = _reference_image(guide, 'guide')
            instruction += f'Use {guide_label} only as a geometry guide for the same view. '

        from nodes import ConditioningZeroOut, VAEDecode, common_ksampler
        if qwen:
            from comfy_extras.nodes_qwen import TextEncodeQwenImage21
        else:
            from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageConditioning

        def render(task, run_seed):
            prompt = instruction + task
            if qwen:
                positive, negative, latent = TextEncodeQwenImage21.execute(
                    clip, prompt, '', vae=vae, resolution=resolution, images=references).result
            else:
                positive, latent = MiniMaxH3ImageConditioning.execute(
                    clip, prompt, width, height, vae=vae, images=references).result
                negative, = ConditioningZeroOut().zero_out(positive)
            samples, = common_ksampler(model, run_seed, sampling_steps, guidance, 'euler', 'simple',
                                       positive, negative, latent)
            decoded, = VAEDecode().decode(vae, samples)
            if decoded.shape[:3] != (1, height, width) or decoded.shape[-1] < 3:
                raise ValueError(f'{backend} returned an unexpected material texture shape {tuple(decoded.shape)}')
            rgb = decoded[..., :3]
            if not rgb.isfinite().all():
                raise ValueError(f'{backend} produced non-finite material pixels')
            return rgb.clamp(0, 1)

        return render(_BASE_COLOR, seed), render(_RMA, (seed + 1) & _MAX_SEED)
