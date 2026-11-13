"""Run Base SuperMat through the installed ComfyUI-SuperPBR nodes."""
import inspect
from pathlib import Path

from . import generation, store


NODES = ('SuperPBRModelLoader', 'SuperPBRDecompose', 'SuperPBRTileSplit', 'SuperPBRTileStitch')
WEIGHTS = (
    ('diffusion_models', 'supermat.pth'),
    ('vae', 'vae-ft-mse-840000-ema-pruned.safetensors'),
    ('text_encoders', 'supermat/sd2-clip.safetensors'),
)
TILE = 512
OVERLAP = 128


def missing_components():
    import folder_paths
    from nodes import NODE_CLASS_MAPPINGS

    missing = [f'{folder}/{name}' for folder, name in WEIGHTS if not folder_paths.get_full_path(folder, name)]
    if any(name not in NODE_CLASS_MAPPINGS for name in NODES):
        missing.append('ComfyUI-SuperPBR (model, decomposition, tile split and stitch nodes)')
    return missing


def checkpoint_identity():
    import folder_paths
    from nodes import NODE_CLASS_MAPPINGS

    missing = missing_components()
    if missing:
        raise FileNotFoundError('SuperMat requires: ' + ', '.join(missing))
    identity = {name: generation.fingerprint_file(folder_paths.get_full_path_or_raise(folder, name))
                for folder, name in WEIGHTS}
    wrapper = Path(inspect.getfile(NODE_CLASS_MAPPINGS['SuperPBRDecompose'])).parent
    for name in ('nodes.py', 'model.py', 'tiles.py'):
        identity[name] = store.file_hash(wrapper / name)
    return identity


def load():
    from nodes import CLIPLoader, NODE_CLASS_MAPPINGS, VAELoader

    model, = NODE_CLASS_MAPPINGS['SuperPBRModelLoader']().load(WEIGHTS[0][1])
    vae, = VAELoader().load_vae(WEIGHTS[1][1])
    clip, = CLIPLoader().load_clip(WEIGHTS[2][1], 'stable_diffusion', 'default')
    return model, vae, clip


def estimate(image, model, vae, clip, resolution, metal_threshold=.5):
    import torch
    from comfy.utils import common_upscale
    from nodes import NODE_CLASS_MAPPINGS

    if not 32 <= resolution <= 4096 or resolution % 32:
        raise ValueError('SuperMat output size must be a multiple of 32 from 32 to 4096.')
    if not 0 <= metal_threshold <= 1:
        raise ValueError('SuperMat metalness threshold must be in [0, 1].')
    h, w = image.shape[1:3]
    # Estimate small outputs at training resolution. Preserve the face aspect ratio.
    scale = max(TILE, resolution) / max(h, w)
    height, width = max(1, round(h * scale)), max(1, round(w * scale))
    decompose = NODE_CLASS_MAPPINGS['SuperPBRDecompose']().decompose
    with torch.inference_mode():
        if (h, w) != (height, width):
            image = common_upscale(image.movedim(-1, 1), width, height, 'lanczos', 'disabled').movedim(1, -1)
        if resolution <= TILE:
            square = image.new_full((1, TILE, TILE, 3), .5)
            y, x = (TILE - height) // 2, (TILE - width) // 2
            square[:, y:y + height, x:x + width] = image
            albedo, orm = decompose(square, model, vae, TILE, batch_chunk=1, clip=clip)
            albedo = albedo[:, y:y + height, x:x + width]
            orm = orm[:, y:y + height, x:x + width]
            if resolution < TILE:
                dst_w = max(1, round(w * resolution / max(h, w)))
                dst_h = max(1, round(h * resolution / max(h, w)))
                albedo, orm = (common_upscale(out.movedim(-1, 1), dst_w, dst_h, 'area', 'disabled').movedim(1, -1)
                               for out in (albedo, orm))
        else:
            split = NODE_CLASS_MAPPINGS['SuperPBRTileSplit']().split
            stitch = NODE_CLASS_MAPPINGS['SuperPBRTileStitch']().stitch
            tiles, _, anchor, grid = split(image, TILE, OVERLAP)
            base_tiles, orm_tiles = decompose(tiles, model, vae, TILE, batch_chunk=2, clip=clip,
                                              anchor_image=anchor, tilesig=grid)
            albedo, orm = stitch(base_tiles, orm_tiles, grid)
        if albedo.shape != orm.shape or albedo.ndim != 4 or albedo.shape[0] != 1 or albedo.shape[-1] != 3:
            raise ValueError('SuperMat returned incompatible material map dimensions.')
        if not albedo.isfinite().all() or not orm.isfinite().all():
            raise ValueError('SuperMat produced non-finite material values.')
        # The trained material target encodes roughness in G and metalness in B.
        # RM packing in the atlas writer keeps the existing ambient occlusion.
        roughness = orm[..., 1].clamp(0, 1)
        rma = torch.stack((roughness, (orm[..., 2] >= metal_threshold).to(roughness.dtype), torch.ones_like(roughness)), dim=-1)
        return albedo.clamp(0, 1), rma
