"""Prepare immutable full-body inputs and a local HF wrapper for existing DINO weights."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--coarse", required=True, type=Path)
    parser.add_argument("--dino-weights", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    import torch
    import trimesh
    from safetensors import safe_open
    from safetensors.torch import load_file
    from transformers import DINOv3ViTConfig, DINOv3ViTModel

    reference = args.output / "source-original.png"
    coarse = args.output / "coarse-original.glb"
    shutil.copy2(args.reference, reference)
    shutil.copy2(args.coarse, coarse)
    assert sha256(reference) == "9d92ad97b34bb949b3aa555afed321b8dae427c0102c862b2afb96745b93390b"
    assert sha256(coarse) == "8aac16b2d7125bc41f76990eb414c46dbd2e5d664f89267791d0151a3113a0b3"
    dino_digest = sha256(args.dino_weights)
    assert dino_digest == "a29ef35101a16966972a0d50732a6f3a608ff7cfffb2afa9bbe9007cb842cc53"

    # Architecture matches the already installed TripoSplat DINOv3 ViT-H/16+.
    # Keep the original safetensors bytes; only provide the HF configuration.
    config = DINOv3ViTConfig(
        hidden_size=1280, intermediate_size=5120, num_hidden_layers=32,
        num_attention_heads=20, num_register_tokens=4, num_channels=3,
        patch_size=16, layer_norm_eps=1e-5, rope_theta=100.0,
        use_gated_mlp=True, hidden_act="silu", query_bias=True,
        key_bias=False, value_bias=True, proj_bias=True, mlp_bias=True,
        image_size=1024, architectures=["DINOv3ViTModel"],
    )
    encoder = args.output / "dinov3-vith-local"
    encoder.mkdir()
    config.save_pretrained(encoder)
    (encoder / "model.safetensors").symlink_to(args.dino_weights.resolve())
    with torch.device("meta"):
        probe = DINOv3ViTModel(config)
    expected = {key: list(value.shape) for key, value in probe.state_dict().items()}
    with safe_open(args.dino_weights, framework="pt", device="cpu") as weights:
        actual = {key: weights.get_slice(key).get_shape() for key in weights.keys()}
    assert actual == expected, {
        "missing": sorted(set(expected) - set(actual)),
        "unexpected": sorted(set(actual) - set(expected)),
        "shapeMismatch": [key for key in expected.keys() & actual.keys() if expected[key] != actual[key]],
    }
    state = load_file(args.dino_weights, device="cpu")
    loaded = probe.load_state_dict(state, strict=True, assign=True)
    assert not loaded.missing_keys and not loaded.unexpected_keys
    del state, probe

    mesh = trimesh.load(coarse, force="mesh", process=False)
    bounds = mesh.bounds
    center = bounds.mean(axis=0)
    extent = float((bounds[1] - bounds[0]).max())
    result = {
        "reference": {"path": str(reference), "sha256": sha256(reference)},
        "coarse": {
            "path": str(coarse), "sha256": sha256(coarse),
            "vertices": len(mesh.vertices), "faces": len(mesh.faces),
            "bounds": bounds.tolist(), "watertight": bool(mesh.is_watertight),
            "provenance": "Pixal3D int8 with TRELLIS.2 shape/texture VAE; original pre-Hi3D mesh",
            "coordinates": "Y-up, front +Z, no additional rotation",
        },
        "normalization": {
            "center": center.tolist(), "inputToCenteredScale": 1.9 / extent,
            "inputToVoxelSpaceScale": 0.95 / extent,
            "nativeToInputScale": extent / 0.95,
            "source": "Direct3D-V3 direct3d/utils/voxelization.py mesh_patches",
            "note": "voxelize additionally multiplies by 0.5; sparse_mc returns coordinates in [-0.5, 0.5]. Additional token-cap scaling is measured during inference.",
        },
        "imageEncoder": {
            "directory": str(encoder), "weights": str(args.dino_weights.resolve()),
            "sha256": dino_digest, "stateDictStrictLoad": True,
            "tensorCount": len(actual), "parameters": sum(math.prod(shape) for shape in actual.values()),
            "source": "https://huggingface.co/VAST-AI/TripoSplat/resolve/56a96e603204ec410c4da60c13ea4fa09a2169a9/clip_vision/dino_v3_vit_h.safetensors",
            "configSource": "ComfyUI comfy/image_encoders/dino3.py DINOV3_VITH_CONFIG; Transformers 4.57.6",
            "weightsModified": False,
        },
    }
    (args.output / "prepared.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
