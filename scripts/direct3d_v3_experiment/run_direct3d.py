"""Run a pinned Direct3D-V3 single stage and record full-body geometry and GPU measurements."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--coarse", type=Path)
    parser.add_argument("--background-removed", action="store_true")
    parser.add_argument("--resolution", type=int, choices=(512, 1024, 1536), default=512)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--cfg", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=46)
    parser.add_argument("--max-tokens", type=int, default=150000)
    parser.add_argument("--target-faces", type=int, default=2000000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    prepared = json.loads(args.prepared.read_text())
    image_path = args.image or Path(prepared["reference"]["path"])
    coarse_path = args.coarse or Path(prepared["coarse"]["path"])
    checkpoint = args.model / f"shape_transformers_{args.resolution}/model.safetensors"
    sys.path.insert(0, str(args.repo))
    import numpy as np
    import torch
    import trimesh
    from PIL import Image
    from direct3d import Direct3DV3Pipeline
    import direct3d.utils.voxelization as voxelization

    source_commit = subprocess.check_output(["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True).strip()
    result = {
        "status": "running", "sourceCommit": source_commit,
        "modelRevision": "1f42e11da6efbc8cc0661c62132b0533c338e3d3",
        "modelDirectory": str(args.model), "checkpoint": str(checkpoint),
        "inputImage": {"path": str(image_path), "sha256": sha256(image_path)},
        "coarseMesh": {"path": str(coarse_path), "sha256": sha256(coarse_path)},
        "imageEncoder": prepared["imageEncoder"],
        "settings": {"resolution": args.resolution, "steps": args.steps,
                     "guidanceScale": args.cfg, "seed": args.seed, "maxTokens": args.max_tokens,
                     "targetFaces": args.target_faces, "dtype": "float16", "singleStage": True},
        "packages": {name: importlib.metadata.version(name) for name in
                     ["torch", "torchvision", "transformers", "flash-attn", "trimesh", "scikit-image", "safetensors"]},
        "gpu": torch.cuda.get_device_name(0), "voxelizations": [], "samplingProgress": [],
    }
    result_path = args.output / "result.json"

    def persist() -> None:
        result_path.write_text(json.dumps(result, indent=2) + "\n")

    persist()
    gpu_stream = (args.output / "gpu.csv").open("w")
    monitor = subprocess.Popen(
        ["nvidia-smi", "--query-gpu=timestamp,memory.used,utilization.gpu", "--format=csv,noheader,nounits", "-lms", "200"],
        stdout=gpu_stream, stderr=subprocess.DEVNULL,
    )
    original_voxelize = voxelization.voxelize

    def observed_voxelize(vertices, faces, dim, patch_size):
        tick = time.perf_counter()
        coordinates = original_voxelize(vertices, faces, dim, patch_size)
        torch.cuda.synchronize()
        bounds = torch.stack((vertices.amin(0), vertices.amax(0))).cpu().tolist()
        measurement = {"resolution": dim, "patchSize": patch_size, "tokens": len(coordinates),
                       "centeredBounds": bounds, "seconds": time.perf_counter() - tick}
        result["voxelizations"].append(measurement)
        print(json.dumps({"voxelization": measurement}), flush=True)
        return coordinates

    def on_step(stage, completed, total):
        item = {"stage": stage, "completed": completed, "total": total, "elapsedSeconds": time.perf_counter() - start}
        result["samplingProgress"].append(item)
        print(json.dumps({"sampling": item}), flush=True)

    voxelization.voxelize = observed_voxelize
    try:
        torch.cuda.reset_peak_memory_stats()
        tick = time.perf_counter()
        pipeline = Direct3DV3Pipeline.from_single_file(
            checkpoint, resolution=args.resolution, config_repo=args.model,
            image_encoder=prepared["imageEncoder"]["directory"], torch_dtype=torch.float16,
        ).to("cuda")
        torch.cuda.synchronize()
        result["modelLoadSeconds"] = time.perf_counter() - tick
        print(json.dumps({"modelLoadedSeconds": result["modelLoadSeconds"]}), flush=True)

        tick = time.perf_counter()
        if args.background_removed:
            foreground = Image.open(image_path).copy()
            assert foreground.mode == "RGBA"
        else:
            foreground = pipeline.remove_background(image_path)
        foreground.save(args.output / "foreground.png")
        result["foregroundSeconds"] = time.perf_counter() - tick
        result["foregroundSha256"] = sha256(args.output / "foreground.png")

        tick = time.perf_counter()
        mesh = pipeline(
            image=foreground, coarse_mesh=coarse_path, resolution=args.resolution,
            num_inference_steps=args.steps, guidance_scale=args.cfg, seed=args.seed,
            max_tokens=args.max_tokens, target_faces=args.target_faces,
            background_removed=True, on_sampling_step=on_step,
        )
        torch.cuda.synchronize()
        result["generationSeconds"] = time.perf_counter() - tick
        raw = args.output / "generated-native.glb"
        mesh.export(raw)

        # Exact inverse of mesh centering + 1.9/extent and voxelize's additional 0.5.
        # get_mesh maps grid coordinates back to [-0.5, 0.5]. Account for token-cap shrinking.
        coarse = trimesh.load(coarse_path, force="mesh", process=False)
        center = coarse.bounds.mean(axis=0)
        extent = float((coarse.bounds[1] - coarse.bounds[0]).max())
        last_bounds = np.asarray(result["voxelizations"][-1]["centeredBounds"])
        token_scale = float((last_bounds[1] - last_bounds[0]).max() / 1.9)
        inverse_scale = extent / (0.95 * token_scale)
        canonical = mesh.copy()
        canonical.vertices = np.asarray(mesh.vertices) * inverse_scale + center
        canonical_path = args.output / "canonical.glb"
        canonical.export(canonical_path)
        assert np.isfinite(canonical.vertices).all() and len(canonical.faces) > 0
        result["normalization"] = {"center": center.tolist(), "inputExtent": extent,
                                   "tokenScale": token_scale, "nativeToInputScale": inverse_scale,
                                   "noAdditionalRotation": True}
        result["output"] = {
            "raw": str(raw), "rawSha256": sha256(raw),
            "canonical": str(canonical_path), "canonicalSha256": sha256(canonical_path),
            "vertices": len(canonical.vertices), "faces": len(canonical.faces),
            "bounds": canonical.bounds.tolist(), "watertight": bool(canonical.is_watertight),
            "appearance": "Geometry only; Direct3D-V3 texture generation is not released.",
        }
        result["status"] = "success"
    except Exception as error:
        result["status"] = "error"
        result["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        raise
    finally:
        voxelization.voxelize = original_voxelize
        monitor.terminate()
        monitor.wait(timeout=10)
        gpu_stream.close()
        values = []
        for line in (args.output / "gpu.csv").read_text().splitlines():
            columns = line.split(",")
            if len(columns) >= 3:
                try:
                    values.append(float(columns[1].strip()))
                except ValueError:
                    pass
        result["gpuMemory"] = {
            "sampledWholeDevicePeakMiB": max(values) if values else None,
            "sampleCount": len(values), "sampleIntervalMilliseconds": 200,
            "torchMaxAllocatedMiB": torch.cuda.max_memory_allocated() / 1024**2,
            "torchMaxReservedMiB": torch.cuda.max_memory_reserved() / 1024**2,
            "note": "nvidia-smi includes other processes and can miss shorter peaks; torch counters cover this process.",
        }
        result["wallSeconds"] = time.perf_counter() - start
        persist()
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
