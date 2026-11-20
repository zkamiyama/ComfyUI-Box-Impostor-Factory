#!/usr/bin/env python3
"""Reproducible ZipSplat prediction and matched-resolution TokenBA experiments.

Requires the inference package from cvg/ZipSplat commit
73ab63f048fcc15ee54ada4ef9cdd79b925611a8 (or an API-compatible version).
The predictor is used unchanged. Low-memory weight casting follows demo/app.py.
Refinement reloads only the Gaussian head, never the 6 GB backbone checkpoint.

Examples:
  python run_zipsplat_experiment.py predict --manifest views.json --out smoke/initial
  python run_zipsplat_experiment.py refine --initial smoke/initial/prediction.pt \
      --long-edge 504 --stock-ba --out smoke/ba504
  python run_zipsplat_experiment.py refine --initial full/initial/prediction.pt \
      --long-edge 1008 --steps 96 --rounds 4 --out full/ba1008

Manifest: {"views": [{"path": "/absolute/view.png", "id": "az000", ...}, ...]}.
All extra view fields are retained. Relative paths resolve against the manifest.
No camera poses from prompts are supplied to the model. A checkpoint contains plain
tensors and JSON-compatible metadata and is safe to load with weights_only=True.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
import traceback


UPSTREAM_COMMIT = "73ab63f048fcc15ee54ada4ef9cdd79b925611a8"
SCHEMA = 1


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def crop_rectangle(width, height, model_width, model_height):
    """Exact integer crop used by the upstream predictor's prepare_inputs()."""
    if width * model_height > model_width * height:
        crop_width, crop_height = round(height * model_width / model_height), height
    else:
        crop_width, crop_height = width, round(width * model_height / model_width)
    left, top = (width - crop_width) // 2, (height - crop_height) // 2
    return left, top, left + crop_width, top + crop_height


def target_hw(encoder_hw, long_edge):
    h, w = encoder_hw
    factor = long_edge / max(h, w)
    return max(1, round(h * factor)), max(1, round(w * factor))


def load_manifest(path, limit=None):
    from PIL import Image

    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    views = data["views"] if isinstance(data, dict) else data
    if limit:
        views = views[:limit]
    if not views:
        raise ValueError("Manifest contains no views")
    out = []
    for i, row in enumerate(views):
        row = {"path": row} if isinstance(row, str) else dict(row)
        p = Path(row["path"])
        p = p if p.is_absolute() else path.parent / p
        p = p.resolve(strict=True)
        digest = sha256(p)
        if row.get("sha256") not in (None, digest):
            raise ValueError(f"Input hash changed: {p}")
        with Image.open(p) as image:
            width, height = image.size
            mode = image.mode
        row.update(path=str(p), sha256=digest, width=width, height=height, mode=mode)
        row.setdefault("id", f"view_{i:03d}")
        out.append(row)
    hashes = [v["sha256"] for v in out]
    if len(set(hashes)) != len(hashes):
        raise ValueError("Duplicate image bytes in view set; use distinct source views")
    return {"source_manifest": str(path), "source_manifest_sha256": sha256(path), "views": out}


def verify_inputs(manifest):
    for row in manifest["views"]:
        if sha256(row["path"]) != row["sha256"]:
            raise ValueError(f"Input hash changed: {row['path']}")


def pack_prediction(pred):
    import torch

    out = {}
    for name, value in pred.items():
        if isinstance(value, torch.Tensor):
            out[name] = {"type": "Tensor", "data": value.detach().cpu()}
        elif hasattr(value, "data_"):
            out[name] = {"type": type(value).__name__, "data": value.data_.detach().cpu()}
        else:
            raise TypeError(f"Unexpected prediction value {name}: {type(value)}")
    return out


def unpack_prediction(data, device="cpu", exclude=()):
    from zipsplat import Camera, Pose
    from zipsplat.gaussians import Gaussians

    wrappers = {"Camera": Camera, "Pose": Pose, "Gaussians": Gaussians}
    out = {}
    for name, row in data.items():
        if name in exclude:
            continue
        tensor = row["data"].to(device)
        out[name] = tensor if row["type"] == "Tensor" else wrappers[row["type"]](tensor)
    return out


def save_checkpoint(path, payload):
    import torch

    temp = Path(path).with_name(Path(path).name + ".tmp")
    torch.save(payload, temp)
    os.replace(temp, path)


def bf16_weights(model):
    """Official demo casting; preserve Gaussian head and backbone.cam_dec as fp32."""
    import torch

    for name, module in model.model.named_children():
        if name == "backbone":
            module.backbone.to(torch.bfloat16)
            module.cam_enc.to(torch.bfloat16)
        elif name != "gaussian_head":
            module.to(torch.bfloat16)
    return model


def head_payload(head):
    return {
        "embed_dim": int(head.gaussian_head[1].in_features),
        "gaussians_per_token": int(head.gaussians_per_token),
        "sh_degree": math.isqrt(int(head.num_sh_coeffs)) - 1,
        "state_dict": {k: v.detach().cpu() for k, v in head.state_dict().items()},
    }


def load_head(data, device):
    from zipsplat.heads import GaussianHead

    head = GaussianHead(data["embed_dim"], data["gaussians_per_token"], data["sh_degree"])
    head.load_state_dict(data["state_dict"])
    return head.to(device).eval().requires_grad_(False)


def images_at_resolution(manifest, pred, long_edge):
    """Reuse the exact encoder crop, with K scaled in the same pixel coordinates."""
    import torch
    from zipsplat import load_image
    from zipsplat.utils import resize_image

    enc_h, enc_w = pred["model_images"].shape[-2:]
    h, w = target_hw((enc_h, enc_w), long_edge)
    crop_records, images = [], []
    for row in manifest["views"]:
        box = crop_rectangle(row["width"], row["height"], enc_w, enc_h)
        left, top, right, bottom = box
        crop_records.append({
            "id": row["id"], "xyxy": list(box), "target_wh": [w, h],
            "upsampled": w > right - left or h > bottom - top,
        })
        if (h, w) != (enc_h, enc_w):
            image = load_image(Path(row["path"]))
            images.append(resize_image(image[:, top:bottom, left:right], h, w))
    device = pred["model_images"].device
    images = pred["model_images"] if (h, w) == (enc_h, enc_w) else torch.stack(images).to(device)
    camera = pred["model_cameras"].scale((w / enc_w, h / enc_h))
    return images, camera, crop_records


def camera_report(pred):
    """Diagnostics only: inferred directions are not calibrated ground-truth angles."""
    import numpy as np

    pose = pred["poses"].cpu()
    centers = pose.t.numpy().astype(float)
    rotations = pose.R.numpy().astype(float)
    directions = rotations[:, :, 2]
    dot = np.clip(directions @ directions.T, -1, 1)
    angles = np.degrees(np.arccos(dot))
    near_pairs = [
        [i, j, float(angles[i, j])] for i in range(len(centers)) for j in range(i + 1, len(centers))
        if angles[i, j] < 5
    ]
    projectors = np.eye(3)[None] - directions[:, :, None] * directions[:, None, :]
    target = np.linalg.lstsq(projectors.sum(0), (projectors @ centers[:, :, None]).sum(0)[:, 0], rcond=None)[0]
    rel = (centers - target) @ rotations[0]
    yaw = np.degrees(np.arctan2(rel[:, 0], -rel[:, 2])) % 360
    yaw = (yaw - yaw[0]) % 360
    sorted_yaw = np.sort(yaw)
    gaps = np.diff(np.r_[sorted_yaw, sorted_yaw[0] + 360])
    distances = np.linalg.norm(centers - target, axis=1)
    ray_errors = np.linalg.norm((projectors @ (target - centers)[:, :, None])[:, :, 0], axis=1)
    radius = float(np.median(distances))
    return {
        "convention": "OpenCV c2w; inferred first-view frame, no calibrated external poses",
        "c2w": pose.Rt.tolist(),
        "model_intrinsics": pred["model_cameras"].data_.cpu().tolist(),
        "original_intrinsics": pred["cameras"].data_.cpu().tolist(),
        "forward_direction_pairwise_degrees": angles.tolist(),
        "near_duplicate_direction_pairs_below_5_degrees": near_pairs,
        "fitted_look_target": target.tolist(),
        "inferred_orbit_yaw_degrees_relative_to_first": yaw.tolist(),
        "inferred_orbit_largest_gap_degrees": float(gaps.max()),
        "median_orbit_radius": radius,
        "median_ray_target_error_over_radius": float(np.median(ray_errors) / max(radius, 1e-12)),
        "camera_center_singular_values": np.linalg.svd(centers - centers.mean(0), compute_uv=False).tolist(),
        "warning": "Orbit coverage is diagnostic only; unreliable when rays do not converge on one object.",
    }


def save_png(path, image):
    import numpy as np
    from PIL import Image

    data = image.detach().cpu().float().clamp(0, 1)
    if data.ndim == 3:
        data = data.permute(1, 2, 0)
        if data.shape[-1] == 1:
            data = data[:, :, 0]
    Image.fromarray(np.rint(data.numpy() * 255).astype(np.uint8)).save(path)


def export_result(pred, manifest, out, render_long_edge=504):
    import numpy as np
    import torch

    out = Path(out)
    pred["gaussians"].save_ply(out / "scene.ply")
    np.save(out / "gaussians.npy", pred["gaussians"].data_.detach().cpu().float().numpy())
    write_json(out / "cameras.json", {**camera_report(pred), "views": manifest["views"]})
    render_dir = out / f"renders_{render_long_edge}"
    render_dir.mkdir(exist_ok=True)
    images, camera, crops = images_at_resolution(manifest, pred, render_long_edge)
    metrics = []
    with torch.no_grad():
        for index, row in enumerate(manifest["views"]):
            rgbd, info = pred["gaussians"].render(camera[index], pred["poses"][index], mode="RGB+ED")
            rgb, depth, alpha = rgbd[0, :3], rgbd[0, 3], info["alphas"][0]
            diff = rgb - images[index]
            mse = float(diff.square().mean().item())
            metrics.append({
                "id": row["id"], "l1_full_image": float(diff.abs().mean().item()),
                "psnr_full_image_db": -10 * math.log10(max(mse, 1e-12)),
                "mean_accumulated_alpha": float(alpha.mean().item()),
            })
            save_png(render_dir / f"{index:03d}_rgb.png", rgb)
            save_png(render_dir / f"{index:03d}_alpha.png", alpha)
            save_png(render_dir / f"{index:03d}_target.png", images[index])
            np.save(render_dir / f"{index:03d}_expected_depth.npy", depth.cpu().numpy())
    write_json(render_dir / "metrics.json", {
        "resolution_long_edge": render_long_edge, "views": metrics, "crops": crops,
        "mean_l1_full_image": sum(m["l1_full_image"] for m in metrics) / len(metrics),
        "mean_psnr_full_image_db": sum(m["psnr_full_image_db"] for m in metrics) / len(metrics),
        "notes": "Input-view reconstruction scores only. Alpha is accumulated opacity, not foreground segmentation. Depth is expected camera Z, not nearest surface.",
    })
    return {
        "gaussian_count": pred["gaussians"].num_gaussians,
        "render_metrics": str(render_dir / "metrics.json"),
        "scene_sha256": sha256(out / "scene.ply"),
    }


def environment(device):
    import torch
    import zipsplat

    packages = {}
    for name in ("torch", "torchvision", "gsplat", "zipsplat", "numpy", "triton", "xformers"):
        with contextlib.suppress(importlib.metadata.PackageNotFoundError):
            packages[name] = importlib.metadata.version(name)
    repository = Path(zipsplat.__file__).resolve().parent.parent
    commit = None
    with contextlib.suppress(Exception):
        commit = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    props = torch.cuda.get_device_properties(device)
    free_memory, total_memory = torch.cuda.mem_get_info(device)
    return {
        "python": sys.version, "platform": platform.platform(), "packages": packages,
        "torch_cuda": torch.version.cuda, "gpu": props.name, "gpu_total_bytes": props.total_memory,
        "cuda_free_bytes_at_start": free_memory, "cuda_total_bytes": total_memory,
        "zipsplat_module": str(zipsplat.__file__), "zipsplat_commit": commit,
        "runner_sha256": sha256(__file__),
    }


def weight_provenance(weights):
    import torch
    from urllib.parse import urlparse
    from zipsplat.predictor import _WEIGHTS_URLS

    resolved = _WEIGHTS_URLS.get(weights, weights)
    path = (Path(torch.hub.get_dir()) / "zipsplat" / Path(urlparse(resolved).path).name
            if resolved.startswith(("http://", "https://")) else Path(resolved))
    result = {"requested": weights, "resolved": resolved}
    if path.is_file():
        result.update(path=str(path.resolve()), size_bytes=path.stat().st_size, sha256=sha256(path))
    return result


class Measurement:
    def __init__(self, device):
        self.device = device

    def __enter__(self):
        import torch
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        import torch
        with contextlib.suppress(Exception):
            torch.cuda.synchronize(self.device)
        self.result = {
            "seconds": time.perf_counter() - self.start,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(self.device),
        }


def make_traced_ba(args, out):
    from zipsplat.token_ba import TokenBA

    class TracedBA(TokenBA):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.losses, self.camera_updates = [], 0

        def _step(self, *values, **kwargs):
            loss = super()._step(*values, **kwargs)
            self.losses.append(float(loss.item()))
            if len(self.losses) % 8 == 0:
                write_json(out / "progress.json", {"optimizer_steps": len(self.losses), "loss": self.losses[-1], "camera_updates": self.camera_updates})
            return loss

        def camera_update(self, *values, **kwargs):
            self.camera_updates += 1
            return super().camera_update(*values, **kwargs)

    return TracedBA(
        rounds=args.rounds, joint_steps=args.steps, verbose=True,
        joint_tol=0.02 if args.stock_ba else None,
        round_tol_deg=0.3 if args.stock_ba else -1.0,
    )


def predict(args, out, report):
    import torch
    from zipsplat import ZipSplat, load_image

    manifest = load_manifest(args.manifest, args.limit)
    write_json(out / "inputs.json", manifest)
    with Measurement(args.device) as loading:
        model = ZipSplat(weights=args.weights).eval().requires_grad_(False)
        if not args.fp32_weights:
            bf16_weights(model)
        model = model.to(args.device)
    report["model_loading"] = loading.result
    report["weights"] = weight_provenance(args.weights)
    images = [load_image(Path(row["path"])) for row in manifest["views"]]
    with Measurement(args.device) as inference:
        pred = model(images, compression=args.compression, ba=False)
    report["feedforward"] = inference.result
    payload = {
        "schema": SCHEMA, "inputs": manifest, "settings": report["settings"],
        "prediction": pack_prediction(pred), "head": head_payload(model.model.gaussian_head),
        "environment": report["environment"],
        "weights": report["weights"],
    }
    save_checkpoint(out / "prediction.pt", payload)
    # Free backbone before rasterizer compilation / diagnostic export.
    del model, images
    torch.cuda.empty_cache()
    with Measurement(args.device) as exporting:
        report["artifacts"] = export_result(pred, manifest, out, args.render_long_edge)
    report["export"] = exporting.result
    report["checkpoint_sha256"] = sha256(out / "prediction.pt")
    report["checkpoint_path"] = str(out / "prediction.pt")


def refine(args, out, report):
    import torch
    from zipsplat.utils import remove_batch_dim

    initial = Path(args.initial).resolve(strict=True)
    payload = torch.load(initial, map_location="cpu", weights_only=True)
    if payload["schema"] != SCHEMA:
        raise ValueError(f"Unsupported checkpoint schema: {payload['schema']}")
    verify_inputs(payload["inputs"])
    write_json(out / "inputs.json", payload["inputs"])
    pred = unpack_prediction(payload["prediction"], args.device, exclude=("gaussians",))
    head = load_head(payload["head"], args.device)
    images, cameras, crops = images_at_resolution(payload["inputs"], pred, args.long_edge)
    write_json(out / "supervision.json", {"crops": crops, "camera_data": cameras.data_.cpu().tolist(), "encoder_hw": list(pred["model_images"].shape[-2:])})
    ba = make_traced_ba(args, out)
    report["initial_checkpoint"] = {"path": str(initial), "sha256": sha256(initial)}
    with Measurement(args.device) as refinement:
        updated = ba(
            scene_tokens=pred["scene_tokens"][None], head=head,
            camera=cameras[None], pose=pred["poses"][None], image=images[None],
            assignments=pred["assignments"][None],
            encoder_camera=pred["model_cameras"][None],
            encoder_hw=pred["model_images"].shape[-2:],
        )
    report["refinement"] = refinement.result
    report["refinement"].update(optimizer_steps=len(ba.losses), camera_updates=ba.camera_updates)
    write_json(out / "losses.json", {"losses": ba.losses, "camera_updates": ba.camera_updates})
    pred = {**pred, **remove_batch_dim(updated)}
    payload["prediction"] = pack_prediction(pred)
    payload["parent_checkpoint_sha256"] = report["initial_checkpoint"]["sha256"]
    payload["refinement_settings"] = report["settings"]
    save_checkpoint(out / "prediction.pt", payload)
    del head, images, updated
    torch.cuda.empty_cache()
    with Measurement(args.device) as exporting:
        report["artifacts"] = export_result(pred, payload["inputs"], out, args.render_long_edge)
    report["export"] = exporting.result
    report["checkpoint_sha256"] = sha256(out / "prediction.pt")
    report["checkpoint_path"] = str(out / "prediction.pt")


def render(args, out, report):
    """Recover diagnostic exports without re-running prediction or refinement."""
    import torch

    checkpoint = Path(args.checkpoint).resolve(strict=True)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload["schema"] != SCHEMA:
        raise ValueError(f"Unsupported checkpoint schema: {payload['schema']}")
    verify_inputs(payload["inputs"])
    write_json(out / "inputs.json", payload["inputs"])
    pred = unpack_prediction(payload["prediction"], args.device)
    with Measurement(args.device) as exporting:
        report["artifacts"] = export_result(pred, payload["inputs"], out, args.render_long_edge)
    report["export"] = exporting.result
    report["checkpoint_sha256"] = sha256(checkpoint)
    report["checkpoint_path"] = str(checkpoint)


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = p.add_subparsers(dest="command", required=True)
    for name in ("predict", "refine", "render"):
        command = commands.add_parser(name)
        command.add_argument("--out", required=True)
        command.add_argument("--device", default="cuda:0")
        command.add_argument("--seed", type=int, default=20261120)
        command.add_argument("--render-long-edge", type=int, default=504)
        if name == "predict":
            command.add_argument("--manifest", required=True)
            command.add_argument("--limit", type=int)
            command.add_argument("--weights", default="zipsplat")
            command.add_argument("--compression", type=float, default=1.0)
            command.add_argument("--fp32-weights", action="store_true")
        elif name == "refine":
            command.add_argument("--initial", required=True)
            command.add_argument("--long-edge", type=int, default=504)
            command.add_argument("--steps", type=int, default=96)
            command.add_argument("--rounds", type=int, default=4)
            command.add_argument("--stock-ba", action="store_true", help="Use stock early stopping; omit for matched-budget resolution comparisons")
        else:
            command.add_argument("--checkpoint", required=True)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    import numpy as np
    import torch

    if args.command == "predict" and not 0 < args.compression <= 1:
        raise ValueError("Compression must be in (0, 1]")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment")
    torch.cuda.set_device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    settings = vars(args).copy()
    settings["out"] = str(out)
    if args.command == "predict":
        settings["manifest_sha256"] = sha256(args.manifest)
    else:
        settings["initial_sha256"] = sha256(args.initial if args.command == "refine" else args.checkpoint)
    report = {
        "status": "running", "settings": settings, "environment": environment(args.device),
        "measurement_notes": {
            "stage_seconds": "Wall time around the named stage with CUDA synchronization; includes CPU work and any first-use compilation inside that stage, not GPU kernel time alone.",
            "refinement_seconds": "TokenBA only, including its progress logging; excludes loading the saved head, checkpoint saving, diagnostic exports, Qwen generation and the initial ZipSplat prediction.",
            "total_seconds": "Stage-dispatch wall time including checkpoint and diagnostic export work; excludes Python imports, CLI/environment setup, initial identity checks and the final JSON report write.",
            "peak_allocated_bytes": "Maximum live tensors tracked by this process's PyTorch CUDA allocator during the named stage, including tensors resident before that stage. Not total device VRAM or nvidia-smi process memory.",
            "peak_reserved_bytes": "Maximum allocator pool reserved by this process's PyTorch CUDA allocator during the named stage. CUDA context, allocations outside this allocator and other processes are not included.",
            "byte_units": "Raw bytes; divide by 1024**3 for GiB.",
        },
    }
    complete = out / "complete.json"
    if complete.exists():
        previous = json.loads(complete.read_text(encoding="utf-8"))
        if previous["settings"] != settings:
            raise RuntimeError("Output already contains a different completed experiment; choose a new directory")
        checkpoint = Path(previous.get("checkpoint_path", out / "prediction.pt"))
        if sha256(checkpoint) != previous["checkpoint_sha256"]:
            raise RuntimeError("Completed checkpoint hash changed")
        verify_inputs(torch.load(checkpoint, map_location="cpu", weights_only=True)["inputs"])
        print(json.dumps({"status": "already_complete", "path": str(complete)}), flush=True)
        return 0
    run_path = out / "run.json"
    if run_path.exists():
        prior = json.loads(run_path.read_text(encoding="utf-8"))
        if prior["settings"] != settings:
            raise RuntimeError("Output contains a different experiment; choose a new directory")
    write_json(run_path, report)
    start = time.perf_counter()
    try:
        {"predict": predict, "refine": refine, "render": render}[args.command](args, out, report)
        report.update(status="complete", total_seconds=time.perf_counter() - start)
        write_json(complete, report)
        write_json(run_path, report)
        print(json.dumps(report, ensure_ascii=False), flush=True)
        return 0
    except Exception as error:
        report.update(status="failed", total_seconds=time.perf_counter() - start,
                      error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        write_json(run_path, report)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
