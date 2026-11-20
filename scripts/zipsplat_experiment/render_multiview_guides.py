#!/usr/bin/env python3
"""Render a fixed perspective ring of full-object TripoSplat guides via ComfyUI."""
import argparse
import hashlib
import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def cameras(bounds, azimuths, elevation, fov, fill):
    low, high = np.array(bounds["min"]), np.array(bounds["max"])
    center = (low + high) / 2
    corners = np.array(list(itertools.product(*zip(low, high)))) - center
    half_tan = math.tan(math.radians(fov) / 2) * fill
    ring = []
    distance = 0.0
    for azimuth in azimuths:
        yaw, pitch = math.radians(azimuth), math.radians(elevation)
        back = np.array([math.sin(yaw) * math.cos(pitch), math.sin(pitch),
                         math.cos(yaw) * math.cos(pitch)])
        right = np.array([math.cos(yaw), 0.0, -math.sin(yaw)])
        up = np.cross(back, right)
        required = corners @ back + np.maximum(np.abs(corners @ right),
                                               np.abs(corners @ up)) / half_tan
        distance = max(distance, float(required.max()))
        ring.append((azimuth, back))
    results = []
    for azimuth, back in ring:
        position = center + back * distance
        camera = {"cameraType": "perspective", "fov": fov, "zoom": 1.0,
                  "position": dict(zip("xyz", map(float, position))),
                  "target": dict(zip("xyz", map(float, center)))}
        results.append({"id": f"yaw{azimuth % 360:03g}", "azimuth_degrees": azimuth,
                        "elevation_degrees": elevation, "camera": camera})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy", type=Path, required=True)
    parser.add_argument("--splat", type=Path, required=True)
    parser.add_argument("--box-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--azimuths", default="0,45,90,135,180,225,270,315")
    parser.add_argument("--elevation", type=float, default=0.0)
    parser.add_argument("--fov", type=float, default=35.0)
    parser.add_argument("--fill", type=float, default=0.88)
    parser.add_argument("--splat-yaw", type=float, default=90.0)
    parser.add_argument("--background", default="#808080")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.box_manifest.read_text())
    azimuths = [float(value) for value in args.azimuths.split(",")]
    views = cameras(manifest["bakeBox"], azimuths, args.elevation, args.fov, args.fill)
    plan = {"method": "native_RenderSplat_perspective", "splat": str(args.splat.resolve()),
            "splat_sha256": digest(args.splat), "box_manifest": str(args.box_manifest.resolve()),
            "box_manifest_sha256": digest(args.box_manifest), "framing_bounds": manifest["bakeBox"],
            "splat_yaw_native_degrees": args.splat_yaw, "size": args.size,
            "background": args.background, "fill": args.fill, "views": views}
    args.output.mkdir(parents=True, exist_ok=True)
    plan_path = args.output / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Use a new directory for changed guide settings")
    write(plan_path, plan)
    if args.plan_only:
        print(json.dumps(plan, indent=2))
        return
    sys.path.insert(0, str(args.comfy))
    # Native modules read Comfy arguments on import, not this script's options.
    sys.argv = [sys.argv[0]]
    import torch
    from comfy_api.latest import Types
    from comfy_extras.nodes_gaussian_splat import RenderSplat, TransformSplat

    with np.load(args.splat, allow_pickle=False) as data:
        splat = Types.SPLAT(*[torch.from_numpy(data[key].copy()) for key in
                             ("positions", "scales", "rotations", "opacities", "sh")])
    splat = TransformSplat.execute(splat=splat, translate_x=0.0, translate_y=0.0,
                                  translate_z=0.0, rotate_x=0.0, rotate_y=args.splat_yaw,
                                  rotate_z=0.0, scale_x=1.0, scale_y=1.0, scale_z=1.0).result[0]
    results = []
    for view in views:
        work = args.output / view["id"]
        work.mkdir(exist_ok=True)
        result_path = work / "result.json"
        if result_path.exists():
            result = json.loads(result_path.read_text())
            if digest(result["image"]) != result["image_sha256"]:
                raise ValueError("Saved guide changed: " + result["image"])
        else:
            start = time.monotonic()
            image_batch, mask_batch = RenderSplat.execute(
                splat=splat, width=args.size, height=args.size, frames=1, splat_scale=1.0,
                sharpen=1.0, headlight_shading=0.0, opacity_threshold=0.0,
                background=args.background, render_style="color", camera_info=view["camera"]).result
            rgb = image_batch[0].detach().float().cpu().numpy()
            alpha = mask_batch[0].detach().float().cpu().numpy()
            # RenderSplat already composites the RGB over the requested background.
            image_path = work / "color.png"
            mask_path = work / "alpha.png"
            Image.fromarray(np.uint8(np.clip(rgb * 255 + 0.5, 0, 255))).save(image_path)
            Image.fromarray(np.uint8(np.clip(alpha * 255 + 0.5, 0, 255))).save(mask_path)
            occupied = alpha > 0.5
            ys, xs = np.nonzero(occupied)
            bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1] if xs.size else None
            result = {**view, "image": str(image_path.resolve()), "mask": str(mask_path.resolve()),
                      "image_sha256": digest(image_path), "mask_sha256": digest(mask_path),
                      "seconds": round(time.monotonic() - start, 3), "foreground_bbox": bbox,
                      "foreground_pixels": int(occupied.sum()),
                      "touches_image_edge": bool(occupied[0].any() or occupied[-1].any()
                                                 or occupied[:, 0].any() or occupied[:, -1].any())}
            write(result_path, result)
        results.append(result)
        write(args.output / "manifest.json", {"plan": str(plan_path.resolve()), "views": results,
              "complete": len(results) == len(views)})
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
