#!/usr/bin/env python3
"""Compare two ZipSplat input-view renders at the same image size.

This measures fitting to the supplied (possibly generated) input images, not
novel-view accuracy or calibrated 3D accuracy. All metrics cover the whole image
or the full object's mask; no face/body-part crops are used.

Example:
  python evaluate_zipsplat_inputs.py --left B/renders_1024 --right C/renders_1024 \
    --masks masks.json --left-label B504 --right-label C1024 --out comparison1024
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rgb(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255


def psnr(mse):
    return -10 * math.log10(mse) if mse > 0 else None


def pixel_metrics(prediction, target, mask):
    count = int(mask.sum())
    if count == 0:
        return {"pixels": 0, "l1": None, "mse": None, "psnr_db": None}
    error = prediction[mask].astype(np.float64) - target[mask].astype(np.float64)
    mse = float(np.mean(error ** 2))
    return {"pixels": count, "l1": float(np.abs(error).mean()), "mse": mse,
            "psnr_db": psnr(mse), "exact_match": mse == 0}


def sobel(image):
    # Luma of the encoded PNG values; this is not linear-light radiometry.
    luma = image @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    gx = ndimage.sobel(luma, axis=1, mode="reflect") / 8
    gy = ndimage.sobel(luma, axis=0, mode="reflect") / 8
    return gx, gy, np.hypot(gx, gy)


def gradient_metrics(prediction, target, mask):
    count = int(mask.sum())
    if count == 0:
        return {"pixels": 0}
    px, py, pm = sobel(prediction)
    tx, ty, tm = sobel(target)
    pmean, tmean = float(pm[mask].mean()), float(tm[mask].mean())
    return {
        "pixels": count,
        "mean_magnitude": pmean,
        "target_mean_magnitude": tmean,
        "mean_magnitude_ratio_to_target": pmean / tmean if tmean > 0 else None,
        "p95_magnitude": float(np.quantile(pm[mask], 0.95)),
        "magnitude_l1_to_target": float(np.abs(pm[mask] - tm[mask]).mean()),
        "vector_l1_to_target": float((np.abs(px[mask] - tx[mask]) + np.abs(py[mask] - ty[mask])).mean() / 2),
    }


def aggregate(rows, branch):
    result = {}
    for region in ("whole", "foreground", "interior"):
        values = [row[branch][region] for row in rows if row[branch][region]["pixels"]]
        count = sum(v["pixels"] for v in values)
        if not count:
            result[region] = {"pixels": 0}
            continue
        mse = sum(v["mse"] * v["pixels"] for v in values) / count
        finite_psnr = [v["psnr_db"] for v in values if v["psnr_db"] is not None]
        result[region] = {
            "pixels": count,
            "pooled_l1": sum(v["l1"] * v["pixels"] for v in values) / count,
            "pooled_mse": mse,
            "pooled_psnr_db": psnr(mse),
            "mean_per_view_l1": sum(v["l1"] for v in values) / len(values),
            "mean_per_view_psnr_db": sum(finite_psnr) / len(finite_psnr) if len(finite_psnr) == len(values) else None,
        }
    values = [row[branch]["sobel_interior"] for row in rows if row[branch]["sobel_interior"]["pixels"]]
    count = sum(v["pixels"] for v in values)
    result["sobel_interior"] = {"pixels": count}
    if count:
        for key in ("mean_magnitude", "target_mean_magnitude", "magnitude_l1_to_target", "vector_l1_to_target"):
            result["sobel_interior"][key] = sum(v[key] * v["pixels"] for v in values) / count
        gradient = result["sobel_interior"]
        gradient["mean_magnitude_ratio_to_target"] = gradient["mean_magnitude"] / gradient["target_mean_magnitude"] if gradient["target_mean_magnitude"] else None
    return result


def render_metadata(directory):
    path = directory / "metrics.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    rows = metadata["views"]
    indices = {row["id"]: index for index, row in enumerate(rows)}
    if len(indices) != len(rows):
        raise ValueError(f"Duplicate view IDs in {path}")
    crops = {row["id"]: row for row in metadata["crops"]}
    return metadata, indices, crops


def foreground_mask(record, manifest_path, crop, target_wh, original_wh=None):
    value = record.get("mask_path", record.get("mask"))
    if value is None:
        raise ValueError(f"No mask_path or mask for {record['id']}")
    path = Path(value)
    path = path if path.is_absolute() else manifest_path.parent / path
    digest = sha(path)
    if record.get("mask_sha256") not in (None, digest):
        raise ValueError(f"Mask hash changed: {path}")
    with Image.open(path) as source:
        image = source.convert("L")
        if original_wh and list(image.size) != list(original_wh):
            raise ValueError(f"Mask does not match original image size: {path}")
        left, top, right, bottom = crop["xyxy"]
        if not 0 <= left < right <= image.width or not 0 <= top < bottom <= image.height:
            raise ValueError(f"Crop extends outside original mask: {path}")
        image = image.crop((left, top, right, bottom)).resize(tuple(target_wh), Image.Resampling.NEAREST)
        mask = np.asarray(image) >= 128
    if not mask.any():
        raise ValueError(f"Empty foreground mask after crop and resize: {path}")
    return mask, {"path": str(path.resolve()), "sha256": digest, "crop_xyxy": crop["xyxy"], "target_wh": list(target_wh)}


def font(size):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def contact_sheet(items, labels, tile, path):
    header, row_label, gap = 76, 32, 8
    width = 5 * tile + 6 * gap
    height = header + len(items) * (tile + row_label + gap) + gap
    canvas = Image.new("RGB", (width, height), "#e9ecf0")
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, 7), "Full-object input-view fit; not unseen-view ground truth", font=font(19), fill="#172033")
    titles = ["Target input", labels[0], labels[1], labels[0] + " abs. error x4", labels[1] + " abs. error x4"]
    for column, title in enumerate(titles):
        draw.text((gap + column * (tile + gap), 43), title, font=font(16), fill="#172033")
    for row_index, (view_id, target, left, right, metrics) in enumerate(items):
        y = header + row_index * (tile + row_label + gap)
        left_l1 = metrics["left"]["foreground"]["l1"]
        right_l1 = metrics["right"]["foreground"]["l1"]
        caption = f"{view_id} | foreground L1: {labels[0]} {left_l1:.5f} / {labels[1]} {right_l1:.5f}"
        draw.text((gap, y), caption, font=font(16), fill="#172033")
        values = [target, left, right, np.abs(left - target) * 4, np.abs(right - target) * 4]
        for column, value in enumerate(values):
            image = Image.fromarray(np.rint(np.clip(value, 0, 1) * 255).astype(np.uint8))
            image.thumbnail((tile, tile), Image.Resampling.LANCZOS)
            x = gap + column * (tile + gap) + (tile - image.width) // 2
            canvas.paste(image, (x, y + row_label + (tile - image.height) // 2))
    canvas.save(path)


def loss_trend(path):
    if path is None:
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    losses = np.asarray(data["losses"], dtype=float)
    result = {"path": str(path.resolve()), "sha256": sha(path), "optimizer_steps": len(losses)}
    if len(losses) >= 16:
        previous, last = float(losses[-16:-8].mean()), float(losses[-8:].mean())
        result.update(previous8_mean=previous, last8_mean=last,
                      last_block_relative_reduction=(previous - last) / previous if previous else None)
    result["interpretation"] = "Each curve uses its own supervision resolution; compare convergence trend, not raw loss magnitude between resolutions."
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--left-label", default="B504")
    parser.add_argument("--right-label", default="C1024")
    parser.add_argument("--left-losses", type=Path)
    parser.add_argument("--right-losses", type=Path)
    parser.add_argument("--erosion", type=int, default=2)
    parser.add_argument("--tile", type=int, default=320)
    args = parser.parse_args()
    if args.erosion < 0 or args.tile < 64:
        raise ValueError("Erosion must be nonnegative and contact tile at least 64 px")
    left_meta, left_ids, left_crops = render_metadata(args.left)
    right_meta, right_ids, right_crops = render_metadata(args.right)
    if set(left_ids) != set(right_ids):
        raise ValueError("Branches contain different input view IDs")
    mask_data = json.loads(args.masks.read_text(encoding="utf-8"))
    mask_rows = mask_data["views"]
    masks = {row["id"]: row for row in mask_rows}
    if len(masks) != len(mask_rows):
        raise ValueError("Duplicate mask view IDs")
    inputs_path = args.left.parent / "inputs.json"
    input_rows = json.loads(inputs_path.read_text(encoding="utf-8"))["views"] if inputs_path.exists() else []
    inputs = {row["id"]: row for row in input_rows}
    rows, items, provenance = [], [], []
    for view_id, left_index in left_ids.items():
        right_index = right_ids[view_id]
        lc, rc = left_crops[view_id], right_crops[view_id]
        if lc["xyxy"] != rc["xyxy"] or lc["target_wh"] != rc["target_wh"]:
            raise ValueError(f"Different crop or evaluation size for {view_id}")
        paths = {"target": args.left / f"{left_index:03d}_target.png",
                 "right_target": args.right / f"{right_index:03d}_target.png",
                 "left": args.left / f"{left_index:03d}_rgb.png",
                 "right": args.right / f"{right_index:03d}_rgb.png"}
        target, other_target, left, right = [rgb(paths[key]) for key in ("target", "right_target", "left", "right")]
        if not np.array_equal(target, other_target) or left.shape != target.shape or right.shape != target.shape:
            raise ValueError(f"Targets or image dimensions differ for {view_id}")
        target_wh = (target.shape[1], target.shape[0])
        if tuple(lc["target_wh"]) != target_wh:
            raise ValueError(f"Render size disagrees with recorded crop metadata for {view_id}")
        original_wh = [inputs[view_id]["width"], inputs[view_id]["height"]] if view_id in inputs else None
        mask, mask_provenance = foreground_mask(masks[view_id], args.masks, lc, target_wh, original_wh)
        interior = ndimage.binary_erosion(mask, structure=np.ones((3, 3), dtype=bool), iterations=args.erosion, border_value=0) if args.erosion else mask.copy()
        regions = {"whole": np.ones(mask.shape, dtype=bool), "foreground": mask, "interior": interior}
        row = {"id": view_id, "size_wh": list(target_wh)}
        for name, prediction in (("left", left), ("right", right)):
            row[name] = {region: pixel_metrics(prediction, target, region_mask) for region, region_mask in regions.items()}
            row[name]["sobel_interior"] = gradient_metrics(prediction, target, interior)
        rows.append(row)
        items.append((view_id, target, left, right, row))
        provenance.append({"id": view_id, "mask": mask_provenance,
                           "images": {key: {"path": str(path.resolve()), "sha256": sha(path)} for key, path in paths.items()}})
    args.out.mkdir(parents=True, exist_ok=True)
    contact_path = args.out / "contact.png"
    contact_sheet(items, (args.left_label, args.right_label), args.tile, contact_path)
    result = {
        "scope": "Input-view fitting to the supplied images; no unseen-view or geometric ground truth",
        "labels": {"left": args.left_label, "right": args.right_label},
        "settings": {"erosion_pixels": args.erosion, "erosion_metric": "Chebyshev (3x3 square repeated)",
                     "pixel_encoding": "8-bit RGB PNG decoded to [0,1]; no linear-light conversion",
                     "psnr_data_range": 1, "exact_match_psnr": "null (infinite)",
                     "sobel": "Rec.709 luma weights on encoded pixels; scipy Sobel / 8, reflect boundary",
                     "gradient_interpretation": "Magnitude is detail/contrast, not accuracy. Gradient mismatch to target is an input-fit error."},
        "summary": {"left": aggregate(rows, "left"), "right": aggregate(rows, "right")},
        "loss_trends": {"left": loss_trend(args.left_losses), "right": loss_trend(args.right_losses)},
        "views": rows, "provenance": provenance,
        "masks_manifest_sha256": sha(args.masks), "helper_sha256": sha(__file__),
        "contact": {"path": str(contact_path.resolve()), "sha256": sha(contact_path)},
    }
    (args.out / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.out), "summary": result["summary"], "loss_trends": result["loss_trends"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
