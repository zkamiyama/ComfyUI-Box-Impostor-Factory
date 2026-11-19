"""Measure front RGB fidelity on one shared opaque, three-pixel-eroded head mask."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_erosion

from build_box_control_front import BG, TEXT, MUTED, FONT, sha, backdrop, aligned_front_crop


def front_entry(capture):
    if capture.get("yawFrames"):
        return next(e for e in capture["yawFrames"] if e["requestedYawDegrees"] == 0)
    return capture["modes"]["lit"]["front"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("work", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--render", action="append", required=True, help="LABEL=absolute capture.json path")
    p.add_argument("--prefix", default="F-J-K-front-fidelity")
    a = p.parse_args()
    sources = {}
    for item in a.render:
        label, path = item.split("=", 1)
        assert label and label not in sources
        sources[label] = Path(path)
    basis_path = a.output/"A-full-comparison-provenance.json"
    basis = json.loads(basis_path.read_text())
    crop = basis["headCrops"]["front"]
    meta = json.loads((a.work/"F-provenance.json").read_text())
    reference_capture = json.loads((a.work/"F-box/capture.json").read_text())
    camera = reference_capture["camera"]
    front = meta["faces"]["+Z"]
    assert sha(front["alignedPath"]) == front["alignedSHA256"]
    ref_image = aligned_front_crop(front["alignedPath"], meta["bakeBox"], crop, camera)
    reference = np.asarray(ref_image)
    arrays, captures, inputs = {}, {}, {}
    observed_cameras = []
    for label, path in sources.items():
        cap = json.loads(path.read_text())
        captures[label] = cap
        assert cap["assetId"] == meta["assetId"]
        assert not cap["errors"] and cap["stageState"] == "Native Box"
        assert sha(cap["html"]) == cap["htmlSHA256"]
        for key in ("projection", "target", "halfExtent", "distance", "pixels", "cssPixels", "deviceScaleFactor"):
            assert cap["camera"][key] == camera[key], (label, key)
        entry = front_entry(cap)
        assert sha(entry["path"]) == entry["sha256"]
        full = Image.open(entry["path"]).convert("RGBA")
        assert full.size == (1024, 1024)
        arrays[label] = np.asarray(full.crop(crop))
        if entry.get("actualCamera"):
            observed_cameras.append((label, entry["actualCamera"]))
        inputs[label] = {"capturePath": str(path), "captureSHA256": sha(path), "front": entry,
                         "html": cap["html"], "htmlSHA256": cap["htmlSHA256"]}
    if observed_cameras:
        anchor_label, anchor = observed_cameras[0]
        for label, actual in observed_cameras[1:]:
            for key in ("matrixWorld", "projectionMatrix"):
                assert np.allclose(actual[key], anchor[key], rtol=0, atol=1e-8), (anchor_label, label, key)
    common = reference[..., 3] > 250
    for array in arrays.values():
        common &= array[..., 3] > 250
    mask = binary_erosion(common, structure=np.ones((3, 3), dtype=bool), iterations=3, border_value=0)
    assert int(mask.sum()) > 1000
    metrics, error_images = {}, {}
    for label, array in arrays.items():
        difference = np.abs(array[..., :3].astype("f8")-reference[..., :3].astype("f8"))
        per_pixel = difference.mean(axis=-1)
        metrics[label] = {"sRGBMAE255": float(difference[mask].mean()),
                          "channelMAE255": difference[mask].mean(axis=0).tolist(),
                          "medianPerPixelMAE255": float(np.median(per_pixel[mask])),
                          "p95PerPixelMAE255": float(np.percentile(per_pixel[mask], 95)),
                          "sampledPixels": int(mask.sum())}
        error_u8 = np.uint8(np.clip(per_pixel/64, 0, 1)*255+.5)
        error_u8[~mask] = 0
        error_images[label] = Image.fromarray(error_u8, "L").convert("RGB")
    mask_path = a.output/f"{a.prefix}-mask.png"
    Image.fromarray(mask.astype("u1")*255, "L").save(mask_path)
    scale = 2
    width, height = ref_image.size
    cell_w, image_h = max(250, width*scale+32), height*scale
    labels = ["RGB reference"]+list(arrays)
    board = Image.new("RGB", (cell_w*len(labels), image_h*2+216), BG)
    draw = ImageDraw.Draw(board)
    font = lambda size: ImageFont.truetype(FONT, size)
    draw.text((20, 14), "Front image fidelity / one identical opaque head mask", font=font(24), fill=TEXT)
    draw.text((20, 49), "sRGB MAE in 0-255 units; shared alpha >250, eroded 3 pixels. This is image fidelity, not 3D accuracy.", font=font(14), fill=MUTED)
    for index, label in enumerate(labels):
        x = index*cell_w+(cell_w-width*scale)//2
        draw.text((index*cell_w+16, 83), label, font=font(18), fill=TEXT)
        if index == 0:
            display, lower = backdrop(ref_image), Image.fromarray(mask.astype("u1")*255, "L").convert("RGB")
            text = f"Shared mask: {int(mask.sum())} pixels"
        else:
            display = backdrop(Image.fromarray(arrays[label], "RGBA"))
            lower = error_images[label]
            text = f"MAE {metrics[label]['sRGBMAE255']:.4f} / 255"
        board.paste(display.resize((width*scale, image_h), Image.Resampling.NEAREST), (x, 112))
        draw.text((index*cell_w+16, image_h+134), text, font=font(15), fill=TEXT)
        board.paste(lower.resize((width*scale, image_h), Image.Resampling.NEAREST), (x, image_h+160))
    draw.text((20, board.height-37), "Lower row: shared metric mask, then fixed-range error maps (black=0, white>=64 mean RGB error).", font=font(14), fill=MUTED)
    draw.text((20, board.height-17), "2x nearest-neighbor display. No alignment optimization, sharpening, or AI postprocessing.", font=font(14), fill=MUTED)
    board_path = a.output/f"{a.prefix}.webp"
    board.save(board_path, "WEBP", lossless=True)
    result = {
        "kind": __doc__, "assetId": meta["assetId"], "camera": camera, "headCrop": crop,
        "reference": {"path": front["alignedPath"], "sha256": front["alignedSHA256"],
                      "sampling": "Premultiplied linear-light bilinear at established orthographic front-camera pixel centers; converted to 8-bit sRGB."},
        "inputs": inputs, "mask": {"path": str(mask_path), "sha256": sha(mask_path),
                                  "alphaThreshold": 250, "operator": "Intersection of reference and every render alpha>250; three iterations of 3x3 binary erosion (Chebyshev radius 3), border=0.",
                                  "pixelsBeforeErosion": int(common.sum()), "pixelsAfterErosion": int(mask.sum())},
        "metrics": metrics, "metricDefinition": "Mean absolute difference across all R/G/B samples in the same eroded mask, on 8-bit sRGB values (0-255).",
        "limitations": ["Image appearance fidelity only; not geometry/depth accuracy or natural-parallax validation.",
                        "The reference is the accepted registered +Z RGB sampled onto the same viewer camera grid.",
                        "No best-fit registration, per-image mask, exposure fitting, or metric-specific alignment is performed."],
        "differenceVisualization": {"range": [0, 64], "outsideMask": 0, "sameScaleForEveryRender": True},
        "output": {"path": str(board_path), "sha256": sha(board_path), "bytes": board_path.stat().st_size},
        "scriptSHA256": sha(__file__),
    }
    proof = a.output/f"{a.prefix}.json"
    proof.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"metrics": metrics, "output": result["output"], "proofSHA256": sha(proof)}))


if __name__ == "__main__":
    main()
