"""Compare numerical native renders and build unretouched diagnostic contact sheets."""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(folder, label):
    return {
        mode: np.fromfile(folder / f"{label}-front-{mode}.rgba32f.bin", dtype="<f4").reshape(1024, 1024, 4)
        for mode in ("albedo", "normal", "depth")
    }


def compare(reference, candidate, near, far, region):
    x0, y0, x1, y1 = region
    sliced = (slice(y0, y1), slice(x0, x1))
    rmask = reference["albedo"][sliced][..., 3] > 0.5
    cmask = candidate["albedo"][sliced][..., 3] > 0.5
    overlap = rmask & cmask
    residual = (candidate["depth"][sliced][..., 0] - reference["depth"][sliced][..., 0]) * (far - near)
    rn = reference["normal"][sliced][..., :3] * 2 - 1
    cn = candidate["normal"][sliced][..., :3] * 2 - 1
    rn /= np.maximum(np.linalg.norm(rn, axis=2, keepdims=True), 1e-10)
    cn /= np.maximum(np.linalg.norm(cn, axis=2, keepdims=True), 1e-10)
    angles = np.degrees(np.arccos(np.clip((rn * cn).sum(2), -1, 1)))[overlap]
    distances = abs(residual[overlap])
    percentiles = (50, 90, 95, 99, 100)
    return {
        "roi": region,
        "overlapPixels": int(overlap.sum()),
        "referencePixels": int(rmask.sum()),
        "candidatePixels": int(cmask.sum()),
        "silhouetteIoU": float(overlap.sum() / np.logical_or(rmask, cmask).sum()),
        "depthWorldAbs": {str(p): float(np.percentile(distances, p)) for p in percentiles},
        "signedDepthMean": float(residual[overlap].mean()),
        "normalAngleDegrees": {str(p): float(np.percentile(angles, p)) for p in percentiles},
        "depthValuesFinite": bool(np.isfinite(distances).all()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--font", default="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    args = ap.parse_args()
    root, output = args.root, args.output
    output.mkdir(parents=True, exist_ok=True)
    reports = {stage: json.loads((root / stage / "diagnostic-report.json").read_text()) for stage in ("F", "H")}
    keys = ("projection", "target", "distance", "halfExtent", "near", "far", "pixels")
    assert all(reports["F"]["camera"][k] == reports["H"]["camera"][k] for k in keys)
    assert reports["F"]["lighting"] == reports["H"]["lighting"]
    assert reports["F"]["scriptSHA256"] == reports["H"]["scriptSHA256"]
    assert reports["F"]["triangles"] == 1000000 and reports["H"]["triangles"] == 999999
    camera = reports["F"]["camera"]
    crop = reports["F"]["headCrop"]
    assert crop == reports["H"]["headCrop"]
    data = {"B-mesh": load(root / "F", "B-mesh"), "F-box": load(root / "F", "F-box"),
            "H-mesh": load(root / "H", "H-mesh"), "H-box": load(root / "H", "H-box")}
    near, far = camera["near"], camera["far"]
    metrics = {}
    for label, ref, candidate in (("F-mesh-vs-Box", "B-mesh", "F-box"), ("H-mesh-vs-Box", "H-mesh", "H-box"),
                                  ("B-vs-H-mesh", "B-mesh", "H-mesh"), ("F-vs-H-Box", "F-box", "H-box")):
        metrics[label] = {region: compare(data[ref], data[candidate], near, far, box)
                          for region, box in (("body", [0, 0, 1024, 1024]), ("head", crop))}

    fonts = {size: ImageFont.truetype(args.font, size) for size in (16, 19, 26)}
    tile = ((crop[2] - crop[0]) * 3, (crop[3] - crop[1]) * 3)
    gap, top = 18, 100
    images = []

    def board(filename, title, subtitle, rows, footer):
        columns = len(rows[0])
        canvas = Image.new("RGB", (columns * tile[0] + (columns + 1) * gap,
                                   len(rows) * (tile[1] + 50) + top + 55), (21, 25, 31))
        draw = ImageDraw.Draw(canvas)
        draw.text((gap, 17), title, font=fonts[26], fill="white")
        draw.text((gap, 54), subtitle, font=fonts[16], fill="#b8c0cc")
        sources = []
        for row, entries in enumerate(rows):
            for column, (stage, label, mode) in enumerate(entries):
                x, y = gap + column * (tile[0] + gap), top + row * (tile[1] + 50)
                path = root / stage / f"{label}-front-{mode}.png"
                image = Image.open(path).convert("RGBA").crop(crop).resize(tile, Image.Resampling.NEAREST)
                draw.text((x, y), f"{label} / {mode}", font=fonts[19], fill="white")
                canvas.paste(image, (x, y + 30), image)
                sources.append({"path": str(path), "sha256": digest(path)})
        draw.text((gap, canvas.height - 37), footer, font=fonts[16], fill="#b8c0cc")
        path = output / filename
        canvas.save(path, quality=96, method=6)
        images.append({"path": str(path), "sha256": digest(path), "sources": sources})

    for stage, mesh in (("F", "B-mesh"), ("H", "H-mesh")):
        board(f"{stage}-diagnostic-heads.webp", f"{stage}: actual mesh and native Box, matched camera and lighting",
              "1024 x 1024 whole-body render; fixed head crop, 3x nearest-neighbor enlargement.",
              [[(stage, label, mode) for mode in ("lit", "albedo", "normal", "depth")]
               for label in (mesh, f"{stage}-box")],
              "Albedo retains native ray intersections and normal-dependent view weights. Depth comes from the actual depth buffer.")
    board("F-H-shading-heads.webp", "F / H: lit appearance and color without lighting",
          "Same whole-body camera, lighting, and head crop. Original native Box maps; no AI image edits or sharpening.",
          [[("F", "F-box", "lit"), ("F", "F-box", "albedo"), ("H", "H-box", "lit"), ("H", "H-box", "albedo")]],
          "Color without lighting still includes native geometry intersections and normal-dependent six-view mixing.")
    board("F-H-normals-heads.webp", "F / H: original mesh normals and native Box normals",
          "Same camera and packed world-normal color scale. Mesh uses original full-triangle Inspector normals.",
          [[("F", "B-mesh", "normal"), ("F", "F-box", "normal"), ("H", "H-mesh", "normal"), ("H", "H-box", "normal")]],
          "Normal differences are not distances to ground truth; no measured reference mesh is available.")
    result = {"method": "paired-native-box-render-diagnostics-v1", "camera": camera, "headCrop": crop,
              "commonLighting": reports["F"]["lighting"], "units": "Canonical object height=1; no measured real-world scale",
              "sourceReports": {s: {"path": str(root / s / "diagnostic-report.json"), "sha256": digest(root / s / "diagnostic-report.json")}
                                for s in ("F", "H")},
              "sameHarnessSHA256": reports["F"]["scriptSHA256"], "metrics": metrics, "images": images,
              "limitations": ["Only front camera evaluated; side and oblique conclusions need separate captures.",
                              "Depth metrics measure Box reconstruction relative to its own mesh, not accuracy to the person.",
                              "Both silhouettes must cover a pixel for depth/normal metrics; edge maxima are reported.",
                              "Mesh normals come from full-triangle Inspector payloads; baked normals can differ.",
                              "Albedo removes lighting but preserves geometry and normal-dependent view selection."]}
    (output / "paired-diagnostics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v["head"] for k, v in metrics.items()}, indent=2))


if __name__ == "__main__":
    main()
