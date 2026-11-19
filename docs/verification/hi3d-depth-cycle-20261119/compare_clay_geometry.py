"""Compare actual A/B or B/C geometry in unchanged official Inspector."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import trimesh

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("work", type=Path)
p.add_argument("output", type=Path)
p.add_argument("--stages", default="A,B", choices=["A,B", "B,C"])
a = p.parse_args()
stages = a.stages.split(",")
is_ab = stages == ["A", "B"]
prefix = "-".join(stages)
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
read = lambda path: json.loads(Path(path).read_text())
views = {"front": [0, 0, 1], "iso": [1, 1, 1], "left": [-1, 0, 0], "right": [1, 0, 0]}
BG, FG, SUB = (21, 24, 30), (237, 241, 247), (177, 192, 211)
font = lambda size: ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
a.output.mkdir(parents=True, exist_ok=True)
crop_proof_path = a.output / "A-full-comparison-provenance.json"
crop_proof = read(crop_proof_path)
crops, records, frames = crop_proof["headCrops"], {}, {}
camera = crop_proof["camera"]
old_meta = read(a.work / "A-partial-pz-provenance.json") if is_ab else None
for stage in stages:
    meta_path = a.work / f"{stage}-provenance.json"
    meta = read(meta_path)
    manifest = read(Path(meta["assetDirectory"]) / "manifest.json")
    assert sha(meta["clay"]["path"]) == meta["clay"]["sha256"]
    assert meta["clay"]["normalPreservation"]["binaryChunksIdentical"]
    assert meta["clay"]["normalPreservation"]["positionNormalIndexBindingsIdentical"]
    assert not meta["clay"]["normalPreservation"]["normalRecomputed"]
    if stage == "A":
        assert meta["clay"]["sha256"] == old_meta["clay"]["sha256"]
        assert meta["proxySHA256"] == old_meta["proxySHA256"]
        assert meta["proxySHA256"] == crop_proof["stages"]["A"]["proxySHA256"]
    scene = trimesh.load(meta["clay"]["path"], force="scene", process=False)
    xyz = np.concatenate([trimesh.transform_points(scene.geometry[scene.graph[node][1]].vertices, scene.graph[node][0])
                          for node in scene.graph.nodes_geometry])
    record = {"metadataPath": str(meta_path), "metadataSHA256": sha(meta_path),
              "sourceAssetId": meta["assetId"], "sourceAssetDirectory": meta["assetDirectory"],
              "sourceProxySHA256": meta["proxySHA256"], "clay": meta["clay"],
              "transform": manifest["transform"], "tightBounds": manifest["tightBounds"],
              "triangles": sum(x["triangles"] for x in meta["clay"]["geometry"]),
              "vertices": sum(x["vertices"] for x in meta["clay"]["geometry"]),
              "views": {}, "cameraProjectionCheck": {}}
    frames[stage] = {}
    for view, vector in views.items():
        folder = "A-partial-pz-clay" if stage == "A" and view == "front" else f"{stage}-clay"
        capture_path = a.work / folder / "capture.json"
        cap = read(capture_path)
        assert not cap["errors"] and "SwiftShader" in cap["graphicsRenderer"]
        assert cap["backend"] == "Three.js / WebGL2" and cap["stageState"] == "Three.js PBR"
        if stage in ("B", "C"):
            assert cap["triangleCounts"]["meshPayload"] == record["triangles"]
            assert cap["triangleCounts"]["expected"] == record["triangles"]
        for key in ("projection", "target", "halfExtent", "cssPixels", "deviceScaleFactor"):
            assert cap["camera"][key] == camera[key], key
        assert sha(cap["html"]) == cap["htmlSHA256"]
        expected_html_sha = old_meta["clay"]["viewerSHA256"] if stage == "A" and view == "front" else meta["clay"]["viewerSHA256"]
        assert cap["htmlSHA256"] == expected_html_sha
        entry = cap["modes"]["lit"][view]
        assert sha(entry["path"]) == entry["sha256"]
        image = Image.open(entry["path"]).convert("RGBA")
        assert image.size == (1024, 1024)
        frames[stage][view] = image
        direction = np.asarray(vector, dtype=float)
        direction /= np.linalg.norm(direction)
        right = np.cross([0., 1., 0.], direction)
        right /= np.linalg.norm(right)
        up = np.cross(direction, right)
        centered = xyz - np.asarray(camera["target"])
        projected = np.column_stack((512 + centered @ right * 512 / camera["halfExtent"],
                                     512 - centered @ up * 512 / camera["halfExtent"]))
        expected = [*projected.min(0).tolist(), *projected.max(0).tolist()]
        yy, xx = np.where(np.asarray(image)[:, :, 3] > 127)
        actual = [int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1]
        error = float(np.abs(np.asarray(expected) - actual).max())
        assert error < 3, (stage, view, error, expected, actual)
        record["cameraProjectionCheck"][view] = {"expectedSilhouetteBounds": expected, "actualAlphaBounds": actual, "maxErrorPx": error}
        record["views"][view] = {"capturePath": str(capture_path), "captureSHA256": sha(capture_path),
                                "capture": cap, "sourceFrame": entry, "reused": stage == "A" and view == "front"}
    records[stage] = record
assert records[stages[0]]["transform"] == records[stages[1]]["transform"]
assert records[stages[0]]["clay"]["material"] == records[stages[1]]["clay"]["material"]
evidence = {
    "kind": ("A coarse versus B Hi3D actual geometry only, before B H3 generation" if is_ab else "B Hi3D versus C RGB-depth-refined actual geometry"),
    "renderer": "Unchanged official 3D-Inspector mesh PBR, CPU SwiftShader",
    "camera": camera, "sameCanonicalTransform": True, "sameGrayMaterial": True,
    "material": records[stages[0]]["clay"]["material"], "actualFrameSize": [1024, 1024],
    "headCrops": crops, "headWorldBounds": crop_proof["headWorldBounds"],
    "headCropProvenance": {"path": str(crop_proof_path), "sha256": sha(crop_proof_path)},
    "stages": records, "panels": {},
    "processing": "Transparent snapshots composited on opaque backdrop. Fixed integer head crops enlarged 2x nearest-neighbor. Whole model reduced to 480px with Lanczos. No sharpening or AI postprocessing.",
    "normalPolicy": ("A source vertex NORMAL is retained. B has no NORMAL accessor and this absence is retained. " if is_ab else "B has no source NORMAL; the official viewer computes normals. C source vertex NORMAL is retained. ") + "No normals are added during clay preparation. Positions, indices, transforms and GLB binary chunks remain unchanged. Official converter obtains trimesh.vertex_normals when source normals are absent and sends them to the mesh renderer. Color and tangent normal textures removed in separate diagnostic GLBs.",
    "limitation": ("This proves geometry appearance before B H3; it is not a completed B RGB or Box result." if is_ab else "This compares actual mesh appearance, including each stage shading normals. It is not ground-truth geometry accuracy or a Box render; consult the separate same-RGB Box controls."),
}
def opaque(image):
    canvas = Image.new("RGBA", image.size, BG + (255,))
    canvas.alpha_composite(image)
    return canvas.convert("RGB")
def save(board, filename):
    path = a.output / filename
    board.save(path, "WEBP", lossless=True)
    assert np.array_equal(np.asarray(Image.open(path).convert("RGB")), np.asarray(board))
    evidence["panels"][filename] = {"sha256": sha(path), "bytes": path.stat().st_size, "size": board.size}
labels = {"A": "A / coarse geometry", "B": "B / Hi3D 1024 / full mesh", "C": "C / RGB-derived depth / full mesh"}
cw = max((c[2] - c[0]) * 2 for c in crops.values()) + 28
ch = max((c[3] - c[1]) * 2 for c in crops.values()) + 48
board = Image.new("RGB", (4 * cw + 40, 2 * ch + 130), BG)
draw = ImageDraw.Draw(board)
draw.text((20, 15), "A / B actual geometry before B image generation" if is_ab else "B / C actual geometry after RGB-depth refinement", font=font(25), fill=FG)
draw.text((20, 54), "Same camera/material/light; full mesh. " + ("A: source normals. B: viewer-computed normals. No geometry smoothing." if is_ab else "B: viewer-computed normals. C: source normals. No image enhancement."), font=font(17), fill=SUB)
for col, view in enumerate(views):
    draw.text((20 + col * cw, 85), view.upper(), font=font(17), fill=SUB)
for row, stage in enumerate(stages):
    y = 114 + row * ch
    draw.text((20, y), labels[stage] + f" / {records[stage]['triangles']:,} triangles", font=font(17), fill=FG)
    for col, view in enumerate(views):
        image = opaque(frames[stage][view].crop(crops[view]))
        image = image.resize((image.width * 2, image.height * 2), Image.Resampling.NEAREST)
        board.paste(image, (20 + col * cw + (cw - image.width) // 2, y + 30))
save(board, f"{prefix}-clay-faces.webp")
body = Image.new("RGB", (1000, 1230), BG)
draw = ImageDraw.Draw(body)
draw.text((20, 15), "A / B geometry only / before B H3" if is_ab else "B / C geometry only / same camera", font=font(25), fill=FG)
draw.text((20, 53), "Identical camera, gray material and light. Preview downsample: Lanczos.", font=font(16), fill=SUB)
for row, stage in enumerate(stages):
    y = 92 + row * 570
    draw.text((20, y), labels[stage] + f" / {records[stage]['triangles']:,} triangles", font=font(18), fill=FG)
    for col, view in enumerate(("front", "iso")):
        image = opaque(frames[stage][view]).resize((480, 480), Image.Resampling.LANCZOS)
        body.paste(image, (20 + col * 490, y + 49))
        draw.text((20 + col * 490, y + 28), view.upper(), font=font(15), fill=SUB)
save(body, f"{prefix}-clay-body.webp")
(a.output / f"{prefix}-clay-comparison-provenance.json").write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
print(json.dumps({"panels": evidence["panels"], "cameraMaxErrorPx": {stage: max(v["maxErrorPx"] for v in rec["cameraProjectionCheck"].values()) for stage, rec in records.items()},
                  "sourceVerticesTriangles": {stage: [rec["vertices"], rec["triangles"]] for stage, rec in records.items()}}))
