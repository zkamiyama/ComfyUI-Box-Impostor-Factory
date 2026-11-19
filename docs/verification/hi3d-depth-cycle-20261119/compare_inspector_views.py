"""Compare unchanged official Inspector captures using fixed world-space head crops."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import trimesh

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("work", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--stages", default="A,A-normal-control")
parser.add_argument("--control-proof", type=Path)
args = parser.parse_args()
stages = args.stages.split(",")
is_control = stages == ["A", "A-normal-control"]
assert is_control or all(stage in ("A", "B", "C") for stage in stages)
assert len(stages) == 1 or stages[0] == "A", "Cross-stage crops must use A as the reference"
args.output.mkdir(parents=True, exist_ok=True)
views = {"front": [0, 0, 1], "iso": [1, 1, 1], "left": [-1, 0, 0], "right": [1, 0, 0]}
labels = {"A": "Full A / original normal texture", "A-normal-control": "Same six RGB views / geometry normals only",
          "B": "B / Hi3D geometry + fresh six-view H3 RGB",
          "C": "C / RGB-depth geometry refinement"}
if not is_control and len(stages) > 1:
    labels["A"] = "A / coarse geometry + fresh six-view H3 RGB"
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
records = {}
for stage in stages:
    meta = json.loads((args.work / f"{stage}-provenance.json").read_text())
    capture_path = args.work / f"{stage}-box/capture.json"
    capture = json.loads(capture_path.read_text())
    assert capture["assetId"] == meta["assetId"]
    assert capture["htmlSHA256"] == meta["box"]["viewerSHA256"]
    assert not capture["errors"] and "SwiftShader" in capture["graphicsRenderer"]
    frames = {}
    for view in views:
        path = args.work / f"{stage}-box/lit-{view}.png"
        assert sha(path) == capture["modes"]["lit"][view]["sha256"]
        frames[view] = Image.open(path).convert("RGBA")
    manifest = json.loads((Path(meta["assetDirectory"]) / "manifest.json").read_text())
    records[stage] = {"metadata": meta, "capture": capture, "captureSHA256": sha(capture_path), "frames": frames, "manifest": manifest}
base = records[stages[0]]
camera = base["capture"]["camera"]
width, height = base["frames"]["front"].size
for record in records.values():
    for field in ("projection", "target", "halfExtent", "cssPixels", "deviceScaleFactor"):
        assert record["capture"]["camera"][field] == camera[field], field
    assert all(image.size == (width, height) for image in record["frames"].values())
manifest = base["manifest"]
scene = trimesh.load(Path(base["metadata"]["assetDirectory"]) / manifest["proxy"], force="scene", process=False)
xyz = np.concatenate([trimesh.transform_points(scene.geometry[scene.graph[node][1]].vertices, scene.graph[node][0])
                      for node in scene.graph.nodes_geometry])
top = manifest["tightBounds"]["max"][1]
object_height = top - manifest["tightBounds"]["min"][1]
upper = xyz[xyz[:, 1] >= top - .08 * object_height]
xmin, xmax = upper[:, 0].min() - .01, upper[:, 0].max() + .01
selected = xyz[(xyz[:, 1] >= top - .18 * object_height) & (xyz[:, 0] >= xmin) & (xyz[:, 0] <= xmax)]
head = np.array([selected.min(0) - [0, .01, .01], selected.max(0) + [0, .01, .01]])
head[:, 0] = [xmin, xmax]
corners = np.array([[head[x, 0], head[y, 1], head[z, 2]] for x in (0, 1) for y in (0, 1) for z in (0, 1)])
crops = {}
for name, vector in views.items():
    direction = np.asarray(vector, dtype=float)
    direction /= np.linalg.norm(direction)
    right = np.cross([0., 1., 0.], direction)
    right /= np.linalg.norm(right)
    up = np.cross(direction, right)
    centered = corners - np.asarray(camera["target"])
    xy = np.column_stack((width / 2 + centered @ right * width / (2 * camera["halfExtent"]),
                          height / 2 - centered @ up * height / (2 * camera["halfExtent"])))
    crops[name] = [*np.floor(xy.min(0)).astype(int).tolist(), *np.ceil(xy.max(0)).astype(int).tolist()]
evidence = {
    "kind": (f"Full six-view {stages[0]}" if len(stages) == 1 else
             "Full A versus same-RGB normalTexture control" if is_control else
             "Same-camera geometry stages: " + ", ".join(stages)),
    "renderer": "Unchanged official 3D-Inspector native Box PBR, CPU SwiftShader",
    "camera": camera, "actualFrameSize": [width, height], "headWorldBounds": head.tolist(), "headCrops": crops,
    "headDefinition": stages[0] + " upper 18 percent height within upper 8 percent X envelope; 0.01 bake-unit padding",
    "stages": {}, "comparisons": {}, "panels": {},
    "processing": "Opaque backdrop for transparent PNGs, integer head crops enlarged 2x nearest-neighbor; full-body previews downsampled with Lanczos. No sharpening or AI postprocessing.",
}
for stage, record in records.items():
    meta = record["metadata"]
    evidence["stages"][stage] = {
        "assetId": meta["assetId"], "assetDirectory": meta["assetDirectory"], "manifestSHA256": meta["manifestSHA256"],
        "proxySHA256": meta["proxySHA256"], "acceptedCount": meta["acceptedCount"],
        "faces": {name: {"status": f["status"], "reason": f["reason"], "qa": f["qa"]} for name, f in meta["faces"].items()},
        "html": meta["box"], "capture": record["capture"], "captureSHA256": record["captureSHA256"],
    }
for stage in stages[1:]:
    other = records[stage]["manifest"]
    equal_maps = {name: description["sha256"] == other["textures"][name]["sha256"]
                  for name, description in manifest["textures"].items() if name != "normalMap"}
    if is_control:
        assert all(equal_maps.values())
    geometry_equal = base["metadata"]["clay"]["geometry"] == records[stage]["metadata"]["clay"]["geometry"]
    if is_control:
        assert geometry_equal
    normal_accessors_equal = (base["metadata"]["clay"]["normalPreservation"]["normalAccessors"]
                              == records[stage]["metadata"]["clay"]["normalPreservation"]["normalAccessors"])
    if is_control:
        assert normal_accessors_equal
        assert manifest["bakeBox"] == other["bakeBox"]
    else:
        assert manifest["transform"] == other["transform"], "Stages are not in the same canonical frame"
    equal_depths = {face: manifest["faces"][face]["depthSha256"] == other["faces"][face]["depthSha256"] for face in manifest["faceOrder"]}
    if is_control:
        assert all(equal_depths.values())
    evidence["comparisons"][stage] = {"allNonNormalMapHashesEqual": equal_maps, "allDepthHashesEqual": equal_depths,
                                     "proxyBytesEqual": manifest["proxySha256"] == other["proxySha256"],
                                     "geometryPositionsIndicesTransformsEqual": geometry_equal,
                                     "vertexNormalAccessorsEqual": normal_accessors_equal, "bakeBoxEqual": manifest["bakeBox"] == other["bakeBox"],
                                     "canonicalFrameTransformEqual": manifest["transform"] == other["transform"],
                                     "bakeBoxes": [manifest["bakeBox"], other["bakeBox"]],
                                     "proxyDifference": ("Control removes normalTexture from material JSON; binary chunks are retained." if is_control
                                                         else "Actual stage geometry changes are retained; no image-based shape alignment."),
                                     "limitation": ("normalMap also affects native cross-view incidence weighting; this is not solely a lighting ablation." if is_control
                                                    else "Appearance includes the stage geometry and its generated RGB; raw RGB and clay evidence must be considered separately."),
                                     "normalRGBOnlyVerification": ("See separately recorded control invariant checks for normal alpha and registered RGB." if is_control else None)}
if args.control_proof:
    assert is_control, "--control-proof applies only to A,A-normal-control"
    proof = json.loads(args.control_proof.read_text())
    assert proof["passed"] and all(proof["checks"].values())
    evidence["controlInvariantProof"] = {"path": str(args.control_proof), "sha256": sha(args.control_proof),
                                        "data": json.loads(args.control_proof.read_text())}
BG = (21, 24, 30)
def opaque(image):
    result = Image.new("RGBA", image.size, BG + (255,))
    result.alpha_composite(image)
    return result.convert("RGB")
font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
font = lambda size: ImageFont.truetype(font_path, size)
cell_width = max((crop[2] - crop[0]) * 2 for crop in crops.values()) + 28
cell_height = max((crop[3] - crop[1]) * 2 for crop in crops.values()) + 48
board = Image.new("RGB", (4 * cell_width + 40, len(stages) * cell_height + 130), BG)
draw = ImageDraw.Draw(board)
title = (f"{stages[0]} / complete six-view Box" if len(stages) == 1 else
         "A / original normal texture vs geometry-normal control" if is_control else
         " / ".join(stages) + " / same-camera Box comparison")
draw.text((20, 15), title, font=font(25), fill=(237, 241, 247))
draw.text((20, 54), "Actual Inspector renders. Fixed camera and crops; no image enhancement.", font=font(17), fill=(177, 192, 211))
for col, view in enumerate(views):
    draw.text((20 + col * cell_width, 85), view.upper(), font=font(17), fill=(218, 226, 236))
for row, stage in enumerate(stages):
    y = 114 + row * cell_height
    draw.text((20, y), labels.get(stage, stage) + f" / accepted {records[stage]['metadata']['acceptedCount']}/6", font=font(17), fill=(237, 241, 247))
    for col, view in enumerate(views):
        crop = crops[view]
        image = opaque(records[stage]["frames"][view].crop(crop))
        image = image.resize((image.width * 2, image.height * 2), Image.Resampling.NEAREST)
        board.paste(image, (20 + col * cell_width + (cell_width - image.width) // 2, y + 30))
prefix = f"{stages[0]}-full" if len(stages) == 1 else "A-normal-control" if is_control else "-".join(stages) + "-stages"
path = args.output / f"{prefix}-faces.webp"
board.save(path, "WEBP", lossless=True)
evidence["panels"][path.name] = {"sha256": sha(path), "size": board.size, "bytes": path.stat().st_size}
body = Image.new("RGB", (1000, 570 * len(stages) + 90), BG)
draw = ImageDraw.Draw(body)
draw.text((20, 15), title + " / whole model", font=font(24), fill=(237, 241, 247))
draw.text((20, 53), "Front and iso from the same saved 1024px frames. Uniform Lanczos preview downsample.", font=font(16), fill=(177, 192, 211))
for row, stage in enumerate(stages):
    y = 92 + row * 570
    draw.text((20, y), labels.get(stage, stage) + f" / accepted {records[stage]['metadata']['acceptedCount']}/6", font=font(18), fill=(237, 241, 247))
    for col, view in enumerate(("front", "iso")):
        image = opaque(records[stage]["frames"][view]).resize((480, 480), Image.Resampling.LANCZOS)
        body.paste(image, (20 + col * 490, y + 49))
        draw.text((20 + col * 490, y + 28), view.upper(), font=font(15), fill=(177, 192, 211))
path = args.output / f"{prefix}-body.webp"
body.save(path, "WEBP", lossless=True)
evidence["panels"][path.name] = {"sha256": sha(path), "size": body.size, "bytes": path.stat().st_size}
(args.output / f"{prefix}-comparison-provenance.json").write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
print(json.dumps({"panels": evidence["panels"], "headCrops": crops, "accepted": {key: value["metadata"]["acceptedCount"] for key, value in records.items()}}))
