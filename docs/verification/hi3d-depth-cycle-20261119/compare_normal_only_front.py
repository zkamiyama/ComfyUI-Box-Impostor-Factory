"""Show the actual B normal-only shading control, with positions and triangles fixed."""
import argparse
import hashlib
import json
import struct
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import trimesh

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("work", type=Path)
p.add_argument("output", type=Path)
p.add_argument("control_glb", type=Path)
p.add_argument("control_proof", type=Path)
a = p.parse_args()
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
read = lambda path: json.loads(Path(path).read_text())
def signature(path):
    scene = trimesh.load(path, force="scene", process=False)
    result = []
    for node in scene.graph.nodes_geometry:
        matrix, name = scene.graph[node]
        mesh = scene.geometry[name]
        result.append({"positions": hashlib.sha256(np.asarray(mesh.vertices, dtype="<f4").tobytes()).hexdigest(),
                       "indices": hashlib.sha256(np.asarray(mesh.faces, dtype="<u4").tobytes()).hexdigest(),
                       "vertices": len(mesh.vertices), "triangles": len(mesh.faces), "transform": matrix.tolist()})
    return sorted(result, key=lambda item: (item["positions"], item["indices"]))
def glb_json(path):
    data = Path(path).read_bytes()
    assert data[:4] == b"glTF"
    length, kind = struct.unpack_from("<II", data, 12)
    assert kind == 0x4E4F534A
    return json.loads(data[20:20+length])
bmeta = read(a.work / "B-provenance.json")
base_tree = glb_json(bmeta["clay"]["path"])
control_tree = glb_json(a.control_glb)
assert base_tree["materials"] == control_tree["materials"]
assert all("NORMAL" in prim["attributes"] for mesh in control_tree["meshes"] for prim in mesh["primitives"])
assert all("NORMAL" not in prim["attributes"] for mesh in base_tree["meshes"] for prim in mesh["primitives"])
before_signature = signature(bmeta["clay"]["path"])
after_signature = signature(a.control_glb)
assert before_signature == after_signature, "Normal control altered geometry"
a_proof_path = a.output / "A-full-comparison-provenance.json"
ap = read(a_proof_path)
crop = ap["headCrops"]["front"]
camera = ap["camera"]
folders = ("A-partial-pz-clay", "B-clay", "B-normal-only-clay")
labels = ("A / coarse geometry", "B / source NORMAL absent", "B / added vertex NORMAL only")
frames, sources = [], []
for folder in folders:
    capture_path = a.work / folder / "capture.json"
    cap = read(capture_path)
    assert not cap["errors"] and "SwiftShader" in cap["graphicsRenderer"]
    for key in ("projection", "target", "halfExtent", "cssPixels", "deviceScaleFactor"):
        assert cap["camera"][key] == camera[key]
    assert cap["backend"] == "Three.js / WebGL2" and cap["stageState"] == "Three.js PBR"
    if folder != "A-partial-pz-clay":
        assert cap.get("triangleCounts", {}).get("meshPayload") == sum(item["triangles"] for item in before_signature), "Cap-limited B preview cannot support this comparison"
    assert sha(cap["html"]) == cap["htmlSHA256"]
    frame = cap["modes"]["lit"]["front"]
    assert sha(frame["path"]) == frame["sha256"]
    image = Image.open(frame["path"]).convert("RGBA")
    assert image.size == (1024, 1024)
    frames.append(image)
    sources.append({"capturePath": str(capture_path), "captureSHA256": sha(capture_path), "capture": cap})
alpha_before = np.asarray(frames[1])[:, :, 3]
alpha_after = np.asarray(frames[2])[:, :, 3]
alpha_differing_pixels = int(np.count_nonzero(alpha_before != alpha_after))
BG, FG, SUB = (21, 24, 30), (237, 241, 247), (177, 192, 211)
font = lambda size: ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
width = (crop[2] - crop[0]) * 3
height = (crop[3] - crop[1]) * 3
cw = max(width + 36, 360)
board = Image.new("RGB", (cw * 3 + 40, height + 190), BG)
draw = ImageDraw.Draw(board)
draw.text((20, 15), "A / B / B normal-only shading control", font=font(25), fill=FG)
draw.text((20, 53), "Same camera, gray material and lighting. Official Inspector; no image enhancement.", font=font(17), fill=SUB)
draw.text((20, 80), "B control: identical vertex positions and triangles; normals are averaged, geometry is not smoothed.", font=font(16), fill=SUB)
for i, (label, source) in enumerate(zip(labels, frames)):
    draw.text((20 + i * cw, 119), label, font=font(18), fill=FG)
    image = source.crop(crop)
    background = Image.new("RGBA", image.size, BG + (255,))
    background.alpha_composite(image)
    image = background.convert("RGB").resize((width, height), Image.Resampling.NEAREST)
    board.paste(image, (20 + i * cw + (cw - width) // 2, 152))
path = a.output / "A-B-normal-only-front.webp"
board.save(path, "WEBP", lossless=True)
assert np.array_equal(np.asarray(Image.open(path).convert("RGB")), np.asarray(board))
proof = {
    "kind": "Pre-H3 B source-normal absence versus added area-weighted vertex normals; geometry positions unchanged",
    "sources": sources, "sameCamera": camera, "headCrop": crop, "headCropReference": {"path": str(a_proof_path), "sha256": sha(a_proof_path)},
    "baseBClay": {"path": bmeta["clay"]["path"], "sha256": bmeta["clay"]["sha256"]},
    "controlGLB": {"path": str(a.control_glb), "sha256": sha(a.control_glb)},
    "bControlAlphaEqual": alpha_differing_pixels == 0, "bControlAlphaDifferingPixels": alpha_differing_pixels,
    "geometryPositionsIndicesTransformsIdentical": True, "materialsIdentical": True, "vertexNormalAdded": True, "geometrySignature": before_signature,
    "normalControlProof": {"path": str(a.control_proof), "sha256": sha(a.control_proof), "data": read(a.control_proof)},
    "processing": "Identical integer front crop enlarged 3x nearest-neighbor. Transparent pixels placed on dark backdrop. No sharpening or AI postprocessing.",
    "panel": {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size, "size": board.size},
    "limitation": "This is a diagnostic shading control, before B H3 and before a complete B Box. It does not add or move geometric detail."
}
(a.output / "A-B-normal-only-front-provenance.json").write_text(json.dumps(proof, indent=2, allow_nan=False) + "\n")
print(json.dumps(proof["panel"]))
