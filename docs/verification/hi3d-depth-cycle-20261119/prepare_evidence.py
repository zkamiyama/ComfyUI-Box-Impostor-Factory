"""Prepare unchanged official Inspector viewers and a material-only clay derivative."""
import argparse
import base64
import re
import zlib
import hashlib
import json
from pathlib import Path
import subprocess
import struct
import sys

import numpy as np
import trimesh

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from box_nodes.archive import pack_asset

def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def viewer_mesh_counts(path):
    html = Path(path).read_text()
    text = re.search(r'<script id="scene-payload"[^>]*>(.*?)</script>', html, re.S).group(1).strip()
    packed = base64.b64decode(text)
    payload = json.loads(zlib.decompress(packed) if packed[:1] == b"\x78" else packed)
    triangles = sum(len(base64.b64decode(mesh.get("indices", ""))) // 12 for mesh in payload["meshes"] if mesh.get("primitive") != "points")
    return {"payloadTriangles": triangles, "payloadOriginal": payload.get("original")}

def glb_chunks(path):
    data = Path(path).read_bytes()
    assert data[:4] == b"glTF"
    chunks = []
    offset = 12
    while offset < len(data):
        length, kind = struct.unpack_from("<II", data, offset)
        chunks.append((kind, data[offset + 8:offset + 8 + length]))
        offset += 8 + length
    return chunks

def normal_accessors(chunks):
    tree = json.loads(next(data for kind, data in chunks if kind == 0x4E4F534A))
    binary = next(data for kind, data in chunks if kind == 0x004E4942)
    result = []
    for mesh in tree["meshes"]:
        for primitive in mesh["primitives"]:
            normal = primitive["attributes"].get("NORMAL")
            if normal is None:
                result.append({"present": False})
                continue
            accessor = tree["accessors"][normal]
            view = tree["bufferViews"][accessor["bufferView"]]
            assert accessor["componentType"] == 5126 and accessor["type"] == "VEC3"
            offset = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
            values = np.ndarray((accessor["count"], 3), dtype="<f4", buffer=binary,
                                offset=offset, strides=(view.get("byteStride", 12), 4))
            result.append({"present": True, "count": accessor["count"],
                           "sha256": hashlib.sha256(values.tobytes()).hexdigest()})
    return result

def write_clay(source_path, output_path):
    chunks = glb_chunks(source_path)
    original = json.loads(next(data for kind, data in chunks if kind == 0x4E4F534A))
    tree = json.loads(json.dumps(original))
    removed = []
    for mesh in tree["meshes"]:
        for primitive in mesh["primitives"]:
            primitive["material"] = 0
            removed.append({key: value for key, value in primitive["attributes"].items() if key.startswith("COLOR_")})
            primitive["attributes"] = {key: value for key, value in primitive["attributes"].items() if not key.startswith("COLOR_")}
    tree["materials"] = [{
        "name": "Diagnostic uniform gray",
        "pbrMetallicRoughness": {"baseColorFactor": [180 / 255] * 3 + [1],
                               "metallicFactor": 0, "roughnessFactor": 1},
    }]
    retained = ("buffers", "bufferViews", "accessors", "nodes", "scenes", "scene", "animations", "skins")
    assert all(tree.get(key) == original.get(key) for key in retained)
    for old_mesh, new_mesh in zip(original["meshes"], tree["meshes"]):
        for old, new in zip(old_mesh["primitives"], new_mesh["primitives"]):
            assert old.get("indices") == new.get("indices")
            assert all(new["attributes"].get(key) == old["attributes"].get(key) for key in ("POSITION", "NORMAL"))
    encoded = json.dumps(tree, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 4)
    output_chunks = [(kind, encoded if kind == 0x4E4F534A else data) for kind, data in chunks]
    body = b"".join(struct.pack("<II", len(data), kind) + data for kind, data in output_chunks)
    output_path.write_bytes(struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body)
    checked = glb_chunks(output_path)
    original_binary = [(kind, data) for kind, data in chunks if kind != 0x4E4F534A]
    assert original_binary == [(kind, data) for kind, data in checked if kind != 0x4E4F534A]
    assert normal_accessors(chunks) == normal_accessors(checked)
    return {
        "method": "GLB JSON material replacement; binary chunks and geometry accessors retained byte-for-byte",
        "normalRecomputed": False, "normalAccessors": normal_accessors(checked),
        "removedColorBindings": removed, "binaryChunksIdentical": True,
        "binaryChunkSHA256": [hashlib.sha256(data).hexdigest() for kind, data in original_binary],
        "positionNormalIndexBindingsIdentical": True, "nodeTransformsIdentical": True,
        "sourceMaterialNormalTextures": [m.get("normalTexture") for m in original.get("materials", [])],
        "materialTextureMode": "All color and tangent normal textures removed; original geometry NORMAL retained",
        "note": "If NORMAL is absent in the source, it remains absent and the official renderer computes shading normals.",
    }

def geometry_signature(scene):
    entries = []
    for node in sorted(scene.graph.nodes_geometry):
        matrix, name = scene.graph[node]
        mesh = scene.geometry[name]
        entries.append({
            "vertices": len(mesh.vertices), "triangles": len(mesh.faces),
            "positionsSHA256": hashlib.sha256(np.asarray(mesh.vertices, dtype="<f4").tobytes()).hexdigest(),
            "indicesSHA256": hashlib.sha256(np.asarray(mesh.faces, dtype="<u4").tobytes()).hexdigest(),
            "transform": np.asarray(matrix).tolist(),
        })
    return sorted(entries, key=lambda entry: (entry["positionsSHA256"], entry["indicesSHA256"]))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("asset", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--stage", required=True, choices=["A", "B", "C", "D", "A-partial-pz", "A-normal-control", "B-normal-only"])
parser.add_argument("--only", choices=["geometry", "all"], default="all")
parser.add_argument("--inspector", type=Path, default=Path("/home/dev/Projects/3D-Inspector-skills/skills/3d-inspector/view.py"))
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
manifest_path = args.asset / "manifest.json"
manifest = json.loads(manifest_path.read_text())
proxy = args.asset / manifest["proxy"]
assert sha(proxy) == manifest["proxySha256"]
source = trimesh.load(proxy, force="scene", process=False)
original_signature = geometry_signature(source)
source_triangles = sum(item["triangles"] for item in original_signature)
triangle_limit = max(250000, source_triangles)
clay_path = args.output / f"{args.stage}-clay.glb"
metadata_path = args.output / f"{args.stage}-provenance.json"
previous = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
history = previous.get("clay", {}).get("preparationHistory", [])
if clay_path.exists() and normal_accessors(glb_chunks(clay_path)) != normal_accessors(glb_chunks(proxy)):
    history.append({"previousGLBSHA256": sha(clay_path), "previousViewerSHA256": previous.get("clay", {}).get("viewerSHA256"),
                    "issue": "Initial trimesh export dropped original NORMAL; replaced before any screenshot was captured."})
normal_preservation = write_clay(proxy, clay_path)
reloaded = trimesh.load(clay_path, force="scene", process=False)
assert geometry_signature(reloaded) == original_signature, "Clay export changed positions, faces or node transforms"
metadata = {
    "stage": args.stage, "assetDirectory": str(args.asset.resolve()),
    "assetId": manifest["asset_id"], "manifestSHA256": sha(manifest_path),
    "proxyPath": str(proxy.resolve()), "proxySHA256": sha(proxy),
    "bakeBox": manifest["bakeBox"], "tightBounds": manifest["tightBounds"],
    "clay": {
        "path": str(clay_path.resolve()), "sha256": sha(clay_path),
        "material": {"baseColorFactor": [180, 180, 180, 255], "metallicFactor": 0, "roughnessFactor": 1},
        "geometryIdenticalAfterReload": True, "geometry": original_signature,
        "normalPreservation": normal_preservation, "preparationHistory": history,
        "note": "Only the material was replaced. Original asset unchanged; no smoothing or remeshing.",
    },
    "faces": {},
    "inspectorLauncher": str(args.inspector), "inspectorLauncherSHA256": sha(args.inspector),
}
qa = json.loads((args.asset / manifest["qa"]).read_text()) if manifest.get("qa") else {}
for name in manifest["faceOrder"]:
    face = manifest["faces"][name]
    edit = face.get("edit", {})
    status = edit.get("status", "baseline")
    item = {"status": status, "reason": edit.get("reason", "not-generated"),
            "source": edit.get("source", "geometry_baseline"),
            "guide": face["guide"], "guideSHA256": face.get("guideSha256"),
            "depth": face["depth"], "depthSHA256": face.get("depthSha256"),
            "qa": qa.get("faces", {}).get(name)}
    if edit.get("view"):
        rgb = args.asset / edit["view"]
        item.update(generatedRGBPath=str(rgb.resolve()), generatedRGBSHA256=sha(rgb))
        aligned = rgb.parent / "aligned.npy"
        if aligned.exists():
            item.update(alignedPath=str(aligned.resolve()), alignedSHA256=sha(aligned))
    metadata["faces"][name] = item
metadata["acceptedCount"] = sum(item["status"] == "accepted" for item in metadata["faces"].values())
clay_viewer = args.output / f"{args.stage}-clay.inspector.html"
subprocess.run([sys.executable, str(args.inspector), str(clay_path), "-o", str(clay_viewer), "--max-triangles", str(triangle_limit)], check=True)
viewer_counts = viewer_mesh_counts(clay_viewer)
assert viewer_counts["payloadTriangles"] == source_triangles, "Official viewer must retain every source triangle"
metadata["clay"].update(viewerTriangleLimit=triangle_limit, sourceTriangles=source_triangles, **viewer_counts)
metadata["clay"]["viewer"] = str(clay_viewer.resolve())
metadata["clay"]["viewerSHA256"] = sha(clay_viewer)
metadata["clay"]["viewerBytes"] = clay_viewer.stat().st_size
if args.only == "all":
    archive = args.output / f"{args.stage}.box.zip"
    pack_asset(args.asset, archive)
    viewer = args.output / f"{args.stage}.inspector.html"
    subprocess.run([sys.executable, str(args.inspector), str(archive), "-o", str(viewer)], check=True)
    metadata["box"] = {
        "archive": str(archive.resolve()), "archiveSHA256": sha(archive),
        "viewer": str(viewer.resolve()), "viewerSHA256": sha(viewer), "viewerBytes": viewer.stat().st_size,
        "colorMapSHA256": manifest["textures"]["colorMap"]["sha256"],
    }
metadata_path = args.output / f"{args.stage}-provenance.json"
metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
print(json.dumps({"stage": args.stage, "metadata": str(metadata_path), "acceptedCount": metadata["acceptedCount"],
                  "clay": {"viewer": str(clay_viewer), "sha256": sha(clay_viewer), "bytes": clay_viewer.stat().st_size}, "box": metadata.get("box")}))