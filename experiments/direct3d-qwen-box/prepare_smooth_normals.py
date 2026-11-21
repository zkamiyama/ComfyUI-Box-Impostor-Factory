#!/usr/bin/env python3
"""Append Inspector-equivalent vertex normals without rewriting source geometry.

This experiment helper supports the verified, single-primitive Direct3D GLB.
It preserves the original BIN chunk as an exact prefix and edits only four
JSON locations.  It never re-exports, repairs, merges, or smooths the geometry.
The separate compare-bakes command checks the native Box structural channels.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import struct
import time

import numpy as np


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_info(path: Path) -> dict:
    raw = path.read_bytes()
    return {"path": str(path.resolve()), "bytes": len(raw), "sha256": sha(raw)}


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def glb_parts(raw: bytes) -> tuple[list[tuple[int, bytes]], dict, str, bytes]:
    magic, version, total = struct.unpack_from("<4sII", raw)
    assert magic == b"glTF" and version == 2 and total == len(raw)
    chunks, offset = [], 12
    while offset < len(raw):
        length, kind = struct.unpack_from("<II", raw, offset)
        payload = raw[offset + 8:offset + 8 + length]
        assert len(payload) == length and length % 4 == 0
        chunks.append((kind, payload))
        offset += 8 + length
    assert offset == len(raw)
    assert [t for t, _ in chunks].count(0x4E4F534A) == 1
    assert [t for t, _ in chunks].count(0x004E4942) == 1
    text = next(b for t, b in chunks if t == 0x4E4F534A).decode("utf-8")
    binary = next(b for t, b in chunks if t == 0x004E4942)
    return chunks, json.loads(text), text, binary


def json_spans(text: str) -> dict:
    """Locate value tokens so untouched JSON, including node transforms, stays raw."""
    spans, decoder = {}, json.JSONDecoder()

    def ws(i):
        while i < len(text) and text[i].isspace():
            i += 1
        return i

    def value(i, path):
        i = ws(i)
        start = i
        if text[i] == "{":
            i = ws(i + 1)
            while text[i] != "}":
                key, end = decoder.raw_decode(text, i)
                i = ws(end)
                assert text[i] == ":"
                i = ws(value(i + 1, path + (key,)))
                if text[i] != ",":
                    break
                i = ws(i + 1)
            assert text[i] == "}"
            i += 1
        elif text[i] == "[":
            i, index = ws(i + 1), 0
            while text[i] != "]":
                i = ws(value(i, path + (index,)))
                index += 1
                if text[i] != ",":
                    break
                i = ws(i + 1)
            assert text[i] == "]"
            i += 1
        else:
            _, i = decoder.raw_decode(text, i)
        spans[path] = (start, i)
        return i

    assert ws(value(0, ())) == len(text)
    return spans


def accessor(doc: dict, binary: bytes, index: int) -> np.ndarray:
    a = doc["accessors"][index]
    assert "sparse" not in a and not a.get("normalized", False)
    v = doc["bufferViews"][a["bufferView"]]
    assert v["buffer"] == 0
    dt = np.dtype({5121: "u1", 5123: "<u2", 5125: "<u4", 5126: "<f4"}[a["componentType"]])
    width = {"SCALAR": 1, "VEC3": 3}[a["type"]]
    stride = v.get("byteStride", width * dt.itemsize)
    start = v.get("byteOffset", 0) + a.get("byteOffset", 0)
    return np.ndarray((a["count"], width), dtype=dt, buffer=binary,
                      offset=start, strides=(stride, dt.itemsize)).copy()


def prepare(args) -> None:
    import trimesh

    started = time.perf_counter()
    source = args.input.resolve()
    raw = source.read_bytes()
    chunks, doc, text, binary = glb_parts(raw)
    assert len(doc["buffers"]) == 1 and "uri" not in doc["buffers"][0]
    assert len(doc["meshes"]) == 1 and len(doc["meshes"][0]["primitives"]) == 1
    primitive = doc["meshes"][0]["primitives"][0]
    assert primitive.get("mode", 4) == 4 and primitive["attributes"].keys() == {"POSITION"}
    assert not doc.get("animations") and not doc.get("skins") and not primitive.get("targets")
    for node in doc.get("nodes", []):
        assert all(k not in node for k in ("matrix", "translation", "rotation", "scale")), "This experiment expects identity nodes"
    positions = accessor(doc, binary, primitive["attributes"]["POSITION"])
    indices = accessor(doc, binary, primitive["indices"])
    assert positions.dtype == np.dtype("<f4") and indices.dtype == np.dtype("<u4")
    assert len(indices) % 3 == 0

    # Exact official build_viewer.py load path; no geometry processing or export.
    scene = trimesh.load(str(source), force="scene", process=False)
    assert len(scene.geometry) == 1
    mesh = next(iter(scene.geometry.values()))
    assert np.array_equal(mesh.vertices, positions)
    assert np.array_equal(mesh.faces, indices.reshape(-1, 3))
    # Inspector also applies the node normal matrix and re-unitizes in float64.
    # The source's checked identity nodes make that matrix exactly identity.
    normal64 = np.asarray(mesh.vertex_normals, dtype=np.float64)
    normal_matrix = np.linalg.pinv(np.eye(3)).T
    normal64 = normal64 @ normal_matrix.T
    normal64 /= np.maximum(np.linalg.norm(normal64, axis=1, keepdims=True), 1e-8)
    normals = np.asarray(normal64, dtype="<f4")
    assert normals.shape == positions.shape and np.isfinite(normals).all()
    lengths = np.linalg.norm(normals.astype(np.float64), axis=1)
    assert np.all((lengths == 0) | (np.abs(lengths - 1) < 1e-6))
    normal_bytes = normals.tobytes(order="C")
    new_binary = binary + normal_bytes
    accessor_index, view_index = len(doc["accessors"]), len(doc["bufferViews"])
    normal_accessor = {"bufferView": view_index, "componentType": 5126,
                       "count": len(normals), "type": "VEC3"}
    normal_view = {"buffer": 0, "byteOffset": len(binary),
                   "byteLength": len(normal_bytes), "target": 34962}
    spans = json_spans(text)
    edits = []
    start, end = spans[("buffers", 0, "byteLength")]
    edits.append((start, end, str(len(new_binary))))
    for path, addition in [(("accessors",), normal_accessor), (("bufferViews",), normal_view)]:
        _, end = spans[path]
        edits.append((end - 1, end - 1, "," + json.dumps(addition, separators=(",", ":"))))
    _, end = spans[("meshes", 0, "primitives", 0, "attributes")]
    edits.append((end - 1, end - 1, ',"NORMAL":' + str(accessor_index)))
    new_text = text
    for start, end, replacement in sorted(edits, reverse=True):
        new_text = new_text[:start] + replacement + new_text[end:]
    json_bytes = new_text.encode("utf-8")
    json_bytes += b" " * (-len(json_bytes) % 4)
    new_chunks = [(kind, json_bytes if kind == 0x4E4F534A else new_binary if kind == 0x004E4942 else payload)
                  for kind, payload in chunks]
    body = b"".join(struct.pack("<II", len(payload), kind) + payload for kind, payload in new_chunks)
    output_bytes = struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body
    _, new_doc, saved_text, saved_binary = glb_parts(output_bytes)
    saved_spans = json_spans(saved_text)
    assert saved_binary[:len(binary)] == binary
    restored = copy.deepcopy(new_doc)
    assert restored["accessors"].pop() == normal_accessor
    assert restored["bufferViews"].pop() == normal_view
    assert restored["meshes"][0]["primitives"][0]["attributes"].pop("NORMAL") == accessor_index
    restored["buffers"][0]["byteLength"] = doc["buffers"][0]["byteLength"]
    assert restored == doc, "Only NORMAL additions and BIN length may change"
    preserved = {}
    for path in [("scene",), ("scenes",), ("nodes",)]:
        old_span, new_span = spans[path], saved_spans[path]
        before, after = text[slice(*old_span)], saved_text[slice(*new_span)]
        assert before == after
        preserved[".".join(path)] = {"rawJsonBytesEqual": True, "sha256": sha(before.encode("utf-8"))}
    accessors_checked = {}
    for name, idx in [("POSITION", primitive["attributes"]["POSITION"]), ("indices", primitive["indices"])]:
        before, after = accessor(doc, binary, idx), accessor(new_doc, saved_binary, idx)
        assert before.tobytes() == after.tobytes()
        accessors_checked[name] = {"rawBytesEqual": True, "sha256": sha(before.tobytes()),
                                   "count": len(before), "dtype": str(before.dtype)}
    assert accessor(new_doc, saved_binary, accessor_index).tobytes() == normal_bytes
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / "canonical.smooth-normals.glb"
    assert not output.exists(), "Refusing to overwrite existing smooth-normal GLB"
    output.write_bytes(output_bytes)
    assert output.read_bytes() == output_bytes
    meta = json.loads(args.identity_meta.read_text())
    assert meta["glbSha256"] == sha(raw)
    identity = np.eye(4).reshape(-1).tolist()
    assert meta["transform"] == identity and meta["inverseTransform"] == identity
    meta["glbSha256"], meta["sourcePath"] = sha(output_bytes), str(output)
    write_json(output_root / "identity-canonical.json", meta)
    inspector = args.inspector_script.resolve()
    report = {"completed": True, "source": file_info(source), "output": file_info(output),
              "identityMeta": file_info(output_root / "identity-canonical.json"),
              "helper": file_info(Path(__file__)), "inspectorScript": file_info(inspector),
              "trimeshVersion": trimesh.__version__,
              "normalMethod": "Official Inspector: trimesh.load(force='scene', process=False).geometry[only].vertex_normals; corner-angle weighting; identity normal matrix; float64 re-unitize; float32 little-endian",
              "positionsAndIndices": accessors_checked, "sceneJson": preserved,
              "originalBinPrefix": {"bytes": len(binary), "sha256": sha(binary), "equal": True},
              "originalDocumentUnchangedExceptNormalAdditionAndBufferLength": True,
              "vertices": len(positions), "triangles": len(indices) // 3,
              "normalData": {"sha256": sha(normal_bytes), "bytes": len(normal_bytes),
                             "zeroLengthCount": int(np.count_nonzero(lengths == 0)),
                             "minLength": float(lengths.min()), "maxLength": float(lengths.max())},
              "elapsedSeconds": time.perf_counter() - started}
    write_json(output_root / "prepare-verification.json", report)
    print(json.dumps(report, ensure_ascii=False))


def compare_bakes(args) -> None:
    from PIL import Image

    before_root, after_root = args.before.resolve(), args.after.resolve()
    before, after = [json.loads((p / "manifest.json").read_text()) for p in [before_root, after_root]]
    report = {"before": file_info(before_root / "manifest.json"),
              "after": file_info(after_root / "manifest.json"), "metadata": {}, "textures": {}, "faces": {}}
    checks = []
    for key in ["schemaVersion", "mode", "layout", "resolution", "tightBounds", "bakeBox", "transform",
                "inverseTransform", "faceOrder", "coordinateSystem", "backend", "versions"]:
        equal = before.get(key) == after.get(key)
        report["metadata"][key] = equal
        checks.append(equal)
    for key in ["colorMap", "normalMap", "intervalMap", "rangeMap"]:
        a, b = before["textures"][key], after["textures"][key]
        raw_a, raw_b = (before_root / a["path"]).read_bytes(), (after_root / b["path"]).read_bytes()
        desc_a = {k: v for k, v in a.items() if k not in {"sha256", "path"}}
        desc_b = {k: v for k, v in b.items() if k not in {"sha256", "path"}}
        assert desc_a == desc_b and len(raw_a) == len(raw_b)
        assert sha(raw_a) == a["sha256"] and sha(raw_b) == b["sha256"]
        shape = (a["height"], a["width"], a["channels"])
        bits_a, bits_b = np.frombuffer(raw_a, "<u2").reshape(shape), np.frombuffer(raw_b, "<u2").reshape(shape)
        different = bits_a != bits_b
        item = {"beforeSha256": sha(raw_a), "afterSha256": sha(raw_b), "rawBytesEqual": raw_a == raw_b,
                "differentUint16Values": int(different.sum()), "alphaUint16Equal": bool(np.array_equal(bits_a[..., 3], bits_b[..., 3])),
                "differentAlphaUint16Values": int(different[..., 3].sum())}
        if key in {"intervalMap", "rangeMap"}:
            checks.append(raw_a == raw_b)
        else:
            checks.append(item["alphaUint16Equal"])
        if key == "normalMap":
            values_a = bits_a.view("<f2").astype(np.float32)
            values_b = bits_b.view("<f2").astype(np.float32)
            item["differentRgbPixels"] = int(np.any(different[..., :3], axis=-1).sum())
            item["maxAbsRgbDifference"] = float(np.max(np.abs(values_a[..., :3] - values_b[..., :3])))
            item["foregroundPixels"] = int(np.count_nonzero(values_a[..., 3] > 0))
        report["textures"][key] = item
    for face in before["faceOrder"]:
        a, b = before["faces"][face], after["faces"][face]
        # Includes camera matrices, frustum, side/up, tile, encodings and pixel counts.
        metadata_equal = {k: v for k, v in a.items() if k not in {"guideSha256", "depthSha256"}} == {k: v for k, v in b.items() if k not in {"guideSha256", "depthSha256"}}
        depth_a, depth_b = (before_root / a["depth"]).read_bytes(), (after_root / b["depth"]).read_bytes()
        assert sha(depth_a) == a["depthSha256"] and sha(depth_b) == b["depthSha256"]
        image_a = np.asarray(Image.open(before_root / a["guide"]).convert("RGBA"))
        image_b = np.asarray(Image.open(after_root / b["guide"]).convert("RGBA"))
        alpha_equal = bool(np.array_equal(image_a[..., 3], image_b[..., 3]))
        item = {"cameraAndFaceMetadataEqual": metadata_equal, "depthBytesEqual": depth_a == depth_b,
                "beforeDepthSha256": sha(depth_a), "afterDepthSha256": sha(depth_b),
                "guideAlphaEqual": alpha_equal, "guideRgbaEqual": bool(np.array_equal(image_a, image_b))}
        checks.extend([metadata_equal, depth_a == depth_b, alpha_equal])
        report["faces"][face] = item
    report["allStructuralChecksPassed"] = all(checks)
    report["normalRgbChanged"] = report["textures"]["normalMap"]["differentRgbPixels"] > 0
    report["renderAdapterBefore"] = before.get("provenance", {}).get("renderAdapter")
    report["renderAdapterAfter"] = after.get("provenance", {}).get("renderAdapter")
    report["scope"] = "Separate bake candidate only; existing native512, prepared guides and Qwen assets are unchanged."
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False))
    assert report["allStructuralChecksPassed"], "Structural mismatch: inspect comparison JSON before using candidate"
    assert report["normalRgbChanged"], "Expected NORMAL attribute to alter normalMap RGB"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--identity-meta", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--inspector-script", type=Path, default=Path("/home/dev/Projects/box-impostor-lab/third_party/3D-Inspector-skills/skills/3d-inspector/scripts/build_viewer.py"))
    p.set_defaults(func=prepare)
    p = sub.add_parser("compare-bakes")
    p.add_argument("--before", type=Path, required=True)
    p.add_argument("--after", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.set_defaults(func=compare_bakes)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
