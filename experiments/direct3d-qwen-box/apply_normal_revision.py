#!/usr/bin/env python3
"""Create isolated normal-only Box revisions and rebind unchanged unlit guides.

Requires the existing prepare_mesh_box.py and compose_faces.py beside this file.
No Qwen result, mesh geometry, original baseline or original guide is modified.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image

from prepare_mesh_box import (
    FACES, asset_files, asset_path, bilinear, bytes_sha, descriptor_array,
    file_record, linear_to_srgb, sha, square_depth, square_mapping,
    srgb_to_linear, write_json,
)
from compose_faces import face_to_square


METHOD = "direct3d-normal-only-revision-v1"
PROXY_SHA = "d7f5e2eb289a4565815782165486485cb9a161094695c1786141c81aea6064ee"


def check_record(record: dict) -> Path:
    path = Path(record["path"]).resolve(strict=True)
    assert sha(path) == record["sha256"]
    if "bytes" in record:
        assert path.stat().st_size == record["bytes"]
    return path


def differences(a, b, path="") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        output = []
        for key in sorted(a.keys() | b.keys()):
            child = f"{path}.{key}" if path else key
            output.extend([child] if key not in a or key not in b else differences(a[key], b[key], child))
        return output
    return [] if a == b else [path]


def revise_baseline(source_root: Path, destination: Path, donor_root: Path,
                    donor: dict, proof_records: dict) -> dict:
    source_manifest = source_root / "manifest.json"
    original = json.loads(source_manifest.read_text())
    before_files = asset_files(source_root)
    assert original["proxySha256"] == PROXY_SHA
    assert sha(asset_path(source_root, original["proxy"])) == PROXY_SHA
    assert original["resolution"] == donor["resolution"] == 512
    for key in ["layout", "tightBounds", "bakeBox", "transform", "inverseTransform", "faceOrder", "coordinateSystem"]:
        assert original[key] == donor[key], f"Donor geometry metadata differs: {key}"
    old_desc, new_desc = original["textures"]["normalMap"], donor["textures"]["normalMap"]
    assert {k: v for k, v in old_desc.items() if k != "sha256"} == {k: v for k, v in new_desc.items() if k != "sha256"}
    old_normal = descriptor_array(source_root, old_desc, bits=True)
    new_normal = descriptor_array(donor_root, new_desc, bits=True)
    old_color = descriptor_array(source_root, original["textures"]["colorMap"], bits=True)
    assert np.array_equal(old_normal[..., 3], new_normal[..., 3])
    assert np.array_equal(old_color[..., 3], new_normal[..., 3])
    rgb_different = int(np.count_nonzero(np.any(old_normal[..., :3] != new_normal[..., :3], axis=-1)))
    assert rgb_different > 0
    for key in ["intervalMap", "rangeMap"]:
        assert original["textures"][key] == donor["textures"][key]
        assert sha(asset_path(source_root, original["textures"][key]["path"])) == original["textures"][key]["sha256"]
    for face in original["faceOrder"]:
        assert original["faces"][face] == donor["faces"][face]
        for path_key, sha_key in [("depth", "depthSha256"), ("guide", "guideSha256")]:
            assert sha(asset_path(source_root, original["faces"][face][path_key])) == original["faces"][face][sha_key]
    input_description = {"method": METHOD, "parentManifest": file_record(source_manifest),
                         "normalDonorManifest": file_record(donor_root / "manifest.json"),
                         "normalSha256": new_desc["sha256"], "helper": file_record(Path(__file__))}
    asset_id = hashlib.sha256(json.dumps(input_description, sort_keys=True).encode()).hexdigest()
    relative_qa = f"qa/normal-only-{asset_id[:16]}.json"
    revised = copy.deepcopy(original)
    revised["textures"]["normalMap"]["sha256"] = new_desc["sha256"]
    revised.update(asset_id=asset_id, parentAsset=original["asset_id"], mode="normal-only-revision", qa=relative_qa)
    allowed = {"textures.normalMap.sha256", "asset_id", "parentAsset", "mode", "qa"}
    manifest_diffs = differences(original, revised)
    assert set(manifest_diffs).issubset(allowed)
    assert "textures.normalMap.sha256" in manifest_diffs
    shutil.copytree(source_root, destination)
    shutil.copyfile(asset_path(donor_root, new_desc["path"]), destination / old_desc["path"])
    (destination / "manifest.json").write_text(json.dumps(revised, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    after_files = asset_files(destination)
    assert before_files.keys() == after_files.keys()
    changed_files = [path for path in before_files if before_files[path] != after_files[path]]
    assert set(changed_files) == {"manifest.json", old_desc["path"]}
    assert after_files[original["proxy"]] == PROXY_SHA
    assert np.array_equal(descriptor_array(destination, revised["textures"]["normalMap"], bits=True)[..., 3], old_normal[..., 3])
    qa = {"completed": True, **input_description, "proofs": proof_records,
          "normalOnlyRevision": True, "changedExistingFiles": changed_files,
          "manifestDifferences": manifest_diffs, "allowedManifestDifferences": sorted(allowed),
          "sourceFiles": before_files, "outputFilesBeforeAddingThisQa": after_files,
          "proxyBytesUnchanged": True, "colorBytesUnchanged": True,
          "allSixDepthAndGuideBytesUnchanged": True, "intervalRangeBytesUnchanged": True,
          "cameraBoundsTransformMetadataUnchanged": True, "normalAlphaUint16DifferenceCount": 0,
          "colorAlphaUint16DifferenceCount": 0, "normalRgbChangedPixels": rgb_different,
          "normalMap": {"before": old_desc["sha256"], "after": new_desc["sha256"]},
          "scope": "Only the normalMap RGB changes; original proxy bytes and all other asset data remain authoritative."}
    qa_path = destination / relative_qa
    qa_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(qa_path, qa)
    final_files = asset_files(destination)
    assert set(final_files) - set(before_files) == {relative_qa}
    assert all(final_files[p] == digest for p, digest in before_files.items()
               if p not in {"manifest.json", old_desc["path"]})
    assert asset_files(source_root) == before_files
    return {"sourceManifest": file_record(source_manifest), "manifest": file_record(destination / "manifest.json"),
            "qa": file_record(qa_path), "normalAlphaUint16DifferenceCount": 0,
            "normalRgbChangedPixels": rgb_different, "allOtherOriginalFilesIdentical": True,
            "originalSourceFilesUnchanged": True}


def png_bytes(array: np.ndarray, rgb=False) -> bytes:
    image = Image.fromarray(array)
    if rgb:
        image = image.convert("RGB")
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def npy_bytes(array: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.save(stream, array.astype("<f4"), allow_pickle=False)
    return stream.getvalue()


def replay_unlit_guide(root: Path, manifest: dict, guide: dict, color_bits: np.ndarray) -> dict:
    """Reproduce all input file bytes using color/alpha/depth only, never normals."""
    assert guide["shading"].get("applied") is False
    assert guide["rgbSelection"]["sourceKey"] == "albedo"
    assert guide["files"]["rgb"]["sha256"] == guide["files"]["albedo"]["sha256"]
    face, resolution = manifest["faces"][guide["face"]], manifest["resolution"]
    tx, ty = face["tile"]
    color = color_bits[ty * resolution:(ty + 1) * resolution,
                       tx * resolution:(tx + 1) * resolution][::-1].view("<f2").astype(np.float32)
    alpha = color[..., 3]
    matrix = np.asarray(guide["nativeToSquarePixel"])
    x = (np.arange(guide["size"][0]) - matrix[0, 2]) / matrix[0, 0]
    y = (np.arange(guide["size"][1]) - matrix[1, 2]) / matrix[1, 1]
    square_alpha = np.clip(bilinear(alpha, x, y), 0, 1)
    saved_alpha = np.load(check_record(guide["files"]["alpha"]), allow_pickle=False)
    assert np.array_equal(square_alpha, saved_alpha)
    background = float(srgb_to_linear(guide["shading"]["backgroundSrgbByte"] / 255)) * (1 - saved_alpha[..., None])
    albedo = bilinear(color[..., :3], x, y) + background
    rgb_bytes = png_bytes(np.uint8(np.rint(linear_to_srgb(albedo) * 255)))
    depth_path = asset_path(root, face["depth"])
    assert sha(depth_path) == face["depthSha256"]
    depth = np.fromfile(depth_path, dtype="<f2").reshape(resolution, resolution).astype(np.float32)
    square_z, depth_preview = square_depth(depth, alpha, x, y)
    _, dx, dy, _ = square_mapping(resolution, 512, guide["physicalSize"], guide["squarePhysicalSpan"])
    _, depth_small = square_depth(depth, alpha, dx, dy)
    regenerated = {"rgb": rgb_bytes, "albedo": rgb_bytes, "alpha": npy_bytes(square_alpha),
                   "alphaPng": png_bytes(np.uint8(np.rint(square_alpha * 255))),
                   "depth": npy_bytes(square_z), "depthPreview": png_bytes(depth_preview, rgb=True),
                   "depth512": png_bytes(depth_small, rgb=True)}
    assert regenerated.keys() == guide["files"].keys()
    records = {}
    for key, raw in regenerated.items():
        original_path = check_record(guide["files"][key])
        original_bytes = original_path.read_bytes()
        assert raw == original_bytes, f"Unlit replay is not byte-identical: {guide['slug']} / {key}"
        records[key] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "allBytesEqual": True}
    return {"normalMapReadByReplay": False, "allSevenFilesByteIdentical": True, "files": records,
            "dependencies": ["colorMap RGB and alpha", "native depth", "fixed camera/affine", "sRGB conversion and fixed background"],
            "method": "In-memory replay of unlit albedo/alpha/depth PNG and NPY encodings; no normal or lighting input"}


def rebind_guides(source_root: Path, destination: Path, prior: dict) -> dict:
    before_files = asset_files(source_root)
    target_manifest = check_record(prior["manifest"])
    target_root = target_manifest.parent
    manifest = json.loads(target_manifest.read_text())
    color_bits = descriptor_array(target_root, manifest["textures"]["colorMap"], bits=True)
    results = []
    destination.mkdir(parents=True, exist_ok=False)
    for face, slug in FACES.items():
        source_json = source_root / slug / "guide.json"
        original = json.loads(source_json.read_text())
        assert original["face"] == face and original["slug"] == slug
        assert original["baselineManifest"] == prior["sourceManifest"]
        check_record(original["baselineManifest"])
        for record in original["files"].values():
            check_record(record)
        replay = replay_unlit_guide(target_root, manifest, original, color_bits)
        revised = copy.deepcopy(original)
        revised["baselineManifest"] = copy.deepcopy(prior["manifest"])
        revised["structure"]["normalMapSha256"] = manifest["textures"]["normalMap"]["sha256"]
        revised["compositionTargetRevision"] = {
            "method": METHOD, "sourceGuide": file_record(source_json),
            "sourceBaseline": original["baselineManifest"], "targetBaseline": prior["manifest"],
            "normalOnlyRevisionQa": prior["qa"], "qwenInputImagesChanged": False,
            "description": "Qwen入力画像は変更なし。composition targetのnormalMapだけ更新。",
            "preservedGuideFiles": "Original absolute paths, byte lengths and SHA256 values are retained.",
            "historicalFields": "sourceManifest, sourceMesh, normalSource and existing provenance describe original guide production; unlit RGB does not use normals.",
            "qwenResults": "Existing Qwen result/prepared/prompt manifests are not edited or relabelled.",
            "unlitReplay": replay,
        }
        expected_changes = {"baselineManifest.path", "baselineManifest.sha256", "baselineManifest.bytes",
                            "structure.normalMapSha256", "compositionTargetRevision"}
        assert set(differences(original, revised)).issubset(expected_changes)
        assert revised["files"] == original["files"]
        face_to_square(manifest, prior["manifest"]["sha256"], revised)
        tx, ty = manifest["faces"][face]["tile"]
        resolution = manifest["resolution"]
        region = color_bits[ty * resolution:(ty + 1) * resolution,
                            tx * resolution:(tx + 1) * resolution][::-1]
        expected_structure = {"proxySha256": manifest["proxySha256"], "depthSha256": manifest["faces"][face]["depthSha256"],
                              "nativeAlphaUint16Sha256": bytes_sha(region[..., 3])}
        expected_structure.update({key + "Sha256": manifest["textures"][key]["sha256"]
                                   for key in ("normalMap", "intervalMap", "rangeMap")})
        assert revised["structure"] == expected_structure
        output_json = destination / slug / "guide.json"
        output_json.parent.mkdir(parents=True)
        write_json(output_json, revised)
        results.append({"face": face, "slug": slug, "sourceGuide": file_record(source_json),
                        "guide": file_record(output_json), "allImageRecordsUnchanged": True,
                        "cameraMappingProxyUnchanged": True, "unlitReplay": replay})
    assert asset_files(source_root) == before_files
    report = {"completed": True, "method": METHOD, "sourceGuides": str(source_root),
              "sourceFiles": before_files, "targetBaseline": prior["manifest"],
              "allQwenInputImageBytesUnchanged": True, "allUnlitReplayBytesIdentical": True,
              "normalMapIsCompositionTargetOnly": True, "guides": results}
    write_json(destination / "result.json", report)
    return {"result": file_record(destination / "result.json"), "guides": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gray", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--smooth-native", type=Path, required=True)
    parser.add_argument("--guides", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    roots = {key: getattr(args, key).resolve(strict=True) for key in ("gray", "prior", "smooth_native", "guides")}
    output_root = args.output_root.resolve(strict=True)
    sources_before = {key: asset_files(root) for key, root in roots.items()}
    donor = json.loads((roots["smooth_native"] / "manifest.json").read_text())
    comparison_path = roots["smooth_native"].parent / "native-comparison.json"
    preparation_path = roots["smooth_native"].parent / "prepare-verification.json"
    comparison, preparation = [json.loads(p.read_text()) for p in (comparison_path, preparation_path)]
    assert comparison["allStructuralChecksPassed"] and comparison["normalRgbChanged"]
    assert comparison["after"]["sha256"] == sha(roots["smooth_native"] / "manifest.json")
    assert preparation["source"]["sha256"] == PROXY_SHA
    assert preparation["output"]["sha256"] == donor["proxySha256"]
    assert sha(asset_path(roots["smooth_native"], donor["proxy"])) == donor["proxySha256"]
    proofs = {"normalPreparation": file_record(preparation_path), "nativeBakeComparison": file_record(comparison_path)}
    revisions = {}
    for key in ("gray", "prior"):
        revisions[key] = revise_baseline(roots[key], output_root / "baselines" / key, roots["smooth_native"], donor, proofs)
        print(json.dumps({"stage": "baseline-ready", "variant": key, **revisions[key]}), flush=True)
    guides = rebind_guides(roots["guides"], output_root / "guides-unlit", revisions["prior"])
    for key, root in roots.items():
        assert asset_files(root) == sources_before[key], f"Original source changed: {key}"
    report = {"completed": True, "method": METHOD, "helper": file_record(Path(__file__)),
              "dependencies": [file_record(Path(__file__).with_name(name)) for name in ("prepare_mesh_box.py", "compose_faces.py")],
              "baselines": revisions, "guides": guides, "allOriginalSourceFilesUnchanged": True,
              "qwenRunManifestsUnmodified": True, "seconds": time.perf_counter() - started}
    write_json(output_root / "normal-revision-result.json", report)
    print(json.dumps({"completed": True, "result": file_record(output_root / "normal-revision-result.json"),
                      "seconds": report["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
