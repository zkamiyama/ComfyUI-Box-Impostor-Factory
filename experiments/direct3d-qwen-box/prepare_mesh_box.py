#!/usr/bin/env python3
"""Make a gray native Box and square Qwen guides without changing its structure.

Input is the existing Factory CLI's six-face mesh bake. No geometry is rendered,
normalized, smoothed or reconstructed here. Only the baseline color atlas RGB is
replaced. Native depth, normals, coverage and proxy bytes remain authoritative.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image


FACES = {"+X": "px", "-X": "nx", "+Y": "py", "-Y": "ny", "+Z": "pz", "-Z": "nz"}
IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bytes_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def srgb_to_linear(value):
    v = np.asarray(value, dtype=np.float32)
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(value):
    v = np.clip(np.asarray(value, dtype=np.float32), 0, 1)
    return np.where(v <= 0.0031308, 12.92 * v, 1.055 * v ** (1 / 2.4) - 0.055)


def asset_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("Invalid asset path: " + str(relative))
    return path


def asset_files(root):
    return {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob("*")) if p.is_file()}


def descriptor_array(root, descriptor, bits=False):
    if descriptor["dtype"] != "float16" or descriptor["endian"] != "little":
        raise ValueError("Expected little-endian half-float atlas")
    if descriptor["rowOrder"] != "increasingShaderV" or descriptor["channels"] != 4:
        raise ValueError("Unexpected native atlas layout")
    path = asset_path(root, descriptor["path"])
    if sha(path) != descriptor["sha256"] or path.stat().st_size != descriptor["byteLength"]:
        raise ValueError("Atlas checksum/length mismatch: " + str(path))
    shape = (descriptor["height"], descriptor["width"], 4)
    return np.fromfile(path, dtype="<u2" if bits else "<f2").reshape(shape)


def bilinear(source, x, y):
    """Tent reconstruction at zero-based pixel centers, with zero outside."""
    h, w = source.shape[:2]
    ix, iy = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    fx, fy = x - ix, y - iy
    channels = source.shape[2:]
    result = np.zeros((len(y), len(x)) + channels, dtype=np.float32)
    for dy, wy in [(0, 1 - fy), (1, fy)]:
        for dx, wx in [(0, 1 - fx), (1, fx)]:
            xx, yy = ix + dx, iy + dy
            valid = ((yy >= 0) & (yy < h))[:, None] & ((xx >= 0) & (xx < w))[None, :]
            weight = (wy[:, None] * wx[None, :] * valid).astype(np.float32)
            if channels:
                weight = weight[..., None]
            result += source[np.clip(yy, 0, h - 1)[:, None], np.clip(xx, 0, w - 1)[None, :]] * weight
    return result


def square_mapping(resolution, size, physical, span):
    width, height = size * np.asarray(physical, dtype=np.float64) / span
    sx, sy = width / resolution, height / resolution
    tx, ty = (size - width) / 2 + sx / 2 - 0.5, (size - height) / 2 + sy / 2 - 0.5
    matrix = np.array([[sx, 0, tx], [0, sy, ty], [0, 0, 1]], dtype=np.float64)
    x, y = (np.arange(size) - tx) / sx, (np.arange(size) - ty) / sy
    return matrix, x, y, [float(width), float(height)]


def square_depth(depth, alpha, x, y):
    valid = np.isfinite(depth) & (alpha > 0)
    weight = bilinear(np.where(valid, alpha, 0), x, y)
    numerator = bilinear(np.where(valid, depth * alpha, 0), x, y)
    output = np.full(weight.shape, np.nan, dtype=np.float32)
    np.divide(numerator, weight, out=output, where=weight > 1e-8)
    preview = np.where(np.isfinite(output), np.clip(1 - output, 0, 1), 0) * np.clip(weight, 0, 1)
    return output, np.uint8(np.rint(np.clip(preview, 0, 1) * 255))


def file_record(path):
    return {"path": str(path.resolve()), "sha256": sha(path), "bytes": path.stat().st_size}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-manifest", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--guide-resolution", type=int, default=1024)
    parser.add_argument("--depth-resolution", type=int, default=512)
    parser.add_argument("--margin", type=float, default=0.02, help="Extra square margin per side, fraction of the native longest span")
    parser.add_argument("--gray", type=int, default=128, help="Baseline unlit albedo, sRGB byte")
    parser.add_argument("--guide-gray", type=int, default=168, help="Lambert guide albedo, sRGB byte")
    parser.add_argument("--background-gray", type=int, default=128)
    args = parser.parse_args()
    if any(not 0 <= x <= 255 for x in [args.gray, args.guide_gray, args.background_gray]):
        parser.error("Gray values must be in [0,255]")
    if not 0 <= args.margin <= 0.25 or not 8 <= args.guide_resolution <= 4096 or not 8 <= args.depth_resolution <= 4096:
        parser.error("Invalid resolution or margin")
    started = time.monotonic()
    manifest_path = args.native_manifest.resolve()
    source_root = manifest_path.parent
    source = json.loads(manifest_path.read_text())
    resolution = source["resolution"]
    if source.get("schemaVersion") != 1 or source.get("layout") != "box" or source.get("faceOrder") != list(FACES):
        raise ValueError("Expected the Factory native six-face Box manifest")
    if source.get("transform") != IDENTITY or source.get("inverseTransform") != IDENTITY:
        raise ValueError("This preparation requires an already canonical mesh baked with identity metadata")
    mesh_record, reference_record = file_record(args.mesh), file_record(args.reference)
    if mesh_record["sha256"] != source["proxySha256"] or sha(asset_path(source_root, source["proxy"])) != mesh_record["sha256"]:
        raise ValueError("Native proxy differs from the supplied canonical mesh")
    if source["baseColor"]["path"] != source["textures"]["colorMap"]["path"]:
        raise ValueError("Expected a native baseline with one color/baseColor atlas")
    input_files = asset_files(source_root)
    color_desc = source["textures"]["colorMap"]
    color_bits = descriptor_array(source_root, color_desc, bits=True)
    normal = descriptor_array(source_root, source["textures"]["normalMap"]).astype(np.float32)
    for descriptor in source["textures"].values():
        descriptor_array(source_root, descriptor)
    color_alpha = color_bits[..., 3].copy()
    settings = {"nativeResolution": resolution, "guideResolution": args.guide_resolution,
                "depthResolution": args.depth_resolution, "margin": args.margin,
                "graySrgbByte": args.gray, "guideGraySrgbByte": args.guide_gray,
                "backgroundSrgbByte": args.background_gray,
                "guideLighting": {"kind": "Lambert from native world-normal atlas", "ambient": 0.35,
                                  "keyIntensity": 0.8, "keyDirectionCamera": [-0.35, 0.5, 0.85]},
                "filter": "bilinear pixel-center sampling, zero outside; premultiplied linear RGB"}
    provenance = {"kind": "native-mesh-square-guides-v1", "sourceManifest": file_record(manifest_path),
                  "sourceMesh": mesh_record, "reference": reference_record,
                  "sourceFiles": input_files, "helper": file_record(Path(__file__)), "settings": settings}
    identity = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    baseline_root = out / "baseline512"
    shutil.copytree(source_root, baseline_root)
    baseline = copy.deepcopy(source)
    linear_gray = float(srgb_to_linear(args.gray / 255))
    alpha_all = color_bits[..., 3].view("<f2").astype(np.float32)
    gray_bits = np.asarray(alpha_all[..., None] * linear_gray, dtype="<f2").view("<u2")
    color_bits[..., :3] = gray_bits
    color_bits.tofile(baseline_root / color_desc["path"])
    if not np.array_equal(color_bits[..., 3], color_alpha):
        raise AssertionError("Coverage half-float bytes changed")
    baseline["asset_id"], baseline["mode"] = identity, "geometry-neutral-gray"
    baseline["textures"]["colorMap"]["sha256"] = sha(baseline_root / color_desc["path"])
    baseline["baseColor"] = copy.deepcopy(baseline["textures"]["colorMap"])
    baseline["provenance"] = {"kind": "native-mesh-gray-rgb-only-v1", "sourceManifest": file_record(manifest_path),
                              "nativeBake": source.get("provenance"), "sourceSha256": mesh_record["sha256"],
                              "helperSha256": provenance["helper"]["sha256"], "graySrgbByte": args.gray}
    baseline_manifest = baseline_root / "manifest.json"
    baseline_manifest.write_text(json.dumps(baseline, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    baseline_files = asset_files(baseline_root)
    preserved = {p: checksum for p, checksum in input_files.items() if p not in ["manifest.json", color_desc["path"]]}
    if any(baseline_files.get(p) != checksum for p, checksum in preserved.items()):
        raise AssertionError("A structural or source-guide file changed")
    baseline_record = file_record(baseline_manifest)
    proof = {"allSourceNoncolorFilesUnchanged": True, "preservedFiles": preserved,
             "allColorAlphaUint16Unchanged": True, "colorAlphaUint16Sha256": bytes_sha(color_alpha),
             "proxyBytesUnchanged": True, "nativeGuidesRetained": "Original native white-albedo guide PNGs remain unchanged in the asset; Qwen clay guides are separate."}
    guide_records = []
    # Publish the +Z guide first so the initial Qwen trial can start immediately.
    for face_id in ["+Z", "+X", "-X", "-Z", "+Y", "-Y"]:
        face = source["faces"][face_id]
        slug = FACES[face_id]
        dest = out / "guides" / slug
        dest.mkdir(parents=True)
        tile_x, tile_y = face["tile"]
        region = np.s_[tile_y * resolution:(tile_y + 1) * resolution,
                       tile_x * resolution:(tile_x + 1) * resolution]
        alpha_bits = color_alpha[region][::-1].copy()
        alpha = alpha_bits.view("<f2").astype(np.float32)
        native_normal = normal[region][::-1].copy()
        depth_path = asset_path(source_root, face["depth"])
        if sha(depth_path) != face["depthSha256"] or sha(asset_path(source_root, face["guide"])) != face["guideSha256"]:
            raise ValueError("Native guide checksum mismatch: " + face_id)
        depth = np.fromfile(depth_path, dtype="<f2").reshape(resolution, resolution).astype(np.float32)
        png_alpha = np.asarray(Image.open(source_root / face["guide"]).convert("RGBA"))[..., 3]
        if not np.array_equal(np.uint8(np.rint(alpha * 255)), png_alpha) or not np.array_equal(np.isfinite(depth), alpha > 0.5):
            raise ValueError("Native atlas/guide/depth alignment mismatch: " + face_id)
        if not np.array_equal(native_normal[..., 3], alpha):
            raise ValueError("Native normal coverage differs: " + face_id)
        normal_vec = np.divide(native_normal[..., :3], alpha[..., None], out=np.zeros_like(native_normal[..., :3]), where=alpha[..., None] > 0) * 2 - 1
        norm = np.linalg.norm(normal_vec, axis=-1, keepdims=True)
        normal_vec = np.divide(normal_vec, norm, out=np.zeros_like(normal_vec), where=norm > 1e-8)
        view = np.asarray(face["viewMatrix"]).reshape(4, 4, order="F")
        light_camera = np.asarray(settings["guideLighting"]["keyDirectionCamera"], dtype=np.float64)
        light_camera /= np.linalg.norm(light_camera)
        light_world = view[:3, :3].T @ light_camera
        shade = 0.35 + 0.8 * np.maximum(0, np.sum(normal_vec * light_world, axis=-1))
        clay = float(srgb_to_linear(args.guide_gray / 255)) * shade * alpha
        left, right, bottom, top, near, far = face["frustum"]
        physical = [right - left, top - bottom]
        span = max(physical) * (1 + 2 * args.margin)
        affine, x, y, content_size = square_mapping(resolution, args.guide_resolution, physical, span)
        square_alpha = np.clip(bilinear(alpha, x, y), 0, 1)
        square_clay = bilinear(clay, x, y) + float(srgb_to_linear(args.background_gray / 255)) * (1 - square_alpha)
        rgb = np.repeat(np.uint8(np.rint(linear_to_srgb(square_clay) * 255))[..., None], 3, axis=-1)
        square_z, depth_image = square_depth(depth, alpha, x, y)
        _, dx, dy, _ = square_mapping(resolution, args.depth_resolution, physical, span)
        _, depth_small = square_depth(depth, alpha, dx, dy)
        Image.fromarray(rgb).save(dest / "rgb.png")
        Image.fromarray(np.uint8(np.rint(square_alpha * 255))).save(dest / "alpha.png")
        Image.fromarray(depth_image).convert("RGB").save(dest / "depth.png")
        Image.fromarray(depth_small).convert("RGB").save(dest / "depth-512.png")
        np.save(dest / "alpha.npy", square_alpha.astype("<f4"), allow_pickle=False)
        np.save(dest / "depth.npy", square_z.astype("<f4"), allow_pickle=False)
        names = {"rgb": "rgb.png", "alpha": "alpha.npy", "alphaPng": "alpha.png", "depth": "depth.npy",
                 "depthPreview": "depth.png", "depth512": "depth-512.png"}
        cx, cy = (left + right) / 2, (bottom + top) / 2
        square_projection = np.array([[2 / span, 0, 0, -2 * cx / span], [0, 2 / span, 0, -2 * cy / span],
                                      [0, 0, -2 / (far - near), -(far + near) / (far - near)], [0, 0, 0, 1]])
        camera = {key: copy.deepcopy(face[key]) for key in ["viewMatrix", "projectionMatrix", "frustum", "up", "side"]}
        structure = {"proxySha256": source["proxySha256"], "depthSha256": face["depthSha256"],
                     "nativeAlphaUint16Sha256": bytes_sha(alpha_bits)}
        structure.update({name + "Sha256": source["textures"][name]["sha256"] for name in ["normalMap", "intervalMap", "rangeMap"]})
        guide = {"kind": "native-mesh-square-guide-v1", "baselineManifest": baseline_record,
                 "sourceManifest": provenance["sourceManifest"], "sourceMesh": mesh_record, "reference": reference_record,
                 "face": face_id, "slug": slug, "tile": face["tile"], "nativeResolution": resolution,
                 "size": [args.guide_resolution, args.guide_resolution], "physicalSize": physical,
                 "squarePhysicalSpan": span, "contentPixels": content_size, "nativeToSquarePixel": affine[:2].tolist(),
                 "squareToNativePixel": np.linalg.inv(affine)[:2].tolist(),
                 "pixelConvention": "top-down rows; zero-based pixel-center indices; no crop or per-face warp",
                 "camera": camera, "squareCamera": {**camera, "projectionMatrix": square_projection.flatten(order="F").tolist(),
                                                      "frustum": [cx-span/2, cx+span/2, cy-span/2, cy+span/2, near, far]},
                 "structure": structure, "files": {key: file_record(dest / name) for key, name in names.items()},
                 "depthEncoding": "linear-axis-0-near-1-far; float32 NaN background; PNG=(1-depth)*coverage, black background",
                 "alphaSource": "Native colorMap alpha only; affine bilinear resampling; no generated mask",
                 "normalSource": "Native normalMap RGB/coverage * 2 - 1; unit world normals; original flat mesh normals",
                 "shading": {**settings["guideLighting"], "keyDirectionWorld": light_world.tolist(),
                             "albedoSrgbByte": args.guide_gray, "backgroundSrgbByte": args.background_gray,
                             "scope": "Qwen RGB guide only; baseline Box albedo and structural maps are independent"},
                 "qa": {"nativeAtlasGuideAlphaExact": True, "nativeDepthValidMatchesAlpha": True,
                        "nativeForegroundPixels": int(np.count_nonzero(alpha)),
                        "squareCoverageSum": float(square_alpha.sum()), "nativeDepthRange": [float(np.nanmin(depth)), float(np.nanmax(depth))]},
                 "helperSha256": provenance["helper"]["sha256"]}
        write_json(dest / "guide.json", guide)
        record = {"face": face_id, "guide": file_record(dest / "guide.json")}
        guide_records.append(record)
        print(json.dumps({"stage": "guide-ready", **record}), flush=True)
    if asset_files(source_root) != input_files or file_record(args.mesh) != mesh_record or file_record(args.reference) != reference_record:
        raise RuntimeError("An input changed during preparation")
    result = {"completed": True, **provenance, "baselineManifest": baseline_record,
              "baselineFiles": baseline_files, "structureProof": proof, "guides": guide_records,
              "seconds": time.monotonic() - started}
    write_json(out / "prepared.json", result)
    print(json.dumps({"stage": "completed", "result": str(out / "prepared.json"), "seconds": result["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
