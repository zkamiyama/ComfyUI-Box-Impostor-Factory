#!/usr/bin/env python3
"""Transfer coarse GLB base color to fixed native Box surfaces and Qwen guides.

Requires NumPy, Pillow, SciPy, trimesh and rtree. The existing preparation and
composition modules must be beside this file. Only new output files are written.
The source GLB/texture is read in place; no model inference or mesh bake is run.
"""

import argparse
import copy
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import shutil
import struct
import time

import numpy as np
from PIL import Image

from prepare_mesh_box import (
    FACES, IDENTITY, asset_files, asset_path, bilinear, bytes_sha,
    descriptor_array, file_record, linear_to_srgb, sha, srgb_to_linear, write_json,
)
from compose_faces import compare_assets, face_to_square, structural_manifest


def glb_document(path):
    data = Path(path).read_bytes()
    if struct.unpack_from("<III", data) != (0x46546C67, 2, len(data)):
        raise ValueError("Expected a complete embedded GLB 2.0")
    chunks, offset = {}, 12
    while offset < len(data):
        length, kind = struct.unpack_from("<II", data, offset)
        chunks[kind] = data[offset + 8:offset + 8 + length]
        offset += 8 + length
    return json.loads(chunks[0x4E4F534A]), chunks[0x004E4942]


def accessor(doc, binary, index):
    spec = doc["accessors"][index]
    if spec.get("sparse") or spec.get("normalized"):
        raise ValueError("This source adapter expects dense unnormalized accessors")
    view = doc["bufferViews"][spec["bufferView"]]
    if view.get("buffer", 0) != 0:
        raise ValueError("Expected embedded buffer")
    dtype = np.dtype({5121: "u1", 5123: "<u2", 5125: "<u4", 5126: "<f4"}[spec["componentType"]])
    count = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[spec["type"]]
    return np.ndarray((spec["count"], count), dtype, buffer=binary,
                      offset=view.get("byteOffset", 0) + spec.get("byteOffset", 0),
                      strides=(view.get("byteStride", count * dtype.itemsize), dtype.itemsize)).copy()


def load_single_world_mesh(path):
    import trimesh
    scene = trimesh.load(path, force="scene", process=False)
    nodes = list(scene.graph.nodes_geometry)
    if len(nodes) != 1:
        raise ValueError("Expected one source geometry node without merging UV seams")
    transform, geometry_name = scene.graph[nodes[0]]
    mesh = scene.geometry[geometry_name].copy()
    mesh.apply_transform(transform)
    return mesh, np.asarray(transform), scene.geometry[geometry_name]


def load_color_source(path):
    doc, binary = glb_document(path)
    primitives = [p for mesh in doc["meshes"] for p in mesh["primitives"]]
    if len(primitives) != 1 or primitives[0].get("mode", 4) != 4:
        raise ValueError("This prior adapter expects the verified single triangle primitive")
    primitive = primitives[0]
    attributes = primitive["attributes"]
    if "COLOR_0" in attributes or doc.get("extensionsRequired"):
        raise ValueError("Vertex colors or required extensions need an explicit appearance adapter")
    material = doc["materials"][primitive["material"]]
    pbr = material["pbrMetallicRoughness"]
    texture_info = pbr["baseColorTexture"]
    if texture_info.get("texCoord", 0) != 0 or texture_info.get("extensions"):
        raise ValueError("Expected untransformed TEXCOORD_0")
    if material.get("alphaMode", "OPAQUE") != "OPAQUE":
        raise ValueError("This prior expects opaque source material; target coverage stays fixed")
    texture = doc["textures"][texture_info["index"]]
    sampler = doc["samplers"][texture["sampler"]]
    if any(sampler.get(k) != v for k, v in {"minFilter": 9729, "magFilter": 9729, "wrapS": 33071, "wrapT": 33071}.items()):
        raise ValueError("Expected the verified LINEAR / CLAMP_TO_EDGE source sampler")
    source_image = doc["images"][texture["source"]]
    view = doc["bufferViews"][source_image["bufferView"]]
    image_bytes = binary[view.get("byteOffset", 0):view.get("byteOffset", 0) + view["byteLength"]]
    with Image.open(io.BytesIO(image_bytes)) as image:
        image_size, image_mode = list(image.size), image.mode
        pixels = np.asarray(image.convert("RGBA"), dtype=np.float32) / 255
    if not np.all(pixels[..., 3] == 1):
        raise ValueError("Expected the verified opaque source image")
    pixels[..., :3] = srgb_to_linear(pixels[..., :3])
    factor = np.asarray(pbr.get("baseColorFactor", [1, 1, 1, 1]), dtype=np.float32)
    if factor.shape != (4,) or not np.isfinite(factor).all() or np.any((factor < 0) | (factor > 1)):
        raise ValueError("Invalid linear baseColorFactor")
    mesh, transform, local_mesh = load_single_world_mesh(path)
    raw_position = accessor(doc, binary, attributes["POSITION"])
    raw_uv = accessor(doc, binary, attributes["TEXCOORD_0"])
    raw_faces = accessor(doc, binary, primitive["indices"]).reshape(-1, 3)
    if not np.array_equal(local_mesh.vertices, raw_position) or not np.array_equal(mesh.faces, raw_faces):
        raise ValueError("GLB loader changed source vertex/triangle order")
    expected_uv = np.column_stack((raw_uv[:, 0], 1 - raw_uv[:, 1]))
    if not np.array_equal(np.asarray(mesh.visual.uv), expected_uv):
        raise ValueError("Trimesh UV convention differs from verified raw glTF v flip")
    if not np.isfinite(mesh.vertices).all() or np.any(mesh.area_faces <= 0):
        raise ValueError("Expected finite source vertices and nondegenerate triangles")
    record = {**file_record(path), "vertices": len(mesh.vertices), "triangles": len(mesh.faces),
              "worldBounds": mesh.bounds.tolist(), "worldFromLocalRows": transform.tolist(),
              "sceneTransformApplications": 1, "vertexAndTriangleOrderPreserved": True,
              "texture": {"embeddedImageSha256": hashlib.sha256(image_bytes).hexdigest(),
                          "size": image_size, "mode": image_mode, "baseColorFactor": factor.tolist(),
                          "sampler": sampler, "uvConvention": "Trimesh v=1-raw glTF v; image row=(1-v)*H-.5",
                          "sampling": "clamped bilinear in linear RGB; x=u*W-.5; factors are linear",
                          "alphaMode": "OPAQUE", "usesMetalRoughnessAoNormalMaps": False}}
    return mesh, np.asarray(mesh.visual.uv, dtype=np.float64), pixels, factor, record


def sample_texture(texture, uv, factor):
    h, w = texture.shape[:2]
    x = np.clip(uv[:, 0] * w - 0.5, 0, w - 1)
    y = np.clip((1 - uv[:, 1]) * h - 0.5, 0, h - 1)
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    value = ((1-fy) * ((1-fx)*texture[y0, x0] + fx*texture[y0, x1])
             + fy * ((1-fx)*texture[y1, x0] + fx*texture[y1, x1]))
    return np.asarray(value * factor, dtype=np.float32)


def surface_points(depth, face, box, pixels):
    """Native depth is box-axis distance, not camera depth or NDC depth."""
    size = depth.shape[0]
    ix, iy = pixels[:, 0], pixels[:, 1]
    d = depth[iy, ix].astype(np.float64)
    lo, hi = np.asarray(box["min"]), np.asarray(box["max"])
    center = (lo + hi) / 2
    side, up = np.asarray(face["side"]), np.asarray(face["up"])
    axis_span = float(np.dot(np.abs(side), hi-lo))
    view = np.asarray(face["viewMatrix"]).reshape(4, 4, order="F")
    projection = np.asarray(face["projectionMatrix"]).reshape(4, 4, order="F")
    camera_to_world = np.linalg.inv(view)
    left, right, bottom, top, near, far = face["frustum"]
    x = left + (ix + .5) / size * (right-left)
    y = top - (iy + .5) / size * (top-bottom)
    margin = float(np.dot(side, camera_to_world[:3, 3]-center) - axis_span/2)
    camera = np.column_stack((x, y, -(margin+d*axis_span), np.ones(len(d))))
    world = (camera @ camera_to_world.T)[:, :3]
    independent = (center + x[:, None]*np.cross(up, side) + y[:, None]*up
                   + (axis_span/2-d*axis_span)[:, None]*side)
    clip = np.column_stack((world, np.ones(len(d)))) @ view.T @ projection.T
    ndc = clip[:, :3] / clip[:, 3, None]
    pixel_again = np.column_stack(((ndc[:, 0]+1)*size/2-.5, (1-ndc[:, 1])*size/2-.5))
    axis_again = .5 - ((world-center) @ side) / axis_span
    errors = {"independentPlaneFormulaMaxWorldError": float(np.max(np.abs(world-independent))),
              "projectionMaxPixelError": float(np.max(np.abs(pixel_again-pixels))),
              "axisDepthRoundTripMaxError": float(np.max(np.abs(axis_again-d))),
              "cameraMargin": margin, "axisSpan": axis_span,
              "near": near, "far": far, "finitePoints": len(d)}
    if errors["independentPlaneFormulaMaxWorldError"] > 1e-9 or errors["projectionMaxPixelError"] > 1e-7 or errors["axisDepthRoundTripMaxError"] > 1e-9:
        raise ValueError("Native depth/camera round-trip failed")
    if np.any(world < lo-1e-8) or np.any(world > hi+1e-8):
        raise ValueError("A reconstructed surface point is outside the bake box")
    return world, errors


def quantiles(values):
    return {name: float(value) for name, value in zip(["p0", "p50", "p90", "p95", "p99", "max"],
                                                      np.quantile(values, [0, .5, .9, .95, .99, 1]))}


def transfer_face(world, coarse, uv, texture, factor, native_normals, chunk_size):
    import trimesh
    colors, distances, cosines = [], [], []
    maximum_barycentric_error = maximum_reconstruction_error = 0.0
    for start in range(0, len(world), chunk_size):
        points = world[start:start+chunk_size]
        nearest, distance, triangle_id = coarse.nearest.on_surface(points)
        triangles = coarse.triangles[triangle_id]
        weights = trimesh.triangles.points_to_barycentric(triangles, nearest)
        if not np.isfinite(weights).all() or np.min(weights) < -1e-6 or np.max(weights) > 1+1e-6:
            raise ValueError("Invalid closest-triangle barycentric coordinates")
        maximum_barycentric_error = max(maximum_barycentric_error, float(np.max(np.abs(weights.sum(axis=1)-1))))
        weights = np.clip(weights, 0, 1)
        weights /= weights.sum(axis=1)[:, None]
        reconstructed = np.einsum("ni,nij->nj", weights, triangles)
        maximum_reconstruction_error = max(maximum_reconstruction_error, float(np.max(np.abs(reconstructed-nearest))))
        interpolated_uv = np.einsum("ni,nij->nj", weights, uv[coarse.faces[triangle_id]])
        colors.append(sample_texture(texture, interpolated_uv, factor)[:, :3])
        distances.append(distance)
        cosines.append(np.sum(coarse.face_normals[triangle_id] * native_normals[start:start+len(points)], axis=1))
    distance = np.concatenate(distances)
    cosines = np.concatenate(cosines)
    height = float(coarse.extents[1])
    report = {"method": "exact closest point on source triangles via trimesh proximity and rtree; barycentric UV",
              "surfaceDistanceWorld": quantiles(distance),
              "surfaceDistanceDividedBySourceHeight": quantiles(distance/height),
              "fractionDistanceAboveHeight": {str(t): float(np.mean(distance > height*t)) for t in [.01, .02, .05]},
              "sourceTriangleVsNativeNormalDot": quantiles(cosines),
              "negativeNormalDotFraction": float(np.mean(cosines < 0)),
              "barycentricSumMaxError": maximum_barycentric_error,
              "closestPointBarycentricMaxWorldError": maximum_reconstruction_error,
              "distanceOrNormalQualityCutoffApplied": False}
    return np.concatenate(colors), report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--guides", type=Path, required=True)
    parser.add_argument("--source-glb", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--validation-samples", type=int, default=256)
    args = parser.parse_args()
    if args.chunk_size < 1 or args.validation_samples < 16:
        parser.error("Use positive chunk size and at least 16 validation samples")
    started = time.monotonic()
    source_manifest = args.baseline_manifest.resolve(strict=True)
    old_root = source_manifest.parent
    manifest = json.loads(source_manifest.read_text())
    if manifest["resolution"] != 512 or manifest["transform"] != IDENTITY or manifest["inverseTransform"] != IDENTITY:
        raise ValueError("Require the fixed native Box512 with identity canonical metadata")
    old_files = asset_files(old_root)
    old_guides = {fid: args.guides.resolve()/slug/"guide.json" for fid, slug in FACES.items()}
    guide_data = {fid: json.loads(path.read_text()) for fid, path in old_guides.items()}
    for fid, guide in guide_data.items():
        face_to_square(manifest, sha(source_manifest), guide)
        for record in guide["files"].values():
            if sha(record["path"]) != record["sha256"]:
                raise ValueError("An original guide changed")
    input_guide_files = {fid: asset_files(path.parent) for fid, path in old_guides.items()}
    color_desc = manifest["textures"]["colorMap"]
    original = descriptor_array(old_root, color_desc, bits=True)
    updated = original.copy()
    normal = descriptor_array(old_root, manifest["textures"]["normalMap"]).astype(np.float32)
    coarse, source_uv, texture, factor, coarse_record = load_color_source(args.source_glb.resolve(strict=True))
    target_path = asset_path(old_root, manifest["proxy"])
    target, target_transform, _ = load_single_world_mesh(target_path)
    if not np.allclose(target_transform, np.eye(4), atol=0, rtol=0) or sha(target_path) != manifest["proxySha256"]:
        raise ValueError("Target mesh transform or hash differs from the immutable baseline")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    reports = {}
    for fid in ["+Z", "+X", "-X", "-Z", "+Y", "-Y"]:
        face_started = time.monotonic()
        face = manifest["faces"][fid]
        tx, ty = face["tile"]
        region = np.s_[ty*512:(ty+1)*512, tx*512:(tx+1)*512]
        old_top = original[region][::-1].view("<f2").astype(np.float32)
        alpha = old_top[..., 3]
        n = normal[region][::-1]
        normals = np.divide(n[..., :3], alpha[..., None], out=np.zeros_like(n[..., :3]), where=alpha[..., None]>0)*2-1
        normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-9)
        depth_path = asset_path(old_root, face["depth"])
        if sha(depth_path) != face["depthSha256"]:
            raise ValueError("Native depth checksum mismatch")
        depth = np.fromfile(depth_path, dtype="<f2").reshape(512, 512)
        if not np.array_equal(np.isfinite(depth), alpha > .5):
            raise ValueError("Native depth and coverage differ")
        iy, ix = np.where(np.isfinite(depth))
        pixels = np.column_stack((ix, iy))
        world, projection_report = surface_points(depth, face, manifest["bakeBox"], pixels)
        selected = np.unique(np.linspace(0, len(world)-1, min(args.validation_samples, len(world)), dtype=int))
        _, target_distance, _ = target.nearest.on_surface(world[selected])
        tolerance = projection_report["axisSpan"]/1024 + 1e-6
        validation = {"samples": len(selected), "distanceWorld": quantiles(target_distance),
                      "conservativeFloat16DepthToleranceWorld": tolerance,
                      "allWithinTolerance": bool(np.all(target_distance <= tolerance)),
                      "targetProxySha256": manifest["proxySha256"]}
        if not validation["allWithinTolerance"]:
            raise ValueError("Reconstructed points do not match the baked target mesh: " + fid)
        rgb, transfer_report = transfer_face(world, coarse, source_uv, texture, factor, normals[iy, ix], args.chunk_size)
        tile = updated[region][::-1]
        tile[iy, ix, :3] = np.asarray(np.clip(rgb, 0, 1)*alpha[iy, ix, None], dtype="<f2").view("<u2")
        reports[fid] = {"face": fid, "foregroundPixels": len(world), "projection": projection_report,
                        "targetSurfaceValidation": validation, "transfer": transfer_report,
                        "seconds": time.monotonic()-face_started}
        print(json.dumps({"stage": "face-albedo-ready", "face": fid, "seconds": reports[fid]["seconds"],
                          "sourceDistanceP95": transfer_report["surfaceDistanceWorld"]["p95"],
                          "targetValidationMax": validation["distanceWorld"]["max"]}), flush=True)
    if not np.array_equal(original[..., 3], updated[..., 3]):
        raise AssertionError("Native coverage bits changed")
    input_record = {"sourceManifest": file_record(source_manifest), "sourceFiles": old_files,
                    "sourceGuides": {fid: file_record(p) for fid, p in old_guides.items()},
                    "sourceGuideFiles": input_guide_files, "coarse": coarse_record,
                    "helpers": {name: file_record(Path(__file__).with_name(name)) for name in
                                [Path(__file__).name, "prepare_mesh_box.py", "compose_faces.py"]},
                    "method": "native-depth-closest-coarse-triangle-uv-v1"}
    asset_id = hashlib.sha256(json.dumps(input_record, sort_keys=True).encode()).hexdigest()
    baseline_root = out/"baseline512"
    shutil.copytree(old_root, baseline_root)
    updated.tofile(baseline_root/color_desc["path"])
    new_manifest = copy.deepcopy(manifest)
    qa_relative = "qa/color-prior-"+asset_id[:16]+".json"
    new_manifest.update(asset_id=asset_id, parentAsset=manifest["asset_id"], mode="coarse-color-prior-rgb-only", qa=qa_relative)
    for desc in [*new_manifest["textures"].values(), new_manifest["baseColor"]]:
        if desc["path"] == color_desc["path"]:
            desc["sha256"] = sha(baseline_root/color_desc["path"])
    new_manifest_path = baseline_root/"manifest.json"
    new_manifest_path.write_text(json.dumps(new_manifest, indent=2, allow_nan=False)+"\n")
    if structural_manifest(manifest) != structural_manifest(new_manifest):
        raise AssertionError("A structural manifest field changed")
    structure_checks = compare_assets(source_manifest, new_manifest_path, list(FACES))
    qa = {**input_record, "faces": reports, "structureChecks": structure_checks,
          "sourceTextureOnlyUsedForRgb": True, "targetGeometryCoverageDepthNormalCameraUnchanged": True,
          "softwareVersions": {name: importlib.metadata.version(name) for name in ["numpy", "scipy", "trimesh", "rtree", "Pillow"]},
          "limitations": ["Nearest source surface may belong to a nearby different body part; distance and normal statistics are diagnostic.",
                          "This transfers the existing coarse appearance; it does not improve target geometry or recover unseen texture detail."]}
    (baseline_root/qa_relative).parent.mkdir(parents=True, exist_ok=True)
    write_json(baseline_root/qa_relative, qa)
    baseline_record = file_record(new_manifest_path)
    result_guides = []
    for fid in ["+Z", "+X", "-X", "-Z", "+Y", "-Y"]:
        old_guide = guide_data[fid]
        dest = out/"guides"/FACES[fid]
        shutil.copytree(old_guides[fid].parent, dest)
        guide = copy.deepcopy(old_guide)
        tx, ty = manifest["faces"][fid]["tile"]
        region = np.s_[ty*512:(ty+1)*512, tx*512:(tx+1)*512]
        color = updated[region][::-1].view("<f2").astype(np.float32)
        alpha = color[..., 3]
        matrix = np.asarray(guide["nativeToSquarePixel"])
        x = (np.arange(guide["size"][0])-matrix[0, 2])/matrix[0, 0]
        y = (np.arange(guide["size"][1])-matrix[1, 2])/matrix[1, 1]
        original_alpha = np.load(old_guide["files"]["alpha"]["path"], allow_pickle=False)
        resampled_alpha = np.clip(bilinear(alpha, x, y), 0, 1)
        alpha_error = float(np.max(np.abs(original_alpha-resampled_alpha)))
        if alpha_error > 1e-7:
            raise ValueError("Prior square guide projection differs from the original")
        n = normal[region][::-1]
        normal_vec = np.divide(n[..., :3], alpha[..., None], out=np.zeros_like(n[..., :3]), where=alpha[..., None]>0)*2-1
        normal_vec /= np.maximum(np.linalg.norm(normal_vec, axis=-1, keepdims=True), 1e-9)
        light = np.asarray(guide["shading"]["keyDirectionWorld"])
        shade = guide["shading"]["ambient"] + guide["shading"]["keyIntensity"]*np.maximum(0, np.sum(normal_vec*light, axis=-1))
        bg = float(srgb_to_linear(guide["shading"]["backgroundSrgbByte"]/255))*(1-original_alpha[..., None])
        albedo = bilinear(color[..., :3], x, y)+bg
        lit = bilinear(color[..., :3]*shade[..., None], x, y)+bg
        Image.fromarray(np.uint8(np.rint(linear_to_srgb(albedo)*255))).save(dest/"albedo.png")
        Image.fromarray(np.uint8(np.rint(linear_to_srgb(lit)*255))).save(dest/"rgb.png")
        guide["baselineManifest"] = baseline_record
        guide["sourceGuide"] = file_record(old_guides[fid])
        guide["files"] = {key: file_record(dest/Path(record["path"]).name) for key, record in old_guide["files"].items()}
        guide["files"]["albedo"] = file_record(dest/"albedo.png")
        for key in ("alpha", "alphaPng", "depth", "depthPreview", "depth512"):
            if guide["files"][key]["sha256"] != old_guide["files"][key]["sha256"]:
                raise AssertionError("A guide alpha/depth file changed")
        guide["shading"].pop("albedoSrgbByte", None)
        guide["shading"]["albedoSource"] = "coarse nearest-triangle baseColor sampled in linear RGB; fixed target alpha"
        guide["shading"]["scope"] = "RGB guide only; albedo.png and baseline color atlas have no added lighting"
        guide["colorPrior"] = {"coarse": coarse_record, "faceDiagnostics": reports[fid],
                               "baselineQa": file_record(baseline_root/qa_relative),
                               "guideAlphaDepthFilesBitIdentical": True, "resampledAlphaMaxDifference": alpha_error}
        guide["helperSha256"] = sha(__file__)
        (dest/"guide.json").write_text(json.dumps(guide, indent=2, allow_nan=False)+"\n")
        face_to_square(new_manifest, baseline_record["sha256"], guide)
        record = {"face": fid, "guide": file_record(dest/"guide.json")}
        result_guides.append(record)
        print(json.dumps({"stage": "guide-ready", **record, "baselineManifest": baseline_record}), flush=True)
    if asset_files(old_root) != old_files or sha(args.source_glb) != coarse_record["sha256"]:
        raise RuntimeError("A source changed during transfer")
    for fid, path in old_guides.items():
        if asset_files(path.parent) != input_guide_files[fid]:
            raise RuntimeError("An original guide changed during transfer")
    final_checks = compare_assets(source_manifest, new_manifest_path, list(FACES))
    result = {"completed": True, **input_record, "baselineManifest": baseline_record,
              "qa": file_record(baseline_root/qa_relative), "structureChecks": final_checks,
              "guides": result_guides, "seconds": time.monotonic()-started}
    write_json(out/"result.json", result)
    print(json.dumps({"completed": True, "result": str(out/"result.json"), "seconds": result["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
