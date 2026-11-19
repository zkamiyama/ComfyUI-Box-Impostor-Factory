"""Compare A's source crop, H and the standalone I head using official captures."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import trimesh

from build_box_control_front import BG, TEXT, MUTED, FONT, backdrop, sha
from compare_fh import CAMERA_KEYS, project, read
from prepare_head_clay import evidence_helpers

VIEWS = {"front": [0, 0, 1], "iso": [1, 1, 1], "right": [1, 0, 0]}
STAGES = ("source", "H", "I")
MODES = ("lit", "normals")
LABELS = {"source": "A / source head and shoulder crop", "H": "H / full-body Hi3D", "I": "I / head-local Hi3D; approximate fit"}
MATERIAL = {"baseColorFactor": [180, 180, 180, 255], "metallicFactor": 0, "roughnessFactor": 1}


def main():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-proof", type=Path, required=True)
    p.add_argument("--result-proof", type=Path, required=True)
    p.add_argument("--h-provenance", type=Path, required=True)
    p.add_argument("--h-capture", type=Path, action="append", required=True)
    p.add_argument("--head-plan", type=Path, required=True)
    p.add_argument("--mapping", type=Path, required=True)
    p.add_argument("--head-bounds-proof", type=Path, default=here / "A-full-comparison-provenance.json")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    inputs = {}
    def checked(path, expected=None):
        path = Path(path).resolve()
        digest = sha(path)
        assert expected is None or digest == expected, f"Hash mismatch: {path}"
        assert str(path) not in inputs or inputs[str(path)] == digest
        inputs[str(path)] = digest
        return path
    def document(path, expected=None):
        return read(checked(path, expected))
    metadata = {"source": document(args.source_proof), "H": document(args.h_provenance),
                "I": document(args.result_proof)}
    plan = document(args.head_plan)
    mapping = document(args.mapping)
    assert not plan["fullBodyMergeAllowed"] and not mapping["fullBodyMergeAllowed"]
    assert mapping["planSha256"] == sha(args.head_plan)
    assert Path(mapping["plan"]).resolve() == args.head_plan.resolve()
    assert mapping["mapping"]["method"] == "isotropic-y-height-bbox-center-fit"
    assert mapping["mapping"]["exactInverse"] is False
    assert mapping["mapping"]["rotationEstimated"] is False
    assert mapping["mapping"]["axisSpecificStretch"] is False
    source_artifact = plan["artifacts"]["head-source.glb"]
    assert metadata["source"]["source"]["sha256"] == source_artifact["sha256"]
    assert Path(metadata["source"]["source"]["path"]).resolve() == Path(source_artifact["path"]).resolve()
    assert metadata["I"]["source"]["sha256"] == mapping["outputMeshSha256"]
    assert Path(metadata["I"]["source"]["path"]).resolve() == Path(mapping["outputMesh"]).resolve()
    checked(mapping["rawMesh"], mapping["rawMeshSha256"])
    coarse = document(plan["sourceCoarse"]["manifest"], plan["sourceCoarse"]["manifestSha256"])
    checked(plan["sourceCoarse"]["path"], plan["sourceCoarse"]["sha256"])
    assert coarse["asset_id"] == plan["sourceCoarse"]["asset"]
    h_manifest = document(Path(metadata["H"]["assetDirectory"]) / "manifest.json", metadata["H"]["manifestSHA256"])
    for key in ("transform", "inverseTransform", "coordinateSystem"):
        assert plan["canonicalFrame"][key] == coarse[key] == h_manifest[key]
    head = np.asarray(document(args.head_bounds_proof)["headWorldBounds"], dtype=float)
    assert head.shape == (2, 3) and np.isfinite(head).all() and np.all(head[1] > head[0])
    helpers_path = here / "prepare_evidence.py"
    helper = evidence_helpers(checked(helpers_path))
    geometry, triangles, normal_policy = {}, {}, {}
    for stage in STAGES:
        meta, clay = metadata[stage], metadata[stage]["clay"]
        assert clay["material"] == MATERIAL
        raw_path = meta["proxyPath"] if stage == "H" else meta["source"]["path"]
        raw_sha = meta["proxySHA256"] if stage == "H" else meta["source"]["sha256"]
        raw = checked(raw_path, raw_sha)
        clay_path = checked(clay["path"], clay["sha256"])
        viewer = checked(clay["viewer"], clay["viewerSHA256"])
        raw_chunks, clay_chunks = helper["glb_chunks"](raw), helper["glb_chunks"](clay_path)
        assert [(k, b) for k, b in raw_chunks if k != 0x4E4F534A] == [(k, b) for k, b in clay_chunks if k != 0x4E4F534A]
        clay_doc = json.loads(next(b for k, b in clay_chunks if k == 0x4E4F534A))
        assert clay_doc["materials"][0]["pbrMetallicRoughness"] == {
            "baseColorFactor": [180 / 255] * 3 + [1], "metallicFactor": 0, "roughnessFactor": 1}
        assert all(q["material"] == 0 and not any(k.startswith("COLOR_") for k in q["attributes"])
                   for mesh in clay_doc["meshes"] for q in mesh["primitives"])
        normals = helper["normal_accessors"](raw_chunks)
        assert normals == helper["normal_accessors"](clay_chunks) == clay["normalPreservation"]["normalAccessors"]
        assert not any(item["present"] for item in normals), "This comparison expects the recorded absent-NORMAL policy"
        normal_policy[stage] = normals
        scene = trimesh.load(clay_path, force="scene", process=False)
        signature = helper["geometry_signature"](scene)
        assert signature == (clay["geometry"] if stage == "H" else meta["source"]["geometry"])
        triangles[stage] = sum(entry["triangles"] for entry in signature)
        counts = helper["viewer_mesh_counts"](viewer)
        assert triangles[stage] == clay["sourceTriangles"] == clay["payloadTriangles"] == counts["payloadTriangles"]
        assert counts["payloadOriginal"]["triangles"] == triangles[stage]
        points = []
        for node in scene.graph.nodes_geometry:
            matrix, name = scene.graph[node]
            mesh = scene.geometry[name]
            points.append(trimesh.transform_points(mesh.vertices[np.unique(mesh.faces)], matrix))
        geometry[stage] = np.concatenate(points)
    assert triangles["I"] == mapping["outputTriangles"]
    capture_paths = {"source": [Path(metadata["source"]["capture"]["path"])],
                     "H": args.h_capture, "I": [Path(metadata["I"]["capture"]["path"])]}
    base = document(capture_paths["source"][0], metadata["source"]["capture"]["sha256"])
    camera = base["camera"]
    assert camera["projection"] == "orthographic" and camera["pixels"] == [1024, 1024]
    reference = checked(base["referenceManifest"], base["referenceManifestSHA256"])
    assert sha(reference) == plan["sourceCoarse"]["manifestSha256"]
    corners = np.array([[head[x, 0], head[y, 1], head[z, 2]] for x in (0, 1) for y in (0, 1) for z in (0, 1)])
    crops = {}
    for view, vector in VIEWS.items():
        xy = project(corners, vector, camera)
        crops[view] = [*np.floor(xy.min(0)).astype(int).tolist(), *np.ceil(xy.max(0)).astype(int).tolist()]
        assert all(0 <= value <= 1024 for value in crops[view])
    frames = {s: {m: {} for m in MODES} for s in STAGES}
    records = {s: {m: {} for m in MODES} for s in STAGES}
    for stage in STAGES:
        for capture_path in capture_paths[stage]:
            expected_hash = metadata[stage]["capture"]["sha256"] if stage != "H" else None
            cap = document(capture_path, expected_hash)
            assert cap["errors"] == [] and cap["hardwareGPUUsed"] is False and "SwiftShader" in cap["graphicsRenderer"]
            assert cap["backend"] == "Three.js / WebGL2" and cap["stageState"] == "Three.js PBR"
            assert cap["htmlSHA256"] == metadata[stage]["clay"]["viewerSHA256"]
            checked(cap["html"], cap["htmlSHA256"])
            assert cap["referenceManifest"] == str(reference) and cap["referenceManifestSHA256"] == sha(reference)
            assert cap["referenceBounds"] == base["referenceBounds"]
            assert cap["canvasPixels"] == [1024, 1024]
            assert all(cap["camera"][key] == camera[key] for key in CAMERA_KEYS)
            assert cap["lighting"] == base["lighting"]
            assert cap["triangleCounts"]["expected"] == cap["triangleCounts"]["meshPayload"] == triangles[stage]
            assert cap["triangleCounts"]["sourceOriginal"]["triangles"] == triangles[stage]
            for mode in MODES:
                for view, entry in cap["modes"].get(mode, {}).items():
                    if view not in VIEWS:
                        continue
                    if view in records[stage][mode]:
                        assert records[stage][mode][view]["image"]["sha256"] == entry["sha256"], "Ambiguous duplicate capture"
                        continue
                    vector = np.asarray(VIEWS[view], dtype=float)
                    vector /= np.linalg.norm(vector)
                    assert np.allclose(entry["direction"], vector, rtol=0, atol=1e-12)
                    assert np.allclose(entry["expectedPosition"], np.asarray(camera["target"]) + vector * camera["distance"], rtol=0, atol=1e-12)
                    with Image.open(checked(entry["path"], entry["sha256"])) as image:
                        frame = image.convert("RGBA")
                    assert frame.size == (1024, 1024)
                    alpha = np.asarray(frame)[..., 3]
                    assert np.any(alpha == 0) and np.any(alpha > 127), "Transparent, nonempty silhouette required"
                    yy, xx = np.where(alpha > 127)
                    actual = [int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1]
                    xy = project(geometry[stage], vector, camera)
                    expected = [*xy.min(0).tolist(), *xy.max(0).tolist()]
                    clipped = min(expected[:2]) < 0 or max(expected[2:]) > 1024
                    error = float(np.abs(np.asarray(expected) - actual).max())
                    if not clipped:
                        assert error < 3, (stage, mode, view, "projected camera bounds", error)
                    conservative = frame.getchannel("A").getbbox()
                    crop = crops[view]
                    outside = int(np.count_nonzero(alpha) - np.count_nonzero(alpha[crop[1]:crop[3], crop[0]:crop[2]]))
                    frames[stage][mode][view] = frame
                    records[stage][mode][view] = {
                        "capture": str(capture_path.resolve()), "captureSHA256": sha(capture_path), "image": entry,
                        "projectedBounds": expected, "alphaBoundsOver127": actual,
                        "alphaBoundsOver0": list(conservative), "geometryExtendsOutsideViewport": clipped,
                        "cameraProjectionValidated": not clipped, "projectionMaxErrorPx": error,
                        "nonzeroAlphaPixelsOutsideHeadROI": outside,
                        "outsideROIInterpretation": "Includes intentionally excluded shoulders/body; not an accuracy score."}
        for mode in MODES:
            assert set(frames[stage][mode]) == set(VIEWS), (stage, mode, "missing required capture")
    context_crops = {}
    for view in VIEWS:
        boxes = [crops[view]] + [records[s][m][view]["alphaBoundsOver0"] for s in ("source", "I") for m in MODES]
        context_crops[view] = [max(0, min(b[0] for b in boxes) - 8), max(0, min(b[1] for b in boxes) - 8),
                               min(1024, max(b[2] for b in boxes) + 8), min(1024, max(b[3] for b in boxes) + 8)]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    names = [f"A-H-I-{kind}-{mode}.webp" for kind in ("head", "context") for mode in MODES]
    assert not any((output / name).exists() for name in names + ["A-H-I-head-provenance.json"]), "Preserve existing comparison evidence"
    panels = {}
    for kind, rois, scale in (("head", crops, 3), ("context", context_crops, 2)):
        cw = max(420, max((b[2] - b[0]) * scale for b in rois.values()))
        row_heights = {v: (b[3] - b[1]) * scale + 40 for v, b in rois.items()}
        for mode in MODES:
            board = Image.new("RGB", (3 * cw + 80, 190 + sum(row_heights.values())), BG)
            draw = ImageDraw.Draw(board)
            font = lambda size: ImageFont.truetype(FONT, size)
            draw.text((20, 15), f"A / H / I {'head' if kind == 'head' else 'bust context'} comparison — {mode}", font=font(25), fill=TEXT)
            note = "Fixed recorded head ROI" if kind == "head" else "Shared ROI contains all visible source/I pixels; H is the full-body model"
            draw.text((20, 52), f"{note}. Common A camera; {scale}x nearest-neighbor enlargement.", font=font(16), fill=MUTED)
            for col, stage in enumerate(STAGES):
                draw.text((20 + col * (cw + 20), 82), LABELS[stage], font=font(17), fill=TEXT)
                draw.text((20 + col * (cw + 20), 105), f"{triangles[stage]:,} full triangles", font=font(14), fill=MUTED)
            y = 132
            for view in VIEWS:
                draw.text((20, y), view.upper(), font=font(16), fill=MUTED)
                roi = rois[view]
                for col, stage in enumerate(STAGES):
                    crop = backdrop(frames[stage][mode][view].crop(roi))
                    crop = crop.resize((crop.width * scale, crop.height * scale), Image.Resampling.NEAREST)
                    board.paste(crop, (20 + col * (cw + 20) + (cw - crop.width) // 2, y + 28))
                y += row_heights[view]
            draw.text((20, board.height - 48), "I uses approximate Y-height / bbox-center registration; matching height is imposed, not a quality measure.", font=font(15), fill=MUTED)
            draw.text((20, board.height - 25), "No source NORMAL accessors; official shading normals depend on each mesh and its open crop boundary.", font=font(15), fill=MUTED)
            path = output / f"A-H-I-{kind}-{mode}.webp"
            board.save(path, "WEBP", lossless=True)
            assert np.array_equal(np.asarray(Image.open(path)), np.asarray(board))
            panels[path.name] = {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size, "size": list(board.size)}
    for path, digest in inputs.items():
        assert sha(path) == digest, f"Input changed during comparison: {path}"
    proof = {"status": "verified existing official captures", "columns": LABELS, "views": VIEWS, "modes": MODES,
             "camera": {key: camera[key] for key in CAMERA_KEYS}, "referenceManifest": str(reference),
             "referenceManifestSHA256": sha(reference), "headWorldBounds": head.tolist(), "headROIs": crops,
             "contextROIs": context_crops, "contextPolicy": "Common union of source/I nonzero-alpha bounds and fixed head ROI, padded8px and limited to viewport. H body extends outside this bust crop.",
             "material": MATERIAL, "normalAccessors": normal_policy, "sourceTriangles": triangles,
             "mapping": mapping, "captures": records, "inputsSHA256": inputs, "panels": panels,
             "processing": "Original frames unchanged; common per-view ROI, neutral alpha backdrop, integer nearest enlargement, lossless WebP.",
             "limitations": ["I mapping is approximate; no anatomical landmark registration or body merge.",
                             "Normal images describe the renderer's per-mesh computed normals, including crop-boundary changes.",
                             "A viewport-clipped mesh cannot pass the full projected-silhouette camera check; clipping is reported explicitly."],
             "scriptSHA256": sha(__file__)}
    proof_path = output / "A-H-I-head-provenance.json"
    proof_path.write_text(json.dumps(proof, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"panels": panels, "proof": str(proof_path), "proofSHA256": sha(proof_path)}))


if __name__ == "__main__":
    main()
