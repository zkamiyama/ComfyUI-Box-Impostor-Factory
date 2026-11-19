"""Capture an isolated canonical head GLB with H's unchanged official clay recipe.

The input can be head-source.glb or a completed head-canonical.glb. This command
does not run inference, map a result, or merge the head into a body.
"""
import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path
import re
import shlex
import struct
import subprocess
import sys
import zlib

import numpy as np
from PIL import Image
import trimesh

from compare_fh import CAMERA_KEYS


def evidence_helpers(path):
    """Reuse original helper definitions without executing its asset-only CLI."""
    names = {"sha", "viewer_mesh_counts", "glb_chunks", "normal_accessors",
             "write_clay", "geometry_signature"}
    tree = ast.parse(path.read_text(), filename=str(path))
    definitions = [node for node in tree.body
                   if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in definitions} == names
    namespace = globals().copy()
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), namespace)
    return {name: namespace[name] for name in names}


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--h-provenance", type=Path, required=True)
    parser.add_argument("--h-capture", type=Path, required=True)
    parser.add_argument("--expected-triangles", type=int)
    parser.add_argument("--inspector", type=Path,
                        default=Path("/home/dev/Projects/3D-Inspector-skills/skills/3d-inspector/view.py"))
    parser.add_argument("--node", default="node")
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    reference, inspector = args.reference.resolve(), args.inspector.resolve()
    h_provenance_path, h_capture_path = args.h_provenance.resolve(), args.h_capture.resolve()
    helper_path, capture_script = here / "prepare_evidence.py", here / "capture_inspector.mjs"
    helper = evidence_helpers(helper_path)
    sha = helper["sha"]
    h = json.loads(h_provenance_path.read_text())
    h_capture = json.loads(h_capture_path.read_text())
    material = {"baseColorFactor": [180, 180, 180, 255], "metallicFactor": 0, "roughnessFactor": 1}
    assert h["clay"]["material"] == material
    assert h_capture["htmlSHA256"] == h["clay"]["viewerSHA256"] == sha(h["clay"]["viewer"])
    assert h_capture["errors"] == [] and h_capture["stageState"] == "Three.js PBR"
    assert h_capture["triangleCounts"]["meshPayload"] == h["clay"]["sourceTriangles"]
    assert h_capture["referenceManifest"] == str(reference)
    assert h_capture["referenceManifestSHA256"] == sha(reference)
    h_normals = helper["normal_accessors"](helper["glb_chunks"](h["clay"]["path"]))
    assert h_normals == h["clay"]["normalPreservation"]["normalAccessors"]
    original_hash = sha(source)
    source_normals = helper["normal_accessors"](helper["glb_chunks"](source))
    source_scene = trimesh.load(source, force="scene", process=False)
    signature = helper["geometry_signature"](source_scene)
    triangles = sum(entry["triangles"] for entry in signature)
    assert triangles > 0
    if args.expected_triangles is not None:
        assert triangles == args.expected_triangles, "Unexpected source triangle count"
    output.mkdir(parents=True, exist_ok=False)
    clay, viewer, captures = output / "clay.glb", output / "clay.inspector.html", output / "captures"
    preservation = helper["write_clay"](source, clay)
    assert helper["geometry_signature"](trimesh.load(clay, force="scene", process=False)) == signature
    assert preservation["binaryChunksIdentical"] and preservation["positionNormalIndexBindingsIdentical"]
    assert not preservation["normalRecomputed"]
    viewer_command = [sys.executable, str(inspector), str(clay), "-o", str(viewer), "--max-triangles", str(triangles)]
    capture_command = [args.node, str(capture_script), str(viewer), str(captures), str(reference),
                       "lit,normals", "front,iso,left,right", str(triangles)]
    subprocess.run(viewer_command, check=True)
    counts = helper["viewer_mesh_counts"](viewer)
    assert counts["payloadTriangles"] == triangles
    assert counts["payloadOriginal"]["triangles"] == triangles
    assert sha(source) == original_hash
    metadata = {
        "stage": args.label, "status": "viewer prepared; capture pending",
        "source": {"path": str(source), "sha256": original_hash, "bytes": source.stat().st_size,
                   "triangles": triangles, "geometry": signature, "bounds": source_scene.bounds.tolist()},
        "clay": {"path": str(clay), "sha256": sha(clay), "material": material,
                 "normalPreservation": preservation, "geometryIdenticalAfterReload": True,
                 "sourceTriangles": triangles, "viewerTriangleLimit": triangles, **counts,
                 "viewer": str(viewer), "viewerSHA256": sha(viewer), "viewerBytes": viewer.stat().st_size},
        "reference": {"manifest": str(reference), "sha256": sha(reference)},
        "comparisonH": {"provenance": str(h_provenance_path), "provenanceSHA256": sha(h_provenance_path),
                        "capture": str(h_capture_path), "captureSHA256": sha(h_capture_path),
                        "materialIdentical": True,
                        "normalAccessors": h_normals},
        "tools": {"preparationScript": {"path": str(Path(__file__).resolve()), "sha256": sha(__file__)},
                  "originalHelpers": {"path": str(helper_path), "sha256": sha(helper_path)},
                  "inspector": {"path": str(inspector), "sha256": sha(inspector)},
                  "captureScript": {"path": str(capture_script), "sha256": sha(capture_script)}},
        "commands": {"viewer": viewer_command, "capture": capture_command,
                     "viewerShell": shlex.join(viewer_command), "captureShell": shlex.join(capture_command)},
        "policy": "Material-only derivative; geometry binary unchanged. Original input and official viewer are unchanged.",
        "normalPolicy": {"sourceAccessors": source_normals,
                         "derivedAccessors": preservation["normalAccessors"], "hAccessors": h_normals,
                         "allAccessorsAbsent": not any(item["present"] for group in
                             (source_normals, preservation["normalAccessors"], h_normals) for item in group),
                         "normalRecomputedBeforeInspector": False,
                         "note": "The recorded accessor flags are read from the actual GLBs. An absent NORMAL remains absent; the official viewer computes shading normals."},
    }
    provenance = output / "provenance.json"
    def save():
        provenance.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    save()
    try:
        subprocess.run(capture_command, check=True)
        capture_path = captures / "capture.json"
        capture = json.loads(capture_path.read_text())
        assert capture["errors"] == [] and capture["hardwareGPUUsed"] is False
        assert "SwiftShader" in capture["graphicsRenderer"]
        assert capture["backend"] == "Three.js / WebGL2" and capture["stageState"] == "Three.js PBR"
        assert capture["triangleCounts"]["expected"] == capture["triangleCounts"]["meshPayload"] == triangles
        assert capture["triangleCounts"]["sourceOriginal"]["triangles"] == triangles
        assert capture["htmlSHA256"] == metadata["clay"]["viewerSHA256"] == sha(viewer)
        assert capture["referenceManifest"] == str(reference)
        assert capture["referenceManifestSHA256"] == sha(reference)
        assert capture["referenceBounds"] == h_capture["referenceBounds"]
        assert all(capture["camera"][key] == h_capture["camera"][key] for key in CAMERA_KEYS)
        assert capture["lighting"] == h_capture["lighting"]
        assert capture["canvasPixels"] == [1024, 1024]
        directions = {"front": [0, 0, 1], "iso": [1, 1, 1], "left": [-1, 0, 0], "right": [1, 0, 0]}
        assert set(capture["modes"]) == {"lit", "normals"}
        for mode in capture["modes"].values():
            assert set(mode) == set(directions)
            for name, entry in mode.items():
                direction = np.asarray(directions[name], dtype=float)
                direction /= np.linalg.norm(direction)
                assert np.allclose(entry["direction"], direction, rtol=0, atol=1e-12)
                assert np.allclose(entry["expectedPosition"], np.asarray(capture["camera"]["target"]) + direction * capture["camera"]["distance"], rtol=0, atol=1e-12)
                assert sha(entry["path"]) == entry["sha256"]
                with Image.open(entry["path"]) as image:
                    assert image.size == (1024, 1024)
        assert sha(source) == original_hash and sha(clay) == metadata["clay"]["sha256"]
        metadata.update(status="captured and verified", sourceUnchanged=True, viewerUnchanged=True,
                        commonCameraWithH=True, sameLightingAsH=True,
                        capture={"path": str(capture_path), "sha256": sha(capture_path), "report": capture})
    except Exception as error:
        metadata.update(status="capture or validation failed", error=repr(error))
        save()
        raise
    save()
    print(json.dumps({"stage": args.label, "status": metadata["status"], "triangles": triangles,
                      "sourceSHA256": original_hash, "claySHA256": metadata["clay"]["sha256"],
                      "viewer": str(viewer), "viewerSHA256": sha(viewer),
                      "provenance": str(provenance), "provenanceSHA256": sha(provenance)}))


if __name__ == "__main__":
    main()
