"""Measure changes between actual meshes/depth maps in the shared canonical frame."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates
import trimesh

# Horizontal image axis/sign, vertical image axis/sign, near-to-far depth axis/sign.
AXES = {
    "+X": (2, -1, 1, -1, 0, -1), "-X": (2, 1, 1, -1, 0, 1),
    "+Y": (0, 1, 2, 1, 1, -1), "-Y": (0, 1, 2, -1, 1, 1),
    "+Z": (0, 1, 1, -1, 2, -1), "-Z": (0, -1, 1, -1, 2, 1),
}
def world(coordinate, low, high, sign):
    return low + coordinate * (high - low) if sign > 0 else high - coordinate * (high - low)

def pixel(coordinate, low, high, sign, resolution):
    fraction = (coordinate - low) / (high - low) if sign > 0 else (high - coordinate) / (high - low)
    return fraction * resolution - .5

def summary(values):
    if not len(values):
        return {"pixels": 0}
    absolute = np.abs(values)
    return {"pixels": len(values), "meanAbsolute": float(absolute.mean()),
            "rmse": float(np.sqrt(np.mean(values * values))),
            "p95Absolute": float(np.quantile(absolute, .95)), "maxAbsolute": float(absolute.max())}

def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("work", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--stages", default="A,B,C")
args = parser.parse_args()
stages = args.stages.split(",")
records = {}
for stage in stages:
    metadata = json.loads((args.work / f"{stage}-provenance.json").read_text())
    asset = Path(metadata["assetDirectory"])
    manifest_path = asset / "manifest.json"
    records[stage] = {"asset": asset, "manifest": json.loads(manifest_path.read_text()), "manifestSHA256": sha(manifest_path)}

base = records[stages[0]]
bm = base["manifest"]
resolution = bm["resolution"]
bb = np.array([bm["bakeBox"]["min"], bm["bakeBox"]["max"]])
yy, xx = np.indices((resolution, resolution), dtype=float)
source_scene = trimesh.load(base["asset"] / bm["proxy"], force="scene", process=False)
source_points = np.concatenate([trimesh.transform_points(source_scene.geometry[source_scene.graph[node][1]].vertices, source_scene.graph[node][0])
                                for node in source_scene.graph.nodes_geometry])
top = bm["tightBounds"]["max"][1]
height = top - bm["tightBounds"]["min"][1]
upper = source_points[source_points[:, 1] >= top - .08 * height]
xmin, xmax = upper[:, 0].min() - .01, upper[:, 0].max() + .01
selected = source_points[(source_points[:, 1] >= top - .18 * height) & (source_points[:, 0] >= xmin) & (source_points[:, 0] <= xmax)]
head = np.array([selected.min(0) - [0, .01, .01], selected.max(0) + [0, .01, .01]])
head[:, 0] = [xmin, xmax]
report = {
    "baselineStage": stages[0], "units": "canonical bake units",
    "interpretation": "Relative change, not an accuracy score. There is no geometric ground truth.",
    "headDefinition": {"bounds": head.tolist(), "method": "A upper 18 percent height within upper 8 percent X envelope; 0.01 bake-unit padding"},
    "stages": {}, "comparisons": {},
}
for stage, record in records.items():
    manifest = record["manifest"]
    mesh = trimesh.load(record["asset"] / manifest["proxy"], force="scene", process=False)
    vertices = triangles = head_vertices = head_triangles = 0
    for node in mesh.graph.nodes_geometry:
        transform, name = mesh.graph[node]
        geometry = mesh.geometry[name]
        xyz = trimesh.transform_points(geometry.vertices, transform)
        vertices += len(xyz)
        triangles += len(geometry.faces)
        head_vertices += int(np.all((xyz >= head[0]) & (xyz <= head[1]), axis=1).sum())
        centers = xyz[geometry.faces].mean(axis=1)
        head_triangles += int(np.all((centers >= head[0]) & (centers <= head[1]), axis=1).sum())
    report["stages"][stage] = {
        "assetId": manifest["asset_id"], "manifestSHA256": record["manifestSHA256"],
        "proxySHA256": manifest["proxySha256"], "bakeBox": manifest["bakeBox"],
        "vertices": vertices, "triangles": triangles, "headVertices": head_vertices, "headTriangles": head_triangles,
    }
for stage in stages[1:]:
    record = records[stage]
    manifest = record["manifest"]
    box = np.array([manifest["bakeBox"]["min"], manifest["bakeBox"]["max"]])
    size = manifest["resolution"]
    result = {}
    for face, (haxis, hsign, vaxis, vsign, daxis, dsign) in AXES.items():
        original = np.fromfile(base["asset"] / bm["faces"][face]["depth"], dtype="<f2").astype(float).reshape(resolution, resolution)
        candidate = np.fromfile(record["asset"] / manifest["faces"][face]["depth"], dtype="<f2").astype(float).reshape(size, size)
        horizontal = world((xx + .5) / resolution, bb[0, haxis], bb[1, haxis], hsign)
        vertical = world((yy + .5) / resolution, bb[0, vaxis], bb[1, vaxis], vsign)
        cols = pixel(horizontal, box[0, haxis], box[1, haxis], hsign, size)
        rows = pixel(vertical, box[0, vaxis], box[1, vaxis], vsign, size)
        original_mask = np.isfinite(original)
        candidate_mask = np.isfinite(candidate)
        nearest_mask = map_coordinates(candidate_mask.astype(float), [rows, cols], order=0, mode="constant", cval=0) > .5
        weights = map_coordinates(candidate_mask.astype(float), [rows, cols], order=1, mode="constant", cval=0)
        surface = world(np.nan_to_num(candidate), box[0, daxis], box[1, daxis], dsign)
        sampled = map_coordinates(surface * candidate_mask, [rows, cols], order=1, mode="constant", cval=0) / np.maximum(weights, 1e-9)
        original_surface = world(original, bb[0, daxis], bb[1, daxis], dsign)
        common = original_mask & (weights >= .999)
        head_mask = ((horizontal >= head[0, haxis]) & (horizontal <= head[1, haxis])
                     & (vertical >= head[0, vaxis]) & (vertical <= head[1, vaxis])
                     & (original_surface >= head[0, daxis]) & (original_surface <= head[1, daxis]))
        difference = sampled - original_surface
        union = original_mask | nearest_mask
        result[face] = {
            "silhouetteIoU": float((original_mask & nearest_mask).sum() / max(1, union.sum())),
            "surfaceDifference": summary(difference[common]),
            "headSurfaceDifference": summary(difference[common & head_mask]),
            "maskSampling": "nearest on baseline physical rays",
            "depthSampling": "bilinear physical surface, all contributing pixels valid",
        }
    report["comparisons"][stage + "-vs-" + stages[0]] = result
args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
print(json.dumps(report["stages"]))
