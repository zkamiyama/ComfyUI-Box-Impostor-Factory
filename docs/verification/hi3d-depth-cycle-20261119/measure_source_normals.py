"""Compare original A vertex normals with area-weighted normals from its actual triangles."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct

import numpy as np
import trimesh

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("prepared", type=Path)
parser.add_argument("clay_provenance", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
recipe = json.loads(args.prepared.read_text())
asset = Path(recipe["baselinePath"])
manifest = json.loads((asset / "manifest.json").read_text())
proxy = asset / manifest["proxy"]
data = proxy.read_bytes()
chunks = {}
offset = 12
while offset < len(data):
    size, kind = struct.unpack_from("<II", data, offset)
    chunks[kind] = data[offset + 8:offset + 8 + size]
    offset += 8 + size
tree = json.loads(chunks[0x4E4F534A])
binary = chunks[0x004E4942]
assert len(tree["meshes"]) == 1 and len(tree["meshes"][0]["primitives"]) == 1
primitive = tree["meshes"][0]["primitives"][0]
def accessor(index):
    item = tree["accessors"][index]
    view = tree["bufferViews"][item["bufferView"]]
    dtype = np.dtype({5126: "<f4", 5125: "<u4", 5123: "<u2"}[item["componentType"]])
    components = {"SCALAR": 1, "VEC3": 3}[item["type"]]
    start = view.get("byteOffset", 0) + item.get("byteOffset", 0)
    return np.ndarray((item["count"], components), dtype=dtype, buffer=binary, offset=start,
                      strides=(view.get("byteStride", components * dtype.itemsize), dtype.itemsize)).copy()
positions = accessor(primitive["attributes"]["POSITION"])
indices = accessor(primitive["indices"]).reshape(-1, 3)
source_normals = accessor(primitive["attributes"]["NORMAL"])
scene = trimesh.load(proxy, force="scene", process=False)
assert len(scene.graph.nodes_geometry) == 1
matrix, name = scene.graph[scene.graph.nodes_geometry[0]]
assert np.array_equal(scene.geometry[name].vertices.astype("<f4"), positions)
xyz = trimesh.transform_points(positions, matrix)
source_world = source_normals.astype(float) @ np.linalg.inv(matrix[:3, :3])
source_length = np.linalg.norm(source_world, axis=1)
source_world /= np.maximum(source_length[:, None], 1e-30)
triangles = xyz[indices]
area_vectors = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
accumulated = np.zeros_like(xyz)
np.add.at(accumulated, indices.reshape(-1), np.repeat(area_vectors, 3, axis=0))
lengths = np.linalg.norm(accumulated, axis=1)
per_index = accumulated / np.maximum(lengths[:, None], 1e-30)
_, inverse = np.unique(positions, axis=0, return_inverse=True)
welded = np.zeros((inverse.max() + 1, 3))
np.add.at(welded, inverse, accumulated)
welded /= np.maximum(np.linalg.norm(welded, axis=1, keepdims=True), 1e-30)
welded = welded[inverse]
top = manifest["tightBounds"]["max"][1]
height = top - manifest["tightBounds"]["min"][1]
upper = xyz[xyz[:, 1] >= top - .08 * height]
xmin, xmax = upper[:, 0].min() - .01, upper[:, 0].max() + .01
selected = xyz[(xyz[:, 1] >= top - .18 * height) & (xyz[:, 0] >= xmin) & (xyz[:, 0] <= xmax)]
bounds = np.array([selected.min(0) - [0, .01, .01], selected.max(0) + [0, .01, .01]])
bounds[:, 0] = [xmin, xmax]
head = np.all((xyz >= bounds[0]) & (xyz <= bounds[1]), axis=1)
valid = (source_length > 1e-20) & (lengths > 1e-20)
def stats(candidate, mask):
    angles = np.degrees(np.arccos(np.clip(np.sum(source_world * candidate, axis=1), -1, 1)))[valid & mask]
    return {"vertices": len(angles), "meanDegrees": float(angles.mean()),
            "p50Degrees": float(np.quantile(angles, .5)), "p95Degrees": float(np.quantile(angles, .95)),
            "maxDegrees": float(angles.max()), "fractionAbove10Degrees": float((angles > 10).mean()),
            "fractionAbove30Degrees": float((angles > 30).mean())}
normal_sha = hashlib.sha256(source_normals.tobytes()).hexdigest()
clay = json.loads(args.clay_provenance.read_text())
assert clay["clay"]["normalPreservation"]["normalAccessors"][0]["sha256"] == normal_sha
report = {
    "sourceProxy": str(proxy), "sourceProxySHA256": hashlib.sha256(data).hexdigest(),
    "sourceNormalSHA256": normal_sha, "headBounds": bounds.tolist(),
    "headDefinition": "A upper 18 percent height within upper 8 percent X envelope; 0.01 bake-unit padding",
    "primaryMethod": "Sum unnormalized triangle cross products per vertex index, then normalize; area-weighted.",
    "secondaryMethod": "Same sums merged only at exactly identical vertex positions; seam diagnostic, not an output modification.",
    "geometryOrNormalsModified": False, "uniformClayPreservesSourceNormals": True,
    "perIndex": {"head": stats(per_index, head), "frontFacingHead": stats(per_index, head & (source_world[:, 2] > .2))},
    "exactPositionWelded": {"head": stats(welded, head), "frontFacingHead": stats(welded, head & (source_world[:, 2] > .2))},
    "sourceMaterialNormalTextures": [m.get("normalTexture") for m in tree.get("materials", [])],
    "clayPreparation": clay["clay"],
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
shutil.copyfile(args.clay_provenance, args.output.parent / "A-clay-preparation-provenance.json")
print(json.dumps({"perIndex": report["perIndex"], "exactPositionWelded": report["exactPositionWelded"],
                  "sourceMaterialNormalTextures": report["sourceMaterialNormalTextures"],
                  "normalPreservation": clay["clay"]["normalPreservation"], "output": str(args.output)}))
