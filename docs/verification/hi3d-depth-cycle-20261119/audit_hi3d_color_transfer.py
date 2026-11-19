"""Read-only audit of source sRGB texture transfer into existing B GLB COLOR_0."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import subprocess

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree
import trimesh
from trimesh.exchange import gltf

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("repo", type=Path)
p.add_argument("source", type=Path)
p.add_argument("target", type=Path)
p.add_argument("output", type=Path)
a = p.parse_args()
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
before_hashes = [sha(a.source), sha(a.target)]
spec = importlib.util.spec_from_file_location("hi3d_postprocess_audit", a.repo / "scripts/hi3d_postprocess.py")
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
source = trimesh.load(a.source, force="mesh", process=False)
target = trimesh.load(a.target, force="mesh", process=False)
data = a.target.read_bytes()
assert data[:4] == b"glTF"
chunks, offset = {}, 12
while offset < len(data):
    length, kind = struct.unpack_from("<II", data, offset)
    chunks[kind] = data[offset+8:offset+8+length]
    offset += 8 + length
tree = json.loads(chunks[0x4E4F534A])
binary = chunks[0x004E4942]
primitive = tree["meshes"][0]["primitives"][0]
accessor = tree["accessors"][primitive["attributes"]["COLOR_0"]]
assert accessor["componentType"] == 5121 and accessor["type"] == "VEC4" and accessor["normalized"]
view = tree["bufferViews"][accessor["bufferView"]]
color = np.ndarray((accessor["count"], 4), dtype=np.uint8, buffer=binary,
                   offset=view.get("byteOffset", 0)+accessor.get("byteOffset", 0),
                   strides=(view.get("byteStride", 4), 1)).copy()
assert len(color) == len(target.vertices)
assert np.array_equal(color, target.visual.vertex_colors)
dist, nearest = cKDTree(source.vertices).query(target.vertices, k=1, workers=1)
source_srgb_bytes = helper.sample_source_basecolor(source)
expected_raw_bytes = source_srgb_bytes[nearest]
delta = color.astype(int) - expected_raw_bytes.astype(int)
image = np.asarray(source.visual.material.baseColorTexture.convert("RGBA"), dtype=np.float64) / 255
linear_image = image.copy()
linear_image[:, :, :3] = np.where(image[:, :, :3] <= .04045, image[:, :, :3] / 12.92, ((image[:, :, :3] + .055) / 1.055) ** 2.4)
uv = np.asarray(source.visual.uv)
coords = np.stack([(1-np.clip(uv[:, 1],0,1))*(image.shape[0]-1), np.clip(uv[:, 0],0,1)*(image.shape[1]-1)])
source_linear = np.stack([map_coordinates(linear_image[:, :, c], coords, order=1, mode="nearest") for c in range(4)], axis=-1)
raw_factor = source.visual.material.baseColorFactor
factor = np.ones(4, dtype=float) if raw_factor is None else np.asarray(raw_factor, dtype=float)
factor = factor / (255 if factor.max() > 1 else 1)
source_linear *= factor
expected_linear = source_linear[nearest]
actual_linear = color[:, :3].astype(float) / 255
def encode_srgb(x):
    return np.where(x <= .0031308, 12.92*x, 1.055*np.maximum(x,0)**(1/2.4)-.055)
correct_display = np.clip(encode_srgb(expected_linear[:, :3]),0,1)*255
wrong_display = np.clip(encode_srgb(actual_linear),0,1)*255
luminance = expected_raw_bytes[:, :3].mean(axis=1)
samples = sorted(set([0,len(color)//4,len(color)//2,len(color)-1] +
                     [int(np.argmin(np.abs(luminance-np.quantile(luminance,q)))) for q in (.1,.5,.9)]))
loader_path = a.repo / "node_modules/three/examples/jsm/loaders/GLTFLoader.js"
js = """
import {readFileSync} from 'node:fs';
import {GLTFLoader} from %s;
const bytes=readFileSync(process.argv[1]);
const gltf=await new GLTFLoader().parseAsync(bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength),'');
const result=[];
gltf.scene.traverse(mesh=>{
  if(!mesh.isMesh)return;
  const c=mesh.geometry.getAttribute('color');
  result.push({count:c.count,arrayType:c.array.constructor.name,normalized:c.normalized,
    vertexColors:mesh.material.vertexColors,flatShading:mesh.material.flatShading,
    materialColor:mesh.material.color.toArray(),hasColorTexture:!!mesh.material.map,
    samples:%s.map(i=>({vertex:i,linearGetterRGBA:[c.getX(i),c.getY(i),c.getZ(i),c.getW(i)]}))});
});
console.log(JSON.stringify(result));
""" % (json.dumps(loader_path.as_uri()), json.dumps(samples))
loaded = json.loads(subprocess.check_output(["node","--input-type=module","-e",js,str(a.target)], text=True))
assert len(loaded)==1
for item in loaded[0]["samples"]:
    assert np.allclose(item["linearGetterRGBA"],color[item["vertex"]]/255,rtol=0,atol=1e-12)
report = {
    "purpose": "Audit only. Existing A/B GLBs and current inference conditions are unchanged.",
    "source": {"path":str(a.source),"sha256":before_hashes[0],"visualKind":source.visual.kind,
               "textureSize":list(source.visual.material.baseColorTexture.size),"baseColorFactorNormalized":factor.tolist()},
    "target": {"path":str(a.target),"sha256":before_hashes[1],"vertices":len(color),"colorAccessor":accessor,
               "materials":tree.get("materials"),"extensionsUsed":tree.get("extensionsUsed",[])},
    "versions":{"trimesh":trimesh.__version__,"three":json.loads((a.repo/"node_modules/three/package.json").read_text())["version"]},
    "sourceCode": [{"path":str(path),"sha256":sha(path)} for path in (a.repo/"scripts/hi3d_postprocess.py",Path(gltf.__file__),loader_path,a.repo/"worker/runtime.mjs")],
    "encodedByteAgreement": {"equalVertices":int(np.all(delta==0,axis=1).sum()),"totalVertices":len(color),
                              "equalFraction":float(np.all(delta==0,axis=1).mean()),"maxAbsoluteByteDifference":int(np.abs(delta).max()),
                              "meanAbsoluteRGBByteDifference":float(np.abs(delta[:,:3]).mean()),
                              "note":"Source lookup repeated on exported float32 coordinates; agreement is measured over every vertex."},
    "linearMismatch": {"meanAbsoluteRGBDifference":float(np.abs(actual_linear-expected_linear[:,:3]).mean()),
                       "hypotheticalNoLightingSRGB8LevelIncreaseQuantiles":np.quantile((wrong_display-correct_display).mean(axis=1),[.05,.5,.95]).tolist(),
                       "note":"Color transfer math, not a rendered-guide measurement. Correct expectation decodes sRGB before bilinear filtering."},
    "samples":[{"vertex":i,"sourceVertex":int(nearest[i]),"sourceUV":uv[nearest[i]].tolist(),
                "sourceSampledSRGBA8":expected_raw_bytes[i].tolist(),"actualStoredRGBA8":color[i].tolist(),
                "GLTFLinearRGB":actual_linear[i].tolist(),"intendedLinearRGB":expected_linear[i,:3].tolist(),
                "predictedDisplaySRGB8WithoutLighting":np.rint(wrong_display[i]).astype(int).tolist(),
                "correctDisplaySRGB8WithoutLighting":np.rint(correct_display[i]).astype(int).tolist()} for i in samples],
    "installedGLTFLoaderActualParse":loaded,
    "specification":"https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html",
    "conclusion":"Existing B COLOR_0 preserves sampled sRGB byte values without decoding. Installed GLTFLoader exposes those bytes/255 as linear vertex color; the texture-to-vertex transfer has a color-space error.",
    "scope":"This explains a systematic brightening of the B guide, not the separate H3 white-statue failure also observed in A.",
}
assert [sha(a.source),sha(a.target)] == before_hashes
report["existingGLBFilesUnchanged"] = True
a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
print(json.dumps({"output":str(a.output),"sha256":sha(a.output),"byteAgreement":report["encodedByteAgreement"],
                  "linearMismatch":report["linearMismatch"],"examples":report["samples"][:3],"loader":loaded}))
