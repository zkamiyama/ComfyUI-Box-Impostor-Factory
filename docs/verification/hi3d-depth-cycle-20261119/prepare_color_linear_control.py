"""Prepare a color-only diagnostic from the saved actual Hi3D raw mesh.
CPU only: no inference, native bake, browser, smoothing, or writes to existing assets.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree
import trimesh

def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def accessor(doc, binary, index):
    spec = doc["accessors"][index]
    assert not spec.get("sparse")
    dt = np.dtype({5120:"i1",5121:"u1",5122:"<i2",5123:"<u2",5125:"<u4",5126:"<f4"}[spec["componentType"]])
    count = {"SCALAR":1,"VEC2":2,"VEC3":3,"VEC4":4}[spec["type"]]
    view = doc["bufferViews"][spec["bufferView"]]
    assert view.get("buffer",0) == 0
    return np.ndarray((spec["count"],count), dtype=dt, buffer=binary,
                      offset=view.get("byteOffset",0)+spec.get("byteOffset",0),
                      strides=(view.get("byteStride",count*dt.itemsize),dt.itemsize)).copy(), spec

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("repo", type=Path)
    p.add_argument("run", type=Path)
    p.add_argument("work", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    sys.path.insert(0,str(a.repo))
    from box_nodes import depth_mesh
    receipt_path = a.run/"B/hi3d-completed.json"
    receipt = json.loads(receipt_path.read_text())
    original = Path(receipt["coarseMesh"])
    baseline = Path(receipt["outputProxy"])
    raw = baseline.parent/"hi3d-raw.glb"
    hashes_before = {str(path):sha(path) for path in (original,baseline,raw)}
    assert hashes_before[str(original)] == receipt["coarseSha256"]
    assert hashes_before[str(baseline)] == receipt["outputProxySha256"]
    assert hashes_before[str(raw)] == receipt["canonicalMetadata"]["refinement"]["hi3dSha256"]
    assert receipt["resolution"] == receipt["canonicalMetadata"]["refinement"]["resolution"] == 1024
    helper_path = a.repo/"scripts/hi3d_postprocess.py"
    spec = importlib.util.spec_from_file_location("hi3d_color_control",helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    mesh, report = helper.prepare(original,raw,smooth=0,max_nearest_p95=.05)
    folder = a.work/"color-linear"
    folder.mkdir(parents=True,exist_ok=True)
    target = folder/"proxy.glb"
    helper.export_prepared_glb(mesh,target)
    old_doc,old_bin = depth_mesh._decode(baseline.read_bytes())
    new_doc,new_bin = depth_mesh._decode(target.read_bytes())
    assert len(old_doc["meshes"]) == len(new_doc["meshes"]) == 1
    oldp,newp = old_doc["meshes"][0]["primitives"][0],new_doc["meshes"][0]["primitives"][0]
    assert len(old_doc["meshes"][0]["primitives"]) == len(new_doc["meshes"][0]["primitives"]) == 1
    assert old_doc.get("nodes") == new_doc.get("nodes")
    assert old_doc.get("scenes") == new_doc.get("scenes")
    assert old_doc.get("materials") == new_doc.get("materials")
    assert set(oldp["attributes"]) == set(newp["attributes"])
    equal_attributes = {}
    for name in oldp["attributes"]:
        if name == "COLOR_0": continue
        x,xspec = accessor(old_doc,old_bin,oldp["attributes"][name])
        y,yspec = accessor(new_doc,new_bin,newp["attributes"][name])
        assert x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes(), name
        equal_attributes[name] = {"byteIdentical":True,"count":len(x),"sha256":hashlib.sha256(x.tobytes()).hexdigest()}
    old_indices,_ = accessor(old_doc,old_bin,oldp["indices"])
    new_indices,_ = accessor(new_doc,new_bin,newp["indices"])
    assert old_indices.dtype == new_indices.dtype and old_indices.tobytes() == new_indices.tobytes()
    old_colors,old_color_spec = accessor(old_doc,old_bin,oldp["attributes"]["COLOR_0"])
    colors,color_spec = accessor(new_doc,new_bin,newp["attributes"]["COLOR_0"])
    assert color_spec["componentType"] == 5126 and color_spec["type"] == "VEC4" and not color_spec.get("normalized",False)
    assert len(colors) == 499920
    assert "NORMAL" not in oldp["attributes"] and "NORMAL" not in newp["attributes"]
    assert report["smooth_iterations"] == 0
    source = trimesh.load(original,force="mesh",process=False)
    source_doc,source_bin = depth_mesh._decode(original.read_bytes())
    sourcep = source_doc["meshes"][0]["primitives"][0]
    assert "COLOR_0" not in sourcep["attributes"], "Independent expectation requires the actual texture-only source"
    texture = np.asarray(source.visual.material.baseColorTexture.convert("RGBA"),dtype=np.float64)/255
    texture[...,:3] = np.where(texture[...,:3] <= .04045,texture[...,:3]/12.92,((texture[...,:3]+.055)/1.055)**2.4)
    uv = np.asarray(source.visual.uv,dtype=np.float64)
    coords = np.stack([(1-np.clip(uv[:,1],0,1))*(texture.shape[0]-1),np.clip(uv[:,0],0,1)*(texture.shape[1]-1)])
    sampled = np.stack([map_coordinates(texture[...,c],coords,order=1,mode="nearest") for c in range(4)],axis=-1)
    factor = np.asarray(source_doc["materials"][sourcep["material"]].get("pbrMetallicRoughness",{}).get("baseColorFactor",[1,1,1,1]),dtype=float)
    sampled *= factor
    _,nearest = cKDTree(source.vertices).query(mesh.vertices,k=1,workers=1)
    expected = sampled[nearest]
    error = np.abs(colors.astype(float)-expected)
    assert error.max() < 5e-7, error.max()
    alpha_error = np.abs(colors[:,3]-old_colors[:,3]/255)
    assert alpha_error.max() < 1e-7
    metadata = copy.deepcopy(receipt["canonicalMetadata"])
    metadata["glbSha256"] = sha(target)
    metadata["tightBounds"] = {"min":mesh.bounds[0].tolist(),"max":mesh.bounds[1].tolist()}
    metadata["refinement"]["colorTransferVersion"] = helper.COLOR_TRANSFER_VERSION
    metadata["refinement"]["colorSpace"] = "linear RGBA FLOAT32; texture RGB decoded before interpolation"
    assert metadata["refinement"]["resolution"] == 1024
    assert metadata["refinement"]["steps"] == 25 and metadata["refinement"]["smoothing"] == 0
    metadata_path = folder/"proxy.json"
    metadata_path.write_text(json.dumps(metadata,indent=2,allow_nan=False)+"\n")
    report["output_sha256"] = sha(target)
    report["inheritedInference"] = receipt["report"]["inference"]
    report_path = folder/"prepare-report.json"
    report_path.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    samples = [0,len(colors)//4,len(colors)//2,len(colors)-1]
    loader_path = a.repo/"node_modules/three/examples/jsm/loaders/GLTFLoader.js"
    js = """import {readFileSync} from 'node:fs';
import {GLTFLoader} from %s;
const bytes=readFileSync(process.argv[1]);
const gltf=await new GLTFLoader().parseAsync(bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength),'');
const result=[];
gltf.scene.traverse(mesh=>{
 if(!mesh.isMesh)return;
 const c=mesh.geometry.getAttribute('color');
 result.push({count:c.count,arrayType:c.array.constructor.name,normalized:c.normalized,
 vertexColors:mesh.material.vertexColors,flatShading:mesh.material.flatShading,
 hasNormal:!!mesh.geometry.getAttribute('normal'),materialColor:mesh.material.color.toArray(),
 samples:%s.map(i=>({vertex:i,rgba:[c.getX(i),c.getY(i),c.getZ(i),c.getW(i)]}))});
});
console.log(JSON.stringify(result));""" % (json.dumps(loader_path.as_uri()),json.dumps(samples))
    loaded = json.loads(subprocess.check_output(["node","--input-type=module","-e",js,str(target)],text=True))
    assert len(loaded)==1 and loaded[0]["arrayType"]=="Float32Array" and loaded[0]["normalized"] is False
    assert loaded[0]["hasNormal"] is False and loaded[0]["flatShading"] is True
    for sample in loaded[0]["samples"]:
        assert np.array_equal(sample["rgba"],colors[sample["vertex"]].astype(float))
    assert {str(path):sha(path) for path in (original,baseline,raw)} == hashes_before
    proof = {
        "purpose":"CPU-only color-transfer preparation. Actual B/C conditions and assets unchanged. No new Hi3D inference, bake or render.",
        "sourceReceipt":{"path":str(receipt_path),"sha256":sha(receipt_path)},
        "inputs":hashes_before,
        "outputs":{"proxy":{"path":str(target),"sha256":sha(target),"bytes":target.stat().st_size},
                   "canonicalMetadata":{"path":str(metadata_path),"sha256":sha(metadata_path)},
                   "prepareReport":{"path":str(report_path),"sha256":sha(report_path)}},
        "retainedInference":{"model":receipt["model"],"resolution":metadata["refinement"]["resolution"],"steps":25,"smoothing":0},
        "geometry":{"attributes":equal_attributes,"indicesByteIdentical":True,
                    "indicesSHA256":hashlib.sha256(new_indices.tobytes()).hexdigest(),"triangles":len(new_indices)//3,
                    "nodeTransformsIdentical":True,"materialsIdentical":True,
                    "sourceNORMALPresent":False,"controlNORMALPresent":False,
                    "vertexPositionsChanged":0,"existingInputFilesUnchanged":True},
        "colors":{"oldAccessor":old_color_spec,"newAccessor":color_spec,
                  "allVertices":len(colors),"expectedMethod":"Independent float64 sRGB decode before bilinear source texture sampling; linear factor; same nearest source-vertex transfer",
                  "maxAbsoluteRGBAError":float(error.max()),"meanAbsoluteRGBError":float(error[:,:3].mean()),
                  "alphaMaxAbsoluteErrorVsOriginalB":float(alpha_error.max()),
                  "samples":[{"vertex":i,"oldEncodedBytes":old_colors[i].tolist(),"newLinearRGBA":colors[i].tolist(),
                              "independentLinearRGBA":expected[i].tolist()} for i in samples]},
        "actualInstalledGLTFLoaderParse":loaded,
        "sourceCode":{"helperPath":str(helper_path),"helperSHA256":sha(helper_path),
                      "loaderPath":str(loader_path),"loaderSHA256":sha(loader_path),"proofScriptSHA256":sha(__file__)},
        "limitation":"This validates representation and geometry invariants. It is not a corrected guide, H3 RGB, or actual Box quality result."
    }
    proof_path = a.output/"B-linear-color-control-proof.json"
    proof_path.write_text(json.dumps(proof,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"proof":str(proof_path),"proofSHA256":sha(proof_path),"outputs":proof["outputs"],
                      "geometry":proof["geometry"],"error":proof["colors"]["maxAbsoluteRGBAError"]}))
if __name__ == "__main__":
    main()
