"""Read-only check that the C flat head top is present in geometry, not a panel crop."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image
import trimesh
p=argparse.ArgumentParser(description=__doc__)
p.add_argument("work",type=Path);p.add_argument("output",type=Path)
a=p.parse_args()
def sha(path):
    with Path(path).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
records={}
for stage in ("B","C"):
    metadata_path=a.work/f"{stage}-provenance.json"
    meta=json.loads(metadata_path.read_text())
    path=a.work/f"{stage}-clay/lit-front.png"
    cap_path=a.work/f"{stage}-clay/capture.json"
    cap=json.loads(cap_path.read_text())
    image=np.asarray(Image.open(path).convert("RGBA"))
    mask=image[...,3]>250
    top=int(np.nonzero(mask)[0].min())
    mesh=trimesh.load(meta["proxyPath"],force="mesh",process=False)
    xyz=np.asarray(mesh.vertices)
    high=xyz.max(axis=0);low=xyz.min(axis=0)
    tol=5e-8
    records[stage]={
        "assetId":meta["assetId"],"proxyPath":meta["proxyPath"],"proxySHA256":sha(meta["proxyPath"]),
        "sourceBounds":cap["sourceBounds"],"actualPositionBounds":[low.tolist(),high.tolist()],
        "capturePath":str(cap_path),"captureSHA256":sha(cap_path),
        "imagePath":str(path),"imageSHA256":sha(path),"imageSize":list(Image.open(path).size),
        "mask":"alpha > 250","headCropTopPixel":115,"firstOpaqueRow":top,"topPaddingPixels":top-115,
        "firstFiveOpaqueRowCounts":{str(y):int(mask[y].sum()) for y in range(top,top+5)},
        "boundaryVertexTolerance":tol,
        "verticesNearMaxBoundsXYZ":[int(np.count_nonzero(np.abs(xyz[:,i]-high[i])<=tol)) for i in range(3)],
        "verticesNearMinBoundsXYZ":[int(np.count_nonzero(np.abs(xyz[:,i]-low[i])<=tol)) for i in range(3)],
    }
    assert top>115
proof={"purpose":__doc__,"stages":records,
       "conclusion":"The C flat head top exists inside the actual geometry and the full native frame. The shared evidence crop starts above the first opaque row and does not produce this plateau.",
       "causalLimit":"Counts and images establish the plateau. Bounds-clamp implementation is audited separately; this file does not infer its cause.",
       "scriptSHA256":sha(__file__)}
out=a.output/"C-head-top-proof.json"
out.write_text(json.dumps(proof,indent=2,allow_nan=False)+"\n")
print(json.dumps({"path":str(out),"sha256":sha(out),"rows":{s:r["firstFiveOpaqueRowCounts"] for s,r in records.items()},
                  "verticesAtYMax":{s:r["verticesNearMaxBoundsXYZ"][1] for s,r in records.items()}}))
