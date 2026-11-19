#!/usr/bin/env python3
"""Deterministic byte-preserving I result archive; no inference or rendering."""
import gzip,hashlib,io,json,os,tarfile
from pathlib import Path
ROOT=Path("/mnt/f/_dev/box-impostor-lab/experiments/geometry-abc-20261119/I-head-local")
REPO=Path("/home/dev/Projects/box-impostor-lab/ComfyUI/custom_nodes/box_impostor_factory")
DOC=REPO/"docs/verification/hi3d-depth-cycle-20261119"
OUT=Path("/mnt/f/_dev/box-impostor-lab/archives/I-head-local-20261119.tar.gz")
def sha(p):
    with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def encode(value):return (json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n").encode()
plan=json.loads((ROOT/"head-plan.json").read_text())
proof=json.loads((ROOT/"evidence/result/provenance.json").read_text())
if not proof["status"].startswith("captured"):
    raise ValueError("Official result capture is not complete.")
if not (proof["clay"]["sourceTriangles"]==proof["clay"]["payloadTriangles"]==proof["clay"]["viewerTriangleLimit"]==1000000):
    raise ValueError("Official result viewer does not contain the full1Mtriangles.")
if not proof["normalPolicy"]["allAccessorsAbsent"] or proof["normalPolicy"]["normalRecomputedBeforeInspector"]:
    raise ValueError("Unexpected comparison normal policy.")
sources={}
def add(path,name,role):
    p=Path(path)
    if not p.is_file() or p.is_symlink():raise ValueError("Expected regular source: "+str(p))
    if name in sources:raise ValueError("Duplicate archive path: "+name)
    sources[name]=(p,role)
for name in ["head-plan.json","head-execution.json","hi3d-inference.json","head-mapping.json",
             "head-topology.json","head-copy-proof.json","head-inference.log","head-source.glb",
             "hi3d-raw.glb","head-canonical.glb","head-reference.png","head-reference-prepared.png",
             "head-reference-context.png"]:
    add(ROOT/name,"I/"+name,"experiment")
for path in sorted((ROOT/"evidence").rglob("*")):
    if path.is_file():
        if path.suffix not in (".json",".png",".glb",".html",".py"):raise ValueError("Unexpected evidence file: "+str(path))
        add(path,"I/"+str(path.relative_to(ROOT)),"official-inspector-evidence")
add(plan["sourceCoarse"]["path"],"inputs/coarse-A/proxy.glb","original-canonical-coarse-input")
add(plan["sourceCoarse"]["manifest"],"inputs/coarse-A/manifest.json","original-input-metadata")
add(plan["sourceReference"]["path"],"inputs/F-front/source.png","original-physical-fullbody-reference")
add(plan["sourceReference"]["manifest"],"inputs/F-front/manifest.json","original-input-metadata")
add(plan["sourceReference"]["qa"],"inputs/F-front/qa-report.json","original-input-metadata")
add(plan["parentHPlan"],"inputs/H-plan.json","parent-control-plan")
for name in ["scripts/refine_head_region.py","scripts/hi3d_infer.py","box_nodes/depth_mesh.py",
             "tests/test_refine_head_region.py"]:
    add(REPO/name,"tools/repo/"+name,"used-helper")
for name in ["run_head_local.py","prepare_head_clay.py","prepare_evidence.py","capture_inspector.mjs",
             "compare_head_clay.py","compare_head_front.py","A-full-comparison-provenance.json"]:
    add(DOC/name,"tools/repo/docs/"+name,"used-helper-or-camera-proof")
for name in ["A-H-I-head-lit.webp","A-H-I-head-normals.webp","A-H-I-context-lit.webp","A-H-I-context-normals.webp","A-H-I-head-provenance.json","F-RGB-H-I-front.webp","F-RGB-H-I-front-provenance.json"]:
    add(DOC/name,"comparison/"+name,"frozen-comparison")
for key,item in plan["upstream"]["sourceFiles"].items():
    p=Path(item["path"])
    if sha(p)!=item["sha256"]:raise ValueError("Changed upstream source: "+str(p))
    add(p,"tools/hi3d/"+str(p.relative_to(plan["upstream"]["root"])),"used-upstream-source")
inspector=proof["tools"]["inspector"]
if sha(inspector["path"])!=inspector["sha256"]:raise ValueError("Changed official Inspector helper.")
add(inspector["path"],"tools/3d-inspector/view.py","official-viewer-generator")
for name in ["check-head-topology.py","archive-head-result.py","verify-head-archive.py"]:
    add(ROOT/name,"tools/audit/"+name,"audit-or-archive-helper")
for key in ("sourceCoarse","sourceReference"):
    if sha(plan[key]["path"])!=plan[key]["sha256"]:raise ValueError("Original input changed.")
if sha(proof["clay"]["viewer"])!=proof["clay"]["viewerSHA256"]:raise ValueError("Result viewer changed.")
rows=[{"archivePath":name,"sourcePath":str(path),"role":role,"bytes":path.stat().st_size,"sha256":sha(path)}
      for name,(path,role) in sorted(sources.items())]
manifest={
 "kind":"I-head-local-portable-evidence-v1","qualityOutcome":"negative trial; no full-body merge",
 "viewerEntry":"I/evidence/result/clay.inspector.html",
 "sourceViewerEntry":"I/evidence/source/clay.inspector.html",
 "officialResultTriangles":1000000,"officialResultVertices":500011,
 "resultViewerSHA256":proof["clay"]["viewerSHA256"],
 "resultProvenanceSHA256":sha(ROOT/"evidence/result/provenance.json"),
 "normalPolicy":proof["normalPolicy"],
 "preservation":"Every listed source file is included byte-for-byte. Source paths remain provenance; all archive paths are relative.",
 "excluded":"All model weights, caches, temporary inference files and unrelated experiments.",
 "sourceInference":"native crop of existing1024fullbody image; no new headRGB generation",
 "mapping":"approximate isotropic bbox/Y-height fit, not an exact upstream inverse",
 "recovery":"Original coarseA proxy and original F+Z source PNG included; model environment is intentionally external.",
 "determinism":{"memberOrder":"lexicographic","tar":"USTAR; uid/gid0; namesempty; mode0644; mtime0","gzip":"mtime0; no filename; level6"},
 "inventoryNote":"This embedded manifest inventories source members; its own hash is in the external full-member inventory.",
 "files":rows}
manifest_bytes=encode(manifest)
manifest_name="archive-manifest.json"
all_rows=rows+[{"archivePath":manifest_name,"sourcePath":None,"role":"embedded-inventory",
              "bytes":len(manifest_bytes),"sha256":hashlib.sha256(manifest_bytes).hexdigest()}]
all_rows.sort(key=lambda row:row["archivePath"])
OUT.parent.mkdir(parents=True,exist_ok=True)
if OUT.exists():raise FileExistsError("Preserve completed archive: "+str(OUT))
temporary=OUT.with_name(OUT.name+".partial")
if temporary.exists():raise FileExistsError("Unexpected partial archive: "+str(temporary))
with temporary.open("xb") as output:
    with gzip.GzipFile(filename="",mode="wb",fileobj=output,mtime=0,compresslevel=6) as gz:
        with tarfile.open(fileobj=gz,mode="w|",format=tarfile.USTAR_FORMAT) as tar:
            for row in all_rows:
                info=tarfile.TarInfo(row["archivePath"]);info.size=row["bytes"]
                info.mode=0o644;info.uid=info.gid=0;info.uname=info.gname="";info.mtime=0
                if row["sourcePath"] is None:
                    tar.addfile(info,io.BytesIO(manifest_bytes))
                else:
                    with Path(row["sourcePath"]).open("rb") as stream:tar.addfile(info,stream)
for row in rows:
    if sha(row["sourcePath"])!=row["sha256"]:raise ValueError("Source changed while archiving: "+row["sourcePath"])
if temporary.stat().st_size>=100_000_000:raise ValueError("Archive exceeds100MB; split before delivery.")
os.replace(temporary,OUT)
inventory={"kind":"I-complete-archive-member-inventory-v1","archive":str(OUT),"members":all_rows}
inventory_path=OUT.with_name("I-head-local-20261119.inventory.json");inventory_path.write_bytes(encode(inventory))
receipt={"kind":"I-deterministic-archive-creation-receipt-v1","archive":str(OUT),"archiveSHA256":sha(OUT),
         "compressedBytes":OUT.stat().st_size,"members":len(all_rows),
         "sourceMemberBytes":sum(row["bytes"] for row in rows),
         "inventory":str(inventory_path),"inventorySHA256":sha(inventory_path),
         "allSourceHashesRecheckedAfterArchiveWrite":True,"independentReopenVerification":"pending",
         "officialResultFullTriangles":1000000,"viewerSHA256":proof["clay"]["viewerSHA256"]}
created=OUT.with_name("I-head-local-20261119.created.json");created.write_bytes(encode(receipt))
print(json.dumps(receipt,indent=2),flush=True)
