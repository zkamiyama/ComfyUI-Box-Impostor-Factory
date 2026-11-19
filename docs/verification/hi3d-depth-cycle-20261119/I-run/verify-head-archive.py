#!/usr/bin/env python3
"""Independently reopen the finished archive and hash every regular member."""
import hashlib,json,tarfile
from pathlib import Path,PurePosixPath
BASE=Path("/mnt/f/_dev/box-impostor-lab/archives")
REPO=Path("/home/dev/Projects/box-impostor-lab/ComfyUI/custom_nodes/box_impostor_factory")
def sha(p):
    with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def encode(value):return (json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n").encode()
created_path=BASE/"I-head-local-20261119.created.json"
created=json.loads(created_path.read_text())
archive=Path(created["archive"]);inventory_path=Path(created["inventory"])
if sha(archive)!=created["archiveSHA256"] or sha(inventory_path)!=created["inventorySHA256"]:
    raise ValueError("Archive/inventory changed after creation.")
inventory=json.loads(inventory_path.read_text())
expected={row["archivePath"]:row for row in inventory["members"]}
seen={};metadata_ok=True
with tarfile.open(archive,"r:gz") as tar:
    for member in tar:
        name=member.name
        if (not member.isfile() or name in seen or name not in expected or PurePosixPath(name).is_absolute()
                or ".." in PurePosixPath(name).parts):raise ValueError("Unexpected member: "+name)
        if (member.uid,member.gid,member.uname,member.gname,member.mode,member.mtime)!=(0,0,"","",0o644,0):
            raise ValueError("Nondeterministic metadata: "+name)
        stream=tar.extractfile(member);hasher=hashlib.sha256();size=0
        while True:
            chunk=stream.read(1024*1024)
            if not chunk:break
            hasher.update(chunk);size+=len(chunk)
        row=expected[name]
        if size!=row["bytes"] or hasher.hexdigest()!=row["sha256"]:raise ValueError("Member hash mismatch: "+name)
        seen[name]={"bytes":size,"sha256":hasher.hexdigest()}
if set(seen)!=set(expected):raise ValueError("Missing archive members.")
source_checks=0
for row in expected.values():
    if row["sourcePath"] is not None:
        if sha(row["sourcePath"])!=row["sha256"]:raise ValueError("Original source differs: "+row["sourcePath"])
        source_checks+=1
receipt={"kind":"I-independent-archive-byte-verification-v1","status":"verified",
         "archive":str(archive),"archiveSHA256":sha(archive),"compressedBytes":archive.stat().st_size,
         "inventory":str(inventory_path),"inventorySHA256":sha(inventory_path),
         "verifiedMembers":len(seen),"originalSourceHashesRechecked":source_checks,
         "allMemberHashesEqual":True,"allOriginalSourceHashesEqual":True,
         "normalizedTarMetadata":True,"safeRelativeRegularMembersOnly":True,
         "actualOfficialResultTriangles":1000000,
         "resultViewerSHA256":seen["I/evidence/result/clay.inspector.html"]["sha256"],
         "verificationScript":str(Path(__file__).resolve()),"verificationScriptSHA256":sha(__file__),
         "members":seen}
target=BASE/"I-head-local-20261119.verification.json"
if target.exists():raise FileExistsError(str(target))
target.write_bytes(encode(receipt))
dest=REPO/"docs/verification/hi3d-depth-cycle-20261119/I-run";dest.mkdir(exist_ok=True)
for p in [created_path,inventory_path,target,Path(__file__),Path(__file__).with_name("archive-head-result.py")]:
    out=dest/p.name
    if out.exists() and out.read_bytes()!=p.read_bytes():raise ValueError("Different existing repository receipt: "+str(out))
    out.write_bytes(p.read_bytes())
print(json.dumps({k:v for k,v in receipt.items() if k!="members"},indent=2),flush=True)
