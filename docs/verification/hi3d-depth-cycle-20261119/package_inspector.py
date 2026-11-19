"""Package exact official Inspector HTML as deterministic gzip; verify round trip."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("work", type=Path)
p.add_argument("stage")
p.add_argument("output", type=Path)
p.add_argument("--kind", choices=["box", "clay"], default="box")
p.add_argument("--no-copy", action="store_true", help="Keep gzip only in work; write the small receipt to output.")
a = p.parse_args()
stem = a.stage if a.kind == "box" else f"{a.stage}-clay"
a.output.mkdir(parents=True, exist_ok=True)
sha = lambda path: hashlib.file_digest(Path(path).open("rb"), "sha256").hexdigest()
meta = json.loads((a.work / f"{a.stage}-provenance.json").read_text())
source = Path(meta[a.kind]["viewer"])
expected_sha = meta[a.kind]["viewerSHA256"]
assert sha(source) == expected_sha
capture = json.loads((a.work / f"{a.stage}-{a.kind}/capture.json").read_text())
assert capture["htmlSHA256"] == expected_sha and not capture["errors"]
if a.kind == "clay":
    triangles = sum(item["triangles"] for item in meta["clay"]["geometry"])
    assert meta["clay"]["payloadTriangles"] == triangles
    assert capture["triangleCounts"]["meshPayload"] == triangles
    assert capture["triangleCounts"]["expected"] == triangles
archive = a.work / f"{stem}.inspector.html.gz"
temporary = archive.with_suffix(archive.suffix + ".partial")
with source.open("rb") as src, temporary.open("wb") as dst:
    with gzip.GzipFile(filename="", fileobj=dst, mode="wb", compresslevel=9, mtime=0) as zipped:
        shutil.copyfileobj(src, zipped, length=1024*1024)
temporary.replace(archive)
digest, restored_bytes = hashlib.sha256(), 0
with gzip.open(archive, "rb") as restored:
    while block := restored.read(1024*1024):
        digest.update(block)
        restored_bytes += len(block)
assert digest.hexdigest() == expected_sha
assert restored_bytes == source.stat().st_size
gzip_sha = sha(archive)
target = archive if a.no_copy else a.output / archive.name
if not a.no_copy:
    free_before = shutil.disk_usage(a.output).free
    assert free_before > archive.stat().st_size + 8*1024*1024, "Insufficient space for gzip delivery copy"
    copy = target.with_suffix(target.suffix + ".partial")
    shutil.copyfile(archive, copy)
    assert sha(copy) == gzip_sha
    copy.replace(target)
assert sha(source) == expected_sha
receipt = {
    "source": source.name, "sourcePath": str(source),
    "htmlBytes": source.stat().st_size, "htmlSHA256": expected_sha,
    "gzip": target.name, "gzipSourcePath": str(archive),
    "gzipBytes": target.stat().st_size, "gzipSHA256": gzip_sha,
    "officialViewerUnmodified": True, "asset": meta["assetId"],
    "decompressedSHA256": digest.hexdigest(), "decompressedBytes": restored_bytes,
    "roundTripByteIdentityVerified": True, "deliveryCopySHA256Verified": None if a.no_copy else True,
    "deliveryArchivePath": str(target), "deliveryCopyMade": not a.no_copy,
    "delivery": "Lossless gzip packaging; decompress to recover exact official HTML",
    "packaging": {"compressionLevel": 9, "mtime": 0, "rawHTMLCopiedToRepo": False},
    "captureStageState": capture["stageState"], "repoFilesystemFreeAfter": shutil.disk_usage(a.output).free,
}
if a.kind == "clay":
    receipt["artifactKind"] = "Full-triangle geometry-only uniform-gray diagnostic; exact official Inspector HTML"
    receipt["sourceTriangles"] = triangles
    receipt["viewerPayloadTriangles"] = capture["triangleCounts"]["meshPayload"]
receipt_path = a.output / f"{stem}-viewer-delivery.json"
receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps({"gzip": str(target), "gzipBytes": receipt["gzipBytes"], "gzipSHA256": gzip_sha,
                  "htmlSHA256": expected_sha, "receipt": str(receipt_path),
                  "freeAfterBytes": receipt["repoFilesystemFreeAfter"]}))
