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
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
sha = lambda path: hashlib.file_digest(Path(path).open("rb"), "sha256").hexdigest()
meta = json.loads((a.work / f"{a.stage}-provenance.json").read_text())
source = Path(meta["box"]["viewer"])
expected_sha = meta["box"]["viewerSHA256"]
assert sha(source) == expected_sha
capture = json.loads((a.work / f"{a.stage}-box/capture.json").read_text())
assert capture["htmlSHA256"] == expected_sha and not capture["errors"]
archive = a.work / f"{a.stage}.inspector.html.gz"
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
target = a.output / archive.name
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
    "roundTripByteIdentityVerified": True, "deliveryCopySHA256Verified": True,
    "delivery": "Lossless gzip packaging; decompress to recover exact official HTML",
    "packaging": {"compressionLevel": 9, "mtime": 0, "rawHTMLCopiedToRepo": False},
    "captureStageState": capture["stageState"], "repoFilesystemFreeAfter": shutil.disk_usage(a.output).free,
}
receipt_path = a.output / f"{a.stage}-viewer-delivery.json"
receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps({"gzip": str(target), "gzipBytes": receipt["gzipBytes"], "gzipSHA256": gzip_sha,
                  "htmlSHA256": expected_sha, "receipt": str(receipt_path),
                  "freeAfterBytes": receipt["repoFilesystemFreeAfter"]}))
