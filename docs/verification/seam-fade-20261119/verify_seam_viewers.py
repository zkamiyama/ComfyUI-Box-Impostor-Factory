"""Verify official seam variants retain the native Box payload and prepare capture controls."""
import argparse
import base64
import hashlib
import json
import re
import subprocess
import zlib
from pathlib import Path


OLD = Path("/mnt/f/_dev/box-impostor-lab/results/hi3d-depth-cycle-20261119")
REPO = Path("/home/dev/Projects/box-impostor-lab/ComfyUI/custom_nodes/box_impostor_factory")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def payload(path):
    text = path.read_text()
    encoded = re.search(r'<script id="scene-payload"[^>]*>([\s\S]*?)</script>', text)[1].strip()
    packed = base64.b64decode(encoded)
    return json.loads(zlib.decompress(packed) if packed[0] == 0x78 else packed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("--capture-first", action="store_true")
    args = parser.parse_args()
    base_path = OLD/"K-view.inspector.html"
    base = payload(base_path)
    checks = {}
    controls = []
    for label, mode in (("L1-view-fade", "view-fade"), ("L2-view-soft", "view-soft")):
        path = args.work/f"{label}.inspector.html"
        data = payload(path)
        checks[label] = {"html": str(path), "htmlSHA256": sha(path), "assetId": data["assetId"],
            "boundsEqualToK": data["bounds"] == base["bounds"],
            "manifestEqualToK": data["nativeBox"]["manifest"] == base["nativeBox"]["manifest"],
            "meshesEqualToK": data["meshes"] == base["meshes"],
            "display": data["nativeBox"]["display"], "maps": {}}
        assert checks[label]["boundsEqualToK"] and checks[label]["manifestEqualToK"] and checks[label]["meshesEqualToK"]
        assert data["assetId"] == base["assetId"]
        assert data["nativeBox"]["display"] == {"lighting": "unlit", "sampling": "native", "colorSampling": mode}
        for name, spec in data["nativeBox"]["manifest"]["textures"].items():
            p = spec["path"]
            raw = base64.b64decode(data["nativeBox"]["files"][p])
            digest = hashlib.sha256(raw).hexdigest()
            equal = raw == base64.b64decode(base["nativeBox"]["files"][p])
            assert equal and digest == spec["sha256"] and len(raw) == spec["byteLength"]
            checks[label]["maps"][name] = {"rawByteEqualToK": equal, "sha256": digest, "bytes": len(raw)}
            del raw
        control_path = args.work/f"qa-{label}-control.json"
        control_path.write_text(json.dumps({"assetId": data["assetId"], "htmlSHA256": sha(path),
                                           "display": data["nativeBox"]["display"]}, indent=2)+"\n")
        controls.append((label, path, control_path))
        del data
    proof = args.work/"viewer-payload-identity.json"
    proof.write_text(json.dumps({"base": str(base_path), "baseSHA256": sha(base_path), "checks": checks,
                                "scriptSHA256": sha(__file__)}, indent=2)+"\n")
    print(json.dumps({"proof": str(proof), "proofSHA256": sha(proof), "allChecksPassed": True}), flush=True)
    del base
    if args.capture_first:
        reference = json.loads((OLD/"K-view-yaw/capture.json").read_text())["referenceManifest"]
        script = REPO/"docs/verification/hi3d-depth-cycle-20261119/capture_inspector_display_yaw.mjs"
        for label, path, control_path in controls:
            subprocess.run(["node", str(script), str(path), str(args.work/f"{label}-first"),
                            reference, "0,-30,45", str(control_path)], check=True, cwd=REPO)
        print("FIRST_PASS_CAPTURE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
