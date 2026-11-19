"""Capture and compare the final source-edge feather variant using established K controls."""
import argparse
import base64
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import compare_seam_variants as comparison
from verify_seam_viewers import OLD, REPO, payload, sha


LABEL = "L3-view-feather"
ROI = (510, 202, 591, 259)
POINTS = {"wedge_dark": (526, 225), "wedge_transition": (527, 225), "neck_skin": (528, 225),
          "chin_above": (541, 224), "chin_below": (541, 225)}


def capture(work, expected_sha):
    path = work/f"{LABEL}.inspector.html"
    assert sha(path) == expected_sha, "Official generated L3 HTML hash differs"
    base_path = OLD/"K-view.inspector.html"
    base, data = payload(base_path), payload(path)
    display = {"lighting": "unlit", "sampling": "native", "colorSampling": "view-feather"}
    checks = {"boundsEqualToK": data["bounds"] == base["bounds"],
              "manifestEqualToK": data["nativeBox"]["manifest"] == base["nativeBox"]["manifest"],
              "meshesEqualToK": data["meshes"] == base["meshes"],
              "assetIdEqualToK": data["assetId"] == base["assetId"],
              "displayMatches": data["nativeBox"]["display"] == display}
    assert all(checks.values()), checks
    maps = {}
    for name, spec in data["nativeBox"]["manifest"]["textures"].items():
        raw = base64.b64decode(data["nativeBox"]["files"][spec["path"]])
        digest = hashlib.sha256(raw).hexdigest()
        equal = raw == base64.b64decode(base["nativeBox"]["files"][spec["path"]])
        assert equal and digest == spec["sha256"] and len(raw) == spec["byteLength"]
        maps[name] = {"byteEqualToK": equal, "sha256": digest, "bytes": len(raw)}
        del raw
    proof = {"html": str(path), "htmlSHA256": expected_sha, "base": str(base_path),
             "baseSHA256": sha(base_path), "assetId": data["assetId"], "checks": checks, "maps": maps,
             "display": display, "scriptSHA256": sha(__file__)}
    (work/"L3-viewer-payload-identity.json").write_text(json.dumps(proof, indent=2)+"\n")
    control = work/f"qa-{LABEL}-control.json"
    control.write_text(json.dumps({"assetId": data["assetId"], "htmlSHA256": expected_sha, "display": display}, indent=2)+"\n")
    del base, data
    print("L3_NATIVE_PAYLOAD_IDENTICAL_TO_K", flush=True)
    reference = json.loads((OLD/"K-view-yaw/capture.json").read_text())["referenceManifest"]
    script = REPO/"docs/verification/hi3d-depth-cycle-20261119/capture_inspector_display_yaw.mjs"
    subprocess.run(["node", str(script), str(path), str(work/f"{LABEL}-first"),
                    reference, "0,-30,45", str(control)], check=True, cwd=REPO)


def neck_comparison(records, output):
    crops, measurements = {}, {}
    for name, record in records.items():
        full = np.asarray(record["images"][-30])
        crop = np.asarray(record["images"][-30].crop(ROI))
        crops[name] = crop
        probes = {key: full[y, x].tolist() for key, (x, y) in POINTS.items()}
        measurements[name] = {"opaquePixels": int((crop[..., 3] > 250).sum()),
            "partialPixels": int(((crop[..., 3] > 5) & (crop[..., 3] <= 250)).sum()),
            "transparentPixels": int((crop[..., 3] <= 5).sum()), "probesRGBA": probes,
            "chinPairMeanAbsoluteRGBDifference": float(np.abs(full[224, 541, :3].astype(float)-full[225, 541, :3]).mean()),
            "chinRowPairDifferences": {str(x): float(np.abs(full[224, x, :3].astype(float)-full[225, x, :3]).mean()) for x in range(537, 547)}}
    width, height, scale = ROI[2]-ROI[0], ROI[3]-ROI[1], 4
    cell = width*scale+24
    board = Image.new("RGB", (len(records)*cell+24, height*scale+184), comparison.BG)
    draw = ImageDraw.Draw(board)
    draw.text((20, 16), "Actual neck seam / -30 deg", font=comparison.font(25), fill=comparison.TEXT)
    draw.text((20, 54), "Same depth and camera; only shader color selection changes.", font=comparison.font(15), fill=comparison.MUTED)
    for col, (name, crop) in enumerate(crops.items()):
        x = 24+col*cell
        draw.text((x, 88), "K / previous" if name == "K-view" else "L3 / source-edge feather", font=comparison.font(18), fill=comparison.TEXT)
        board.paste(Image.fromarray(crop[..., :3]).resize((width*scale, height*scale), Image.Resampling.NEAREST), (x,122))
    draw.text((20, board.height-37), f"Original image crop {list(ROI)}; 4x nearest-neighbor.", font=comparison.font(14), fill=comparison.MUTED)
    draw.text((20, board.height-17), "Saved actual renders. No image blur, retouching or alignment.", font=comparison.font(14), fill=comparison.MUTED)
    path = output/"L3-neck-comparison.png"
    board.save(path)
    delta = np.abs(crops[LABEL][..., :3].astype(float)-crops["K-view"][..., :3])
    return path, {"roiFullImage": ROI, "pixels": width*height, "probeCoordinates": POINTS,
                  "measurements": measurements, "meanRGBByteDifferenceToK": float(delta.mean()),
                  "changedPixelsToK": int(np.any(delta != 0, axis=-1).sum()),
                  "interpretation": "The fixed chin-pair contrast describes a tested seam, not geometry accuracy. Inspect both the neck crop and +45 degree face ghost separately."}


def compare(work):
    output = work/"L3-evidence"
    output.mkdir(parents=True, exist_ok=True)
    comparison.LABELS[LABEL] = "L3 / source-edge feather"
    angles = [0, -30, 45]
    records = {"K-view": comparison.load("K-view", [OLD/"K-view-front/capture.json", OLD/"K-view-yaw/capture.json"], angles),
               LABEL: comparison.load(LABEL, [work/f"{LABEL}-first/capture.json"], angles)}
    checks = comparison.invariants(records, angles)
    outputs = comparison.static_boards(records, angles, output)
    front, front_metrics = comparison.front_fidelity(records, output)
    outputs.append(front)
    renamed = []
    for path in outputs:
        new = path.with_name("L3-"+path.name)
        path.replace(new)
        renamed.append(new)
    neck_path, neck = neck_comparison(records, output)
    renamed.append(neck_path)
    proof = {"kind": __doc__, "anglesDegrees": angles, "assetId": records["K-view"]["reports"][0]["assetId"],
        "invariantChecks": checks, "frontFidelity": front_metrics, "neckComparison": neck,
        "differencesToK": comparison.differences(records, angles),
        "reports": {name: record["reports"] for name, record in records.items()},
        "frames": {name: record["frames"] for name, record in records.items()},
        "outputs": [{"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size,
                     "size": list(Image.open(path).size)} for path in renamed],
        "scriptSHA256": sha(__file__), "comparisonHelperSHA256": sha(comparison.__file__),
        "processing": "Same saved PNG captures and fixed camera. PNG boards use integer crops and nearest-neighbor enlargement only."}
    path = output/"L3-initial-provenance.json"
    path.write_text(json.dumps(proof, indent=2)+"\n")
    print(json.dumps({"proof": str(path), "proofSHA256": sha(path), "outputs": proof["outputs"],
                      "frontMetrics": front_metrics["metrics"], "neck": neck}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("stage", choices=("capture", "compare"))
    parser.add_argument("--expected-sha")
    args = parser.parse_args()
    if args.stage == "capture":
        assert args.expected_sha, "An exact root-provided official HTML SHA is required"
        capture(args.work, args.expected_sha)
    else:
        compare(args.work)


if __name__ == "__main__":
    main()
