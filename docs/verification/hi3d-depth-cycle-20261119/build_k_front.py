"""Actual J/K front comparison: same F geometry/maps, only native RGB sampling changes."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from build_box_control_front import BG, TEXT, MUTED, FONT, sha, backdrop, aligned_front_crop


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("work", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    meta = json.loads((a.work/"F-provenance.json").read_text())
    capture_paths = {"J": a.work/"I-box/capture.json",
                     "K-view": a.work/"K-view-front/capture.json",
                     "K-dominant": a.work/"K-dominant-front/capture.json"}
    captures = {label: json.loads(path.read_text()) for label, path in capture_paths.items()}
    camera = captures["J"]["camera"]
    crop_proof = a.output/"A-full-comparison-provenance.json"
    crop = json.loads(crop_proof.read_text())["headCrops"]["front"]
    front = meta["faces"]["+Z"]
    assert sha(front["alignedPath"]) == front["alignedSHA256"]
    rgb = aligned_front_crop(front["alignedPath"], meta["bakeBox"], crop, camera)
    images = [backdrop(rgb)]
    actual, full, inputs = {}, {}, {}
    for label, cap in captures.items():
        assert cap["assetId"] == meta["assetId"]
        assert not cap["errors"] and cap["stageState"] == "Native Box"
        assert "SwiftShader" in cap["graphicsRenderer"]
        assert sha(cap["html"]) == cap["htmlSHA256"]
        for key in ("projection", "target", "halfExtent", "distance", "pixels", "cssPixels", "deviceScaleFactor"):
            assert cap["camera"][key] == camera[key], (label, key)
        entry = cap["modes"]["lit"]["front"] if label == "J" else next(e for e in cap["yawFrames"] if e["requestedYawDegrees"] == 0)
        assert sha(entry["path"]) == entry["sha256"]
        im = Image.open(entry["path"]).convert("RGBA")
        assert im.size == (1024, 1024)
        actual[label] = entry["actualCamera"]
        full[label] = np.asarray(im)
        images.append(backdrop(im.crop(crop)))
        inputs[label] = {"capture": {"path": str(capture_paths[label]), "sha256": sha(capture_paths[label])},
                         "html": {"path": cap["html"], "sha256": cap["htmlSHA256"]}, "frame": entry}
    alpha = {}
    for label in ("K-view", "K-dominant"):
        for key in ("matrixWorld", "projectionMatrix"):
            assert np.allclose(actual[label][key], actual["J"][key], rtol=0, atol=1e-8), (label, key)
        equal = np.array_equal(full[label][..., 3], full["J"][..., 3])
        assert equal, (label, "rendered alpha changed")
        alpha[label] = {"byteExactToJ": equal, "differentPixels": 0}
    enlarged = [im.resize((im.width*3, im.height*3), Image.Resampling.NEAREST) for im in images]
    board = Image.new("RGB", (1520, enlarged[0].height+184), BG)
    draw = ImageDraw.Draw(board)
    font = lambda size: ImageFont.truetype(FONT, size)
    draw.text((24, 18), "Same F geometry and textures / native RGB sampling controls", font=font(26), fill=TEXT)
    draw.text((24, 55), "All three Box renders use unlit + NoToneMapping and the same camera. Official HTML outputs are unchanged.", font=font(17), fill=MUTED)
    labels = ("H3 +Z / registered RGB", "J / native blend", "K-view / view weights", "K-dominant / one view")
    for index, (im, label) in enumerate(zip(enlarged, labels)):
        draw.text((index*380+16, 91), label, font=font(18), fill=TEXT)
        board.paste(im, (index*380+(380-im.width)//2, 125))
    draw.text((24, board.height-47), "3x nearest-neighbor; no sharpening or AI correction. Native depth raycasting is retained in every Box render.", font=font(15), fill=MUTED)
    draw.text((24, board.height-24), "This tests display color selection. It does not repair geometry or guarantee natural parallax at other angles.", font=font(15), fill=MUTED)
    path = a.output/"K-front-stages.webp"
    board.save(path, "WEBP", lossless=True)
    assert np.array_equal(np.asarray(Image.open(path)), np.asarray(board))
    proof = {
        "kind": __doc__, "assetId": meta["assetId"], "camera": camera, "headCrop": crop,
        "inputs": inputs, "actualCameras": actual, "renderedAlpha": alpha,
        "registeredRGB": {"path": front["alignedPath"], "sha256": front["alignedSHA256"],
                          "sampling": "Established premultiplied linear-light bilinear at identical camera pixel centers; 8-bit sRGB."},
        "payloadProofs": {mode: {"path": str(a.work/"K"/f"{mode}-payload-proof.json"), "sha256": sha(a.work/"K"/f"{mode}-payload-proof.json")} for mode in ("view", "dominant")},
        "displayChange": "J native RGB blend versus K view-weighted or dominant-view RGB; all unlit and NoToneMapping.",
        "output": {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size, "size": list(board.size)},
        "processing": "Identical integer crop, 3x nearest-neighbor scaling, lossless WebP; no enhancement.",
        "scriptSHA256": sha(__file__),
    }
    out = a.output/"K-front-provenance.json"
    out.write_text(json.dumps(proof, indent=2)+"\n")
    print(json.dumps({"output": proof["output"], "proofSHA256": sha(out), "alpha": alpha}))


if __name__ == "__main__":
    main()
