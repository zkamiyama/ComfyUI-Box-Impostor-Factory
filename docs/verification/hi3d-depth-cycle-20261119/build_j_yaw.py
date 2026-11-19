"""Seven measured-yaw J renders; fixed common world-head crop and untouched PNG evidence."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from build_box_control_front import BG, TEXT, MUTED, FONT, sha, backdrop, aligned_front_crop


def projected_roi(world_bounds, captures):
    corners = np.array([(*p, 1.0) for p in itertools.product(*zip(*world_bounds))])
    pixels = []
    for entry in captures:
        camera = entry["actualCamera"]
        view = np.array(camera["matrixWorldInverse"]).reshape(4, 4, order="F")
        projection = np.array(camera["projectionMatrix"]).reshape(4, 4, order="F")
        clip = (projection @ view @ corners.T).T
        ndc = clip[:, :3] / clip[:, 3:4]
        pixels.append(np.column_stack(((ndc[:, 0]+1)*512, (1-ndc[:, 1])*512)))
    all_pixels = np.concatenate(pixels)
    low = np.floor(all_pixels.min(axis=0)).astype(int)
    high = np.ceil(all_pixels.max(axis=0)).astype(int)
    return [int(low[0]), int(low[1]), int(high[0]), int(high[1])]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("work", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    capture_path = a.work / "J-yaw/capture.json"
    cap = json.loads(capture_path.read_text())
    assert not cap["errors"] and "SwiftShader" in cap["graphicsRenderer"]
    assert cap["stageState"] == "Native Box"
    assert sha(cap["html"]) == cap["htmlSHA256"]
    expected_yaws = [-45, -30, -15, 0, 15, 30, 45]
    entries = cap["yawFrames"]
    assert [e["requestedYawDegrees"] for e in entries] == expected_yaws
    basis_path = a.output / "A-full-comparison-provenance.json"
    basis = json.loads(basis_path.read_text())
    crop = projected_roi(basis["headWorldBounds"], entries)
    assert 0 <= crop[0] < crop[2] <= 1024 and 0 <= crop[1] < crop[3] <= 1024
    images = []
    full_images = []
    for entry in entries:
        assert sha(entry["path"]) == entry["sha256"]
        assert abs(entry["actualYawDegrees"]-entry["requestedYawDegrees"]) < .3
        assert abs(entry["actualPitchDegrees"]) < .01
        im = Image.open(entry["path"]).convert("RGBA")
        assert im.size == (1024, 1024)
        full_images.append(im)
        images.append(backdrop(im.crop(crop)))
    meta = json.loads((a.work/"F-provenance.json").read_text())
    front = meta["faces"]["+Z"]
    assert sha(front["alignedPath"]) == front["alignedSHA256"]
    images.append(backdrop(aligned_front_crop(front["alignedPath"], meta["bakeBox"], crop, cap["camera"])))
    scale = 2
    w = (crop[2]-crop[0])*scale
    h = (crop[3]-crop[1])*scale
    cell_w, cell_h = w+24, h+63
    board = Image.new("RGB", (cell_w*4+24, cell_h*2+132), BG)
    draw = ImageDraw.Draw(board)
    font = lambda size: ImageFont.truetype(FONT, size)
    draw.text((24, 16), "J / actual native Box at measured yaw angles", font=font(27), fill=TEXT)
    draw.text((24, 53), "Unlit + NoToneMapping; same F maps and native blending. Mouse rotation; unchanged official HTML.", font=font(17), fill=MUTED)
    for index, im in enumerate(images):
        x = 12+(index % 4)*cell_w
        y = 87+(index // 4)*cell_h
        if index < 7:
            label = f"Yaw {entries[index]['actualYawDegrees']:+.6f} deg"
        else:
            label = "Reference: H3 +Z / registered RGB"
        draw.text((x+12, y), label, font=font(19), fill=TEXT)
        draw.text((x+12, y+25), "Actual rendered frame" if index < 7 else "Front reference only; no 3D render", font=font(14), fill=MUTED)
        board.paste(im.resize((w, h), Image.Resampling.NEAREST), (x+12, y+48))
    draw.text((24, board.height-27), f"One fixed crop {crop}; 2x nearest-neighbor; no sharpening, AI correction, or synthesized intermediate views.", font=font(16), fill=MUTED)
    face_path = a.output / "J-yaw-faces.webp"
    board.save(face_path, "WEBP", lossless=True)
    assert np.array_equal(np.asarray(Image.open(face_path)), np.asarray(board))
    body = Image.new("RGB", (7*280+24, 366), BG)
    bd = ImageDraw.Draw(body)
    bd.text((24, 15), "J / whole model, identical orthographic camera extent and target", font=font(23), fill=TEXT)
    for i, im in enumerate(full_images):
        x = 12+i*280
        bd.text((x+12, 55), f"{entries[i]['actualYawDegrees']:+.6f} deg", font=font(18), fill=TEXT)
        body.paste(backdrop(im).resize((256, 256), Image.Resampling.NEAREST), (x+12, 86))
    bd.text((24, 346), "1024 x 1024 actual renders shown at 1/4 size with nearest-neighbor. All native depth/raycast data retained.", font=font(14), fill=MUTED)
    body_path = a.output / "J-yaw-body.webp"
    body.save(body_path, "WEBP", lossless=True)
    previous = a.work / "audit/yaw-front-toggle/partial-capture.json"
    repeated = []
    if previous.exists():
        prior = json.loads(previous.read_text())
        for before in prior.get("yawFrames", []):
            now = next(e for e in entries if e["requestedYawDegrees"] == before["requestedYawDegrees"])
            repeated.append({"yaw": before["requestedYawDegrees"], "priorSHA256": before["sha256"], "finalSHA256": now["sha256"], "byteExact": before["sha256"] == now["sha256"]})
    zero = full_images[expected_yaws.index(0)]
    original_front = json.loads((a.work/"I-box/capture.json").read_text())["modes"]["lit"]["front"]
    front_equal = np.array_equal(np.asarray(zero), np.asarray(Image.open(original_front["path"]).convert("RGBA")))
    result = {
        "kind": __doc__, "assetId": cap["assetId"], "display": cap["display"],
        "capture": {"path": str(capture_path), "sha256": sha(capture_path)},
        "html": {"path": cap["html"], "sha256": cap["htmlSHA256"]},
        "camera": cap["camera"], "cameraAngleDefinition": cap["rotation"]["angleSource"],
        "referenceBounds": {"path": str(basis_path), "sha256": sha(basis_path), "headWorldBounds": basis["headWorldBounds"]},
        "fixedCrop": crop, "cropMethod": "Union of the same world-space head bounding box projected by every actual captured camera; one integer ROI reused for every panel.",
        "frames": entries, "frontPixelsEqualApprovedJFront": bool(front_equal),
        "firstAttemptRepeatVerification": repeated,
        "registeredRGB": {"path": front["alignedPath"], "sha256": front["alignedSHA256"], "sampling": "Premultiplied linear-light bilinear at common front-camera pixel centers; converted to sRGB."},
        "outputs": [{"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size, "size": list(Image.open(path).size)} for path in (face_path, body_path)],
        "processing": "Actual official snapshot PNGs. Fixed integer crop, nearest-neighbor scaling, lossless WebP. No image enhancement or generated views.",
        "scriptSHA256": sha(__file__),
    }
    proof = a.output / "J-yaw-provenance.json"
    proof.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"outputs": result["outputs"], "proofSHA256": sha(proof), "fixedCrop": crop, "frontPixelsEqual": front_equal, "repeatedSHAExact": all(e["byteExact"] for e in repeated)}))


if __name__ == "__main__":
    main()
