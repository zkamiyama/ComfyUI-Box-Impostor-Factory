"""Build a small real-render J yaw loop; no generated or interpolated frames."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from build_box_control_front import BG, TEXT, MUTED, FONT, sha, backdrop


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("work", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    reports = [a.work/"J-yaw/capture.json", a.work/"J-loop-unique/capture.json"]
    captures = [json.loads(path.read_text()) for path in reports]
    angles = [i*2.5 for i in range(-6, 7)]
    entries = {}
    for capture in captures:
        assert not capture["errors"] and "SwiftShader" in capture["graphicsRenderer"]
        assert capture["stageState"] == "Native Box"
        assert sha(capture["html"]) == capture["htmlSHA256"]
        for key in ("projection", "target", "halfExtent", "distance", "pixels", "cssPixels", "deviceScaleFactor"):
            assert capture["camera"][key] == captures[0]["camera"][key]
        for entry in capture["yawFrames"]:
            if entry["requestedYawDegrees"] in angles:
                entries[entry["requestedYawDegrees"]] = entry
    assert sorted(entries) == angles
    crop_proof = a.output/"J-yaw-provenance.json"
    crop = json.loads(crop_proof.read_text())["fixedCrop"]
    sequence = angles + angles[-2:0:-1]
    assert len(sequence) == 24
    font = lambda size: ImageFont.truetype(FONT, size)
    face_frames, body_frames, raw = {}, {}, {}
    for angle in angles:
        entry = entries[angle]
        assert sha(entry["path"]) == entry["sha256"]
        assert abs(entry["actualYawDegrees"]-angle) < .3
        assert abs(entry["actualPitchDegrees"]) < .01
        source = Image.open(entry["path"]).convert("RGBA")
        raw[angle] = np.asarray(source)
        face = backdrop(source.crop(crop))
        face = face.resize((face.width*2, face.height*2), Image.Resampling.NEAREST)
        board = Image.new("RGB", (face.width+32, face.height+108), BG)
        draw = ImageDraw.Draw(board)
        draw.text((16, 10), "J / native Box, unlit", font=font(21), fill=TEXT)
        draw.text((16, 39), f"Measured camera yaw {entry['actualYawDegrees']:+.6f} deg", font=font(16), fill=MUTED)
        board.paste(face, (16, 68))
        draw.text((16, board.height-31), "Actual frames / 2x nearest / playback 10 fps", font=font(15), fill=MUTED)
        draw.text((16, board.height-13), "Playback timing is not measured render performance.", font=font(12), fill=MUTED)
        face_frames[angle] = board
        body = backdrop(source).resize((512, 512), Image.Resampling.NEAREST)
        board = Image.new("RGB", (544, 620), BG)
        draw = ImageDraw.Draw(board)
        draw.text((16, 10), "J / native Box, unlit / whole model", font=font(21), fill=TEXT)
        draw.text((16, 39), f"Measured camera yaw {entry['actualYawDegrees']:+.6f} deg", font=font(16), fill=MUTED)
        board.paste(body, (16, 68))
        draw.text((16, 589), "Actual frames / 1/2 nearest / playback 10 fps", font=font(15), fill=MUTED)
        draw.text((16, 607), "Playback timing is not measured render performance.", font=font(12), fill=MUTED)
        body_frames[angle] = board
    outputs = []
    for label, frames in (("face", face_frames), ("body", body_frames)):
        frames_in_order = [frames[angle] for angle in sequence]
        path = a.output / f"J-parallax-{label}-loop.webp"
        frames_in_order[0].save(path, "WEBP", save_all=True, append_images=frames_in_order[1:],
                                duration=100, loop=0, lossless=True, quality=100, method=6)
        with Image.open(path) as verify:
            assert verify.n_frames == len(sequence)
            for index, expected in enumerate(frames_in_order):
                verify.seek(index)
                assert np.array_equal(np.asarray(verify.convert("RGB")), np.asarray(expected)), index
        outputs.append({"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size,
                        "dimensions": list(frames_in_order[0].size), "frames": len(sequence),
                        "losslessDecodedPixelEquality": True})
    contact = Image.new("RGB", (5*300, 3*194+66), BG)
    draw = ImageDraw.Draw(contact)
    draw.text((16, 10), "J loop / 13 actual rendered angles, same fixed head ROI", font=font(24), fill=TEXT)
    for index, angle in enumerate(angles):
        x, y = (index%5)*300, 52+(index//5)*194
        draw.text((x+12, y), f"{entries[angle]['actualYawDegrees']:+.6f} deg", font=font(17), fill=TEXT)
        contact.paste(backdrop(Image.fromarray(raw[angle], "RGBA").crop(crop)), (x+12, y+26))
    contact_path = a.work/"J-loop-contact.webp"
    contact.save(contact_path, "WEBP", lossless=True)
    adjacent = []
    for a0, a1 in zip(angles, angles[1:]):
        im0, im1 = raw[a0], raw[a1]
        x0, y0, x1, y1 = crop
        m0, m1 = im0[y0:y1, x0:x1, 3] > 250, im1[y0:y1, x0:x1, 3] > 250
        common = m0 & m1
        delta = np.abs(im0[y0:y1, x0:x1, :3].astype("f4")-im1[y0:y1, x0:x1, :3].astype("f4"))
        adjacent.append({"fromYaw": a0, "toYaw": a1,
                         "opaqueMaskIoU": float(np.count_nonzero(m0 & m1)/np.count_nonzero(m0 | m1)),
                         "commonOpaqueRGBMAE255": float(delta[common].mean())})
    result = {
        "kind": __doc__, "display": captures[0]["display"], "assetId": captures[0]["assetId"],
        "captureReports": [{"path": str(path), "sha256": sha(path)} for path in reports],
        "html": {"path": captures[0]["html"], "sha256": captures[0]["htmlSHA256"]},
        "fixedCrop": crop, "cropProvenance": {"path": str(crop_proof), "sha256": sha(crop_proof)},
        "uniqueAngles": angles, "uniqueFrames": [entries[angle] for angle in angles],
        "sequenceRequestedYawDegrees": sequence,
        "sequenceActualYawDegrees": [entries[angle]["actualYawDegrees"] for angle in sequence],
        "playback": {"millisecondsPerFrame": 100, "fps": 10, "totalMilliseconds": 2400, "loop": "infinite",
                     "performanceClaim": False, "note": "Playback timing is chosen for inspection, not actual render speed."},
        "frameReuse": "The -15, 0, +15 degree PNGs are reused from the approved seven-angle static capture. Only ten missing 2.5-degree views were newly rendered. Return motion repeats those same real PNGs.",
        "processing": "Integer crop, nearest-neighbor spatial scaling, lossless animated WebP. No temporal interpolation, AI edits, optical flow, sharpening, or simulated geometry.",
        "adjacentFrameDiagnostics": adjacent,
        "metricLimitations": "Adjacent RGB/alpha changes include genuine camera parallax; these are not 3D accuracy or artifact-severity scores.",
        "contactSheet": {"path": str(contact_path), "sha256": sha(contact_path)},
        "outputs": outputs, "scriptSHA256": sha(__file__),
    }
    proof = a.output/"J-parallax-loop-provenance.json"
    proof.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"outputs": outputs, "proofSHA256": sha(proof), "contact": str(contact_path)}))


if __name__ == "__main__":
    main()
