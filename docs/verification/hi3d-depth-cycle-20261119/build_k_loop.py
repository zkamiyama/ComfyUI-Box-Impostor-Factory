"""J/K-view small-parallax comparison from actual snapshots; no synthesized frames."""
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
    paths = {
        "J": [a.work/"J-yaw/capture.json", a.work/"J-loop-unique/capture.json"],
        "K-view": [a.work/"K-view-front/capture.json", a.work/"K-view-yaw/capture.json",
                   a.work/"K-view-loop-unique/capture.json"],
    }
    angles = [i*2.5 for i in range(-6, 7)]
    sequence = angles + angles[-2:0:-1]
    frames, reports, source_reports = {}, {}, {}
    camera = None
    for label in paths:
        frames[label], reports[label], source_reports[label] = {}, [], []
        for path in paths[label]:
            cap = json.loads(path.read_text())
            assert not cap["errors"] and "SwiftShader" in cap["graphicsRenderer"]
            assert cap["stageState"] == "Native Box"
            assert sha(cap["html"]) == cap["htmlSHA256"]
            if camera is None:
                camera = cap["camera"]
            for key in ("projection", "target", "halfExtent", "distance", "pixels", "cssPixels", "deviceScaleFactor"):
                assert cap["camera"][key] == camera[key], (label, key)
            reports[label].append(cap)
            source_reports[label].append({"path": str(path), "sha256": sha(path)})
            for entry in cap["yawFrames"]:
                angle = entry["requestedYawDegrees"]
                if angle in angles:
                    assert angle not in frames[label]
                    assert sha(entry["path"]) == entry["sha256"]
                    frames[label][angle] = entry
        assert sorted(frames[label]) == angles
    crop_proof = a.output/"J-yaw-provenance.json"
    crop = json.loads(crop_proof.read_text())["fixedCrop"]
    raw, checks = {}, {}
    for label in frames:
        raw[label] = {}
        for angle, entry in frames[label].items():
            im = Image.open(entry["path"]).convert("RGBA")
            assert im.size == (1024, 1024)
            raw[label][angle] = im
    for angle in angles:
        jc, kc = frames["J"][angle]["actualCamera"], frames["K-view"][angle]["actualCamera"]
        matrix_differences = {}
        for key in ("matrixWorld", "projectionMatrix"):
            delta = float(np.max(np.abs(np.asarray(jc[key])-kc[key])))
            assert delta < 1e-8, (angle, key, delta)
            matrix_differences[key] = delta
        alpha_equal = np.array_equal(np.asarray(raw["J"][angle])[..., 3], np.asarray(raw["K-view"][angle])[..., 3])
        assert alpha_equal, angle
        checks[str(angle)] = {"cameraMaximumAbsDifferences": matrix_differences, "renderedAlphaByteExact": alpha_equal,
                              "actualYawJ": frames["J"][angle]["actualYawDegrees"],
                              "actualYawKView": frames["K-view"][angle]["actualYawDegrees"]}
    font = lambda size: ImageFont.truetype(FONT, size)
    cw, ch = (crop[2]-crop[0])*2, (crop[3]-crop[1])*2
    face_boards, body_boards = {}, {}
    for angle in angles:
        board = Image.new("RGB", (2*(cw+24)+24, ch+148), BG)
        draw = ImageDraw.Draw(board)
        draw.text((24, 12), "Same actual small parallax / J versus K-view", font=font(25), fill=TEXT)
        for column, label in enumerate(("J", "K-view")):
            x = 24+column*(cw+24)
            detail = "J / native RGB blend" if label == "J" else "K-view / view-weighted RGB"
            draw.text((x, 47), detail, font=font(20), fill=TEXT)
            draw.text((x, 74), f"Measured yaw {frames[label][angle]['actualYawDegrees']:+.6f} deg", font=font(16), fill=MUTED)
            patch = backdrop(raw[label][angle].crop(crop)).resize((cw, ch), Image.Resampling.NEAREST)
            board.paste(patch, (x, 103))
        draw.text((24, board.height-31), "Actual 2.5-degree frames / identical fixed crop / 2x nearest / chosen playback: 10 fps", font=font(16), fill=MUTED)
        draw.text((24, board.height-13), "No temporal interpolation or AI correction. Playback timing does not measure rendering performance.", font=font(12), fill=MUTED)
        face_boards[angle] = board
        body = Image.new("RGB", (544, 620), BG)
        draw = ImageDraw.Draw(body)
        draw.text((16, 10), "K-view / unlit / actual whole model", font=font(21), fill=TEXT)
        draw.text((16, 39), f"Measured yaw {frames['K-view'][angle]['actualYawDegrees']:+.6f} deg", font=font(16), fill=MUTED)
        body.paste(backdrop(raw["K-view"][angle]).resize((512, 512), Image.Resampling.NEAREST), (16, 68))
        draw.text((16, 589), "Actual frames / 1/2 nearest / playback 10 fps", font=font(15), fill=MUTED)
        draw.text((16, 607), "Playback timing is not measured render performance.", font=font(12), fill=MUTED)
        body_boards[angle] = body
    outputs = []
    for filename, boards in (("J-K-parallax-face-loop.webp", face_boards),
                             ("K-view-parallax-body-loop.webp", body_boards)):
        images = [boards[angle] for angle in sequence]
        path = a.output/filename
        images[0].save(path, "WEBP", save_all=True, append_images=images[1:], duration=100, loop=0,
                       lossless=True, quality=100, method=4)
        with Image.open(path) as verify:
            assert verify.n_frames == 24
            for index, expected in enumerate(images):
                verify.seek(index)
                assert np.array_equal(np.asarray(verify.convert("RGB")), np.asarray(expected)), index
        outputs.append({"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size,
                        "size": list(images[0].size), "frames": 24, "losslessDecodedPixelEquality": True})
    selected = [-15, -7.5, 0, 7.5, 15]
    contact = Image.new("RGB", (600, 5*206+70), BG)
    draw = ImageDraw.Draw(contact)
    draw.text((16, 12), "J / K-view: real small-angle frames", font=font(24), fill=TEXT)
    for row, angle in enumerate(selected):
        for column, label in enumerate(("J", "K-view")):
            x, y = 12+column*300, 58+row*206
            draw.text((x+12, y), f"{label} / {frames[label][angle]['actualYawDegrees']:+.6f}", font=font(17), fill=TEXT)
            contact.paste(backdrop(raw[label][angle].crop(crop)), (x+12, y+30))
    contact_path = a.work/"K-view-loop-contact.webp"
    contact.save(contact_path, "WEBP", lossless=True)
    result = {
        "kind": __doc__, "assetId": reports["J"][0]["assetId"], "camera": camera, "fixedCrop": crop,
        "cropProvenance": {"path": str(crop_proof), "sha256": sha(crop_proof)},
        "captureReports": source_reports, "uniqueFrames": frames,
        "all13AngleInvariants": checks,
        "html": {label: {"path": reports[label][0]["html"], "sha256": reports[label][0]["htmlSHA256"]} for label in reports},
        "sequenceRequestedYawDegrees": sequence,
        "playback": {"millisecondsPerFrame": 100, "fps": 10, "totalMilliseconds": 2400,
                     "performanceClaim": False, "note": "Chosen playback timing, not measured render speed."},
        "reuse": "All J images are reused. K-view -15/0/+15 PNGs are reused from the existing static proof; only ten missing 2.5-degree angles were newly rendered. Return frames repeat those real snapshots.",
        "processing": "Same fixed integer crop. Nearest-neighbor spatial scaling, lossless animated WebP. No generated views, interpolation, image alignment, optical flow, or sharpening.",
        "encoding": {"lossless": True, "method": 4, "quality": 100, "allDecodedFramesPixelExact": True},
        "limitations": ["Only +/-15-degree small parallax is covered by this loop.",
                        "This display control leaves the source geometry and coverage defects unchanged.",
                        "The separately captured 45-degree views show unresolved cross-view RGB disagreement."],
        "contactSheet": {"path": str(contact_path), "sha256": sha(contact_path)},
        "outputs": outputs, "scriptSHA256": sha(__file__),
    }
    proof = a.output/"J-K-parallax-loop-provenance.json"
    proof.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"outputs": outputs, "proofSHA256": sha(proof), "contact": str(contact_path)}))


if __name__ == "__main__":
    main()
