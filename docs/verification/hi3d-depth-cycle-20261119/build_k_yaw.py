"""Compare actual J/K seven-yaw captures with one fixed crop and common camera."""
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
    labels = ["J", "K-view", "K-dominant"]
    yaws = [-45, -30, -15, 0, 15, 30, 45]
    report_paths = {"J": [a.work/"J-yaw/capture.json"],
                    "K-view": [a.work/"K-view-front/capture.json", a.work/"K-view-yaw/capture.json"],
                    "K-dominant": [a.work/"K-dominant-front/capture.json", a.work/"K-dominant-yaw/capture.json"]}
    entries, reports, report_sources, images = {}, {}, {}, {}
    for label in labels:
        entries[label], reports[label], report_sources[label], images[label] = {}, [], [], {}
        for path in report_paths[label]:
            cap = json.loads(path.read_text())
            assert not cap["errors"] and "SwiftShader" in cap["graphicsRenderer"]
            assert cap["stageState"] == "Native Box"
            assert sha(cap["html"]) == cap["htmlSHA256"]
            reports[label].append(cap)
            report_sources[label].append({"path": str(path), "sha256": sha(path)})
            for entry in cap["yawFrames"]:
                angle = entry["requestedYawDegrees"]
                assert angle not in entries[label]
                assert sha(entry["path"]) == entry["sha256"]
                entries[label][angle] = entry
                im = Image.open(entry["path"]).convert("RGBA")
                assert im.size == (1024, 1024)
                images[label][angle] = im
        assert sorted(entries[label]) == yaws
    basis_path = a.output/"J-yaw-provenance.json"
    basis = json.loads(basis_path.read_text())
    crop = basis["fixedCrop"]
    camera = reports["J"][0]["camera"]
    checks = {}
    for label in labels:
        for cap in reports[label]:
            assert cap["assetId"] == reports["J"][0]["assetId"]
            for key in ("projection", "target", "halfExtent", "distance", "pixels", "cssPixels", "deviceScaleFactor"):
                assert cap["camera"][key] == camera[key], (label, key)
        for angle in yaws:
            anchor = entries["J"][angle]["actualCamera"]
            actual = entries[label][angle]["actualCamera"]
            for key in ("matrixWorld", "projectionMatrix"):
                assert np.allclose(actual[key], anchor[key], rtol=0, atol=1e-8), (label, angle, key)
            alpha_equal = np.array_equal(np.asarray(images[label][angle])[..., 3],
                                         np.asarray(images["J"][angle])[..., 3])
            assert alpha_equal, (label, angle, "alpha")
            checks[f"{label}/{angle}"] = {
                "actualYawDegrees": entries[label][angle]["actualYawDegrees"],
                "actualPitchDegrees": entries[label][angle]["actualPitchDegrees"],
                "alphaByteExactToJ": alpha_equal,
                "maxCameraMatrixAbsDifference": float(np.max(np.abs(np.asarray(actual["matrixWorld"])-anchor["matrixWorld"])))
            }
    cw, ch = (crop[2]-crop[0])*2, (crop[3]-crop[1])*2
    cell_w, row_h = cw+24, ch+66
    board = Image.new("RGB", (3*cell_w+24, 7*row_h+126), BG)
    draw = ImageDraw.Draw(board)
    font = lambda size: ImageFont.truetype(FONT, size)
    draw.text((24, 16), "Same real views / J native blend / K view-directed color", font=font(27), fill=TEXT)
    draw.text((24, 55), "F geometry/maps fixed; all unlit + NoToneMapping. Identical camera matrices and rendered alpha at each angle.", font=font(17), fill=MUTED)
    for row, angle in enumerate(yaws):
        for col, label in enumerate(labels):
            x, y = 12+col*cell_w, 92+row*row_h
            actual_yaw = entries[label][angle]["actualYawDegrees"]
            draw.text((x+12, y), f"{label} / {actual_yaw:+.6f} deg", font=font(21), fill=TEXT)
            detail = {"J": "Native RGB blend", "K-view": "Continuous view weights", "K-dominant": "Dominant view color"}[label]
            draw.text((x+12, y+27), detail, font=font(14), fill=MUTED)
            im = backdrop(images[label][angle].crop(crop)).resize((cw, ch), Image.Resampling.NEAREST)
            board.paste(im, (x+12, y+48))
    draw.text((24, board.height-26), f"Fixed crop {crop}; 2x nearest-neighbor. No AI edits, image alignment, sharpening, or generated viewpoints.", font=font(16), fill=MUTED)
    faces_path = a.output/"J-K-yaw-faces.webp"
    board.save(faces_path, "WEBP", lossless=True)
    body = Image.new("RGB", (7*280+24, 3*314+88), BG)
    bd = ImageDraw.Draw(body)
    bd.text((24, 16), "J / K-view / K-dominant: actual whole-model views", font=font(25), fill=TEXT)
    for row, label in enumerate(labels):
        for col, angle in enumerate(yaws):
            x, y = 12+col*280, 60+row*314
            bd.text((x+12, y), f"{label} / {entries[label][angle]['actualYawDegrees']:+.3f}", font=font(17), fill=TEXT)
            im = backdrop(images[label][angle]).resize((256, 256), Image.Resampling.NEAREST)
            body.paste(im, (x+12, y+32))
    bd.text((24, body.height-22), "1024 x 1024 snapshots shown at 1/4 nearest-neighbor; common camera and native depth/raycast data.", font=font(15), fill=MUTED)
    body_path = a.output/"J-K-yaw-body.webp"
    body.save(body_path, "WEBP", lossless=True)
    mode_differences = {}
    for angle in yaws:
        v = np.asarray(images["K-view"][angle])
        d = np.asarray(images["K-dominant"][angle])
        delta = np.abs(v[..., :3].astype("i2")-d[..., :3].astype("i2"))
        mode_differences[str(angle)] = {"RGBByteExact": bool(np.array_equal(v[..., :3], d[..., :3])),
                                        "RGBDifferentPixels": int(np.count_nonzero(np.any(delta, axis=-1)))}
    result = {
        "kind": __doc__, "assetId": reports["J"][0]["assetId"], "camera": camera,
        "cropSource": {"path": str(basis_path), "sha256": sha(basis_path)}, "fixedCrop": crop,
        "reports": report_sources, "frames": entries, "invariantChecks": checks,
        "colorModeDifferences": mode_differences,
        "frontReuse": "K 0-degree PNGs reused byte-for-byte from the separately approved front capture; only the other six angles were rendered in the yaw sessions.",
        "outputs": [{"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size, "size": list(Image.open(path).size)} for path in (faces_path, body_path)],
        "processing": "Actual rendered snapshots. Same integer crop at each angle. Nearest-neighbor scaling and lossless WebP; no image enhancement.",
        "limitations": "Static views show appearance and geometry-dependent coverage; temporal continuity and dominant-face boundary switching require separate motion/boundary tests.",
        "scriptSHA256": sha(__file__),
    }
    proof = a.output/"J-K-yaw-provenance.json"
    proof.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"outputs": result["outputs"], "proofSHA256": sha(proof), "modeDifferences": mode_differences}))


if __name__ == "__main__":
    main()
