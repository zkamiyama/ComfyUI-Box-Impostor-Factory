"""Final CPU-only RGB/F/J/K-view front board from existing verified snapshots."""
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
    paths = {"F": a.work/"F-box/capture.json",
             "J": a.work/"I-box/capture.json",
             "K-view": a.work/"K-view-front/capture.json"}
    captures = {label: json.loads(path.read_text()) for label, path in paths.items()}
    camera = captures["J"]["camera"]
    crop_proof = a.output/"A-full-comparison-provenance.json"
    crop = json.loads(crop_proof.read_text())["headCrops"]["front"]
    front = meta["faces"]["+Z"]
    assert sha(front["alignedPath"]) == front["alignedSHA256"]
    rgb = aligned_front_crop(front["alignedPath"], meta["bakeBox"], crop, camera)
    images, inputs, arrays = [backdrop(rgb)], {}, {}
    actual = {}
    for label, cap in captures.items():
        assert cap["assetId"] == meta["assetId"]
        assert not cap["errors"] and cap["stageState"] == "Native Box"
        assert "SwiftShader" in cap["graphicsRenderer"]
        assert sha(cap["html"]) == cap["htmlSHA256"]
        for key in ("projection", "target", "halfExtent", "distance", "pixels", "cssPixels", "deviceScaleFactor"):
            assert cap["camera"][key] == camera[key], (label, key)
        entry = next(e for e in cap["yawFrames"] if e["requestedYawDegrees"] == 0) if label == "K-view" else cap["modes"]["lit"]["front"]
        assert sha(entry["path"]) == entry["sha256"]
        source = Image.open(entry["path"]).convert("RGBA")
        assert source.size == (1024, 1024)
        arrays[label] = np.asarray(source)
        images.append(backdrop(source.crop(crop)))
        actual[label] = entry.get("actualCamera")
        inputs[label] = {"capture": {"path": str(paths[label]), "sha256": sha(paths[label])},
                         "html": {"path": cap["html"], "sha256": cap["htmlSHA256"]}, "frame": entry}
    for label in ("J", "K-view"):
        assert np.array_equal(arrays[label][..., 3], arrays["F"][..., 3]), label
    for key in ("matrixWorld", "projectionMatrix"):
        assert np.allclose(actual["J"][key], actual["K-view"][key], rtol=0, atol=1e-8), key
    enlarged = [im.resize((im.width*3, im.height*3), Image.Resampling.NEAREST) for im in images]
    board = Image.new("RGB", (1520, enlarged[0].height+184), BG)
    draw = ImageDraw.Draw(board)
    font = lambda size: ImageFont.truetype(FONT, size)
    draw.text((24, 18), "From the generated RGB to the actual Box: F / J / K-view", font=font(26), fill=TEXT)
    draw.text((24, 55), "Same geometry, textures, camera and crop. Existing original snapshots; no additional rendering.", font=font(17), fill=MUTED)
    labels = ("Registered RGB / +Z", "F / original lighting", "J / unlit", "K-view / view selection")
    panel_checks = []
    previous_sources = [
        (a.output/"K-front-stages.webp", 0),
        (a.output/"F-J-front-stages.webp", 1),
        (a.output/"K-front-stages.webp", 1),
        (a.output/"K-front-stages.webp", 2),
    ]
    for index, (im, label) in enumerate(zip(enlarged, labels)):
        x = index*380+(380-im.width)//2
        draw.text((index*380+16, 91), label, font=font(18), fill=TEXT)
        board.paste(im, (x, 125))
        prior_path, prior_column = previous_sources[index]
        prior_x = prior_column*380+(380-im.width)//2
        prior = Image.open(prior_path).convert("RGB").crop((prior_x, 125, prior_x+im.width, 125+im.height))
        assert np.array_equal(np.asarray(im), np.asarray(prior)), (index, prior_path)
        panel_checks.append({"column": index, "priorBoard": str(prior_path), "priorBoardSHA256": sha(prior_path),
                             "priorColumn": prior_column, "imagePanelPixelsByteExact": True})
    draw.text((24, board.height-47), "3x nearest-neighbor. No sharpening or AI correction. Original PNG/HTML files are unchanged.", font=font(15), fill=MUTED)
    draw.text((24, board.height-24), "F to J removes lighting/tone mapping; J to K changes RGB view selection. Native geometry/depth stay fixed.", font=font(15), fill=MUTED)
    path = a.output/"RGB-F-J-K-front.webp"
    board.save(path, "WEBP", lossless=True)
    assert np.array_equal(np.asarray(Image.open(path)), np.asarray(board))
    for source in inputs.values():
        assert sha(source["frame"]["path"]) == source["frame"]["sha256"]
        assert sha(source["html"]["path"]) == source["html"]["sha256"]
    residual_path = a.output/"K-front-residual-classification.json"
    proof = {
        "kind": __doc__, "assetId": meta["assetId"], "camera": camera, "headCrop": crop,
        "labels": {"display": list(labels), "requestedJapanese": ["登録済みRGB", "F元の照明表示", "J照明なし", "K-view視線方向選択"]},
        "inputs": inputs, "actualCameras": actual,
        "cameraEvidence": "All configured camera fields are equal. J/K have observed actual render matrices which agree. Legacy F capture recorded the same verified camera setup without the later read-only matrix observer.",
        "renderedAlpha": {"F_J_KViewByteExact": True},
        "registeredRGB": {"path": front["alignedPath"], "sha256": front["alignedSHA256"],
                          "sampling": "Same established premultiplied linear-light bilinear at front-camera pixel centers; 8-bit sRGB."},
        "priorPanelPixelEquality": panel_checks,
        "residualDiagnosis": {"path": str(residual_path), "sha256": sha(residual_path),
                              "note": "Small internal nose/cheek/chin marks were classified as opaque RGB residuals, not transparent holes in the source-only eroded head region."} if residual_path.exists() else None,
        "output": {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size, "size": list(board.size)},
        "processing": "CPU composition only: same integer crop, 3x nearest-neighbor, lossless WebP. Every image panel equals the already reviewed comparison pixels. No new browser/render/image generation.",
        "scriptSHA256": sha(__file__),
    }
    out = a.output/"RGB-F-J-K-front-provenance.json"
    out.write_text(json.dumps(proof, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({"output": proof["output"], "proofSHA256": sha(out), "allImagePanelsEqualPriorBoards": True}))


if __name__ == "__main__":
    main()
