"""Same-F-data lit/unlit front comparison from unchanged official Inspector outputs."""
import argparse
import copy
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
    captures = {s: json.loads((a.work/f"{folder}-box/capture.json").read_text()) for s, folder in (("F", "F"), ("J", "I"))}
    camera = captures["F"]["camera"]
    crop_source = a.output/"A-full-comparison-provenance.json"
    crop = json.loads(crop_source.read_text())["headCrops"]["front"]
    for s in ("F", "J"):
        cap = captures[s]
        assert cap["assetId"] == meta["assetId"]
        assert not cap["errors"] and "SwiftShader" in cap["graphicsRenderer"]
        assert cap["stageState"] == "Native Box"
        for key in ("projection", "target", "halfExtent", "distance", "pixels", "cssPixels", "deviceScaleFactor"):
            assert cap["camera"][key] == camera[key], (s, key)
        assert sha(cap["html"]) == cap["htmlSHA256"]
    assert captures["J"]["htmlSHA256"] == "2fd2f8910ee1dd9f097fd3dc91ec6bb772574f3592be7d7b919aa9306e901d0e"
    actual = captures["J"]["modes"]["lit"]["front"]["actualCamera"]
    assert actual["type"] == "OrthographicCamera"
    expected_position = captures["J"]["modes"]["lit"]["front"]["expectedPosition"]
    position_error = float(np.max(np.abs(np.asarray(actual["position"])-expected_position)))
    extent_error = float(max(abs(actual["top"]-camera["halfExtent"]), abs(actual["right"]-camera["halfExtent"])))
    assert position_error < 1e-4 and extent_error < 1e-4, (position_error, extent_error)
    front = meta["faces"]["+Z"]
    assert sha(front["alignedPath"]) == front["alignedSHA256"]
    rgb = aligned_front_crop(front["alignedPath"], meta["bakeBox"], crop, camera)
    images, full, source_images = [backdrop(rgb)], {}, {}
    for s in ("F", "J"):
        entry = captures[s]["modes"]["lit"]["front"]
        assert sha(entry["path"]) == entry["sha256"]
        im = Image.open(entry["path"]).convert("RGBA")
        assert im.size == (1024, 1024)
        full[s] = np.asarray(im)
        images.append(backdrop(im.crop(crop)))
        source_images[s] = entry
    enlarged = [im.resize((im.width*3, im.height*3), Image.Resampling.NEAREST) for im in images]
    board = Image.new("RGB", (1140, enlarged[0].height+184), BG)
    draw = ImageDraw.Draw(board)
    font = lambda size: ImageFont.truetype(FONT, size)
    draw.text((24, 18), "Same F data / native lighting / unlit display", font=font(25), fill=TEXT)
    draw.text((24, 55), "Unchanged official HTML outputs; identical native maps, blend, geometry and viewer camera.", font=font(16), fill=MUTED)
    labels = ("H3 +Z / registered RGB", "F / native PBR + ACES", "J / unlit + NoToneMapping")
    for i, (im, label) in enumerate(zip(enlarged, labels)):
        draw.text((i*380+16, 91), label, font=font(16), fill=TEXT)
        board.paste(im, (i*380+(380-im.width)//2, 125))
    draw.text((24, board.height-47), "3x nearest-neighbor; no sharpening or AI postprocessing. Native depth raycasting is retained.", font=font(14), fill=MUTED)
    draw.text((24, board.height-24), "This tests display shading only. Native cross-view color blending remains active; geometry is not repaired.", font=font(14), fill=MUTED)
    path = a.output/"F-J-front-stages.webp"
    board.save(path, "WEBP", lossless=True)
    assert np.array_equal(np.asarray(Image.open(path)), np.asarray(board))
    alpha_delta = full["J"][..., 3].astype("i2")-full["F"][..., 3].astype("i2")
    payload_proof = a.work/"I/inspector-payload-proof.json"
    assert payload_proof.exists()
    result = {
        "kind": __doc__, "assetId": meta["assetId"], "camera": camera, "headCrop": crop,
        "sourceImages": source_images,
        "actualCameraJ": actual, "actualCameraPositionError": position_error, "actualCameraExtentError": extent_error,
        "renderedAlpha": {"byteExact": bool(np.array_equal(full["F"][..., 3], full["J"][..., 3])),
                          "differentPixelCount": int(np.count_nonzero(alpha_delta)),
                          "maxAbsDifference": int(np.abs(alpha_delta).max())},
        "registeredRGB": {"path": front["alignedPath"], "sha256": front["alignedSHA256"],
                          "sampling": "Premultiplied linear-light bilinear at common camera pixel centers; sRGB conversion"},
        "html": {s: {"path": captures[s]["html"], "sha256": captures[s]["htmlSHA256"]} for s in ("F", "J")},
        "payloadProof": {"path": str(payload_proof), "sha256": sha(payload_proof)},
        "displayChange": "Official --box-lighting unlit; native color blend retained; lighting and ACES tone mapping removed",
        "cameraObserver": "Public factory and renderer.render read-only wrappers; unchanged input arguments and return values",
        "limitations": ["Front view only; yaw/parallax quality not yet tested.", "This does not improve source geometry."],
        "processing": "Identical integer crop, 3x nearest-neighbor, lossless WebP. No image enhancement.",
        "output": {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size, "size": list(board.size)},
        "scriptSHA256": sha(__file__),
    }
    out = a.output/"J-front-provenance.json"
    out.write_text(json.dumps(result, indent=2)+"\n")
    i_meta = copy.deepcopy(meta)
    i_meta.update(stage="J", displayControl=result["displayChange"], inputPayloadProof=result["payloadProof"])
    i_meta["box"].update(viewer=captures["J"]["html"], viewerSHA256=captures["J"]["htmlSHA256"],
                         viewerBytes=Path(captures["J"]["html"]).stat().st_size)
    (a.work/"J-provenance.json").write_text(json.dumps(i_meta, indent=2)+"\n")
    print(json.dumps({"output": result["output"], "proofSHA256": sha(out), "alpha": result["renderedAlpha"],
                      "actualCameraPositionError": position_error, "actualCameraExtentError": extent_error}))


if __name__ == "__main__":
    main()
