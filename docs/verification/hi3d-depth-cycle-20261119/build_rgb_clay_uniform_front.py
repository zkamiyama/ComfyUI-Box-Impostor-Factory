"""Reuse verified captures to compare RGB, full-mesh clay and uniform/color Box.
No browser is started. Mesh and native Box have different material/light settings.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from build_box_control_front import BG, TEXT, MUTED, FONT, sha, backdrop, aligned_front_crop


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    metadata = {s: json.loads((args.work / f"{s}-provenance.json").read_text()) for s in ("B", "E")}
    captures = {name: json.loads((args.work / f"{name}/capture.json").read_text())
                for name in ("B-clay", "E-box", "B-box")}
    prior_proof_path = args.output / "E-uniform-color-proof.json"
    prior_proof = json.loads(prior_proof_path.read_text())
    assert metadata["B"]["assetId"] == prior_proof["sourceAsset"]
    assert metadata["E"]["assetId"] == prior_proof["assetId"]
    assert prior_proof["colorAlphaBitExact"] and prior_proof["sourceFilesUnchanged"]
    assert all(prior_proof["allOtherOriginalFilesByteExact"].values())
    assert metadata["E"]["proxySHA256"] == metadata["B"]["proxySHA256"]
    camera = captures["B-box"]["camera"]
    common_path = args.output / "A-full-comparison-provenance.json"
    crop = json.loads(common_path.read_text())["headCrops"]["front"]
    source_proofs = {}
    images = []
    front = metadata["B"]["faces"]["+Z"]
    assert sha(front["alignedPath"]) == front["alignedSHA256"]
    rgb = aligned_front_crop(front["alignedPath"], metadata["B"]["bakeBox"], crop, camera)
    images.append(backdrop(rgb))
    for name in ("B-clay", "E-box", "B-box"):
        stage, kind = name.split("-")
        cap = captures[name]
        assert not cap["errors"]
        assert cap["referenceManifestSHA256"] == captures["B-box"]["referenceManifestSHA256"]
        assert cap["htmlSHA256"] == metadata[stage][kind]["viewerSHA256"]
        for key in ("projection", "target", "halfExtent", "distance", "pixels", "cssPixels", "deviceScaleFactor"):
            assert cap["camera"][key] == camera[key], (name, key)
        entry = cap["modes"]["lit"]["front"]
        assert sha(entry["path"]) == entry["sha256"]
        im = Image.open(entry["path"]).convert("RGBA")
        assert im.size == tuple(camera["pixels"]) == (1024, 1024)
        images.append(backdrop(im.crop(crop)))
        source_proofs[name] = {
            "capturePath": str(args.work / f"{name}/capture.json"),
            "captureSHA256": sha(args.work / f"{name}/capture.json"),
            "image": entry, "htmlSHA256": cap["htmlSHA256"], "stageState": cap["stageState"],
            "assetId": metadata[stage]["assetId"],
        }
    tri = captures["B-clay"]["triangleCounts"]
    assert tri["expected"] == tri["meshPayload"] == metadata["B"]["clay"]["sourceTriangles"] == 1000000
    assert metadata["B"]["clay"]["geometryIdenticalAfterReload"]
    enlarged = [im.resize((im.width * 3, im.height * 3), Image.Resampling.NEAREST) for im in images]
    prior_path = args.output / "B-stage.webp"
    prior_json = args.output / "B-evidence.json"
    previous = json.loads(prior_json.read_text())
    assert sha(prior_path) == previous["panels"][prior_path.name]["sha256"]
    width, height = enlarged[0].size
    px = (380 - width) // 2
    prior_rgb = Image.open(prior_path).convert("RGB").crop((px, 125, px + width, 125 + height))
    assert np.array_equal(np.asarray(prior_rgb), np.asarray(enlarged[0]))
    board = Image.new("RGB", (1520, height + 214), BG)
    draw = ImageDraw.Draw(board)
    font = lambda size: ImageFont.truetype(FONT, size)
    draw.text((24, 18), "B: generated RGB / full mesh / uniform Box / color Box", font=font(25), fill=TEXT)
    draw.text((24, 54), "Same front camera and crop. Existing official Inspector renders; no new rendering.", font=font(16), fill=MUTED)
    labels = [
        ("H3 +Z / registered RGB", "2D color reference"),
        ("B / full 1M-triangle clay", "Mesh material, lights and normals"),
        ("E / uniform-color Box", "Native Box; RGB = 0.18 x alpha"),
        ("B / original-color Box", "Native Box; original six RGB views"),
    ]
    for i, (im, (title, subtitle)) in enumerate(zip(enlarged, labels)):
        draw.text((i * 380 + 16, 91), title, font=font(18), fill=TEXT)
        draw.text((i * 380 + 16, 119), subtitle, font=font(14), fill=MUTED)
        board.paste(im, (i * 380 + (380 - im.width) // 2, 149))
    draw.text((24, board.height - 47), "Mesh and Box use different material, lighting and normal policies; this is not a matched-lighting measurement.", font=font(14), fill=MUTED)
    draw.text((24, board.height - 24), "3x nearest-neighbor. RGB uses the same camera pixel grid. No sharpening or AI enhancement.", font=font(14), fill=MUTED)
    path = args.output / "B-E-rgb-clay-box-front.webp"
    board.save(path, "WEBP", lossless=True)
    assert np.array_equal(np.asarray(Image.open(path)), np.asarray(board))
    result = {
        "kind": "Registered B RGB, full B mesh clay, E uniform native Box, B color native Box",
        "camera": camera, "crop": crop, "cropSource": str(common_path), "cropSourceSHA256": sha(common_path),
        "sourceCaptures": source_proofs,
        "registeredRGB": {"path": front["alignedPath"], "sha256": front["alignedSHA256"],
                          "sampling": "Linear-light premultiplied RGBA bilinear at common camera pixel centers; sRGB conversion",
                          "pixelsEqualPriorBStage": True, "priorBStageSHA256": sha(prior_path)},
        "clay": {"sourceTriangles": 1000000, "payloadTriangles": tri["meshPayload"],
                 "geometryIdenticalAfterReload": True,
                 "normalPolicy": "B GLB has no NORMAL. Official mesh converter computes shading normals; native Box uses baked maps.",
                 "materialAndLighting": "Mesh and Box material/light settings differ; no quantitative matched-lighting claim"},
        "uniformColorProof": {"path": str(prior_proof_path), "sha256": sha(prior_proof_path)},
        "processing": {"integerCrop": True, "resize": "3x nearest-neighbor", "sharpening": False,
                       "aiPostprocessing": False, "losslessWebPDecodedPixelsEqual": True},
        "output": {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size, "size": list(board.size)},
        "scriptSHA256": sha(__file__),
    }
    proof_path = args.output / "B-E-rgb-clay-box-front-provenance.json"
    proof_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["output"]))
    print(json.dumps({"proof": str(proof_path), "sha256": sha(proof_path)}))


if __name__ == "__main__":
    main()
