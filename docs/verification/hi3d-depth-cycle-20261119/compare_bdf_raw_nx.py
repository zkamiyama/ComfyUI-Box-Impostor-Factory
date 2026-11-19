"""Compare B/D/F raw -X outputs at one source-pixel crop, before Box."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("f_raw", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--expected-f-sha", required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    read = lambda path: json.loads(Path(path).read_text())
    ref_path = a.output / "A-six-view-source-provenance.json"
    crop = read(ref_path)["headCrops"]["-X"]
    stages = ("B", "D", "F")
    paths = {"B": a.run/"B/nx/generated.png", "D": a.run/"D/nx/generated.png", "F": a.f_raw}
    expected = {"B": "b1252bf14ca410884ef8f2cb2214893a372e994b59f31291995d73cf239f5103",
                "D": "290d900dc6a71d2542ff2fe17c77bb8db590bfe116d86b6caab693ab2a922025",
                "F": a.expected_f_sha}
    recipes = {s: read(a.run/s/"prepared.json") for s in stages}
    keys = ("models", "sampler", "pictureSize", "outputSize", "baseline", "proxySha256", "bakeBox", "transform")
    for key in keys:
        assert all(recipes[s][key] == recipes["B"][key] for s in stages), key
    assert all(recipes[s]["seeds"]["-X"] == 102 for s in stages)
    pictures = {s: recipes[s]["faces"]["-X"]["pictures"] for s in stages}
    for role in ("guide", "source"):
        for s in stages:
            item = pictures[s][role]
            assert sha(item["path"]) == item["sha256"] == pictures["B"][role]["sha256"]
    prompts = {s: recipes[s]["faces"]["-X"]["prompt"] for s in stages}
    assert prompts["B"] == prompts["D"]
    assert prompts["F"] != prompts["D"]
    experiment = recipes["F"]["experiment"]
    assert experiment["vlmPictureIndices"] == experiment["vaePictureIndices"] == [1, 2]
    assert experiment["promptIdenticalToD"] is False
    expected_prompt = prompts["D"]
    for change in experiment["promptChanges"]:
        assert change["before"] in expected_prompt
        expected_prompt = expected_prompt.replace(change["before"], change["after"])
    assert expected_prompt == prompts["F"]
    records, images = {}, {}
    for s, path in paths.items():
        assert sha(path) == expected[s]
        copy = a.output/f"{s}-nx-raw.png"
        if not copy.exists():
            shutil.copyfile(path, copy)
        assert sha(copy) == sha(path)
        images[s] = Image.open(path).convert("RGB")
        assert images[s].size == (1024, 1024)
        recipe_path = a.run/s/"prepared.json"
        records[s] = {
            "rawPath": str(path), "rawSHA256": sha(path), "unchangedCopy": str(copy), "copySHA256": sha(copy),
            "recipePath": str(recipe_path), "recipeSHA256": sha(recipe_path),
            "conditioning": recipes[s]["conditioning"], "seed": 102,
            "colorPictures": {role: pictures[s][role] for role in ("guide", "source")},
            "prompt": prompts[s], "promptSHA256": hashlib.sha256(prompts[s].encode()).hexdigest(),
            "experiment": recipes[s].get("experiment"),
        }
    bg, fg, sub = (21, 24, 30), (237, 241, 247), (177, 192, 211)
    font = lambda size: ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    width, height = (crop[2]-crop[0])*2, (crop[3]-crop[1])*2
    cw = max(width+24, 470)
    board = Image.new("RGB", (3*cw+40, height+560), bg)
    draw = ImageDraw.Draw(board)
    draw.text((20, 16), "B / D / F -X AI raw outputs, before Box composition", font=font(26), fill=fg)
    draw.text((20, 55), "Same two color PNGs, geometry, model, sampler and seed. Structural conditioning is reduced.", font=font(17), fill=sub)
    draw.text((20, 80), "F also removes the prompt instructions for depth, normals and mask; its prompt is different from D.", font=font(16), fill=sub)
    labels = {"B": "B / VLM 5 + VAE 5", "D": "D / VLM 5 + VAE first 2", "F": "F / VLM 2 + VAE 2 colors"}
    for i, stage in enumerate(stages):
        x = 20+i*cw
        draw.text((x, 116), labels[stage], font=font(20), fill=fg)
        crop_image = images[stage].crop(crop).resize((width, height), Image.Resampling.NEAREST)
        board.paste(crop_image, (x+(cw-width)//2, 150))
        draw.text((x, height+174), "Complete raw frame / 1/4 nearest", font=font(16), fill=sub)
        board.paste(images[stage].resize((256, 256), Image.Resampling.NEAREST), (x+(cw-256)//2, height+206))
    draw.text((20, height+495), f"Head crop {crop}: 2x nearest-neighbor. No registration, sharpening or color edits.", font=font(16), fill=sub)
    draw.text((20, height+523), "Raw PNGs copied unchanged. 1024px / -X seed 102 / 6 steps / CFG 1. One direction only.", font=font(16), fill=sub)
    out = a.output/"B-D-F-nx-raw-proof.webp"
    board.save(out, "WEBP", lossless=True)
    decoded = np.asarray(Image.open(out).convert("RGB"))
    assert np.array_equal(decoded, np.asarray(board))
    for i, stage in enumerate(stages):
        x = 20+i*cw+(cw-width)//2
        assert np.array_equal(decoded[150:150+height:2, x:x+width:2], np.asarray(images[stage].crop(crop)))
    proof = {
        "kind": __doc__, "isBoxRender": False, "stages": records, "headCrop": crop,
        "headCropReference": {"path": str(ref_path), "sha256": sha(ref_path)},
        "controlledSettingsEqual": {key: recipes["B"][key] for key in keys},
        "colorPictureSHA256": {role: pictures["B"][role]["sha256"] for role in ("guide", "source")},
        "declaredChanges": {
            "B_to_D": "VAE pictures reduced from 5 to first 2; VLM pictures and prompt unchanged",
            "D_to_F": "Structural VLM pictures and corresponding prompt instructions removed; two VAE color pictures unchanged",
        },
        "promptDifferenceVerifiedFromRecipe": True,
        "pixelPreservationVerifiedAfterWebPDecode": True,
        "processing": "Same integer crop, 2x nearest; full raw frames 1/4 nearest. No registration, sharpening, recoloring or AI processing.",
        "limitation": "One raw direction; does not establish final Box quality or six-view consistency.",
        "panel": {"path": str(out), "sha256": sha(out), "bytes": out.stat().st_size, "size": list(board.size)},
        "scriptSHA256": sha(__file__),
    }
    proof_path = a.output/"B-D-F-nx-raw-proof-provenance.json"
    proof_path.write_text(json.dumps(proof, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"panel": proof["panel"], "provenanceSHA256": sha(proof_path),
                      "fRawSHA256": sha(a.f_raw)}))


if __name__ == "__main__":
    main()
