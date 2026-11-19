"""Compare A/B raw -X images with an identical source-pixel crop, before Box."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFont

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("run", type=Path)
p.add_argument("b_raw", type=Path)
p.add_argument("output", type=Path)
p.add_argument("--expected-b-sha256", required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
read = lambda path: json.loads(Path(path).read_text())
reference_path = a.output / "A-six-view-source-provenance.json"
reference = read(reference_path)
crop = reference["headCrops"]["-X"]
paths = {"A": a.run / "A/nx/generated.png", "B": a.b_raw}
assert sha(paths["A"]) == reference["faces"]["-X"]["raw"]["sha256"]
assert sha(paths["B"]) == a.expected_b_sha256
recipes = {stage: read(a.run / stage / "prepared.json") for stage in ("A", "B")}
for key in ("models", "sampler", "conditioning", "pictureOrder", "pictureSize", "outputSize"):
    assert recipes["A"][key] == recipes["B"][key], key
assert recipes["A"]["seeds"]["-X"] == recipes["B"]["seeds"]["-X"]
assert recipes["A"]["faces"]["-X"]["pictures"]["source"]["sha256"] == recipes["B"]["faces"]["-X"]["pictures"]["source"]["sha256"]
records, images = {}, {}
for stage, raw_path in paths.items():
    recipe_path = a.run / stage / "prepared.json"
    copied = a.output / f"{stage}-nx-raw.png"
    shutil.copyfile(raw_path, copied)
    assert sha(copied) == sha(raw_path)
    images[stage] = Image.open(raw_path).convert("RGB")
    assert images[stage].size == (1024, 1024)
    records[stage] = {"rawPath": str(raw_path), "rawSHA256": sha(raw_path),
                      "unchangedCopy": str(copied), "copySHA256": sha(copied), "rawBytes": copied.stat().st_size,
                      "recipePath": str(recipe_path), "recipeSHA256": sha(recipe_path),
                      "guide": recipes[stage]["faces"]["-X"]["pictures"]["guide"],
                      "seed": recipes[stage]["seeds"]["-X"]}
BG, FG, SUB = (21, 24, 30), (237, 241, 247), (177, 192, 211)
font = lambda size: ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
w, h = (crop[2] - crop[0]) * 2, (crop[3] - crop[1]) * 2
cw = max(w + 24, 470)
board = Image.new("RGB", (2 * cw + 40, h + 530), BG)
draw = ImageDraw.Draw(board)
draw.text((20, 16), "A / B -X raw outputs, before Box composition", font=font(26), fill=FG)
draw.text((20, 55), "Same H3 settings and seed. Identical head crop; complete raw frames below.", font=font(17), fill=SUB)
labels = {"A": "A / coarse geometry input", "B": "B / Hi3D geometry input"}
for i, stage in enumerate(("A", "B")):
    x = 20 + i * cw
    draw.text((x, 102), labels[stage], font=font(20), fill=FG)
    image = images[stage].crop(crop).resize((w, h), Image.Resampling.NEAREST)
    board.paste(image, (x + (cw - w) // 2, 136))
    draw.text((x, h + 156), "Complete raw frame / 1/4 nearest", font=font(16), fill=SUB)
    board.paste(images[stage].resize((256, 256), Image.Resampling.NEAREST), (x + (cw - 256) // 2, h + 188))
draw.text((20, h + 465), f"Head crop {crop}: 2x nearest-neighbor. No registration, sharpening or color edits.", font=font(16), fill=SUB)
draw.text((20, h + 493), f"Raw PNGs copied unchanged. 1024px / -X seed {records['A']['seed']} / 6 steps / CFG 1.", font=font(16), fill=SUB)
output = a.output / "A-B-nx-raw-proof.webp"
board.save(output, "WEBP", lossless=True)
decoded = np.asarray(Image.open(output).convert("RGB"))
for i, stage in enumerate(("A", "B")):
    x = 20 + i * cw + (cw - w) // 2
    assert np.array_equal(decoded[136:136+h:2, x:x+w:2], np.asarray(images[stage].crop(crop)))
proof = {
    "kind": "A versus B raw -X RGB, before Box composition", "isBoxRender": False,
    "stages": records, "headCrop": crop, "headCropReference": {"path": str(reference_path), "sha256": sha(reference_path)},
    "controlledSettingsEqual": {key: recipes["A"][key] for key in ("models", "sampler", "conditioning", "pictureOrder", "pictureSize", "outputSize")},
    "sameReferencePhotoSHA256": recipes["A"]["faces"]["-X"]["pictures"]["source"]["sha256"],
    "processing": "Same integer pixel crop enlarged 2x nearest-neighbor. Full-frame inset 1/4 nearest-neighbor. No registration, sharpening, recoloring or AI postprocessing.",
    "pixelPreservationVerifiedAfterWebPDecode": True,
    "limitation": "Compares raw generation of one direction. It does not validate six-view consistency, registration, geometry accuracy or Box rendering.",
    "panel": {"path": str(output), "sha256": sha(output), "bytes": output.stat().st_size, "size": board.size}
}
(a.output / "A-B-nx-raw-proof-provenance.json").write_text(json.dumps(proof, indent=2) + "\n")
print(json.dumps({"panel": proof["panel"], "raws": records}))
