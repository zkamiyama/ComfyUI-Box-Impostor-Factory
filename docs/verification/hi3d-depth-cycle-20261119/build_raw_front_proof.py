"""Compare the recorded input guide with the untouched first H3 +Z output."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFont

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("run", type=Path)
parser.add_argument("raw", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--stage", choices=("A", "B"), default="A")
parser.add_argument("--expected-raw-sha256")
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
prepared_path = args.run / args.stage / "prepared.json"
recipe = json.loads(prepared_path.read_text())
assert "w4a8" in recipe["models"]["model"].lower()
guide_record = recipe["faces"]["+Z"]["pictures"]["guide"]
guide_path = Path(guide_record["path"])
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
assert sha(guide_path) == guide_record["sha256"]
if args.expected_raw_sha256:
    assert sha(args.raw) == args.expected_raw_sha256
raw_copy = args.output / f"{args.stage}-pz-raw.png"
shutil.copyfile(args.raw, raw_copy)
assert sha(raw_copy) == sha(args.raw)
guide = Image.open(guide_path).convert("RGB")
raw = Image.open(args.raw).convert("RGB")
assert guide.size == raw.size == (1024, 1024)
crop = [440, 10, 600, 210]
panel = Image.new("RGB", (996, 612), (21, 24, 30))
draw = ImageDraw.Draw(panel)
fp = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
font = lambda size: ImageFont.truetype(fp, size)
draw.text((20, 16), f"{args.stage} / AI-generated RGB, before Box composition", font=font(28), fill=(238, 241, 246))
draw.text((20, 56), f"Fresh +Z H3 output only. This is not a Box render or the completed six-view {args.stage} asset.", font=font(17), fill=(190, 201, 216))
for x, title, image in [(20, "Input coarse mesh guide" if args.stage == "A" else "Input Hi3D mesh guide", guide), (360, "New H3 RGB / untouched pixels", raw)]:
    draw.text((x, 103), title, font=font(17), fill=(238, 241, 246))
    panel.paste(image.crop(crop).resize((320, 400), Image.Resampling.NEAREST), (x, 136))
draw.text((720, 103), "Complete raw frame", font=font(17), fill=(238, 241, 246))
panel.paste(raw.resize((256, 256), Image.Resampling.NEAREST), (720, 136))
draw.text((720, 408), "1/4 nearest-neighbor", font=font(14), fill=(190, 201, 216))
draw.text((720, 432), "Full PNG saved unchanged", font=font(14), fill=(190, 201, 216))
draw.text((20, 553), "Both head crops: [440,10,600,210], 2x nearest-neighbor. No sharpening or color edits.", font=font(17), fill=(190, 201, 216))
draw.text((20, 580), f"H3 W4A8 / INT8 text encoder / 1024 px / +Z seed {recipe['seeds']['+Z']} / 6 steps / CFG 1", font=font(16), fill=(158, 173, 193))
output = args.output / f"{args.stage}-raw-front-proof.webp"
panel.save(output, "WEBP", lossless=True)
decoded = np.asarray(Image.open(output).convert("RGB"))
for x, image in [(20, guide), (360, raw)]:
    assert np.array_equal(decoded[136:536:2, x:x + 320:2], np.asarray(image.crop(crop)))
metadata = {
    "stage": f"{args.stage} raw +Z only, before Box composition", "isBoxRender": False, "isCompletedSixViewAsset": False,
    "preparedPath": str(prepared_path), "preparedSHA256": sha(prepared_path),
    "recipe": {key: recipe[key] for key in ("models", "seeds", "sampler", "conditioning", "pictureOrder", "pictureSize", "outputSize", "viewConsistency", "materialMode")},
    "rawSource": str(args.raw), "rawSHA256": sha(args.raw),
    "rawUnchangedCopy": str(raw_copy), "rawCopySHA256": sha(raw_copy), "rawBytes": raw_copy.stat().st_size,
    "guideSource": str(guide_path), "guideSHA256": sha(guide_path), "sourceSize": raw.size,
    "headCropPixels": crop, "headDisplay": "2x nearest-neighbor", "fullFrameDisplay": "1/4 nearest-neighbor",
    "processing": "Integer crops and nearest-neighbor sampling only. No registration, sharpening, recoloring or AI postprocessing.",
    "pixelPreservationVerifiedAfterWebPDecode": True,
    "proof": {"path": str(output), "sha256": sha(output), "size": panel.size, "bytes": output.stat().st_size},
}
(args.output / f"{args.stage}-raw-front-proof-provenance.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
print(json.dumps({"proof": metadata["proof"], "rawCopy": str(raw_copy), "rawSHA256": metadata["rawSHA256"], "rawBytes": metadata["rawBytes"]}))
