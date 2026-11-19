"""Document actual six-view inputs and raw H3 outputs before Box composition."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("run", type=Path)
p.add_argument("asset", type=Path)
p.add_argument("output", type=Path)
p.add_argument("--stage", default="A")
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
m = json.loads((a.asset / "manifest.json").read_text())
qa = json.loads((a.asset / m["qa"]).read_text())
recipe = a.run / a.stage / "prepared.json"
BG, FG, SUB = (21, 24, 30), (237, 241, 247), (177, 192, 211)
font = lambda size: ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
names = {"+X": "px", "-X": "nx", "+Y": "py", "-Y": "ny", "+Z": "pz", "-Z": "nz"}
records, images, crops = {}, {}, {}
for face, key in names.items():
    entry = m["faces"][face]
    prov = entry["edit"]["provenance"]
    guide = a.run / "input" / a.stage / key / "guide.png"
    mask = a.run / "input" / a.stage / key / "mask.png"
    raw = a.run / a.stage / key / "generated.png"
    assert sha(guide) == prov["recipe"]["pictures"]["guide"]["sha256"]
    assert sha(mask) == prov["recipe"]["pictures"]["mask"]["sha256"]
    pair = [Image.open(path).convert("RGB") for path in (guide, raw)]
    assert pair[0].size == pair[1].size == (1024, 1024)
    images[face] = pair
    records[face] = {
        "status": entry["edit"]["status"], "reason": entry["edit"]["reason"],
        "maskIoU": qa["faces"][face]["after"]["iou"],
        "guide": {"path": str(guide), "sha256": sha(guide), "size": pair[0].size},
        "raw": {"path": str(raw), "sha256": sha(raw), "size": pair[1].size},
        "mask": {"path": str(mask), "sha256": sha(mask)},
        "seed": prov["recipeSeed"], "graphSha256": prov["graphSha256"],
    }
    if face in ("+X", "-X", "+Z", "-Z"):
        pixels = np.asarray(Image.open(mask).convert("L"))
        yy, xx = np.where(pixels[:205] > 127)
        assert len(xx)
        crops[face] = [max(0, int(xx.min()) - 8), 0, min(1024, int(xx.max()) + 9), 213]
out = {"assetId": m["asset_id"], "assetManifestSHA256": sha(a.asset / "manifest.json"),
       "stage": a.stage, "recipePath": str(recipe), "recipeSHA256": sha(recipe),
       "kind": "Actual guide inputs versus raw AI output before matting and Box composition",
       "faces": records, "panels": {}, "headCrops": crops,
       "processing": "Full frames reduced 1024 to 256 by nearest-neighbor. Head crops enlarged exactly 2x nearest-neighbor. No sharpening, correction, AI postprocessing, or raw-image overwrite.",
       "headCropDefinition": "Guide mask foreground >127 within top 205 rows; horizontal bounds plus 8px padding, y0..213. Same exact rectangle applied to raw output; no raw registration.",
       "acceptanceMeaning": "Accepted is the existing silhouette QA result, not validation of view direction, color, identity, or facial detail."}

def save(board, name):
    path = a.output / name
    board.save(path, "WEBP", lossless=True)
    assert np.array_equal(np.asarray(Image.open(path).convert("RGB")), np.asarray(board))
    out["panels"][name] = {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size, "size": board.size}

board = Image.new("RGB", (1660, 790), BG)
draw = ImageDraw.Draw(board)
draw.text((20, 14), f"{a.stage} / six actual guide inputs and raw AI outputs", font=font(26), fill=FG)
draw.text((20, 52), "Before matting and Box composition. Full frames at 1/4 size, nearest-neighbor; no enhancement.", font=font(18), fill=SUB)
draw.text((20, 79), "Accepted = silhouette test passed. It does not guarantee correct facial detail or consistency between views.", font=font(17), fill=SUB)
for i, face in enumerate(names):
    x, y = 20 + (i % 3) * 548, 122 + (i // 3) * 330
    r = records[face]
    draw.text((x, y), f"{face} / {r['status']} / mask IoU {r['maskIoU']:.4f}", font=font(18), fill=FG)
    for col, (label, image) in enumerate(zip(("GUIDE INPUT", "RAW AI OUTPUT"), images[face])):
        draw.text((x + col * 266, y + 27), label, font=font(16), fill=SUB)
        board.paste(image.resize((256, 256), Image.Resampling.NEAREST), (x + col * 266, y + 51))
save(board, f"{a.stage}-six-view-guide-raw.webp")
vertical = ("+Z", "+X", "-X", "-Z")
cw = max((c[2] - c[0]) * 2 for c in crops.values()) + 20
row_h = 490
heads = Image.new("RGB", (4 * cw + 40, 112 + 2 * row_h), BG)
draw = ImageDraw.Draw(heads)
draw.text((20, 14), f"{a.stage} / raw head detail before Box composition", font=font(25), fill=FG)
draw.text((20, 52), "Same guide-defined crop in each pair. 2x nearest-neighbor; no registration or enhancement.", font=font(17), fill=SUB)
for i, face in enumerate(vertical):
    x, y = 20 + (i % 2) * 2 * cw, 98 + (i // 2) * row_h
    draw.text((x, y), f"{face} / {records[face]['status']} / guide then raw AI", font=font(18), fill=FG)
    for col, image in enumerate(images[face]):
        image = image.crop(crops[face])
        image = image.resize((image.width * 2, image.height * 2), Image.Resampling.NEAREST)
        heads.paste(image, (x + col * cw + (cw - image.width) // 2, y + 33))
save(heads, f"{a.stage}-raw-head-guide-comparison.webp")
(a.output / f"{a.stage}-six-view-source-provenance.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps({"panels": out["panels"], "headCrops": crops,
                  "faces": {k: {"status": v["status"], "maskIoU": v["maskIoU"], "rawSHA256": v["raw"]["sha256"]} for k, v in records.items()}}))
