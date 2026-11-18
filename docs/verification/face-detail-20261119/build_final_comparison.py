"""Reproduce the native Box comparison; add --render to regenerate all six views."""
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
CASES = [
    ("before", "Before / 512 px per face", "AI front rejected; original result",
     "4a8ccaf59bda137a8e34c8eb03833a4c9d91f93b4846fd68ece1e981b9b8c6c0"),
    ("fixed-512", "Fixed / 512 px per face", "AI front retained; views reconciled",
     "77cb3973214fbef0a54257982a516cc8e4eb7481357a48477f2f42d956c86b70"),
    ("fixed-1024", "Fixed / 1024 px per face", "AI front retained; views reconciled",
     "1b8a756f88c84570b17ab582caa55291f29df99b9e1a4800be00b8ab99a57c52"),
]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--assets-root", type=Path, default=REPO.parents[1] / "output/box_impostor")
parser.add_argument("--render", action="store_true")
args = parser.parse_args()

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

if args.render:
    env = {**os.environ, "BI_SOFTWARE_GPU": "1"}
    for slug, _, _, aid in CASES:
        for view, selected in [("front", "+Z"), ("diagonal", "diagonal")]:
            subprocess.run(["node", str(HERE / "capture_native.mjs"), str(args.assets_root / aid),
                            str(HERE / "native" / (slug + "-" + view)), selected], env=env, check=True)

def font(size):
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)

def panel(path, crop, size, sampling):
    image = Image.open(path).convert("RGBA").crop(crop).resize(size, sampling)
    background = Image.new("RGBA", size, (23, 27, 29, 255))
    background.alpha_composite(image)
    return background.convert("RGB")

canvas = Image.new("RGB", (1320, 1520), (240, 242, 245))
draw = ImageDraw.Draw(canvas)
draw.text((30, 20), "Actual Box Impostor / before and after", fill=(20, 28, 35), font=font(30))
draw.text((30, 60), "Identical camera, renderer and lighting. All panels are native renders of the saved Box assets.",
          fill=(55, 65, 75), font=font(18))
provenance = {"method": "native Box renderer; unmodified standard shader", "projection": "orthographic",
              "canvasPixels": [1024, 1024], "deviceScaleFactor": 1,
              "camera": "explicit front (+Z) or diagonal, then fit after setting 1024 x 1024 stage",
              "faceCrop": {"xyxy": [440, 120, 600, 300], "displaySize": [400, 450], "resampling": "nearest", "scale": 2.5},
              "bodyCrop": {"xyxy": [240, 100, 800, 940], "displaySize": [196, 294], "resampling": "Lanczos downsampling"},
              "sharpening": False, "generatedImageSubstitution": False, "cases": []}
for index, (slug, title, note, aid) in enumerate(CASES):
    x = 30 + index * 430
    manifest_path = args.assets_root / aid / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["asset_id"] == aid
    item = {"label": title, "assetId": aid, "resolution": manifest["resolution"],
            "manifestSHA256": sha(manifest_path), "proxySHA256": manifest["proxySha256"],
            "bakeBox": manifest["bakeBox"],
            "frontEditStatus": manifest["faces"]["+Z"].get("edit", {}).get("status"),
            "textureSHA256": {k: v["sha256"] for k, v in manifest["textures"].items()}, "renders": []}
    draw.text((x, 96), title, fill=(16, 30, 42), font=font(23))
    draw.text((x, 125), note, fill=(60, 72, 82), font=font(16))
    for row, mode in enumerate(("lit", "albedo")):
        y = 185 + row * 500
        label = "FACE / LIT / 2.5x nearest" if mode == "lit" else "FACE / ALBEDO / 2.5x nearest"
        draw.text((x, y - 28), label, fill=(60, 72, 82), font=font(16))
        path = HERE / "native" / (slug + "-front") / ("standard-" + mode + ".png")
        assert Image.open(path).size == (1024, 1024)
        canvas.paste(panel(path, (440, 120, 600, 300), (400, 450), Image.Resampling.NEAREST), (x, y))
    for j, view in enumerate(("front", "diagonal")):
        folder = HERE / "native" / (slug + "-" + view)
        evidence = json.loads((folder / "render-evidence.json").read_text())
        assert evidence["asset"].rstrip("/").endswith(aid)
        assert evidence["width"] == evidence["height"] == 1024
        standard = [v for v in evidence["variants"] if v["variant"] == "standard"]
        assert len(standard) == 1 and not standard[0]["errors"], evidence
        assert evidence["view"] == ("+Z" if view == "front" else "diagonal")
        for mode in ("lit", "albedo"):
            path = folder / ("standard-" + mode + ".png")
            item["renders"].append({"view": view, "shading": mode, "path": path.relative_to(HERE).as_posix(),
                                    "sha256": sha(path), "renderEvidence": evidence})
        draw.text((x + j * 204, 1158), view.upper() + " / lit", fill=(60, 72, 82), font=font(16))
        canvas.paste(panel(folder / "standard-lit.png", (240, 100, 800, 940), (196, 294),
                           Image.Resampling.LANCZOS), (x + j * 204, 1185))
    provenance["cases"].append(item)

draw.text((30, 1493), "Face crops: identical pixel coordinates, nearest-neighbour enlargement, no sharpening. Body thumbnails: reduced.",
          fill=(65, 75, 85), font=font(15))
output = HERE / "final-comparison.webp"
canvas.save(output, lossless=True, method=6)
provenance["sameProxy"] = len({c["proxySHA256"] for c in provenance["cases"]}) == 1
provenance["sameBakeBox"] = all(c["bakeBox"] == provenance["cases"][0]["bakeBox"] for c in provenance["cases"])
provenance["outputSHA256"] = sha(output)
provenance["sourceCodeSHA256"] = {str(p.relative_to(REPO)): sha(p) for p in
    (Path(__file__).resolve(), HERE / "capture_native.mjs", REPO / "web/viewer.html",
     REPO / "worker/chromium.mjs", REPO / "node_modules/three-impostor/Impostor.js")}
(HERE / "final-comparison-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
print(output)
print("sameProxy", provenance["sameProxy"], "sameBakeBox", provenance["sameBakeBox"])
