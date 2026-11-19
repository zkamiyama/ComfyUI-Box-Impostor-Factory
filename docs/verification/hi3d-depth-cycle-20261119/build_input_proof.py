"""Make a pixel-preserving head panel from the recorded A input images."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("run", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
prepared_path = args.run / "A/prepared.json"
recipe = json.loads(prepared_path.read_text())
assert "w4a8" in recipe["models"]["model"].lower(), "Use the current official A recipe"
face = recipe["faces"]["+Z"]
crops = {"source": [430, 35, 590, 235], "guide": [440, 10, 600, 210],
         "axis-depth": [440, 10, 600, 210], "normals": [440, 10, 600, 210]}
labels = {"source": "Reference photograph", "guide": "Coarse mesh RGB guide",
          "axis-depth": "Axis depth / near = light", "normals": "World normals / RGB = XYZ"}
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
font = lambda size: ImageFont.truetype(font_path, size)
panel = Image.new("RGB", (1368, 612), (21, 24, 30))
draw = ImageDraw.Draw(panel)
draw.text((20, 16), "A INPUTS ONLY / before RGB generation", font=font(29), fill=(238, 241, 246))
draw.text((20, 56), "Actual +Z input files from A/prepared.json. These are inputs, not a generated result.", font=font(18), fill=(190, 201, 216))
metadata = {
    "kind": "actual A inputs; no generated result", "face": "+Z",
    "preparedPath": str(prepared_path), "preparedSHA256": sha(prepared_path),
    "recipe": {key: recipe[key] for key in ("branch", "baseline", "proxySha256", "conditioning", "pictureOrder",
               "pictureSize", "outputSize", "depthEncoding", "normalEncoding", "models", "seeds", "sampler",
               "viewConsistency", "materialMode")},
    "faceCanvas": face["canvas"], "viewMatrix": face["viewMatrix"], "projectionMatrix": face["projectionMatrix"],
    "cropPolicy": "Photograph and guide use separate visually selected head crops. Guide/depth/normals share identical coordinates. No image registration is claimed.",
    "processing": "Integer crop and 2x nearest-neighbor pixel replication only; no color adjustment, sharpening or AI processing.",
    "inputs": {},
}
verification = []
for i, (name, crop) in enumerate(crops.items()):
    item = face["pictures"][name]
    path = Path(item["path"])
    digest = sha(path)
    assert digest == item["sha256"], f"Input changed since prepared.json: {name}"
    original = Image.open(path).convert("RGB")
    patch = original.crop(crop)
    x, y = 20 + i * 336, 136
    draw.text((x, 101), labels[name], font=font(18), fill=(238, 241, 246))
    draw.text((x, 119), f"crop {crop[0]},{crop[1]} : {crop[2]},{crop[3]}", font=font(12), fill=(158, 173, 193))
    panel.paste(patch.resize((320, 400), Image.Resampling.NEAREST), (x, y))
    metadata["inputs"][name] = {"path": str(path), "sha256": digest, "size": original.size, "cropPixels": crop,
                               "panelPosition": [x, y], "displaySize": [320, 400]}
    verification.append((x, y, np.asarray(patch)))
draw.text((20, 552), "2x nearest-neighbor. Original pixel colors preserved. The photograph has a different camera and crop.", font=font(17), fill=(190, 201, 216))
seed = recipe["seeds"]["+Z"]
draw.text((20, 580), f"Recorded recipe: H3 W4A8 / INT8 text encoder / 1024 px / +Z seed {seed} / 6 steps / CFG 1", font=font(16), fill=(158, 173, 193))
output = args.output / "A-input-heads.webp"
panel.save(output, "WEBP", lossless=True)
decoded = np.asarray(Image.open(output).convert("RGB"))
for x, y, expected in verification:
    assert np.array_equal(decoded[y:y + 400:2, x:x + 320:2], expected), "Rendered proof altered source pixels"
metadata["pixelPreservationVerifiedAfterWebPDecode"] = True
metadata["proof"] = {"path": str(output), "sha256": sha(output), "size": panel.size, "bytes": output.stat().st_size}
(args.output / "A-input-heads-provenance.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
print(json.dumps(metadata["proof"]))
