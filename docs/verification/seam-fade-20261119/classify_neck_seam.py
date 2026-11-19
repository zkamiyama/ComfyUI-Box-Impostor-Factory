"""Classify the visible -30 degree neck seam directly from saved RGBA pixels."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import label as connected_components


WORK = Path("/mnt/f/_dev/box-impostor-lab/results/seam-fade-20261119")
OLD = Path("/mnt/f/_dev/box-impostor-lab/results/hi3d-depth-cycle-20261119")
ROI = (510, 202, 591, 259)
POINTS = {"wedge_dark": (526, 225), "wedge_transition": (527, 225), "neck_skin": (528, 225),
          "chin_above": (541, 224), "chin_below": (541, 225)}
PATHS = {"K-view": OLD/"K-view-yaw/yaw-m30_000.png",
         "L1-view-fade": WORK/"L1-view-fade-first/yaw-m30_000.png",
         "L2-view-soft": WORK/"L2-view-soft-first/yaw-m30_000.png"}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    inputs, arrays, counts = {}, {}, {}
    x0, y0, x1, y1 = ROI
    for stage, path in PATHS.items():
        full = np.asarray(Image.open(path).convert("RGBA"))
        arrays[stage] = full[y0:y1, x0:x1]
        crop = arrays[stage]
        dark = (crop[..., 3] > 250) & (crop[..., :3].mean(axis=-1) < 60)
        labels, n = connected_components(dark, structure=np.ones((3, 3), bool))
        components = []
        for i in range(1, n+1):
            y, x = np.where(labels == i)
            components.append({"pixels": int(len(x)), "bboxFullImage": [int(x.min()+x0), int(y.min()+y0),
                int(x.max()+1+x0), int(y.max()+1+y0)]})
        components.sort(key=lambda item: -item["pixels"])
        inputs[stage] = {"path": str(path), "sha256": sha(path),
                         "probesRGBA": {name: full[y, x].tolist() for name, (x, y) in POINTS.items()}}
        counts[stage] = {"roiPixels": int(crop.shape[0]*crop.shape[1]),
            "transparentAlpha0to5": int((crop[..., 3] <= 5).sum()),
            "partialAlpha6to250": int(((crop[..., 3] > 5) & (crop[..., 3] <= 250)).sum()),
            "opaqueAlpha251to255": int((crop[..., 3] > 250).sum()),
            "opaqueDarkComponents": components}
    for stage, crop in arrays.items():
        assert np.array_equal(crop[..., 3], arrays["K-view"][..., 3])
        difference = np.abs(crop[..., :3].astype(float)-arrays["K-view"][..., :3])
        counts[stage]["meanRGBByteDifferenceToK"] = float(difference.mean())
        counts[stage]["differentPixelsToK"] = int(np.any(difference != 0, axis=-1).sum())
    bg = (21, 24, 30)
    fg = (237, 241, 247)
    muted = (177, 192, 211)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = lambda n: ImageFont.truetype(font_path, n)
    scale, gap = 6, 24
    width, height = (x1-x0)*scale, (y1-y0)*scale
    board = Image.new("RGB", (3*(width+gap)+gap, height+194), bg)
    draw = ImageDraw.Draw(board)
    draw.text((24, 16), "Neck seam / -30 deg / actual saved RGBA", font=font(26), fill=fg)
    draw.text((24, 55), "All 4,617 pixels in this ROI are opaque in all three variants. The dark wedge is RGB, not transparent background.", font=font(16), fill=muted)
    for i, (stage, crop) in enumerate(arrays.items()):
        x = gap+i*(width+gap)
        draw.text((x, 92), stage, font=font(20), fill=fg)
        board.paste(Image.fromarray(crop[..., :3]).resize((width, height), Image.Resampling.NEAREST), (x,126))
        p = inputs[stage]["probesRGBA"]["wedge_dark"]
        draw.text((x, height+140), f"Pixel (526,225): RGB {p[:3]}, alpha {p[3]}", font=font(15), fill=fg)
    draw.text((24, board.height-23), f"Same ROI {list(ROI)}, enlarged 6x nearest-neighbor. Darkness is a color measurement, not a ground-truth error label.", font=font(15), fill=muted)
    output = WORK/"evidence/neck-seam-classification.png"
    board.save(output)
    proof = {"kind": __doc__, "roiFullImage": ROI, "inputs": inputs, "counts": counts,
        "probeCoordinates": POINTS, "alphaByteIdentical": True,
        "darkComponentDefinition": "Mean of 8-bit sRGB channels <60 and alpha>250; 8-connected components within the fixed ROI. This threshold only locates dark patches, not errors.",
        "output": {"path": str(output), "sha256": sha(output), "bytes": output.stat().st_size},
        "scriptSHA256": sha(__file__), "renderingPerformed": False,
        "limitations": ["No reference image exists for this oblique view; the measurement classifies RGBA, not 3D accuracy.",
            "This ROI contains no transparency; other silhouettes or hair-edge gaps are outside this statement.",
            "Opaque pixels do not by themselves distinguish source image content, projection, candidate validity, or native fallback."]}
    path = WORK/"evidence/neck-seam-classification.json"
    path.write_text(json.dumps(proof, indent=2)+"\n")
    print(json.dumps({"proof": str(path), "proofSHA256": sha(path), "counts": counts, "output": proof["output"]}))


if __name__ == "__main__":
    main()
