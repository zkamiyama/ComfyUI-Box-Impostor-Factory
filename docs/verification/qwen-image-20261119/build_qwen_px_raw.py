"""Show raw +X guide/H3/Qwen images at fixed coordinates, without alignment or retouching."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


RUN = Path("/mnt/f/_dev/box-impostor-lab/experiments/geometry-abc-20261119")
F_ASSET = RUN/"output/box_impostor/4e8090f307b4958de8a94877ad391b3b0e355fa208d7ded8340a229e773b1920"
CROP = (339, 0, 558, 213)
BG, FG, MUTED = (21, 24, 30), (237, 241, 247), (177, 192, 211)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text())
    assert sha(args.candidate) == receipt["rawSha256"], "Candidate differs from completed generation receipt"
    prior_receipt = RUN/"B/px/result.json"
    previous = json.loads(prior_receipt.read_text())
    paths = {"Geometry guide / +X": RUN/"input/B/px/guide.png", "Previous H3 / +X": RUN/"B/px/generated.png",
             "Qwen Image 2.1 / +X": args.candidate}
    assert sha(paths["Previous H3 / +X"]) == previous["rawSha256"]
    images = {name: Image.open(path).convert("RGB") for name, path in paths.items()}
    assert all(image.size == (1024, 1024) for image in images.values()), "Fixed comparison requires 1024 square source canvases"
    original = np.load(F_ASSET/"views/px/source.npy", allow_pickle=False)
    original_rgb = np.rint(np.clip(original[..., :3], 0, 1)*255).astype("uint8")
    original_matches = np.array_equal(original_rgb, np.asarray(images["Previous H3 / +X"]))
    assert original_matches, "Previous raw H3 image is not the current F +X source"
    args.output.mkdir(parents=True, exist_ok=True)
    font = lambda size: ImageFont.truetype(FONT, size)
    width, height = (CROP[2]-CROP[0])*2, (CROP[3]-CROP[1])*2
    cell = width+24
    board = Image.new("RGB", (3*cell+24, height+154), BG)
    draw = ImageDraw.Draw(board)
    draw.text((24, 16), "Same +X input canvas / actual generated images", font=font(25), fill=FG)
    draw.text((24, 55), "Fixed head crop. Inspect hidden-face visibility and profile direction against the guide.", font=font(17), fill=MUTED)
    for i, (name, image) in enumerate(images.items()):
        x = 24+i*cell
        draw.text((x, 90), name, font=font(20), fill=FG)
        board.paste(image.crop(CROP).resize((width, height), Image.Resampling.NEAREST), (x,122))
    draw.text((24, board.height-24), f"Crop {list(CROP)} from 1024px images; 2x nearest-neighbor. No registration, sharpening or AI post-edit.", font=font(15), fill=MUTED)
    head = args.output/"Q1-px-raw-head-comparison.png"
    board.save(head)
    body = Image.new("RGB", (1104, 438), BG)
    draw = ImageDraw.Draw(body)
    draw.text((24, 16), "Full raw +X canvases / same input guide", font=font(25), fill=FG)
    for i, (name, image) in enumerate(images.items()):
        x = 24+i*360
        draw.text((x, 62), name, font=font(18), fill=FG)
        body.paste(image.resize((336,336), Image.Resampling.NEAREST), (x,90))
    full = args.output/"Q1-px-raw-body-comparison.png"
    body.save(full)
    proof = {"kind": __doc__, "fixedHeadCrop": CROP, "sourceDimensions": [1024,1024],
        "originalFAsset": str(F_ASSET), "previousH3RGBEqualsFSource": original_matches,
        "inputs": {name: {"path": str(path), "sha256": sha(path)} for name, path in paths.items()},
        "candidateReceipt": {"path": str(args.receipt), "sha256": sha(args.receipt), "promptId": receipt.get("prompt_id")},
        "previousReceipt": {"path": str(prior_receipt), "sha256": sha(prior_receipt)},
        "outputs": [{"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size,
                     "size": list(Image.open(path).size)} for path in (head, full)],
        "processing": "Raw RGB PNG pixels only. Fixed source coordinates, nearest-neighbor display. No alignment, pose correction or pixel retouching.",
        "scriptSHA256": sha(__file__),
        "limitations": "This comparison shows generated source content before compositing. It does not prove cross-view 3D consistency or shape accuracy."}
    result = args.output/"Q1-px-raw-provenance.json"
    result.write_text(json.dumps(proof, indent=2)+"\n")
    print(json.dumps({"proof": str(result), "proofSHA256": sha(result), "outputs": proof["outputs"]}))


if __name__ == "__main__":
    main()
