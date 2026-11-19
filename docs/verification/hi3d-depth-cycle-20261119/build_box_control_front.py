"""Compare registered B front RGB with actual B and B normal-only Inspector renders.
No AI processing or image enhancement. Reuses identical orthographic pixel sampling.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import map_coordinates

BG = (21, 24, 30)
TEXT = (233, 237, 243)
MUTED = (169, 180, 195)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def backdrop(image):
    image = image.convert("RGBA")
    result = Image.new("RGBA", image.size, BG + (255,))
    result.alpha_composite(image)
    return result.convert("RGB")

def aligned_front_crop(path, bounds, crop, camera):
    rgba = np.load(path, allow_pickle=False)
    height, width = rgba.shape[:2]
    x0, y0, x1, y1 = crop
    xx, yy = np.meshgrid(np.arange(x0, x1) + .5, np.arange(y0, y1) + .5)
    scale = 2 * camera["halfExtent"] / 1024
    world_x = camera["target"][0] + (xx - 512) * scale
    world_y = camera["target"][1] - (yy - 512) * scale
    sx = (world_x - bounds["min"][0]) / (bounds["max"][0] - bounds["min"][0]) * width - .5
    sy = (bounds["max"][1] - world_y) / (bounds["max"][1] - bounds["min"][1]) * height - .5
    premult = rgba.copy()
    premult[..., :3] *= premult[..., 3:4]
    sample = np.stack([map_coordinates(premult[..., c], [sy, sx], order=1,
                                       mode="constant", cval=0) for c in range(4)], axis=-1)
    rgb = np.clip(sample[..., :3] / np.maximum(sample[..., 3:4], 1e-6), 0, 1)
    srgb = np.where(rgb <= .0031308, rgb * 12.92, 1.055 * rgb ** (1 / 2.4) - .055)
    result = np.concatenate([srgb, np.clip(sample[..., 3:4], 0, 1)], axis=-1)
    return Image.fromarray(np.uint8(np.clip(result, 0, 1) * 255 + .5), "RGBA")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--include-c", action="store_true")
    args = parser.parse_args()
    stages = ["B", "B-normal-only"] + (["C"] if args.include_c else [])
    proof_path = args.output / ("B-B-normal-only-C-stages-comparison-provenance.json" if args.include_c else "B-normal-only-comparison-provenance.json")
    proof = json.loads(proof_path.read_text())
    crop = proof["headCrops"]["front"]
    camera = proof["camera"]
    metadata = {s: json.loads((args.work / f"{s}-provenance.json").read_text()) for s in stages}
    captures = {s: json.loads((args.work / f"{s}-box/capture.json").read_text()) for s in stages}
    for s in stages:
        assert not captures[s]["errors"]
        assert captures[s]["htmlSHA256"] == metadata[s]["box"]["viewerSHA256"]
        assert proof["stages"][s]["assetId"] == metadata[s]["assetId"]
        for key in ("projection", "target", "halfExtent", "distance", "pixels", "cssPixels", "deviceScaleFactor"):
            assert captures[s]["camera"][key] == camera[key], (s, key)
    front = metadata["B"]["faces"]["+Z"]
    assert sha(front["alignedPath"]) == front["alignedSHA256"]
    rgb = aligned_front_crop(front["alignedPath"], metadata["B"]["bakeBox"], crop, camera)
    images = [backdrop(rgb)]
    sources = {}
    for s in stages:
        path = args.work / f"{s}-box/lit-front.png"
        im = Image.open(path)
        assert im.size == tuple(camera["pixels"]) == (1024, 1024)
        images.append(backdrop(im.crop(crop)))
        sources[s] = {"path": str(path), "sha256": sha(path), "assetId": metadata[s]["assetId"],
                      "htmlSHA256": captures[s]["htmlSHA256"], "capture": str(args.work / f"{s}-box/capture.json")}
    width, height = rgb.size
    enlarged = [im.resize((width * 3, height * 3), Image.Resampling.NEAREST) for im in images]
    column_width = max(380, width * 3 + 32)
    panel_height = height * 3 + 184
    prior_path = args.output / "B-stage.webp"
    prior_proof_path = args.output / "B-evidence.json"
    prior = json.loads(prior_proof_path.read_text())
    assert sha(prior_path) == prior["panels"][prior_path.name]["sha256"]
    px = (column_width - width * 3) // 2
    prior_rgb = Image.open(prior_path).convert("RGB").crop((px, 125, px + width * 3, 125 + height * 3))
    assert np.array_equal(np.asarray(prior_rgb), np.asarray(enlarged[0])), "Registered RGB differs from B stage evidence"
    panel = Image.new("RGB", (column_width * len(images), panel_height), BG)
    draw = ImageDraw.Draw(panel)
    def font(size): return ImageFont.truetype(FONT, size)
    draw.text((24, 18), "B / normal-only / C: same registered RGB, actual Box" if args.include_c else "B / fixed RGB and geometry; vertex-normal control", font=font(25), fill=TEXT)
    draw.text((24, 55), "Actual official Inspector Box renders | same A-based camera | no image enhancement", font=font(16), fill=MUTED)
    labels = ["H3 +Z / registered RGB", "B / actual Box", "B / vertex-normal control"] + (["C / RGB-depth-refined geometry"] if args.include_c else [])
    for i, (im, label) in enumerate(zip(enlarged, labels)):
        draw.text((i * column_width + 16, 91), label, font=font(16), fill=TEXT)
        panel.paste(im, (i * column_width + (column_width - im.width) // 2, 125))
    draw.text((24, panel_height - 47), "Head crops: 3x nearest-neighbor. RGB is sampled on the same camera pixel grid; no sharpening.", font=font(14), fill=MUTED)
    draw.text((24, panel_height - 24), "Normal-only: same geometry. C: depth-refined geometry. Both retain B registered RGB; coverage may change in C." if args.include_c else "Control changes vertex-normal data only; the normal map affects lighting and cross-view weights.", font=font(14), fill=MUTED)
    target = args.output / ("C-front-stages.webp" if args.include_c else "B-normal-only-front-stages.webp")
    panel.save(target, "WEBP", lossless=True)
    assert np.array_equal(np.asarray(Image.open(target)), np.asarray(panel))
    external = {}
    for name in ("B-normal-only-rebake-evaluation.json", "B-normal-only-refinement.json", "B-normal-only-normal-alpha-proof.json"):
        p = args.output / name
        assert p.is_file()
        external[name] = {"path": str(p), "sha256": sha(p)}
    result = {
        "kind": "Registered B front RGB versus actual B, normal-only control and C geometry" if args.include_c else "Registered B front RGB versus actual B and same-geometry normal-only Box",
        "renderer": "Unchanged official 3D-Inspector native Box PBR, CPU SwiftShader",
        "crop": crop, "camera": camera, "sourceImages": sources,
        "registeredRGB": {"path": front["alignedPath"], "sha256": front["alignedSHA256"],
                          "bakeBox": metadata["B"]["bakeBox"],
                          "sampling": "Linear-light premultiplied RGBA bilinear sample at the common orthographic pixel centers; sRGB conversion",
                          "priorBStagePath": str(prior_path), "priorBStageSHA256": sha(prior_path),
                          "pixelsEqualPriorBStage": True, "priorProofSHA256": sha(prior_proof_path)},
        "controlProvenance": {"path": str(proof_path), "sha256": sha(proof_path)},
        "nativeControlProofs": external,
        "normalMapMeaning": "Normal RGB controls both native PBR lighting and incidence-dependent cross-view weighting; this is not a pure lighting-only control.",
        "imageProcessing": {"cropOnlyRenders": True, "resize": "3x nearest-neighbor", "backdropRGB": list(BG),
                            "sharpening": False, "aiPostprocessing": False, "losslessWebPDecodedPixelsEqual": True},
        "output": {"path": str(target), "sha256": sha(target), "bytes": target.stat().st_size, "size": list(panel.size)},
        "script": {"path": str(Path(__file__).resolve()), "sha256": sha(__file__)},
    }
    output_proof = args.output / ("C-front-stages-provenance.json" if args.include_c else "B-normal-only-front-stages-provenance.json")
    output_proof.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["output"]))
    print(json.dumps({"provenance": str(output_proof), "sha256": sha(output_proof)}))

if __name__ == "__main__":
    main()
