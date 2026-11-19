"""Compact registered F RGB / H clay / I clay front evidence; no image enhancement."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from build_box_control_front import BG, TEXT, MUTED, FONT, sha, backdrop, aligned_front_crop
from compare_fh import read


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("comparison_proof", type=Path)
    p.add_argument("f_provenance", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    proof, f = read(a.comparison_proof), read(a.f_provenance)
    assert proof["status"] == "verified existing official captures"
    camera, crop = proof["camera"], proof["headROIs"]["front"]
    frame_record = f["faces"]["+Z"]
    assert sha(frame_record["alignedPath"]) == frame_record["alignedSHA256"]
    assert sha(frame_record["generatedRGBPath"]) == frame_record["generatedRGBSHA256"]
    manifest_path = Path(f["assetDirectory"]) / "manifest.json"
    assert sha(manifest_path) == f["manifestSHA256"]
    manifest, reference = read(manifest_path), read(proof["referenceManifest"])
    assert sha(proof["referenceManifest"]) == proof["referenceManifestSHA256"]
    for key in ("transform", "inverseTransform", "coordinateSystem"):
        assert manifest[key] == reference[key]
    images = [backdrop(aligned_front_crop(frame_record["alignedPath"], f["bakeBox"], crop, camera))]
    records = {}
    for stage in ("H", "I"):
        record = proof["captures"][stage]["lit"]["front"]
        assert record["cameraProjectionValidated"]
        entry = record["image"]
        assert sha(entry["path"]) == entry["sha256"]
        with Image.open(entry["path"]) as image:
            images.append(backdrop(image.crop(crop)))
        records[stage] = record
    scale, pad = 4, 24
    width, height = images[0].size
    board = Image.new("RGB", (3 * width * scale + 4 * pad, height * scale + 184), BG)
    draw = ImageDraw.Draw(board)
    font = lambda size: ImageFont.truetype(FONT, size)
    draw.text((pad, 14), "Registered front RGB / H / I", font=font(25), fill=TEXT)
    draw.text((pad, 51), "Same A camera and fixed head ROI. 4x nearest-neighbor enlargement.", font=font(16), fill=MUTED)
    labels = ("F +Z / registered RGB", "H / full-body Hi3D clay", "I / head-local Hi3D clay")
    for col, (label, image) in enumerate(zip(labels, images)):
        x = pad + col * (width * scale + pad)
        draw.text((x, 87), label, font=font(18), fill=TEXT)
        board.paste(image.resize((width * scale, height * scale), Image.Resampling.NEAREST), (x, 120))
    draw.text((pad, board.height - 50), "I position uses approximate Y-height / bbox-center fit; anatomical landmark alignment is not guaranteed.", font=font(15), fill=MUTED)
    topology = proof["mapping"]["outputBoundary"]
    draw.text((pad, board.height - 26), f"I topology: {topology['boundaryEdges']} boundary edges, {topology['nonmanifoldEdges']} nonmanifold edges. No repair or body merge.", font=font(15), fill=MUTED)
    a.output.mkdir(parents=True, exist_ok=True)
    image_path, metadata_path = a.output / "F-RGB-H-I-front.webp", a.output / "F-RGB-H-I-front-provenance.json"
    assert not image_path.exists() and not metadata_path.exists()
    board.save(image_path, "WEBP", lossless=True)
    assert np.array_equal(np.asarray(Image.open(image_path)), np.asarray(board))
    edge_flags = {s: {m: {v: any(x in (0, 1024) for x in row["alphaBoundsOver0"])
                         for v, row in views.items()} for m, views in modes.items()}
                  for s, modes in proof["captures"].items()}
    metadata = {
        "comparisonProof": str(a.comparison_proof.resolve()), "comparisonProofSHA256": sha(a.comparison_proof),
        "fProvenance": str(a.f_provenance.resolve()), "fProvenanceSHA256": sha(a.f_provenance),
        "referenceRGB": frame_record, "camera": camera, "headROI": crop, "clayCaptures": records,
        "rgbProjection": "Existing aligned_front_crop helper: registered F +Z linear RGBA sampled on the common A orthographic pixel grid, then sRGB encoded.",
        "rgbLimitation": "This display is the registered F +Z reference, not a newly generated face image or I-specific resynthesis.",
        "mapping": proof["mapping"]["mapping"], "outputBoundary": topology,
        "alphaTouchesViewportBoundary": edge_flags,
        "boundaryFlagMeaning": "A true flag means potential raster clipping; it does not by itself prove lost geometry.",
        "processing": "Identical head ROI;4x nearest;neutral backdrop;lossless WebP. Original captures unchanged.",
        "image": {"path": str(image_path.resolve()), "sha256": sha(image_path), "size": list(board.size)},
        "scriptSHA256": sha(__file__),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"image": metadata["image"], "proof": str(metadata_path), "proofSHA256": sha(metadata_path)}))


if __name__ == "__main__":
    main()
