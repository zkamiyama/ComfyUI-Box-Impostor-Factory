"""Compare unchanged Inspector snapshots for the seam-fade experiment."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


BG = (21, 24, 30)
TEXT = (237, 241, 247)
MUTED = (177, 192, 211)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
HEAD = (378, 115, 654, 271)
OLD = Path("/mnt/f/_dev/box-impostor-lab/results/hi3d-depth-cycle-20261119")
OLD_EVIDENCE = Path("/home/dev/Projects/box-impostor-lab/ComfyUI/custom_nodes/box_impostor_factory/docs/verification/hi3d-depth-cycle-20261119")
LABELS = {"K-view": "K / existing view blend", "L1-view-fade": "L1 / visibility fade",
          "L2-view-soft": "L2 / wider fade + softer angular blend"}
ROIS = {"head": HEAD, "neck_m30": (512, 202, 587, 258),
        "hair_p45": (411, 132, 501, 236), "skirt_p45": (462, 425, 548, 559)}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def font(size):
    return ImageFont.truetype(FONT, size)


def backdrop(image):
    result = Image.new("RGBA", image.size, BG + (255,))
    result.alpha_composite(image)
    return result.convert("RGB")


def opaque_mask(images, crop):
    arrays = [np.asarray(image) for image in images]
    mask = np.logical_and.reduce([array[..., 3] > 250 for array in arrays])
    region = np.zeros(mask.shape, dtype=bool)
    x0, y0, x1, y1 = crop
    region[y0:y1, x0:x1] = True
    mask &= region
    return np.asarray(Image.fromarray(mask.astype("uint8") * 255).filter(ImageFilter.MinFilter(7))) > 0


def load(label, paths, angles):
    frames, images, reports = {}, {}, []
    for path in paths:
        cap = json.loads(path.read_text())
        assert not cap["errors"] and "SwiftShader" in cap["graphicsRenderer"]
        assert cap["stageState"] == "Native Box"
        assert sha(cap["html"]) == cap["htmlSHA256"]
        reports.append({"path": str(path), "sha256": sha(path), "html": cap["html"],
                        "htmlSHA256": cap["htmlSHA256"], "display": cap.get("display"),
                        "assetId": cap["assetId"], "camera": cap["camera"]})
        for entry in cap["yawFrames"]:
            angle = entry["requestedYawDegrees"]
            if angle not in angles:
                continue
            if angle in frames:
                assert frames[angle]["sha256"] == entry["sha256"], (label, angle, "duplicate differs")
                continue
            assert sha(entry["path"]) == entry["sha256"]
            im = Image.open(entry["path"]).convert("RGBA")
            assert im.size == (1024, 1024)
            frames[angle], images[angle] = entry, im
    assert set(frames) == set(angles), (label, sorted(frames), angles)
    return {"frames": frames, "images": images, "reports": reports}


def invariants(records, angles):
    base = records["K-view"]
    checks = {}
    for label, record in records.items():
        assert all(report["assetId"] == base["reports"][0]["assetId"] for report in record["reports"])
        for angle in angles:
            actual = record["frames"][angle]["actualCamera"]
            anchor = base["frames"][angle]["actualCamera"]
            differences = {}
            for key in ("matrixWorld", "projectionMatrix"):
                differences[key] = float(np.max(np.abs(np.asarray(actual[key])-anchor[key])))
                assert differences[key] <= 1e-8, (label, angle, key, differences[key])
            equal = np.array_equal(np.asarray(record["images"][angle])[..., 3],
                                   np.asarray(base["images"][angle])[..., 3])
            assert equal, (label, angle, "alpha differs")
            checks[f"{label}/{angle}"] = {"alphaByteExactToK": equal, "maxMatrixDifference": differences,
                                          "actualYawDegrees": record["frames"][angle]["actualYawDegrees"]}
    return checks


def static_boards(records, angles, output):
    labels = list(records)
    cw, ch = (HEAD[2]-HEAD[0])*2, (HEAD[3]-HEAD[1])*2
    board = Image.new("RGB", (len(labels)*(cw+24)+24, len(angles)*(ch+66)+126), BG)
    draw = ImageDraw.Draw(board)
    draw.text((24, 16), "Actual Box Impostor / same geometry, RGB, depth and camera", font=font(25), fill=TEXT)
    draw.text((24, 55), "Only RGB blending changes. Raw 1024px captures; fixed crop; nearest-neighbor enlargement.", font=font(17), fill=MUTED)
    for row, angle in enumerate(angles):
        for col, label in enumerate(labels):
            x, y = 24+col*(cw+24), 92+row*(ch+66)
            draw.text((x, y), LABELS[label], font=font(19), fill=TEXT)
            draw.text((x, y+26), f"Yaw {records[label]['frames'][angle]['actualYawDegrees']:+.6f} deg", font=font(15), fill=MUTED)
            im = backdrop(records[label]["images"][angle].crop(HEAD)).resize((cw, ch), Image.Resampling.NEAREST)
            board.paste(im, (x, y+48))
    draw.text((24, board.height-26), f"Head crop {list(HEAD)}. No AI edits, sharpening, optical flow or generated frames.", font=font(16), fill=MUTED)
    faces = output/"seam-fade-faces.png"
    board.save(faces)
    body = Image.new("RGB", (len(labels)*360+24, len(angles)*394+104), BG)
    bd = ImageDraw.Draw(body)
    bd.text((24, 16), "Same actual captures / whole model", font=font(25), fill=TEXT)
    for row, angle in enumerate(angles):
        for col, label in enumerate(labels):
            x, y = 24+col*360, 62+row*394
            bd.text((x, y), f"{label} / {angle:+g} deg", font=font(18), fill=TEXT)
            im = backdrop(records[label]["images"][angle]).resize((336, 336), Image.Resampling.NEAREST)
            body.paste(im, (x, y+30))
    bd.text((24, body.height-24), "Common native alpha and camera. 1024px captures shown at 336px for layout.", font=font(15), fill=MUTED)
    body_path = output/"seam-fade-body.png"
    body.save(body_path)
    return [faces, body_path]


def differences(records, angles):
    base = records["K-view"]
    result = {}
    for angle in angles:
        mask = opaque_mask([record["images"][angle] for record in records.values()], HEAD)
        assert mask.any()
        source = np.asarray(base["images"][angle])[..., :3].astype(float)
        for label, record in records.items():
            delta = np.abs(np.asarray(record["images"][angle])[..., :3].astype(float)-source)[mask]
            result[f"{label}/{angle}"] = {"region": "common opaque head, eroded 3px", "pixels": int(mask.sum()),
                "meanRGBByteDifferenceToK": float(delta.mean()), "maxRGBByteDifferenceToK": float(delta.max()),
                "differentPixelsToK": int(np.count_nonzero(np.any(delta != 0, axis=1)))}
    return result


def front_fidelity(records, output):
    old = json.loads((OLD_EVIDENCE/"F-J-K-front-fidelity.json").read_text())
    residual = json.loads((OLD_EVIDENCE/"K-front-residual-classification.json").read_text())
    crop = old["headCrop"]
    ref_entry = residual["outputs"]["K-front-reference-rgba.png"]
    inner_entry = residual["outputs"]["K-front-source-interior-mask.png"]
    assert sha(ref_entry["path"]) == ref_entry["sha256"]
    assert sha(inner_entry["path"]) == inner_entry["sha256"]
    assert sha(old["mask"]["path"]) == old["mask"]["sha256"]
    reference_image = Image.open(ref_entry["path"]).convert("RGBA")
    reference = np.asarray(reference_image)
    mask = np.asarray(Image.open(old["mask"]["path"]).convert("L")) > 0
    inner = np.asarray(Image.open(inner_entry["path"]).convert("L")) > 0
    assert int(mask.sum()) == 10936 and int(inner.sum()) == 10953
    face = np.zeros(mask.shape, bool)
    face_rect = residual["faceRectangleFullImage"]
    face[face_rect[1]-crop[1]:face_rect[3]-crop[1], face_rect[0]-crop[0]:face_rect[2]-crop[0]] = True
    face &= inner
    results, displays = {}, {"RGB reference": reference_image}
    for name, record in records.items():
        image = record["images"][0].crop(crop)
        array = np.asarray(image)
        difference = np.abs(array[..., :3].astype(float)-reference[..., :3])
        mae = difference.mean(axis=-1)
        item = {"sRGBMAE255FixedPriorMask": float(difference[mask].mean()), "sampledPixels": int(mask.sum()),
                "sourceInterior": {}}
        for region_name, region in (("head", inner), ("faceRectangle", face)):
            opaque = region & (array[..., 3] > 250)
            item["sourceInterior"][region_name] = {"pixels": int(region.sum()),
                "transparentAlpha0to5": int((region & (array[..., 3] <= 5)).sum()),
                "partialAlpha6to250": int((region & (array[..., 3] > 5) & (array[..., 3] <= 250)).sum()),
                "opaqueMeanRGBErrorOver16": int((opaque & (mae > 16)).sum())}
        results[name] = item
        displays[name] = image
    assert abs(results["K-view"]["sRGBMAE255FixedPriorMask"]-old["metrics"]["K-view"]["sRGBMAE255"]) < 1e-10
    width, height = reference_image.size
    scale = 3
    cell = width*scale+32
    board = Image.new("RGB", (len(displays)*cell+16, height*scale+184), BG)
    draw = ImageDraw.Draw(board)
    draw.text((20, 16), "Front preservation / exact previous reference and masks", font=font(25), fill=TEXT)
    draw.text((20, 55), "Fixed 10,936-pixel fidelity mask. Separate 10,953-pixel source-only interior for residual counts.", font=font(16), fill=MUTED)
    for i, (name, image) in enumerate(displays.items()):
        x = 16+i*cell
        draw.text((x, 88), name, font=font(20), fill=TEXT)
        board.paste(backdrop(image).resize((width*scale, height*scale), Image.Resampling.NEAREST), (x,120))
        if name in results:
            draw.text((x, height*scale+134), f"MAE {results[name]['sRGBMAE255FixedPriorMask']:.4f} /255", font=font(18), fill=TEXT)
    path = output/"seam-fade-front-preservation.png"
    board.save(path)
    return path, {"referenceImage": ref_entry, "fidelityMask": old["mask"], "residualMask": inner_entry,
                  "headCrop": crop, "faceRectangle": face_rect, "metrics": results,
                  "interpretation": "Same registered +Z image on the same camera grid and unchanged historical masks. Image fidelity and residual classification, not 3D ground-truth accuracy."}


def sweep(records, angles, output):
    labels = list(records)
    metrics = {}
    for first, second in zip(angles, angles[1:]):
        mask = opaque_mask([record["images"][angle] for record in records.values() for angle in (first, second)], HEAD)
        assert mask.any()
        for label, record in records.items():
            step = abs(record["frames"][second]["actualYawDegrees"]-record["frames"][first]["actualYawDegrees"])
            a = np.asarray(record["images"][first])[..., :3].astype(float)
            b = np.asarray(record["images"][second])[..., :3].astype(float)
            delta = np.abs(b-a)[mask].mean(axis=1)/step
            metrics[f"{label}/{first}:{second}"] = {"pixels": int(mask.sum()), "actualStepDegrees": step,
                "meanRGBByteChangePerDegree": float(delta.mean()), "p95PixelMeanChangePerDegree": float(np.percentile(delta, 95))}
    sequence = angles + angles[-2:0:-1]
    frames = []
    for angle in sequence:
        frame = Image.new("RGB", (len(labels)*576+24, 460), BG)
        draw = ImageDraw.Draw(frame)
        draw.text((24, 16), f"Actual boundary sweep / {angle:+g} deg", font=font(24), fill=TEXT)
        for col, label in enumerate(labels):
            x = 24+576*col
            draw.text((x, 56), LABELS[label], font=font(19), fill=TEXT)
            frame.paste(backdrop(records[label]["images"][angle].crop(HEAD)).resize((552,312), Image.Resampling.NEAREST), (x,88))
        draw.text((24, 424), "Same geometry and camera. Actual rendered views; 10 fps playback, no temporal interpolation.", font=font(15), fill=MUTED)
        frames.append(frame)
    gif = output/"seam-fade-boundary-loop.gif"
    frames[0].save(gif, save_all=True, append_images=frames[1:], duration=100, loop=0, disposal=2, optimize=False)
    decoded = Image.open(gif)
    assert decoded.n_frames == len(sequence)
    return [gif], {"screenSpaceTemporalChange": metrics, "sequenceDegrees": sequence, "actualViews": len(angles),
        "playbackFrames": len(sequence), "playbackFPS": 10,
        "interpretation": "RGB byte change per measured degree on common opaque head interiors. Includes legitimate projection motion and texture changes; lower values alone do not prove better geometry or visual quality. Ghosted/blurred images can score lower.",
        "gif": "GIF preview uses palette quantization. All statistics use original full-color PNG captures."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--stage", choices=("initial", "sweep"), default="initial")
    parser.add_argument("--variants", default="L1-view-fade,L2-view-soft")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    variants = args.variants.split(",")
    assert all(label in LABELS and label != "K-view" for label in variants)
    angles = [0, -30, 45] if args.stage == "initial" else list(range(40,51))
    paths = {"K-view": [OLD/"K-view-front/capture.json", OLD/"K-view-yaw/capture.json"]}
    if args.stage == "sweep":
        paths["K-view"].append(args.work/"K-view-sweep/capture.json")
    for label in variants:
        paths[label] = [args.work/f"{label}-first/capture.json"]
        if args.stage == "sweep":
            paths[label].append(args.work/f"{label}-sweep/capture.json")
    records = {label: load(label, source, angles) for label, source in paths.items()}
    checks = invariants(records, angles)
    if args.stage == "initial":
        outputs = static_boards(records, angles, args.output)
        front_path, front_metrics = front_fidelity(records, args.output)
        outputs.append(front_path)
        extra = {"frontFidelity": front_metrics}
    else:
        outputs, extra = sweep(records, angles, args.output)
    result = {"kind": __doc__, "stage": args.stage, "scriptSHA256": sha(__file__), "anglesDegrees": angles,
        "assetId": records["K-view"]["reports"][0]["assetId"], "fixedHeadCrop": HEAD, "inspectionROIs": ROIS,
        "invariantChecks": checks, "differencesToK": differences(records, angles),
        "reports": {label: record["reports"] for label, record in records.items()},
        "frames": {label: record["frames"] for label, record in records.items()},
        "outputs": [{"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size,
                     "size": list(Image.open(path).size)} for path in outputs], **extra}
    proof = args.output/f"seam-fade-{args.stage}-provenance.json"
    proof.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"proof": str(proof), "proofSHA256": sha(proof), "outputs": result["outputs"],
                      "differencesToK": result["differencesToK"]}))


if __name__ == "__main__":
    main()
