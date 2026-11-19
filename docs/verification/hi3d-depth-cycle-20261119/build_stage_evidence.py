"""Build small, labeled proof panels from actual Inspector PNG snapshots."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import map_coordinates
import trimesh

BG = (21, 24, 30)
TEXT = (233, 237, 243)
MUTED = (169, 180, 195)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
LABELS = {
    "A": "A / coarse mesh + fresh H3 RGB",
    "A-partial-pz": "A partial / new H3 +Z only; five other faces still baseline",
    "B": "B / Hi3D mesh + fresh H3 RGB",
    "C": "C / RGB-derived depth fused into mesh",
}
def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def font(size):
    return ImageFont.truetype(FONT, size)

def backdrop(image):
    image = image.convert("RGBA")
    background = Image.new("RGBA", image.size, BG + (255,))
    background.alpha_composite(image)
    return background.convert("RGB")

def reference_head(manifest_path):
    manifest = json.loads(manifest_path.read_text())
    scene = trimesh.load(manifest_path.parent / manifest["proxy"], force="scene", process=False)
    points = []
    for node in scene.graph.nodes_geometry:
        transform, name = scene.graph[node]
        points.append(trimesh.transform_points(scene.geometry[name].vertices, transform))
    points = np.concatenate(points)
    top = manifest["tightBounds"]["max"][1]
    height = top - manifest["tightBounds"]["min"][1]
    upper = points[points[:, 1] >= top - .08 * height]
    xmin, xmax = upper[:, 0].min() - .01, upper[:, 0].max() + .01
    selected = points[(points[:, 1] >= top - .18 * height) & (points[:, 0] >= xmin) & (points[:, 0] <= xmax)]
    bounds = np.array([selected.min(0) - [0, .01, .01], selected.max(0) + [0, .01, .01]])
    bounds[:, 0] = [xmin, xmax]
    return bounds

def crop_for_view(head, camera, direction):
    direction = np.asarray(direction, dtype=float)
    direction /= np.linalg.norm(direction)
    right = np.cross([0., 1., 0.], direction)
    right /= np.linalg.norm(right)
    up = np.cross(direction, right)
    corners = np.array([[head[x, 0], head[y, 1], head[z, 2]] for x in (0, 1) for y in (0, 1) for z in (0, 1)])
    centered = corners - np.asarray(camera["target"])
    scale = 1024 / (2 * camera["halfExtent"])
    pixels = np.column_stack((512 + centered @ right * scale, 512 - centered @ up * scale))
    low = np.floor(pixels.min(0)).astype(int)
    high = np.ceil(pixels.max(0)).astype(int)
    return [int(low[0]), int(low[1]), int(high[0]), int(high[1])]

def aligned_front_crop(path, bounds, crop, camera):
    rgba = np.load(path, allow_pickle=False)
    height, width = rgba.shape[:2]
    x0, y0, x1, y1 = crop
    xx, yy = np.meshgrid(np.arange(x0, x1) + .5, np.arange(y0, y1) + .5)
    scale = 2 * camera["halfExtent"] / 1024
    world_x = camera["target"][0] + (xx - 512) * scale
    world_y = camera["target"][1] - (yy - 512) * scale
    source_x = (world_x - bounds["min"][0]) / (bounds["max"][0] - bounds["min"][0]) * width - .5
    source_y = (bounds["max"][1] - world_y) / (bounds["max"][1] - bounds["min"][1]) * height - .5
    premult = rgba.copy()
    premult[..., :3] *= premult[..., 3:4]
    sample = np.stack([map_coordinates(premult[..., c], [source_y, source_x], order=1, mode="constant", cval=0) for c in range(4)], axis=-1)
    rgb = np.clip(sample[..., :3] / np.maximum(sample[..., 3:4], 1e-6), 0, 1)
    srgb = np.where(rgb <= .0031308, rgb * 12.92, 1.055 * rgb ** (1 / 2.4) - .055)
    result = np.concatenate([srgb, np.clip(sample[..., 3:4], 0, 1)], axis=-1)
    return Image.fromarray(np.uint8(np.clip(result, 0, 1) * 255 + .5), "RGBA")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("work", type=Path)
parser.add_argument("stage", choices=["A", "B", "C", "A-partial-pz"])
parser.add_argument("reference_manifest", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--front-only", action="store_true")
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
stage = args.stage
metadata_path = args.work / f"{stage}-provenance.json"
metadata = json.loads(metadata_path.read_text())
box_capture = json.loads((args.work / f"{stage}-box" / "capture.json").read_text())
clay_capture = json.loads((args.work / f"{stage}-clay" / "capture.json").read_text())
assert box_capture["camera"]["target"] == clay_capture["camera"]["target"]
assert box_capture["camera"]["halfExtent"] == clay_capture["camera"]["halfExtent"]
camera = box_capture["camera"]
head = reference_head(args.reference_manifest)
front_crop = crop_for_view(head, camera, [0, 0, 1])
evidence = {
    "stage": stage, "label": LABELS[stage], "assetId": metadata["assetId"],
    "metadata": metadata, "boxCapture": box_capture, "clayCapture": clay_capture,
    "headRegion": {"definition": "A mesh upper 18 percent of height, restricted to the X envelope of its upper 8 percent; 0.01 bake-unit padding",
                   "bounds": head.tolist(), "frontCropPixels": front_crop},
    "panels": {}, "sourceImages": {},
}
clay_front = Image.open(args.work / f"{stage}-clay" / "lit-front.png").convert("RGBA")
actual_bbox = clay_front.getchannel("A").point(lambda a: 255 if a >= 128 else 0).getbbox()
source_bounds = clay_capture["sourceBounds"]
pixel_scale = 1024 / (2 * camera["halfExtent"])
expected_bbox = [
    512 + (source_bounds["min"][0] - camera["target"][0]) * pixel_scale,
    512 - (source_bounds["max"][1] - camera["target"][1]) * pixel_scale,
    512 + (source_bounds["max"][0] - camera["target"][0]) * pixel_scale,
    512 - (source_bounds["min"][1] - camera["target"][1]) * pixel_scale,
]
pixel_error = max(abs(a - b) for a, b in zip(actual_bbox, expected_bbox))
evidence["cameraCalibration"] = {
    "method": "Opaque clay front alpha bounds vs projected source mesh bounds",
    "expectedPixelBounds": expected_bbox, "observedPixelBounds": actual_bbox,
    "maxBoundaryErrorPixels": pixel_error, "tolerancePixels": 2.0,
}
assert pixel_error <= 2.0, "UI camera did not reproduce the reference projection"
front = metadata["faces"]["+Z"]
rgb = aligned_front_crop(front["alignedPath"], metadata["bakeBox"], front_crop, camera) if front.get("alignedPath") else None
box_image = Image.open(args.work / f"{stage}-box" / "lit-front.png").crop(front_crop)
clay_image = Image.open(args.work / f"{stage}-clay" / "lit-front.png").crop(front_crop)
images = [rgb, box_image, clay_image]
labels = ["H3 +Z / registered RGB", "Actual partial Box" if stage == "A-partial-pz" else "Actual Box / official Inspector", "Same mesh / uniform gray"]
column_width = max(380, (front_crop[2] - front_crop[0]) * 3 + 32)
panel_height = (front_crop[3] - front_crop[1]) * 3 + 184
panel = Image.new("RGB", (column_width * 3, panel_height), BG)
draw = ImageDraw.Draw(panel)
draw.text((24, 18), LABELS[stage], font=font(25), fill=TEXT)
subtitle = ("Intermediate diagnostic, not completed A. " if stage == "A-partial-pz" else "") + f"RGB accepted: {metadata['acceptedCount']}/6 | same A-based camera"
draw.text((24, 55), subtitle, font=font(16), fill=MUTED)
for i, (source, label) in enumerate(zip(images, labels)):
    draw.text((i * column_width + 16, 91), label, font=font(16), fill=TEXT)
    if source is None:
        draw.text((i * column_width + 16, 145), "RGB rejected / fallback used", font=font(17), fill=(255, 139, 139))
        continue
    enlarged = backdrop(source).resize((source.width * 3, source.height * 3), Image.Resampling.NEAREST)
    panel.paste(enlarged, (i * column_width + (column_width - enlarged.width) // 2, 125))
draw.text((24, panel_height - 47), "Head crops: 3x nearest-neighbor; no sharpening. Registered RGB uses the same camera pixel grid.", font=font(14), fill=MUTED)
draw.text((24, panel_height - 24), "Clay: original geometry normals retained; color and tangent normal textures removed.", font=font(14), fill=MUTED)
path = args.output / f"{stage}-stage.webp"
panel.save(path, "WEBP", lossless=True)
evidence["panels"][path.name] = {"sha256": sha(path), "size": panel.size, "resize": "nearest-neighbor 3x", "rgbPlaneSampling": "linear-light premultiplied RGBA bilinear sample at the same orthographic pixel centers"}

if args.front_only:
    for kind in ("box", "clay"):
        source_path = args.work / f"{stage}-{kind}" / "lit-front.png"
        evidence["sourceImages"][str(source_path)] = sha(source_path)
    (args.output / f"{stage}-evidence.json").write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"stage": stage, "crop": front_crop, "cameraCalibration": evidence["cameraCalibration"],
                      "accepted": metadata["acceptedCount"], "panels": list(evidence["panels"])}))
    raise SystemExit(0)

views = ("front", "iso", "left", "right")
view_panel = Image.new("RGB", (1280, 750), BG)
draw = ImageDraw.Draw(view_panel)
draw.text((20, 14), LABELS[stage] + " / actual 3D views", font=font(24), fill=TEXT)
for row, kind in enumerate(("box", "clay")):
    for col, view in enumerate(views):
        image_path = args.work / f"{stage}-{kind}" / f"lit-{view}.png"
        image = Image.open(image_path)
        assert image.size == (1024, 1024)
        thumb = backdrop(image).resize((320, 320), Image.Resampling.LANCZOS)
        view_panel.paste(thumb, (col * 320, 80 + row * 335))
        draw.text((col * 320 + 14, 59 + row * 335), f"{kind.capitalize()} / {view}", font=font(16), fill=TEXT)
        evidence["sourceImages"][str(image_path)] = sha(image_path)
path = args.output / f"{stage}-views.webp"
view_panel.save(path, "WEBP", lossless=True)
evidence["panels"][path.name] = {"sha256": sha(path), "size": view_panel.size, "resize": "Lanczos 1024 to 320"}

contact = Image.new("RGB", (1260, 260), BG)
draw = ImageDraw.Draw(contact)
for i, (name, item) in enumerate(metadata["faces"].items()):
    draw.text((i * 210 + 10, 8), name + " / " + item["status"], font=font(15), fill=TEXT if item["status"] == "accepted" else (255, 139, 139))
    if item.get("generatedRGBPath"):
        image = backdrop(Image.open(item["generatedRGBPath"]))
        image.thumbnail((200, 200), Image.Resampling.LANCZOS)
        contact.paste(image, (i * 210 + (210 - image.width) // 2, 35 + (200 - image.height) // 2))
        evidence["sourceImages"][item["generatedRGBPath"]] = item["generatedRGBSHA256"]
path = args.output / f"{stage}-six-rgb.webp"
contact.save(path, "WEBP", lossless=True)
evidence["panels"][path.name] = {"sha256": sha(path), "size": contact.size, "resize": "Lanczos fit in 200-square cells"}
(args.output / f"{stage}-evidence.json").write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
print(json.dumps({"stage": stage, "crop": front_crop, "headBounds": head.tolist(), "accepted": metadata["acceptedCount"],
                  "panels": list(evidence["panels"])}))