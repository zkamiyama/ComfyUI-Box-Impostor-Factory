"""F/H visual proof: common world-frame viewer camera, preserved source RGB, actual geometry.
H has different native bake cameras and camera-affine RGB resampling. Neither
aligned RGB bytes nor the final color atlas are required to equal F.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import trimesh

from build_box_control_front import BG, TEXT, MUTED, FONT, sha, backdrop, aligned_front_crop

VIEWS = {"front": [0, 0, 1], "left": [-1, 0, 0], "back": [0, 0, -1], "right": [1, 0, 0]}
CAMERA_KEYS = ("projection", "target", "halfExtent", "distance", "pixels", "cssPixels", "deviceScaleFactor")


def read(path):
    return json.loads(Path(path).read_text())


def basis(vector):
    direction = np.asarray(vector, dtype=float)
    direction /= np.linalg.norm(direction)
    right = np.cross([0., 1., 0.], direction)
    right /= np.linalg.norm(right)
    return right, np.cross(direction, right)


def project(xyz, vector, camera):
    right, up = basis(vector)
    centered = xyz - np.asarray(camera["target"])
    return np.column_stack((512 + centered @ right * 512 / camera["halfExtent"],
                            512 - centered @ up * 512 / camera["halfExtent"]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("work", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--front-only", action="store_true")
    a = p.parse_args()
    requested_views = {"front": VIEWS["front"]} if a.front_only else VIEWS
    meta = {s: read(a.work/f"{s}-provenance.json") for s in ("B", "F", "H")}
    manifests = {s: read(Path(meta[s]["assetDirectory"])/"manifest.json") for s in ("F", "H")}
    assert meta["F"]["proxySHA256"] == meta["B"]["proxySHA256"]
    assert meta["F"]["clay"]["geometry"] == meta["B"]["clay"]["geometry"]
    assert meta["F"]["clay"]["sha256"] == meta["B"]["clay"]["sha256"]
    assert manifests["F"]["transform"] == manifests["H"]["transform"], "Canonical transforms differ"
    source_proof = {}
    for face in manifests["F"]["faceOrder"]:
        entries = {s: meta[s]["faces"][face] for s in ("F", "H")}
        pngs = {s: Path(entries[s]["generatedRGBPath"]) for s in ("F", "H")}
        npys = {s: pngs[s].parent/"source.npy" for s in ("F", "H")}
        png_hash = {s: sha(pngs[s]) for s in ("F", "H")}
        npy_hash = {s: sha(npys[s]) for s in ("F", "H")}
        assert png_hash["F"] == png_hash["H"], (face, "source PNG changed")
        assert npy_hash["F"] == npy_hash["H"], (face, "source NPY changed")
        assert all(png_hash[s] == entries[s]["generatedRGBSHA256"] for s in ("F", "H"))
        source_proof[face] = {
            "sourcePNG": {s: {"path": str(pngs[s]), "sha256": png_hash[s]} for s in ("F", "H")},
            "sourceNPY": {s: {"path": str(npys[s]), "sha256": npy_hash[s]} for s in ("F", "H")},
            "sourcePNGAndNPYByteExact": True,
            "alignedSHA256": {s: entries[s].get("alignedSHA256") for s in ("F", "H")},
            "alignedBytesEqual": entries["F"].get("alignedSHA256") == entries["H"].get("alignedSHA256"),
            "nativeCamera": {s: {key: manifests[s]["faces"][face].get(key)
                                   for key in ("viewMatrix", "projectionMatrix")} for s in ("F", "H")},
        }
    base_capture = read(a.work/"F-box/capture.json")
    camera = base_capture["camera"]
    crop_file = a.output/"A-full-comparison-provenance.json"
    head = np.asarray(read(crop_file)["headWorldBounds"])
    corners = np.array([[head[x, 0], head[y, 1], head[z, 2]]
                        for x in (0, 1) for y in (0, 1) for z in (0, 1)])
    crops = {}
    for name, vector in VIEWS.items():
        xy = project(corners, vector, camera)
        crops[name] = [*np.floor(xy.min(0)).astype(int).tolist(), *np.ceil(xy.max(0)).astype(int).tolist()]
    frames, records = {}, {}
    cache, checked_html = {}, set()
    for kind in (("box",) if a.front_only else ("box", "clay")):
        frames[kind], records[kind] = {}, {}
        for stage in ("F", "H"):
            frames[kind][stage], records[kind][stage] = {}, {}
            asset_meta = meta["B"] if kind == "clay" and stage == "F" else meta[stage]
            geometry = None
            if kind == "clay":
                clay = asset_meta["clay"]
                assert sha(clay["path"]) == clay["sha256"]
                assert clay["normalPreservation"]["binaryChunksIdentical"]
                assert clay["normalPreservation"]["positionNormalIndexBindingsIdentical"]
                assert not clay["normalPreservation"]["normalRecomputed"]
                scene = trimesh.load(clay["path"], force="scene", process=False)
                geometry = np.concatenate([trimesh.transform_points(scene.geometry[scene.graph[node][1]].vertices,
                                                                       scene.graph[node][0])
                                           for node in scene.graph.nodes_geometry])
                triangles = sum(len(scene.geometry[scene.graph[node][1]].faces) for node in scene.graph.nodes_geometry)
                assert triangles == clay["sourceTriangles"] == clay["payloadTriangles"]
            for view, vector in requested_views.items():
                if stage == "H":
                    folder = f"H-{kind}"
                elif kind == "box":
                    folder = "F-box-back" if view == "back" else "F-box"
                else:
                    folder = "B-clay-back" if view == "back" else "B-clay"
                capture_path = a.work/folder/"capture.json"
                if str(capture_path) not in cache:
                    cache[str(capture_path)] = read(capture_path)
                cap = cache[str(capture_path)]
                assert not cap["errors"] and "SwiftShader" in cap["graphicsRenderer"]
                assert cap["backend"] == "Three.js / WebGL2"
                assert cap["stageState"] == ("Native Box" if kind == "box" else "Three.js PBR")
                assert cap["htmlSHA256"] == asset_meta[kind]["viewerSHA256"]
                if cap["html"] not in checked_html:
                    assert sha(cap["html"]) == cap["htmlSHA256"]
                    checked_html.add(cap["html"])
                for key in CAMERA_KEYS:
                    assert cap["camera"][key] == camera[key], (stage, kind, view, key)
                if kind == "box":
                    assert cap["assetId"] == meta[stage]["assetId"]
                else:
                    assert cap["triangleCounts"]["meshPayload"] == cap["triangleCounts"]["expected"] == triangles
                entry = cap["modes"]["lit"][view]
                assert sha(entry["path"]) == entry["sha256"]
                assert np.allclose(entry["direction"], vector, atol=1e-12)
                frame = Image.open(entry["path"]).convert("RGBA")
                assert frame.size == (1024, 1024)
                frames[kind][stage][view] = frame
                rec = {"capturePath": str(capture_path), "captureSHA256": sha(capture_path),
                       "image": entry, "html": cap["html"], "htmlSHA256": cap["htmlSHA256"],
                       "camera": cap["camera"], "lighting": cap["lighting"],
                       "reusedExistingFrame": stage == "F" and view != "back"}
                if kind == "clay":
                    projected = project(geometry, vector, camera)
                    expected = [*projected.min(0).tolist(), *projected.max(0).tolist()]
                    yy, xx = np.where(np.asarray(frame)[..., 3] > 127)
                    actual = [int(xx.min()), int(yy.min()), int(xx.max())+1, int(yy.max())+1]
                    error = float(np.abs(np.asarray(expected)-actual).max())
                    assert error < 3, (stage, view, error)
                    rec.update(sourceTriangles=triangles, payloadTriangles=cap["triangleCounts"]["meshPayload"],
                               projectedBounds=expected, alphaBounds=actual, projectionMaxErrorPx=error)
                records[kind][stage][view] = rec
    assert meta["F"]["clay"]["material"] == meta["H"]["clay"]["material"]
    font = lambda size: ImageFont.truetype(FONT, size)
    panels = {}
    def save(board, name):
        path = a.output/name
        board.save(path, "WEBP", lossless=True)
        assert np.array_equal(np.asarray(Image.open(path)), np.asarray(board))
        panels[name] = {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size, "size": list(board.size)}
    labels = {"F": "F / previous Hi3D reference", "H": "H / AI front used as Hi3D reference"}
    cw = max((crop[2]-crop[0])*2 for crop in crops.values())+28
    ch = max((crop[3]-crop[1])*2 for crop in crops.values())+48
    for kind in (() if a.front_only else ("box", "clay")):
        board = Image.new("RGB", (4*cw+40, 2*ch+130), BG)
        draw = ImageDraw.Draw(board)
        draw.text((20, 15), "F / H actual " + ("Box" if kind == "box" else "full-mesh clay") + " / common world-frame camera", font=font(24), fill=TEXT)
        subtitle = ("Same six source RGB files; H has new bake cameras and camera-affine resampling."
                    if kind == "box" else "Same camera, gray material and light. Full source triangles; source normal policy retained.")
        draw.text((20, 54), subtitle, font=font(16), fill=MUTED)
        for col, view in enumerate(VIEWS):
            draw.text((20+col*cw, 85), view.upper(), font=font(16), fill=MUTED)
        for row, stage in enumerate(("F", "H")):
            y = 114+row*ch
            title = labels[stage]
            if kind == "clay":
                title += f" / {records[kind][stage]['front']['sourceTriangles']:,} triangles"
            draw.text((20, y), title, font=font(17), fill=TEXT)
            for col, view in enumerate(VIEWS):
                im = backdrop(frames[kind][stage][view].crop(crops[view]))
                im = im.resize((im.width*2, im.height*2), Image.Resampling.NEAREST)
                board.paste(im, (20+col*cw+(cw-im.width)//2, y+30))
        save(board, f"F-H-{kind}-faces.webp")
        body = Image.new("RGB", (1000, 1230), BG)
        draw = ImageDraw.Draw(body)
        draw.text((20, 15), f"F / H actual {kind} / front and back", font=font(25), fill=TEXT)
        draw.text((20, 53), "Common viewer camera. Whole-frame preview: 480px Lanczos; no enhancement.", font=font(16), fill=MUTED)
        for row, stage in enumerate(("F", "H")):
            y = 92+row*570
            draw.text((20, y), labels[stage], font=font(18), fill=TEXT)
            for col, view in enumerate(("front", "back")):
                im = backdrop(frames[kind][stage][view]).resize((480, 480), Image.Resampling.LANCZOS)
                body.paste(im, (20+col*490, y+49))
                draw.text((20+col*490, y+28), view.upper(), font=font(15), fill=MUTED)
        save(body, f"F-H-{kind}-body.webp")
    front = meta["F"]["faces"]["+Z"]
    assert sha(front["alignedPath"]) == front["alignedSHA256"]
    rgb = backdrop(aligned_front_crop(front["alignedPath"], meta["F"]["bakeBox"], crops["front"], camera))
    images = [rgb, backdrop(frames["box"]["F"]["front"].crop(crops["front"])),
              backdrop(frames["box"]["H"]["front"].crop(crops["front"]))]
    enlarged = [im.resize((im.width*3, im.height*3), Image.Resampling.NEAREST) for im in images]
    board = Image.new("RGB", (1140, enlarged[0].height+184), BG)
    draw = ImageDraw.Draw(board)
    draw.text((24, 18), "Same front color source / F Box / H Box", font=font(25), fill=TEXT)
    draw.text((24, 55), "Actual official Inspector; same viewer camera and light, with different geometry.", font=font(16), fill=MUTED)
    for i, (im, label) in enumerate(zip(enlarged, ("F +Z / registered RGB", "F / previous Hi3D reference", "H / AI front Hi3D reference"))):
        draw.text((i*380+16, 91), label, font=font(16), fill=TEXT)
        board.paste(im, (i*380+(380-im.width)//2, 125))
    draw.text((24, board.height-47), "3x nearest-neighbor. Source RGB is shown on the common camera pixel grid; no sharpening.", font=font(14), fill=MUTED)
    draw.text((24, board.height-24), "H preserves the six source RGB files; new native cameras require one additional camera-affine resampling.", font=font(14), fill=MUTED)
    save(board, "H-front-stages.webp")
    run = Path(meta["H"]["assetDirectory"]).parents[2]
    artifacts = [a.work/"H/result.json", a.work/"H/plan.json", run/"H/composed.json"]
    proof = {
        "kind": __doc__, "scope": "front only" if a.front_only else "four-view Box and clay", "camera": camera, "headWorldBounds": head.tolist(), "headCrops": crops,
        "headCropSource": {"path": str(crop_file), "sha256": sha(crop_file)},
        "assets": {s: {"assetId": meta[s]["assetId"], "manifestSHA256": meta[s]["manifestSHA256"],
                       "proxySHA256": meta[s]["proxySHA256"], "bakeBox": meta[s]["bakeBox"],
                       "clayNormalPolicy": meta[s]["clay"]["normalPreservation"]} for s in ("F", "H")},
        "sameCanonicalTransform": True, "sourceRGB": source_proof,
        "rgbMapping": "F source PNG and source NPY bytes retained. H uses a known camera-affine bilinear resample for its actual native cameras; aligned bytes and color atlas may differ.",
        "FClayReuse": {"sameProxySHA256AsB": True, "sameClayGLBSHA256": meta["B"]["clay"]["sha256"],
                      "normalPolicy": "Original NORMAL absence retained; official mesh converter computes normals."},
        "captures": records, "panels": panels,
        "pipelineArtifacts": {str(path): sha(path) for path in artifacts if path.exists()},
        "processing": "Fixed integer head ROI. 2x nearest for face boards, 3x nearest for front-stage board. Whole frames 480px Lanczos. No sharpening or AI postprocessing.",
        "limitation": "Mesh and native Box are distinct official rendering paths. Camera and lighting comparisons apply within each path, not across them.",
        "scriptSHA256": sha(__file__),
    }
    out = a.output/("H-front-provenance.json" if a.front_only else "F-H-comparison-provenance.json")
    out.write_text(json.dumps(proof, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"panels": panels, "proof": str(out), "proofSHA256": sha(out),
                      "triangles": {s: meta[s]["clay"]["sourceTriangles"] for s in ("F", "H")}}))


if __name__ == "__main__":
    main()
