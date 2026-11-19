"""E diagnostic: remove differences between the six RGB views, retaining the lit Box renderer.

Run self-check before authorization. create and capture are separate commands and
must only be run after the experiment's explicit GO; neither runs by default.
"""
import argparse
import copy
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from box_nodes import store
from box_nodes.archive import pack_asset

VERSION = "uniform-color-rgb-v1"
SOURCE_B = "ce53eb5429973a519e264d5752fd90348697797cd50f0903a4993076398ba1a2"
GRAY = .18
COLOR_PATH = "diagnostic/uniform-color.f16.bin"
PROOF_PATH = "diagnostic/uniform-color-proof.json"
BG = (21, 24, 30)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def uniform_rgb(rgba):
    if rgba.dtype != np.dtype("<f2") or rgba.shape[-1] != 4:
        raise ValueError("Expected little-endian FLOAT16 RGBA.")
    alpha = rgba[..., 3].astype("f8")
    if not np.isfinite(alpha).all() or np.any(alpha < 0) or np.any(alpha > 1):
        raise ValueError("Coverage alpha must be finite and in [0,1].")
    result = rgba.copy()
    premultiplied = (GRAY * alpha).astype("<f2")
    result[..., :3] = premultiplied[..., None]
    assert np.array_equal(result.view("<u2")[..., 3], rgba.view("<u2")[..., 3])
    return result


def self_check(report_path=None):
    alpha = np.array([-0., 0., np.nextafter(np.float16(0), np.float16(1)),
                      .001, .0625, .125, .5, 1.], dtype="<f2").reshape(2, 4)
    source = np.arange(32, dtype="f4").reshape(2, 4, 4).astype("<f2") / np.float16(32)
    source[..., 3] = alpha
    before = source.tobytes()
    result = uniform_rgb(source)
    expected = np.asarray([np.float16(float(a) * GRAY) for a in alpha.flat], dtype="<f2").reshape(alpha.shape)
    assert source.tobytes() == before
    assert result[..., 3].tobytes() == source[..., 3].tobytes()
    assert np.array_equal(result[..., 0].view("<u2"), expected.view("<u2"))
    assert np.array_equal(result[..., 0], result[..., 1]) and np.array_equal(result[..., 1], result[..., 2])
    assert result[1, 2, 0] == np.float16(.09)
    assert result[1, 3, 0] == np.float16(.18)
    assert result[0, 0, 3].view("<u2") == 0x8000
    valid = alpha >= .001
    straight_error = np.abs(result[..., 0][valid].astype("f8") / alpha[valid].astype("f8") - GRAY)
    assert straight_error.max() < .0002
    report = {
        "kind": "Synthetic-array premultiplication check only; no real asset or render",
        "version": VERSION, "grayLinear": GRAY, "passed": True,
        "alphaInput": alpha.astype("f8").tolist(),
        "expectedPremultipliedRGB": expected.astype("f8").tolist(),
        "alphaBitsPreserved": True, "negativeZeroAlphaPreserved": True,
        "sourceArrayUnchanged": True, "grayIsLinearNoSRGBConversion": True,
        "arithmetic": "float64 multiplication, one rounding to FLOAT16 RGB; alpha copied without casting",
        "maxStraightColorErrorAlphaAtLeast001": float(straight_error.max()),
        "scriptSHA256": sha(__file__),
    }
    if report_path:
        write_json(report_path, report)
    print(json.dumps(report))


def inventory(directory):
    return {str(path.relative_to(directory)): sha(path)
            for path in sorted(Path(directory).rglob("*")) if path.is_file()}


def create_asset(source, asset_root, external_report):
    source = source.resolve()
    asset_root = asset_root.resolve()
    manifest = store.validate(source, SOURCE_B)
    assert manifest["asset_id"] == SOURCE_B
    assert asset_root != source and not asset_root.is_relative_to(source)
    before = inventory(source)
    color = manifest["textures"]["colorMap"]
    assert color["dtype"] == "float16" and color["channels"] == 4
    assert color["colorSpace"] == "linear-srgb" and color["alpha"] == "premultiplied-coverage"
    rgba = np.fromfile(store.contained(source, color["path"]), dtype="<f2").reshape(color["height"], color["width"], 4)
    result = uniform_rgb(rgba)
    result_bytes = result.tobytes()
    result_sha = hashlib.sha256(result_bytes).hexdigest()
    identity = store.key(VERSION, SOURCE_B, before["manifest.json"], GRAY, result_sha)
    destination = asset_root / identity
    new_manifest = copy.deepcopy(manifest)
    new_color = {**color, "path": COLOR_PATH, "sha256": result_sha}
    new_manifest["textures"]["colorMap"] = new_color
    new_manifest.update(asset_id=identity, parentAsset=SOURCE_B, mode="diagnostic_uniform_color")
    new_manifest["diagnostic"] = {
        "method": VERSION, "sourceAsset": SOURCE_B, "grayLinear": GRAY,
        "operation": "All six colorMap RGB channels = grayLinear * original alpha; one FLOAT16 rounding",
        "alpha": "Original FLOAT16 bit patterns retained",
        "meaning": "Six-view RGB differences removed; geometry, normals and active lighting retained",
        "proof": PROOF_PATH,
    }
    assert COLOR_PATH not in before and PROOF_PATH not in before
    asset_root.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        assert store.validate(destination, identity) == new_manifest
        proof = json.loads((destination / PROOF_PATH).read_text())
        assert proof["sourceFilesUnchanged"] and inventory(source) == before
    else:
        with tempfile.TemporaryDirectory(prefix=".uniform-color-", dir=asset_root) as temp:
            staging = Path(temp) / "asset"
            shutil.copytree(source, staging)
            color_path = store.contained(staging, COLOR_PATH)
            color_path.parent.mkdir(parents=True, exist_ok=True)
            color_path.write_bytes(result_bytes)
            write_json(staging / "manifest.json", new_manifest)
            validated = store.validate(staging, identity)
            assert validated == new_manifest
            after_copy = inventory(staging)
            unchanged = {name: after_copy[name] == digest for name, digest in before.items() if name != "manifest.json"}
            assert all(unchanged.values())
            non_color = {name: new_manifest["textures"][name] == item
                         for name, item in manifest["textures"].items() if name != "colorMap"}
            assert all(non_color.values())
            same_frame = {name: new_manifest[name] == manifest[name] for name in
                          ("layout", "resolution", "faceOrder", "faces", "bakeBox", "tightBounds",
                           "transform", "inverseTransform", "proxy", "proxySha256", "maps", "baseColor")}
            assert all(same_frame.values())
            out = np.fromfile(color_path, dtype="<f2").reshape(rgba.shape)
            old_alpha_bits = rgba.view("<u2")[..., 3]
            new_alpha_bits = out.view("<u2")[..., 3]
            assert old_alpha_bits.tobytes() == new_alpha_bits.tobytes()
            assert np.array_equal(out.view("<u2")[..., :3], result.view("<u2")[..., :3])
            source_after = inventory(source)
            assert source_after == before
            valid = rgba[..., 3].astype("f8") >= .001
            straight = out[..., 0][valid].astype("f8") / rgba[..., 3][valid].astype("f8")
            proof = {
                "kind": "E / six-view RGB difference control, actual lit Box data",
                "method": VERSION, "sourceAsset": SOURCE_B, "assetId": identity,
                "sourceDirectory": str(source), "assetDirectory": str(destination), "grayLinear": GRAY,
                "sourceManifestSHA256": before["manifest.json"], "manifestSHA256": sha(staging / "manifest.json"),
                "sourceColorDescriptor": color, "diagnosticColorDescriptor": new_color,
                "colorAlphaBitExact": True,
                "colorAlphaSHA256": hashlib.sha256(old_alpha_bits.tobytes()).hexdigest(),
                "premultiplication": "RGB16 = float16(0.18 * float64(original alpha16)); alpha16 bit patterns unchanged",
                "sourceColorFileRetained": True, "allOtherOriginalFilesByteExact": unchanged,
                "sourceFileHashesBefore": before, "sourceFilesUnchanged": True,
                "nonColorTextureDescriptorsExact": non_color, "geometryCameraFrameMetadataExact": same_frame,
                "allRGBChannelsEqual": bool(np.array_equal(out[..., 0], out[..., 1]) and np.array_equal(out[..., 1], out[..., 2])),
                "maxStraightRGBErrorAlphaAtLeast001": float(np.max(np.abs(straight - GRAY))),
                "lowAlphaTexels": int(np.count_nonzero(rgba[..., 3] < .001)),
                "lowAlphaNote": "Native shader divides by max(alpha,.001); its original edge and weight behavior remains active.",
                "semanticLimit": "This removes six-view RGB differences. It is not an unlit render or a direct evaluation of mesh accuracy.",
                "scriptSHA256": sha(__file__),
            }
            write_json(staging / PROOF_PATH, proof)
            staging.rename(destination)
    write_json(external_report, proof)
    print(json.dumps({"assetId": identity, "directory": str(destination),
                      "proof": str(external_report), "proofSHA256": sha(external_report),
                      "colorMapSHA256": result_sha, "sourceUnchanged": True}))


def capture_asset(asset, work, output, inspector):
    manifest = store.validate(asset)
    assert manifest["diagnostic"]["method"] == VERSION and manifest["diagnostic"]["sourceAsset"] == SOURCE_B
    proof = json.loads((asset / PROOF_PATH).read_text())
    assert proof["colorAlphaBitExact"] and all(proof["allOtherOriginalFilesByteExact"].values())
    base = json.loads((work / "B-provenance.json").read_text())
    base_capture = json.loads((work / "B-box/capture.json").read_text())
    assert base["assetId"] == SOURCE_B and not base_capture["errors"]
    assert manifest["proxySha256"] == base["proxySHA256"]
    reference = Path(base_capture["referenceManifest"])
    assert sha(reference) == base_capture["referenceManifestSHA256"]
    work.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    archive = work / "E.box.zip"
    pack_asset(asset, archive)
    viewer = work / "E.inspector.html"
    subprocess.run([sys.executable, str(inspector), str(archive), "-o", str(viewer)], check=True)
    html_sha = sha(viewer)
    metadata = copy.deepcopy(base)
    metadata.update(stage="E", assetId=manifest["asset_id"], assetDirectory=str(asset.resolve()),
                    manifestSHA256=sha(asset/"manifest.json"), proxyPath=str((asset/manifest["proxy"]).resolve()),
                    diagnostic=manifest["diagnostic"], diagnosticProof=proof,
                    inspectorLauncher=str(inspector), inspectorLauncherSHA256=sha(inspector))
    metadata["box"] = {"archive":str(archive),"archiveSHA256":sha(archive),"viewer":str(viewer),
                       "viewerSHA256":html_sha,"viewerBytes":viewer.stat().st_size,
                       "colorMapSHA256":manifest["textures"]["colorMap"]["sha256"]}
    metadata["clayReuse"] = "B geometry is byte-identical; no E clay generated or rendered"
    write_json(work/"E-provenance.json",metadata)
    subprocess.run(["node",str(output/"capture_inspector.mjs"),str(viewer),str(work/"E-box"),
                    str(reference),"lit","front,left"],check=True)
    assert sha(viewer) == html_sha
    capture = json.loads((work/"E-box/capture.json").read_text())
    assert not capture["errors"] and "SwiftShader" in capture["graphicsRenderer"]
    assert capture["assetId"] == manifest["asset_id"]
    assert capture["stageState"] == base_capture["stageState"]
    assert capture["backend"] == base_capture["backend"] == "Three.js / WebGL2"
    for key in ("projection","target","halfExtent","distance","pixels","cssPixels","deviceScaleFactor"):
        assert capture["camera"][key] == base_capture["camera"][key], key
    crop_source = output/"A-full-comparison-provenance.json"
    crops = json.loads(crop_source.read_text())["headCrops"]
    views = ("front","left")
    frames = {}
    frame_proofs = {}
    for view in views:
        pair = {}
        for stage,cap in (("B",base_capture),("E",capture)):
            entry = cap["modes"]["lit"][view]
            assert sha(entry["path"]) == entry["sha256"]
            pair[stage] = Image.open(entry["path"]).convert("RGBA")
            assert pair[stage].size == (1024,1024)
            frames[stage,view] = pair[stage]
        assert np.array_equal(np.asarray(pair["B"])[...,3],np.asarray(pair["E"])[...,3])
        frame_proofs[view] = {"B":base_capture["modes"]["lit"][view],"E":capture["modes"]["lit"][view],
                             "renderedAlphaByteExact":True}
    font = lambda size: ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",size)
    cw=max((crops[v][2]-crops[v][0])*3 for v in views)+32
    rh=max((crops[v][3]-crops[v][1])*3 for v in views)+55
    board=Image.new("RGB",(cw*2+40,rh*2+130),BG)
    draw=ImageDraw.Draw(board)
    draw.text((20,16),"B / E: six-view RGB difference control",font=font(25),fill=(235,239,245))
    draw.text((20,55),"Actual lit Box; same geometry, alpha, normals, depth, camera and light.",font=font(16),fill=(175,188,205))
    for col,view in enumerate(views):draw.text((20+col*cw,88),view.upper(),font=font(16),fill=(175,188,205))
    for row,stage in enumerate(("B","E")):
        y=115+row*rh
        label="B / original six RGB views" if stage=="B" else "E / uniform linear RGB 0.18; original alpha"
        draw.text((20,y),label,font=font(18),fill=(235,239,245))
        for col,view in enumerate(views):
            im=frames[stage,view].crop(crops[view])
            opaque=Image.new("RGBA",im.size,BG+(255,));opaque.alpha_composite(im)
            im=opaque.convert("RGB").resize((im.width*3,im.height*3),Image.Resampling.NEAREST)
            board.paste(im,(20+col*cw+(cw-im.width)//2,y+32))
    panel=output/"B-E-uniform-color-faces.webp"
    board.save(panel,"WEBP",lossless=True)
    assert np.array_equal(np.asarray(Image.open(panel)),np.asarray(board))
    result={"kind":"B/E lit native Box; six-view RGB differences removed",
            "sourceAsset":SOURCE_B,"diagnosticAsset":manifest["asset_id"],
            "diagnosticProofPath":str(asset/PROOF_PATH),"diagnosticProofSHA256":sha(asset/PROOF_PATH),
            "camera":capture["camera"],"frameProofs":frame_proofs,
            "headCrops":{v:crops[v] for v in views},"cropSourceSHA256":sha(crop_source),
            "processing":"Identical integer crops; 3x nearest-neighbor; no sharpening or AI processing",
            "lighting":"Active official native PBR; unchanged. This is not unlit.",
            "html":metadata["box"],"htmlUnchangedAfterCapture":True,
            "output":{"path":str(panel),"sha256":sha(panel),"bytes":panel.stat().st_size,"size":list(board.size)},
            "scriptSHA256":sha(__file__)}
    write_json(output/"B-E-uniform-color-comparison-provenance.json",result)
    print(json.dumps(result["output"]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    check=sub.add_parser("self-check")
    check.add_argument("--report",type=Path)
    create=sub.add_parser("create")
    create.add_argument("--source",type=Path,required=True)
    create.add_argument("--asset-root",type=Path,required=True)
    create.add_argument("--report",type=Path,required=True)
    capture=sub.add_parser("capture")
    capture.add_argument("--asset",type=Path,required=True)
    capture.add_argument("--work",type=Path,required=True)
    capture.add_argument("--output",type=Path,required=True)
    capture.add_argument("--inspector",type=Path,default=Path("/home/dev/Projects/3D-Inspector-skills/skills/3d-inspector/view.py"))
    a=p.parse_args()
    if a.command=="self-check":self_check(a.report)
    elif a.command=="create":create_asset(a.source,a.asset_root,a.report)
    else:capture_asset(a.asset,a.work,a.output,a.inspector)
if __name__=="__main__":main()
