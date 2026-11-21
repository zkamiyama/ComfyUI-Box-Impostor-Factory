#!/usr/bin/env python3
"""Make an immutable six-face guide variant using an existing albedo PNG as RGB.

Copies exact image bytes. Baseline, geometry, cameras, alpha and depth do not
change. Requires only the Python standard library; no inference is performed.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil


SLUGS = ["pz", "px", "nx", "nz", "py", "ny"]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path):
    return {"path": str(path.resolve()), "sha256": sha(path), "bytes": path.stat().st_size}


def hashes(root):
    return {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob("*")) if p.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guides", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-key", choices=["albedo"], default="albedo")
    args = parser.parse_args()
    source_root, output = args.guides.resolve(strict=True), args.output.resolve()
    if output.is_relative_to(source_root):
        raise ValueError("Output must be outside the immutable source guides")
    input_hashes = hashes(source_root)
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for slug in SLUGS:
        source = source_root/slug
        source_json = source/"guide.json"
        original = json.loads(source_json.read_text())
        if original.get("kind") != "native-mesh-square-guide-v1" or original["slug"] != slug:
            raise ValueError("Unexpected source guide")
        baseline = original["baselineManifest"]
        if sha(baseline["path"]) != baseline["sha256"]:
            raise ValueError("The immutable baseline changed")
        for item in original["files"].values():
            if sha(item["path"]) != item["sha256"]:
                raise ValueError("A source guide image changed")
        selected = original["files"][args.source_key]
        destination = output/slug
        shutil.copytree(source, destination)
        rgb_path = destination/"rgb.png"
        shutil.copyfile(selected["path"], rgb_path)
        guide = copy.deepcopy(original)
        guide["files"] = {key: record(destination/Path(value["path"]).name)
                          for key, value in original["files"].items()}
        guide["files"]["rgb"] = record(rgb_path)
        if guide["files"]["rgb"]["sha256"] != selected["sha256"]:
            raise AssertionError("Selected RGB bytes differ")
        if any(guide["files"][key]["sha256"] != value["sha256"]
               for key, value in original["files"].items() if key != "rgb"):
            raise AssertionError("An alpha, depth or other non-RGB guide file changed")
        guide["sourceGuide"] = record(source_json)
        guide["rgbSelection"] = {"sourceKey": args.source_key, "sourceImage": selected,
                                  "sourceGuide": record(source_json),
                                  "previousGuideSource": original.get("sourceGuide"),
                                  "sourceShading": original.get("shading"),
                                  "method": "exact byte copy; no resampling, recoloring or relighting",
                                  "helperSha256": sha(__file__)}
        guide["shading"] = {"applied": False, "kind": "no added illumination",
                            "albedoSource": "existing albedo.png from coarse baseColor transfer",
                            "backgroundSrgbByte": original["shading"]["backgroundSrgbByte"],
                            "scope": "Qwen RGB guide only; baseline color atlas unchanged"}
        allowed = {"files", "sourceGuide", "rgbSelection", "shading"}
        if {k:v for k,v in guide.items() if k not in allowed} != {k:v for k,v in original.items() if k not in allowed}:
            raise AssertionError("A baseline/camera/structure/coordinate field changed")
        (destination/"guide.json").write_text(json.dumps(guide, indent=2, allow_nan=False)+"\n")
        result = {"face": guide["face"], "guide": record(destination/"guide.json"),
                  "rgb": guide["files"]["rgb"], "baselineManifest": baseline,
                  "allNonRgbFileHashesUnchanged": True, "cameraAndStructureUnchanged": True}
        results.append(result)
        print(json.dumps({"stage": "guide-ready", **result}), flush=True)
    if hashes(source_root) != input_hashes:
        raise RuntimeError("An original guide changed during selection")
    report = {"completed": True, "method": "immutable-guide-rgb-selection-v1", "sourceKey": args.source_key,
              "sourceGuides": str(source_root), "sourceFiles": input_hashes, "helper": record(Path(__file__)),
              "guides": results, "outputFiles": hashes(output)}
    with (output/"result.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"completed": True, "result": str(output/"result.json")}), flush=True)


if __name__ == "__main__":
    main()
