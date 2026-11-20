#!/usr/bin/env python3
"""Run a frozen two-reference AnyAngle view set through a local ComfyUI server."""
import argparse
import copy
import hashlib
import json
import re
import shutil
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from PIL import Image, ImageOps


AUTHOR_PROMPT = "Change the camera angle from <image2> to <image1>."


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def request(server, endpoint, value=None):
    data = None if value is None else json.dumps(value).encode()
    req = urllib.request.Request(server.rstrip("/") + endpoint, data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=90) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"ComfyUI HTTP {error.code}: "
                           + error.read().decode(errors="replace")[:4000]) from error


def prepare_image(source, destination, size):
    """Fit the complete supplied image; no subject or face crop."""
    with Image.open(source) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        image = ImageOps.pad(image, (size, size), method=Image.Resampling.LANCZOS,
                             color=(128, 128, 128), centering=(0.5, 0.5))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        with Image.open(destination) as existing:
            if existing.size != image.size or existing.convert("RGB").tobytes() != image.tobytes():
                raise ValueError("Frozen input changed: " + str(destination))
    else:
        image.save(destination)


def prepare(args):
    guide_manifest = read(args.guides)
    views = guide_manifest["views"]
    if args.views:
        selected = args.views.split(",")
        by_id = {view["id"]: view for view in views}
        views = [by_id[name] for name in selected]
    template = read(args.template)
    required = {"1": "LoadImage", "2": "LoadImage", "10": "TextEncodeQwenImage21",
                "12": "KSampler", "14": "SaveImage", "18": "SaveImage",
                "19": "LoraLoaderModelOnly", "9": "QwenImage21Cache"}
    for node, kind in required.items():
        if template[node]["class_type"] != kind:
            raise ValueError(f"Template node {node} must be {kind}")
    input_dir = args.comfy_input / args.input_prefix
    source_input = input_dir / "original.png"
    prepare_image(args.original, source_input, args.size)
    jobs = []
    for index, view in enumerate(views):
        name = view["id"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError("View ID must be a simple directory name: " + name)
        guide_path = Path(view["image"])
        if not guide_path.is_absolute():
            guide_path = args.guides.parent / guide_path
        guide_input = input_dir / name / "guide.png"
        prepare_image(guide_path, guide_input, args.size)
        graph = copy.deepcopy(template)
        # Author's released connection diagram: original=image_1, coarse target=image_2.
        graph["1"]["inputs"]["image"] = str(source_input.relative_to(args.comfy_input))
        graph["2"]["inputs"]["image"] = str(guide_input.relative_to(args.comfy_input))
        enc = graph["10"]["inputs"]
        for key in list(enc):
            if key.startswith("images."):
                del enc[key]
        enc.update({"prompt": args.prompt, "negative_prompt": "", "resolution": args.size,
                    "images.image_1": ["1", 0], "images.image_2": ["2", 0]})
        graph.pop("3", None)
        graph["12"]["inputs"].update(steps=args.steps, cfg=3.0, seed=args.seed + index,
                                      sampler_name="euler", scheduler="simple", denoise=1.0)
        graph["19"]["inputs"]["strength_model"] = 1.0
        graph["9"]["inputs"].update(device="cpu", dtype=args.cache_dtype)
        graph["14"]["inputs"]["filename_prefix"] = f"{args.input_prefix}/{name}/raw"
        graph["18"]["inputs"]["filename_prefix"] = f"{args.input_prefix}/{name}/mask"
        work = args.output / "views" / name
        job = {"id": name, "seed": args.seed + index, "guide": str(guide_path.resolve()),
               "guide_sha256": digest(guide_path), "prepared_guide_sha256": digest(guide_input),
               "camera_conditioning": view.get("camera"), "camera_is_calibration": False,
               "graph": graph}
        frozen = work / "prepared.json"
        if frozen.exists() and read(frozen) != job:
            raise ValueError("Existing preparation differs: " + str(frozen))
        write(frozen, job)
        write(work / "prompt.json", graph)
        jobs.append(job)
    record = {"original": str(args.original.resolve()), "original_sha256": digest(args.original),
              "prepared_original_sha256": digest(source_input),
              "template_sha256": digest(args.template), "guides_sha256": digest(args.guides),
              "size": args.size, "steps": args.steps, "cfg": 3.0, "lora_strength": 1.0,
              "seed_base": args.seed, "cache_dtype": args.cache_dtype, "prompt": args.prompt,
              "reference_order": ["original", "coarse_target"],
              "view_ids": [job["id"] for job in jobs], "whole_image_fit": "pad_no_crop",
              "view_consistency": "independent_generation_from_one_fixed_original"}
    frozen = args.output / "prepared.json"
    if frozen.exists() and read(frozen) != record:
        raise ValueError("Use a new output directory for changed experiment settings")
    write(frozen, record)
    return jobs, record


def run_job(args, job):
    work = args.output / "views" / job["id"]
    result_path = work / "result.json"
    if result_path.exists():
        result = read(result_path)
        for key in ("image", "mask"):
            if digest(result[key]) != result[key + "_sha256"]:
                raise ValueError("Completed output changed: " + result[key])
        return result
    submission_path = work / "submitted.json"
    if submission_path.exists():
        submitted = read(submission_path)
        if submitted["prompt_sha256"] != digest(work / "prompt.json"):
            raise ValueError("Submitted prompt hash changed")
        prompt_id = submitted["prompt_id"]
        history = request(args.server, "/history/" + prompt_id)
        if prompt_id not in history:
            queue = request(args.server, "/queue")
            queued = [item[1] for key in ("queue_running", "queue_pending") for item in queue[key]]
            if prompt_id not in queued:
                raise RuntimeError("Prior submission is absent from queue and history; "
                                   "inspect before creating a new run: " + prompt_id)
    else:
        queue = request(args.server, "/queue")
        if queue["queue_running"] or queue["queue_pending"]:
            raise RuntimeError("ComfyUI has another queued job; keep GPU stages sequential")
        response = request(args.server, "/prompt", {"prompt": job["graph"],
                           "client_id": str(uuid.uuid4()), "extra_data": {"experiment": args.input_prefix}})
        if response.get("node_errors"):
            raise RuntimeError(str(response))
        prompt_id = response["prompt_id"]
        submitted = {"prompt_id": prompt_id, "submitted_epoch": time.time(),
                     "prompt_sha256": digest(work / "prompt.json"), "response": response}
        write(submission_path, submitted)
    print(json.dumps({"stage": "generating", "view": job["id"], "prompt_id": prompt_id}), flush=True)
    last_status = 0
    while True:
        history = request(args.server, "/history/" + prompt_id)
        if prompt_id in history:
            entry = history[prompt_id]
            break
        elapsed = time.time() - submitted["submitted_epoch"]
        if elapsed > args.timeout:
            raise TimeoutError("Prompt is still running; resume the same output later: " + prompt_id)
        if elapsed - last_status >= 30:
            print(json.dumps({"view": job["id"], "elapsed_s": round(elapsed)}), flush=True)
            last_status = elapsed
        time.sleep(3)
    write(work / "history.json", entry)
    status = entry.get("status", {})
    if not status.get("completed") or status.get("status_str") != "success":
        raise RuntimeError("Generation failed: " + str(status))
    if entry["prompt"][2] != job["graph"]:
        raise ValueError("History graph differs from submitted graph")
    cached = [str(node) for event, values in status["messages"] if event == "execution_cached"
              for node in values.get("nodes", [])]
    if "12" in cached:
        raise ValueError("Sampler was cached; generated-view evidence would not be fresh")
    result = {"id": job["id"], "seed": job["seed"], "guide": job["guide"],
              "prompt_id": prompt_id, "prompt_sha256": digest(work / "prompt.json"),
              "seconds": round(time.time() - submitted["submitted_epoch"], 3),
              "camera_conditioning": job["camera_conditioning"], "camera_is_calibration": False}
    for node, key, name in (("14", "image", "generated.png"), ("18", "mask", "mask.png")):
        image = entry["outputs"][node]["images"][0]
        if image["type"] != "output":
            raise ValueError("Expected an output image")
        source = (args.comfy_output / image.get("subfolder", "") / image["filename"]).resolve()
        if not source.is_relative_to(args.comfy_output.resolve()):
            raise ValueError("Output path escaped the configured output directory")
        target = work / name
        shutil.copyfile(source, target)
        with Image.open(target) as opened:
            if opened.size != (args.size, args.size):
                raise ValueError(f"Unexpected generated dimensions {opened.size}")
        result[key] = str(target.resolve())
        result[key + "_sha256"] = digest(target)
    result["path"] = result["image"]
    result["sha256"] = result["image_sha256"]
    write(result_path, result)
    print(json.dumps({"stage": "complete", **result}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--guides", type=Path, required=True, help="JSON with views:[{id,image,camera}]")
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--comfy-input", type=Path, required=True)
    parser.add_argument("--comfy-output", type=Path, required=True)
    parser.add_argument("--server", default="http://127.0.0.1:8198")
    parser.add_argument("--input-prefix", default="zipsplat-multiview-20261120")
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026112100)
    parser.add_argument("--cache-dtype", choices=("default", "int8", "int4"), default="default")
    parser.add_argument("--prompt", default=AUTHOR_PROMPT)
    parser.add_argument("--views", help="Comma-separated IDs, preserving this order")
    parser.add_argument("--max-views", type=int, help="Run only the first N frozen views; resume without this limit")
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.input_prefix):
        parser.error("--input-prefix must be a simple directory name")
    if args.size < 32 or args.size % 32:
        parser.error("--size must be a positive multiple of 32")
    jobs, recipe = prepare(args)
    if args.prepare_only:
        print(json.dumps({"prepared": str(args.output), "views": len(jobs)}))
        return
    results = []
    for job in jobs[:args.max_views]:
        results.append(run_job(args, job))
        write(args.output / "manifest.json", {"recipe": recipe, "views": results,
              "complete": len(results) == len(jobs)})
    print(json.dumps({"complete": str(args.output / "manifest.json")}), flush=True)


if __name__ == "__main__":
    main()
