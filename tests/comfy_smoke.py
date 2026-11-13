"""Run the registered nodes through a real, CPU-only ComfyUI HTTP queue."""
import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image

from check_workflow_types import check_workflows
from splat_native_smoke import write_splat_fixture, check_splat

ROOT = Path(__file__).resolve().parents[1]

def run(comfy, output):
    output.mkdir(parents=True, exist_ok=True)
    inputs = output / "input"
    inputs.mkdir(exist_ok=True)
    write_splat_fixture(inputs)
    fixture = inputs / "asymmetric.glb"
    subprocess.run(["node", str(ROOT / "tests/make_fixture.mjs"), str(fixture)], check=True)
    subprocess.run(["node", str(ROOT / "tests/make_open_frame.mjs"), str(inputs / 'corrected.glb')], check=True)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    origin = f"http://127.0.0.1:{port}"

    def request(path, data=None):
        req = Request(origin + path, data=json.dumps(data).encode() if data is not None else None,
                      headers={"Content-Type": "application/json"})
        try:
            with urlopen(req, timeout=15) as result:
                return result.read()
        except HTTPError as exc:
            raise RuntimeError(f"{path} HTTP {exc.code}: {exc.read().decode()[:4000]}") from exc

    def queue(graph):
        receipt = json.loads(request("/prompt", {"prompt": graph, "client_id": "box-smoke"}))
        prompt_id = receipt["prompt_id"]
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline:
            data = json.loads(request("/history/" + prompt_id)).get(prompt_id)
            if data:
                if data["status"]["status_str"] != "success":
                    raise RuntimeError(json.dumps(data["status"])[-4000:])
                return data["outputs"]
            time.sleep(.5)
        raise TimeoutError("ComfyUI did not finish the queue item")

    def node(class_type, **inputs_):
        return {"class_type": class_type, "inputs": inputs_}

    log_path = output / "comfy.log"
    with log_path.open("w") as log:
        cmd = [sys.executable, "main.py", "--cpu", "--listen", "127.0.0.1", "--port", str(port),
               "--input-directory", str(inputs), "--output-directory", str(output / "output"),
               "--disable-auto-launch", "--disable-all-custom-nodes", "--whitelist-custom-nodes", "box_impostor_factory"]
        proc = subprocess.Popen(cmd, cwd=comfy, stdout=log, stderr=subprocess.STDOUT)
        try:
            for _ in range(180):
                if proc.poll() is not None:
                    raise RuntimeError("ComfyUI stopped during startup")
                try:
                    request("/system_stats")
                    break
                except URLError:
                    time.sleep(1)
            else:
                raise TimeoutError("ComfyUI startup timed out")
            registry_proof = check_workflows(json.loads(request('/object_info')), ROOT / 'workflows')
            (output / 'workflow-nodes.json').write_text(json.dumps(registry_proof, indent=2))
            print('Runtime workflow node checks', json.dumps(registry_proof), flush=True)
            required = ["BICanonicalizeProxy", "BIBakeBox", "BIFaceImage", "BIPreviewBox", "BIRenderBox", "BILoadBox"]
            for name in required:
                if name not in json.loads(request("/object_info/" + name)):
                    raise RuntimeError("ComfyUI did not register " + name)
            common = {
                "1": node("BICanonicalizeProxy", filename=fixture.name),
                "2": node("BIBakeBox", proxy=["1",0], resolution=128, backend="webgl2"),
                "3": node("BIPreviewBox", asset=["2",0]),
                "4": node("BISaveBox", asset=["2",0], name="fixture"),
            }
            render_check = queue({**common,
                "5": node("BIRenderBox", asset=["2",0], width=96, height=80, view="+Z",
                          projection="orthographic", sampling="reduce_bleeding", backend="webgl2"),
                "6": node("JoinImageWithAlpha", image=["5", 0], alpha=["5", 1]),
                "7": node("SaveImage", images=["6", 0], filename_prefix="box-render")})
            saved, = render_check["7"]["images"]
            path = output / 'output' / saved['subfolder'] / saved['filename']
            with Image.open(path) as image:
                assert image.mode == 'RGBA' and image.size == (96, 80)
                rgba = np.asarray(image)
            assert rgba[0, 0, 3] == 0 and rgba[40, 48, 3] == 255
            captures = output / 'output/box_impostor/.stages/renders'
            # Compare the actual node outputs after Comfy's mask inversion and SaveImage quantization.
            assert any(np.abs(rgba.astype(int) - np.asarray(Image.open(p), dtype=int)).max() <= 1
                       for p in captures.glob('*/render.png') if Image.open(p).size == (96, 80))
            first = queue(common)
            viewer_path = first["3"]["box_impostor"][0]
            manifest_path = viewer_path.split("?manifest=",1)[1]
            baseline = json.loads(request(manifest_path))
            check_splat(queue, node, inputs, output, fixture.name, baseline, request)
            assert len(request(viewer_path)) > 1000
            zipped = queue({
                '1': node('BILoadBox', source='fixture'),
                '2': node('BIPackBox', asset=['1', 0], name='native-smoke'),
                '3': node('BILoadBox', source=['2', 0]),
                '4': node('BIPreviewBox', asset=['3', 0]),
            })
            assert zipped['4']['box_impostor'][0] == viewer_path
            for rel in ("web/viewer.mjs","worker/runtime.mjs","vendor/three/build/three.webgpu.js"):
                assert len(request("/box-impostor/" + rel)) > 100
            # Use the actual guide for one manual face, pass through the native queue and reload the export.
            edited = queue({
                **{k:v for k,v in common.items() if k in ("1","2")},
                "5": node("BIMakeSixGuides", proxy=["1",0], resolution=128, backend="webgl2"),
                "6": node("BIFaceImage", image=["5",0], mask_mode="rgba"),
                "7": node("BIFaceInputs", px=["6",0]),
                "8": node("BIAlignAndValidateFaces", faces=["7",0], guides=["5",6],
                          min_iou=.85, max_shift=.1, max_scale_change=.15),
                "9": node("BIBakeBox", proxy=["1",0], resolution=128, backend="webgl2", faces=["8",0]),
                "10": node("BIPreviewBox", asset=["9",0]),
                "11": node("BILoadBox", source="fixture"),
                "12": node("BIPreviewBox", asset=["11",0]),
            })
            assert edited["12"]["box_impostor"][0] == viewer_path, "reload returned a different baseline"
            path = edited["10"]["box_impostor"][0].split("?manifest=",1)[1]
            updated = json.loads(request(path))
            assert updated["parentAsset"] == baseline["asset_id"]
            assert updated["faces"]["+X"]["edit"]["status"] == "accepted"
            for name in ("intervalMap","rangeMap","normalMap"):
                assert updated["textures"][name] == baseline["textures"][name]
            print(json.dumps({"ok":True,"baseline":baseline["asset_id"],"manual":updated["asset_id"]}))
            # The frontend browser must use the same CPU/software renderer as
            # the Box worker (e.g. Windows Edge from a WSL ComfyUI process).
            browser_node = os.environ.get('BI_NODE') or 'node'
            subprocess.run([browser_node, str(ROOT / 'tests/comfy_frontend_smoke.mjs'), origin,
                            str(inputs), str(output / 'frontend')], cwd=ROOT, check=True, timeout=960)
            # The actual frontend independently numbers Pixal3D nodes for each containing workflow.
            sys.path.insert(0, str(ROOT))
            sys.path.insert(0, str(comfy))
            import folder_paths
            from box_nodes.generation import lineage
            folder_paths.set_input_directory(str(inputs))
            folder_paths.set_output_directory(str(output / 'output'))
            keys = []
            for name in ('02_box_from_reference_qwen21', '03_box_from_reference_h3', '05_pixal3d_baseline'):
                graph = json.loads((output/'frontend'/(name+'.api.json')).read_text())
                proxy = next(i for i, n in graph.items() if n['class_type']=='BICanonicalizeProxy')
                keys.append(lineage(graph,proxy,('model_3d',))[0])
            assert len(set(keys)) == 1, 'Independently expanded Pixal3D subgraphs must share one cache entry'
            with Image.open(output / 'frontend/rendered-box.png') as image:
                assert image.mode == 'RGBA' and image.size == (512, 512)
                assert image.getpixel((0, 0))[3] == 0 and image.getpixel((256, 256))[3] == 255
        finally:
            proc.terminate()
            try: proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--comfy", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    opts = ap.parse_args()
    run(opts.comfy.resolve(), opts.output.resolve())
