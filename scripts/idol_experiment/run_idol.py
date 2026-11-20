#!/usr/bin/env python3
"""Run the saved TripoSplat graph with Idol weights; preserve preprocessing.

Defaults only prepare prompt/metadata. Root may submit explicitly:
  python run_idol.py --steps 20 --execute --url http://127.0.0.1:8198
  python run_idol.py --steps 50 --execute --url http://127.0.0.1:8198
Use --collect PROMPT_ID after an interrupted client; it never submits a job.
Run with the ComfyUI Python environment (NumPy is needed for mask validation).
"""

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import uuid

BASE = Path('/mnt/f/_dev/box-impostor-lab/experiments/triposplat-person-20261120')
OUTPUT = Path('/mnt/f/_dev/box-impostor-lab/experiments/idol-triposplat-20261120')
COMFY = Path('/home/dev/Projects/box-impostor-lab/ComfyUI')
STAGES = Path('/mnt/f/mnt/output/box_impostor/.stages')
OLD_REFERENCE = 'bb85341d2c3caf034417b3d16fed1dc6b78cf4fee9a93ea031f00bc7ec07efa3'
SOURCE_SHA = '9d92ad97b34bb949b3aa555afed321b8dae427c0102c862b2afb96745b93390b'
OLD_CACHE = STAGES / 'triposplat/ecb8433166915206286d50ab299fe2faa73d7e7b028f8ea19fa36cd3d41fad54'
EXPECTED_WEIGHTS = {
    'diffusion_models/idol_fp16.safetensors': '9abf6a99fc6bdad701901b12de8fe1b256f02cae4948f7489d25d86b11b52a93',
    'vae/idol_decoder_fp16.safetensors': '6a146a1e6b2653f74422f7ea835eb8715c547c7de8fc4545051f15c247ae54a9',
    'clip_vision/dino_v3_vit_h.safetensors': 'a29ef35101a16966972a0d50732a6f3a608ff7cfffb2afa9bbe9007cb842cc53',
    'vae/flux2-vae.safetensors': 'd64f3a68e1cc4f9f4e29b6e0da38a0204fe9a49f2d4053f0ec1fa1ca02f9c4b5',
}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def key(*parts):
    return hashlib.sha256(json.dumps(parts, sort_keys=True, allow_nan=False,
                                    separators=(',', ':')).encode()).hexdigest()


def request(url, endpoint, value=None):
    data = None if value is None else json.dumps(value).encode()
    req = urllib.request.Request(url.rstrip('/') + endpoint, data=data,
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=90) as response:
        return json.load(response)


def build_prompt(source, steps):
    original = json.loads(source.read_text())
    graph = copy.deepcopy(original)
    assert graph['8']['inputs']['unet_name'] == 'triposplat_fp16.safetensors'
    assert graph['11']['inputs']['vae_name'] == 'triposplat_vae_decoder_fp16.safetensors'
    assert graph['10']['inputs']['seed'] == 46
    assert graph['12']['inputs']['seed'] == 790219963981395
    assert graph['12']['inputs']['num_gaussians'] == 262144
    graph['8']['inputs']['unet_name'] = 'idol_fp16.safetensors'
    graph['11']['inputs']['vae_name'] = 'idol_decoder_fp16.safetensors'
    graph['10']['inputs']['steps'] = steps
    # This is the complete allowed difference: no nodes, edges, masks or cameras change.
    restored = copy.deepcopy(graph)
    for node, field in [('8', 'unet_name'), ('11', 'vae_name'), ('10', 'steps')]:
        restored[node]['inputs'][field] = original[node]['inputs'][field]
    assert restored == original
    assert graph['2']['inputs']['bg_removal_name'] == 'birefnet.safetensors'
    assert graph['4']['inputs']['mask_mode'] == 'foreground'
    assert graph['5']['inputs']['erode_radius'] == 1
    assert graph['5']['inputs']['size'] == 1024
    assert graph['13']['class_type'] == 'BICachedTripoSplat'
    assert graph['20']['inputs']['format'] == 'ply'
    return graph


def graph_hashes(graph, weights):
    """Exact current/legacy generation.lineage hashes for the splat socket.

    No Comfy import is needed. This graph terminates at BIPrepareReference;
    image_inputs is therefore empty, as in the original implementation.
    """
    nodes = {}
    def visit(link):
        if not (isinstance(link, list) and len(link) == 2 and str(link[0]) in graph):
            return
        node = str(link[0])
        if node in nodes:
            return
        item = graph[node]
        nodes[node] = {'type': item['class_type']}
        if item['class_type'] == 'BIPrepareReference':
            return
        nodes[node]['inputs'] = item['inputs']
        for value in item['inputs'].values():
            visit(value)
    bindings = {'splat': graph['13']['inputs']['splat']}
    visit(bindings['splat'])
    indices, ordered = {}, []
    def encode(value):
        if not (isinstance(value, list) and len(value) == 2 and
                isinstance(value[0], (str, int)) and type(value[1]) is int and str(value[0]) in nodes):
            return {'literal': value}
        node = str(value[0])
        if node not in indices:
            indices[node] = len(ordered)
            raw = nodes[node]
            item = {'type': raw['type']}
            ordered.append(item)
            if 'inputs' in raw:
                item['inputs'] = {k: encode(raw['inputs'][k]) for k in sorted(raw['inputs'])}
        return {'node': indices[node], 'output': value[1]}
    roots = {k: encode(bindings[k]) for k in sorted(bindings)}
    return [key('graph-v2', roots, ordered, weights, {}), key(bindings, nodes, weights, {})]


def find_cache(stages, hashes, weights):
    matches = []
    for path in sorted((stages / 'triposplat').glob('*/result.json')):
        meta = json.loads(path.read_text())
        if meta.get('graphSha256') in hashes and meta.get('weights') == weights:
            matches.append((path.parent, meta))
    # Cache migration can retain two directories with byte-identical receipts.
    unique = {(m['sha256'], m['referenceSha256']): (p, m) for p, m in matches}
    if len(unique) != 1:
        raise RuntimeError('Expected one exact graph/weight decode; candidates: ' +
                           str([str(p) for p, _ in matches]))
    return next(iter(unique.values()))


def ply_reference(value):
    if isinstance(value, dict):
        if str(value.get('filename', '')).lower().endswith('.ply'):
            yield value
        for child in value.values():
            yield from ply_reference(child)
    elif isinstance(value, list):
        for child in value:
            yield from ply_reference(child)


def compare_reference(stages, current_sha, run):
    import numpy as np
    old_path = STAGES / 'faces' / OLD_REFERENCE / 'source.npy'
    new_path = stages / 'faces' / current_sha / 'source.npy'
    old, new = np.load(old_path, allow_pickle=False), np.load(new_path, allow_pickle=False)
    digest = lambda a: key('face-v1', a.shape, hashlib.sha256(a.tobytes()).hexdigest())
    assert digest(old) == OLD_REFERENCE, 'Original reference data does not match its cache identity'
    assert digest(new) == current_sha, 'New reference data does not match its cache identity'
    same_shape = old.shape == new.shape
    same = same_shape and np.array_equal(old, new)
    same_alpha = same_shape and np.array_equal(old[..., 3], new[..., 3])
    result = {
        'oldReferenceSha256': OLD_REFERENCE, 'newReferenceSha256': current_sha,
        'sameCacheIdentity': current_sha == OLD_REFERENCE,
        'rgbAndCoverageFloat32ExactlyEqual': bool(same),
        'coverageFloat32ExactlyEqual': bool(same_alpha),
        'oldShape': list(old.shape), 'newShape': list(new.shape),
        'alphaChangedPixels': int(np.count_nonzero(old[..., 3] != new[..., 3])) if same_shape else None,
        'alphaMaxAbsoluteDifference': float(np.max(np.abs(old[..., 3] - new[..., 3]))) if same_shape else None,
        'interpretation': ('BIPrepareReference RGB and coverage are byte-identical to the original Tripo run.'
                           if same else 'Prepared RGB/coverage differs from the original Tripo run; this is a comparison confound.'),
        'scope': 'This validates PrepareReference output, not all downstream model conditioning internals.',
    }
    shutil.copyfile(new_path, run / 'prepared-reference.npy')
    png = new_path.with_suffix('.png')
    if png.exists():
        shutil.copyfile(png, run / 'prepared-reference.png')
    save(run / 'reference-comparison.json', result)
    return result


def gpu_summary(path, interval):
    samples = []
    for row in csv.reader(path.read_text(errors='replace').splitlines()):
        try:
            samples.append({'timestamp': row[0].strip(), 'gpuIndex': int(row[1]),
                            'memoryUsedMiB': float(row[2]), 'memoryTotalMiB': float(row[3])})
        except (ValueError, IndexError):
            pass
    return {'measurement': 'nvidia-smi sampled whole-device used VRAM; not PyTorch allocator peak',
            'intervalMilliseconds': interval, 'sampleCount': len(samples),
            'peakSampledUsedMiB': max((x['memoryUsedMiB'] for x in samples), default=None),
            'firstSample': samples[0] if samples else None,
            'lastSample': samples[-1] if samples else None,
            'limitation': 'Can miss peaks between samples and includes other processes and driver memory.'}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--steps', type=int, choices=(20, 50), required=True)
    p.add_argument('--url', default='http://127.0.0.1:8198')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true', help='Submit one inference, wait and collect')
    mode.add_argument('--collect', metavar='PROMPT_ID', help='Collect a previously submitted prompt; no submission')
    p.add_argument('--output', type=Path, default=OUTPUT)
    p.add_argument('--source-prompt', type=Path, default=BASE / 'minimal-prompt.json')
    p.add_argument('--comfy-root', type=Path, default=COMFY)
    p.add_argument('--stages', type=Path, default=STAGES)
    p.add_argument('--run-name')
    p.add_argument('--gpu', default='0')
    p.add_argument('--vram-interval-ms', type=int, default=200)
    p.add_argument('--timeout-seconds', type=float, default=3600)
    args = p.parse_args()
    run = args.output / (args.run_name or f'idol-{args.steps}steps-seed46')
    run.mkdir(parents=True, exist_ok=True)
    if args.execute and (run / 'receipt.json').exists():
        raise RuntimeError('Existing submission receipt: use --collect PROMPT_ID or a different --run-name')
    graph = build_prompt(args.source_prompt, args.steps)
    save(run / 'prompt.json', graph)
    source = BASE / 'inputs/source-original.png'
    assert sha(source) == SOURCE_SHA
    comfy_input = args.comfy_root / 'input' / graph['1']['inputs']['image']
    assert sha(comfy_input) == SOURCE_SHA, 'Comfy LoadImage input differs from the original reference'
    weights = {name: sha(args.comfy_root / 'models' / name) for name in EXPECTED_WEIGHTS}
    assert weights == EXPECTED_WEIGHTS, 'Expected original shared weights and verified Idol weights'
    hashes = graph_hashes(graph, weights)
    source_files = [args.comfy_root / 'comfy_extras/nodes_triposplat.py',
                    args.comfy_root / 'comfy_version.py',
                    *sorted((args.comfy_root / 'comfy/ldm/triposplat').glob('*.py'))]
    implementation = {path.name: sha(path) for path in source_files if path.is_file()}
    old_receipt = json.loads((OLD_CACHE / 'result.json').read_text())
    save(run / 'preflight.json', {
        'source': str(source), 'sourceSha256': SOURCE_SHA, 'comfyInput': str(comfy_input),
        'sourcePrompt': str(args.source_prompt), 'sourcePromptSha256': sha(args.source_prompt),
        'promptSha256': sha(run / 'prompt.json'),
        'submittedGraphHashesBeforeComfyTypeValidation': hashes,
        'weights': weights, 'implementation': implementation, 'steps': args.steps,
        'onlyChangedInputs': ['8.unet_name', '11.vae_name', '10.steps'],
        'biRefNetAndPreprocessingGraphPreserved': True, 'oldReferenceSha256': OLD_REFERENCE,
        'preprocessingCodeHashMatchesOldRun': implementation.get('nodes_triposplat.py') == old_receipt['implementation'].get('nodes_triposplat.py'),
        'maskCaveat': 'Identical mask graph does not itself prove identical masks; compare saved prepared RGBA after execution.',
        'stagesRoot': str(args.stages), 'server': args.url,
    })
    if not args.execute and not args.collect:
        print(json.dumps({'prepared': str(run), 'submitted': False}), flush=True)
        return
    started = time.monotonic()
    monitor = None
    gpu_stream = error_stream = None
    prompt_id = args.collect
    try:
        if args.execute:
            queue = request(args.url, '/queue')
            save(run / 'queue-before.json', queue)
            if queue.get('queue_running') or queue.get('queue_pending'):
                raise RuntimeError('Comfy queue is not idle; refusing an ambiguous timed comparison')
            gpu_stream = (run / 'gpu-memory.csv').open('w')
            error_stream = (run / 'gpu-memory.stderr.txt').open('w')
            monitor = subprocess.Popen(['nvidia-smi', '-i', args.gpu,
                '--query-gpu=timestamp,index,memory.used,memory.total', '--format=csv,noheader,nounits',
                f'--loop-ms={args.vram_interval_ms}'], stdout=gpu_stream, stderr=error_stream)
            submitted_at = time.time()
            reply = request(args.url, '/prompt', {'prompt': graph, 'client_id': 'idol-study-' + str(uuid.uuid4())})
            prompt_id = reply.get('prompt_id')
            save(run / 'receipt.json', {'promptId': prompt_id, 'serverReply': reply,
                                       'submittedAtUnixSeconds': submitted_at,
                                       'promptSha256': sha(run / 'prompt.json')})
            if not prompt_id or reply.get('node_errors'):
                raise RuntimeError('Prompt was rejected: ' + json.dumps(reply))
            print(json.dumps({'submitted': prompt_id, 'run': str(run), 'steps': args.steps}), flush=True)
        last_progress = 0
        while True:
            history = request(args.url, '/history/' + urllib.parse.quote(prompt_id, safe=''))
            if prompt_id in history:
                entry = history[prompt_id]
                save(run / 'history.json', entry)
                if entry.get('status', {}).get('completed') or entry.get('status', {}).get('status_str') == 'error':
                    break
            elapsed = time.monotonic() - started
            if elapsed > args.timeout_seconds:
                raise TimeoutError('Client collection timeout; job was not interrupted. Resume with --collect ' + prompt_id)
            if elapsed - last_progress >= 15:
                print(json.dumps({'waiting': prompt_id, 'elapsedSeconds': round(elapsed, 1)}), flush=True)
                last_progress = elapsed
            time.sleep(1)
        inference_wall = time.monotonic() - started
        if entry.get('status', {}).get('status_str') != 'success':
            raise RuntimeError('Inference failed; see history.json')
    finally:
        if monitor is not None:
            monitor.terminate()
            try:
                monitor.wait(timeout=5)
            except subprocess.TimeoutExpired:
                monitor.kill()
                monitor.wait()
            gpu_stream.close()
            error_stream.close()
            save(run / 'gpu-memory-summary.json', gpu_summary(run / 'gpu-memory.csv', args.vram_interval_ms))
    # Comfy validates e.g. cfg=3 -> 3.0 and denoise=1 -> 1.0 before its lineage
    # cache sees the prompt. JSON hashes distinguish these representations.
    # Derive the cache identity from history's executed graph, not the request.
    executed = entry['prompt'][2]
    expected_content = {node: {'class_type': n['class_type'], 'inputs': n['inputs']}
                        for node, n in graph.items()}
    actual_content = {node: {'class_type': n['class_type'], 'inputs': n['inputs']}
                      for node, n in executed.items()}
    # Numeric equality permits only equivalent int/float schema conversion.
    assert actual_content == expected_content, 'History prompt differs from the requested Idol graph'
    save(run / 'execution-prompt.json', executed)
    hashes = graph_hashes(executed, weights)
    stage, cache_meta = find_cache(args.stages, hashes, weights)
    assert sha(stage / 'splat.npz') == cache_meta['sha256']
    assert cache_meta['gaussians'] == 262144
    shutil.copyfile(stage / 'splat.npz', run / 'splat.npz')
    save(run / 'decode-cache.json', {'stage': str(stage), **cache_meta})
    # PreviewGaussianSplat exposes a temp PLY through Comfy's ordinary /view route.
    refs = list(ply_reference(entry.get('outputs', {}).get('21', {})))
    unique = {(r['filename'], r.get('subfolder', ''), r.get('type', 'temp')) for r in refs}
    if len(unique) != 1:
        raise RuntimeError('Expected one PLY in node 21 history; see history.json: ' + str(refs))
    filename, subfolder, file_type = next(iter(unique))
    query = urllib.parse.urlencode({'filename': filename, 'subfolder': subfolder, 'type': file_type})
    with urllib.request.urlopen(args.url.rstrip('/') + '/view?' + query, timeout=90) as response:
        with (run / 'splat.ply').open('wb') as dest:
            shutil.copyfileobj(response, dest)
    with (run / 'splat.ply').open('rb') as stream:
        assert stream.read(4) == b'ply\n', 'Downloaded file is not PLY'
    comparison = compare_reference(args.stages, cache_meta['referenceSha256'], run)
    messages = entry.get('status', {}).get('messages', [])
    times = {name: data.get('timestamp') for name, data in messages if isinstance(data, dict)}
    start, end = times.get('execution_start'), times.get('execution_success')
    summary = {
        'completed': True, 'promptId': prompt_id, 'run': str(run), 'steps': args.steps,
        'clientWaitSeconds': inference_wall,
        'serverExecutionSeconds': (end - start) / 1000 if start is not None and end is not None else None,
        'collectionOnly': bool(args.collect), 'stage': str(stage),
        'executedGraphHashes': hashes,
        'gaussians': cache_meta['gaussians'], 'npzSha256': sha(run / 'splat.npz'),
        'plySha256': sha(run / 'splat.ply'), 'plyBytes': (run / 'splat.ply').stat().st_size,
        'referenceComparison': comparison,
        'executionCachedMessages': [data for name, data in messages if name == 'execution_cached'],
        'vram': json.loads((run / 'gpu-memory-summary.json').read_text()) if (run / 'gpu-memory-summary.json').exists() else None,
        'orientation': 'Raw native decoded splat; identity TransformSplat. Previous comparison applied +90 degrees around Y separately.',
    }
    save(run / 'result.json', summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
