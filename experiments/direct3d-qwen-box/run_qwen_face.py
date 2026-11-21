#!/usr/bin/env python3
"""Colorize one Direct3D-V3 Box face using a full-view RGB guide, photo and depth.

Default: prepare only. --execute submits one job. --collect PROMPT_ID retrieves
an existing job without resubmitting. Only Pillow and the standard library are
needed in this client; ComfyUI owns inference.
"""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import time
import urllib.parse
import urllib.request
import uuid

from PIL import Image

DIRECT3D = Path('/mnt/f/_dev/box-impostor-lab/experiments/direct3d-v3-20261121')
TEMPLATE = Path(__file__).with_name('qwen_success_template.json')
TEMPLATE_SHA = '7c3d96f0ea0b7485f130ecd847cfe1bd78e4a09effa4864a10700ffec2c04bce'
REFERENCE_SHA = '9d92ad97b34bb949b3aa555afed321b8dae427c0102c862b2afb96745b93390b'
FACE_IDS = {'px': '+X', 'nx': '-X', 'py': '+Y', 'ny': '-Y', 'pz': '+Z', 'nz': '-Z'}
FACE_PROMPTS = {
    'pz': 'This is the +Z front camera view. Preserve the full-body framing in image 1 and the '
          'existing head direction; show only facial features visible from this exact camera.',
    'nz': 'This is the -Z back camera view. Show the rear surfaces and the back of the hair and '
          'clothing. Do not add eyes, a face, nose or mouth on the back of the head. Do not turn '
          'the person toward the camera. Preserve the full-body framing in image 1.',
    'px': 'This is the +X side camera view looking toward -X. Preserve the full-body framing, '
          'side silhouette and exact head orientation in image 1. Do not invent a front-facing '
          'face or reveal an eye or limb hidden from this camera.',
    'nx': 'This is the -X side camera view looking toward +X. Preserve the full-body framing, '
          'side silhouette and exact head orientation in image 1. Do not invent a front-facing '
          'face or reveal an eye or limb hidden from this camera.',
    'py': 'This is the +Y top camera view looking downward. Keep its overhead projection and '
          'the exact visible surfaces and occlusions. Hidden body parts must stay hidden; do '
          'not rearrange the person into an upright full-body view or expose the face.',
    'ny': 'This is the -Y bottom camera view looking upward. Keep its underside projection and '
          'the exact visible surfaces and occlusions, including shoe soles where visible. '
          'Hidden body parts must stay hidden; do not rearrange the person into an upright '
          'full-body view or expose the face.',
}
DEFAULTS = {'profile': 'anchored', 'denoise': .9, 'loraStrength': 0., 'size': 1024,
            'steps': 10, 'cfg': 3, 'seed': 2026113031, 'sampler': 'euler', 'scheduler': 'simple'}


def face_prompt(face):
    return (
        'Colorize the neutral-clay render in image 1 using the appearance of the person in image 2. '
        'Image 1 defines the exact camera, projection, framing, head direction, pose, silhouette, '
        'body proportions, limb positions, clothing outline, occlusions and background gaps '
        'between arms and torso and between the legs. Preserve those structures at the same '
        'image coordinates. Image 2 is only an appearance reference for the same person: blonde '
        'hair, gray knit sweater dress, natural skin colors and shoes. Do not copy its pose, '
        'camera angle, head direction, expression, body proportions or framing. Image 3 is the '
        'corresponding Z-depth of image 1, brighter nearer and black background. Use it to '
        'understand the existing structure and occlusions; never copy depth values as material '
        'colors. Replace the clay appearance with natural photographic skin, hair strands, '
        'coherent gray knit fabric and shoe materials wherever those surfaces are visible. '
        'Preserve the complete uncropped camera frame and all visible subject coverage. Keep '
        'occluded surfaces hidden and keep the same uniform neutral gray background. '
        + FACE_PROMPTS[face] + ' Do not rotate the person, move body parts, close background '
        'gaps, zoom, crop, add subjects, text, graphics, panels or diagrams. This is surface '
        'colorization of the geometry already visible in image 1.'
    )


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def request(server, endpoint, value=None):
    data = None if value is None else json.dumps(value).encode()
    req = urllib.request.Request(server.rstrip('/') + endpoint, data=data,
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=90) as response:
        return json.load(response)


def graph_from_success(template, prefix, size, face, profile='anchored', denoise=.9,
                       lora_strength=0., prompt=None):
    original = json.loads(template.read_text())
    graph = copy.deepcopy(original)
    assert original['6']['inputs']['unet_name'] == 'qwen_image_2.1_int8_convrot.safetensors'
    assert original['7']['inputs']['clip_name'] == 'qwen3vl_8b_int8_convrot.safetensors'
    assert original['8']['inputs']['vae_name'] == 'qwen_image_2.1_vae_bf16.safetensors'
    assert original['19']['inputs']['lora_name'] == 'QI2.1_AnyAngle.safetensors'
    assert original['19']['inputs']['strength_model'] == 1
    assert original['9']['inputs'] == {'model': ['19', 0], 'device': 'cpu', 'dtype': 'int8'}
    sampler = original['12']['inputs']
    assert {k: sampler[k] for k in ('steps', 'cfg', 'sampler_name', 'scheduler', 'denoise', 'seed')} == {
        'steps': 10, 'cfg': 3, 'sampler_name': 'euler', 'scheduler': 'simple', 'denoise': 1, 'seed': 2026113031}
    enc = original['10']['inputs']
    assert enc['resolution'] == 0 and enc['negative_prompt'] == ''
    assert [enc[f'images.image_{i}'] for i in (1, 2, 3)] == [['1', 0], ['2', 0], ['3', 0]]
    assert '2048 pixel square' in enc['prompt']
    graph['1']['inputs']['image'] = prefix + '/target-color.png'
    graph['2']['inputs']['image'] = prefix + '/original.png'
    graph['3']['inputs']['image'] = prefix + '/target-depth.png'
    graph['10']['inputs']['prompt'] = enc['prompt'].replace('2048 pixel square', f'{size} pixel square')
    graph['14']['inputs']['filename_prefix'] = prefix + '/raw'
    graph['18']['inputs']['filename_prefix'] = prefix + '/mask'
    # Preserve the complete graph, including model/cache, image order, and mask branch.
    restored = copy.deepcopy(graph)
    for node, field in [('1', 'image'), ('2', 'image'), ('3', 'image'), ('10', 'prompt'),
                        ('14', 'filename_prefix'), ('18', 'filename_prefix')]:
        restored[node]['inputs'][field] = original[node]['inputs'][field]
    assert restored == original
    if profile == 'anchored':
        graph['20'] = {'class_type': 'VAEEncode', 'inputs': {'pixels': ['1', 0], 'vae': ['8', 0]}}
        graph['12']['inputs']['latent_image'] = ['20', 0]
        graph['12']['inputs']['denoise'] = denoise
    graph['10']['inputs']['prompt'] = face_prompt(face) if prompt is None else prompt
    graph['19']['inputs']['strength_model'] = lora_strength
    return graph


def prepare(args):
    if not args.target_rgb or not args.target_depth:
        raise ValueError('--target-rgb and --target-depth are required unless collecting an existing run')
    assert sha(args.template) == TEMPLATE_SHA, 'Use the exact successful front-2kopt template'
    assert sha(args.original) == REFERENCE_SHA, 'Original reference differs from the previous experiment'
    prefix = PurePosixPath(args.input_prefix)
    if prefix.is_absolute() or '..' in prefix.parts or '\\' in args.input_prefix:
        raise ValueError('Input prefix must be a relative POSIX path without traversal')
    custom_prompt = args.prompt_file.read_text(encoding='utf-8').strip() if args.prompt_file else None
    if custom_prompt is not None and not custom_prompt:
        raise ValueError('--prompt-file must contain a nonempty UTF-8 prompt')
    graph = graph_from_success(args.template, prefix.as_posix(), args.size, args.face,
                               args.profile, args.denoise, args.lora_strength, custom_prompt)
    destination = args.comfy_input / prefix
    destination.mkdir(parents=True, exist_ok=True)
    inputs = {}
    for name, path, dimensions in [('target-color.png', args.target_rgb, (args.size, args.size)),
                                   ('target-depth.png', args.target_depth, (512, 512)),
                                   ('original.png', args.original, (1024, 1024))]:
        path = path.resolve(strict=True)
        with Image.open(path) as im:
            assert im.size == dimensions, f'{name}: expected {dimensions}, got {im.size}'
            if name == 'target-color.png':
                assert im.mode == 'RGB', 'Supply the square RGB guide already composited on neutral gray'
            mode = im.mode
        dest = destination / name
        digest = sha(path)
        if dest.exists():
            assert sha(dest) == digest, 'Frozen Comfy input differs: ' + str(dest)
        else:
            shutil.copyfile(path, dest)
        inputs[name] = {'source': str(path), 'comfyInput': str(dest), 'sha256': digest,
                        'size': list(dimensions), 'mode': mode}
    record = {
        'case': 'Direct3D-V3 Box face colorization', 'face': FACE_IDS[args.face], 'faceSlug': args.face,
        'sourceTemplate': str(args.template), 'sourceTemplateSha256': TEMPLATE_SHA,
        'sourceSuccessfulPromptId': '1f0e2f3a-6651-4db2-89e6-c8e956fa740a',
        'inputs': inputs, 'inputOrder': ['target full-view RGB', 'original reference photograph', 'target depth'],
        'size': args.size, 'depthSize': 512, 'textEncodeResolution': 0,
        'steps': 10, 'cfg': 3, 'seed': 2026113031, 'sampler': 'euler', 'scheduler': 'simple',
        'cache': {'device': 'cpu', 'dtype': 'int8'}, 'loraStrength': graph['19']['inputs']['strength_model'],
        'changesFromSuccessful2K': [f'Image 1 is {args.size} square instead of 2048 square',
            'Face-specific Direct3D surface-colorization prompt replaces the camera-change prompt',
            'New target RGB/depth, frozen input paths, and output prefixes',
            f'AnyAngle LoRA strength is {graph["19"]["inputs"]["strength_model"]}'],
        'unchanged': ['1024 square original photograph', '512 square depth reference size',
            'reference order', 'resolution=0 preserves each reference size', '10 steps, CFG 3, seed',
            'base model, text encoder, VAE, and CPU int8 KV cache', 'BiRefNet output-mask branch'],
        'server': args.server,
        'profile': args.profile, 'runnerSha256': sha(__file__),
        'defaults': DEFAULTS, 'prompt': graph['10']['inputs']['prompt'],
        'promptSource': {'kind': 'file', 'path': str(args.prompt_file.resolve()),
                         'sha256': sha(args.prompt_file)} if args.prompt_file else {'kind': 'built-in-face-prompt'},
        'denoise': graph['12']['inputs']['denoise'],
        'initialLatent': 'VAE-encoded target RGB' if args.profile == 'anchored' else 'zeros from TextEncodeQwenImage21',
    }
    if args.profile == 'anchored':
        record['changesFromSuccessful2K'].extend([
            'VAEEncode target RGB is the initial sampling latent instead of zeros',
            f'Sampler denoise is {args.denoise}',
            'Sampler starts from the target full-view RGB rather than an empty latent'])
    if args.target_alpha:
        record['targetAlpha'] = {'path': str(args.target_alpha.resolve(strict=True)),
                                 'sha256': sha(args.target_alpha),
                                 'usage': 'Registration/QA only; not fed to Qwen and never replaces Box alpha'}
    if args.box_manifest:
        record['boxManifest'] = {'path': str(args.box_manifest.resolve(strict=True)),
                                 'sha256': sha(args.box_manifest)}
    prepared = args.output / 'prepared.json'
    if prepared.exists():
        assert json.loads(prepared.read_text()) == record, 'Prepared inputs changed; use a new output directory'
    prompt = args.output / 'prompt.json'
    if prompt.exists():
        assert json.loads(prompt.read_text()) == graph, 'Prompt changed; use a new output directory'
    write(prepared, record)
    write(prompt, graph)
    return graph, record


def memory_summary(path):
    values = []
    for row in csv.reader(path.read_text(errors='replace').splitlines()):
        try:
            values.append(float(row[2]))
        except (ValueError, IndexError):
            pass
    return {'measurement': 'nvidia-smi sampled whole-device used VRAM',
            'intervalMilliseconds': 200, 'sampleCount': len(values),
            'peakSampledUsedMiB': max(values) if values else None,
            'limitation': 'Includes driver/other processes; not allocator peak; may miss short peaks.'}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--face', choices=tuple(FACE_IDS), required=True, help='Canonical Box face; required for prepare, execute and collect')
    p.add_argument('--prompt-file', type=Path, help='Optional complete UTF-8 prompt override; frozen in prompt.json')
    p.add_argument('--target-rgb', type=Path)
    p.add_argument('--target-depth', type=Path)
    p.add_argument('--target-alpha', type=Path)
    p.add_argument('--box-manifest', type=Path)
    p.add_argument('--original', type=Path, default=DIRECT3D / 'inputs/fullbody/source-original.png')
    p.add_argument('--template', type=Path, default=TEMPLATE)
    p.add_argument('--comfy-input', type=Path, default=Path('/home/dev/Projects/box-impostor-lab/ComfyUI/input'))
    p.add_argument('--input-prefix', help='Unique relative Comfy input path; defaults to experiment/face/output-directory-name')
    p.add_argument('--size', type=int, choices=(1024, 2048), default=1024)
    p.add_argument('--profile', choices=('legacy', 'anchored'), default='anchored', help='anchored uses VAE-encoded target RGB; legacy uses empty latent at denoise 1')
    p.add_argument('--denoise', type=float, default=.9, help='Applied only by the anchored profile')
    p.add_argument('--lora-strength', type=float, default=0., help='AnyAngle model strength; default 0 disables it')
    p.add_argument('--server', default='http://127.0.0.1:8198')
    p.add_argument('--timeout-seconds', type=float, default=7200)
    p.add_argument('--gpu', default='0')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--collect', metavar='PROMPT_ID')
    args = p.parse_args()
    if not 0 < args.denoise <= 1:
        p.error('--denoise must be in (0,1]')
    if not 0 <= args.lora_strength <= 1:
        p.error('--lora-strength must be in [0,1]')
    args.output = args.output.resolve()
    if args.input_prefix is None:
        args.input_prefix = 'direct3d-qwen-box-20261121/' + args.face + '/' + args.output.name
    args.output.mkdir(parents=True, exist_ok=True)
    if args.collect:
        graph = json.loads((args.output / 'prompt.json').read_text())
        prepared = json.loads((args.output / 'prepared.json').read_text())
        assert prepared['faceSlug'] == args.face and prepared['face'] == FACE_IDS[args.face], 'Collect face differs from frozen preparation'
        receipt = json.loads((args.output / 'submitted.json').read_text())
        assert receipt['promptId'] == args.collect
        assert sha(args.output / 'prompt.json') == receipt['promptSha256']
    else:
        if args.execute and (args.output / 'submitted.json').exists():
            raise RuntimeError('Already submitted; use --collect with its promptId, never resubmit this run')
        graph, prepared = prepare(args)
        if not args.execute:
            print(json.dumps({'prepared': str(args.output), 'submitted': False}), flush=True)
            return
    prompt_id = args.collect
    monitor = stream = errors = None
    client_started = time.monotonic()
    try:
        if args.execute:
            queue = request(args.server, '/queue')
            write(args.output / 'queue-before.json', queue)
            if queue.get('queue_running') or queue.get('queue_pending'):
                raise RuntimeError('Comfy queue is not idle')
            stream = (args.output / 'gpu-memory.csv').open('w')
            errors = (args.output / 'gpu-memory.stderr.txt').open('w')
            monitor = subprocess.Popen(['nvidia-smi', '-i', args.gpu,
                '--query-gpu=timestamp,index,memory.used,memory.total', '--format=csv,noheader,nounits',
                '--loop-ms=200'], stdout=stream, stderr=errors)
            submitted_at = time.time()
            reply = request(args.server, '/prompt', {'prompt': graph, 'client_id': 'direct3d-qwen-' + str(uuid.uuid4())})
            prompt_id = reply.get('prompt_id')
            receipt = {'promptId': prompt_id, 'serverReply': reply, 'submittedAtUnixSeconds': submitted_at,
                       'promptSha256': sha(args.output / 'prompt.json')}
            write(args.output / 'submitted.json', receipt)
            if not prompt_id or reply.get('node_errors'):
                raise RuntimeError('Prompt rejected: ' + json.dumps(reply))
            print(json.dumps({'submitted': prompt_id, 'output': str(args.output)}), flush=True)
        last_notice = 0
        while True:
            history = request(args.server, '/history/' + urllib.parse.quote(prompt_id, safe=''))
            if prompt_id in history:
                entry = history[prompt_id]
                write(args.output / 'history.json', entry)
                if entry.get('status', {}).get('completed') or entry.get('status', {}).get('status_str') == 'error':
                    break
            elapsed = time.monotonic() - client_started
            if elapsed >= args.timeout_seconds:
                raise TimeoutError('Collection timeout; inference was not interrupted. Resume using --collect ' + prompt_id)
            if elapsed - last_notice >= 30:
                print(json.dumps({'waiting': prompt_id, 'clientWaitSeconds': round(elapsed, 1)}), flush=True)
                last_notice = elapsed
            time.sleep(2)
        client_wait = time.monotonic() - client_started
    finally:
        if monitor is not None:
            monitor.terminate()
            try:
                monitor.wait(timeout=5)
            except subprocess.TimeoutExpired:
                monitor.kill()
                monitor.wait()
            stream.close()
            errors.close()
            write(args.output / 'gpu-memory-summary.json', memory_summary(args.output / 'gpu-memory.csv'))
    assert entry['status']['status_str'] == 'success', 'Comfy execution failed; inspect history.json'
    actual = entry['prompt'][2]
    assert actual == graph, 'Executed graph differs from submitted graph (numeric int/float equality allowed)'
    write(args.output / 'execution-prompt.json', actual)
    messages = entry['status'].get('messages', [])
    cached = [str(node) for event, data in messages if event == 'execution_cached' for node in data.get('nodes', [])]
    assert '12' not in cached, 'Sampler was cached; this is not a fresh Qwen edit'
    images = {}
    for node, kind in [('14', 'generated'), ('18', 'mask')]:
        records = entry.get('outputs', {}).get(node, {}).get('images', [])
        assert len(records) == 1, f'Expected one image at SaveImage node {node}'
        record = records[0]
        query = urllib.parse.urlencode({k: record.get(k, '') for k in ('filename', 'subfolder', 'type')})
        destination = args.output / (kind + '.png')
        with urllib.request.urlopen(args.server.rstrip('/') + '/view?' + query, timeout=90) as response:
            with destination.open('wb') as out:
                shutil.copyfileobj(response, out)
        with Image.open(destination) as image:
            assert image.size == (prepared['size'], prepared['size']), 'Generated dimensions differ'
        images[kind] = {'path': str(destination), 'sha256': sha(destination), 'bytes': destination.stat().st_size}
    timestamps = {event: data.get('timestamp') for event, data in messages if isinstance(data, dict)}
    start, end = timestamps.get('execution_start'), timestamps.get('execution_success')
    result = {'completed': True, 'case': 'Direct3D-V3 Box face colorization candidate',
        'face': prepared['face'], 'faceSlug': prepared['faceSlug'], 'promptId': prompt_id,
        'promptSHA256': sha(args.output / 'prompt.json'), 'images': images,
        'model': 'Qwen Image 2.1 int8', 'lora': 'QI2.1_AnyAngle', 'loraStrength': prepared['loraStrength'],
        'size': prepared['size'], 'steps': graph['12']['inputs']['steps'],
        'cfg': graph['12']['inputs']['cfg'], 'seed': graph['12']['inputs']['seed'],
        'sampler': graph['12']['inputs']['sampler_name'], 'scheduler': graph['12']['inputs']['scheduler'],
        'prompt': graph['10']['inputs']['prompt'], 'promptSource': prepared['promptSource'],
        'defaults': prepared['defaults'], 'inputOrder': prepared['inputOrder'],
        'server': prepared['server'], 'runnerSha256': sha(__file__),
        'profile': prepared.get('profile', 'legacy'), 'denoise': prepared.get('denoise', 1),
        'initialLatent': prepared.get('initialLatent', 'zeros from TextEncodeQwenImage21'),
        'serverExecutionSeconds': (end - start) / 1000 if start is not None and end is not None else None,
        'clientWaitSeconds': client_wait, 'collectionOnly': bool(args.collect),
        'cachedNodes': cached, 'inputsSHA256': {k: v['sha256'] for k, v in prepared['inputs'].items()},
        'vram': json.loads((args.output / 'gpu-memory-summary.json').read_text())
                if (args.output / 'gpu-memory-summary.json').exists() else None,
        'boxModified': False,
        'nextStage': 'Review this full-view candidate, register generated RGB/mask against its guide, and compose only selected face RGB while preserving Direct3D structural alpha and geometry.'}
    write(args.output / 'result.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
