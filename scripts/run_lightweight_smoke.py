"""Exercise six real img2img passes and the final Box renderer on a running ComfyUI.

Put the test GLB under ComfyUI/input and an SD 1.x checkpoint under
ComfyUI/models/checkpoints. Then run, from the custom node directory:

  python scripts/run_lightweight_smoke.py --mesh asymmetric.glb --checkpoint model.safetensors --run

Without --run this writes the API graph and checks installed nodes and checkpoint.
The test uses one checkpoint and one image per sampler. The guide supplies coverage,
so this exercises appearance generation and integration, not silhouette prediction.
"""
import argparse
import hashlib
import io
import json
import re
import tempfile
import time
from pathlib import Path, PurePosixPath
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlencode, urlsplit
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image

FACES = ('+X', '-X', '+Y', '-Y', '+Z', '-Z')
SOCKETS = ('px', 'nx', 'py', 'ny', 'pz', 'nz')
DEFAULT_CHECKPOINT = 'v1-5-pruned-emaonly-fp16.safetensors'
DEFAULT_PROMPT = 'a cobalt blue ceramic sculpture, detailed painted turquoise and gold patterns, studio product photograph'
MATERIAL_TEXTURES = {'none': (), 'material': ('materialMap',), 'emissive': ('emissiveMap',),
                     'material+emissive': ('materialMap', 'emissiveMap')}
GENERATOR = 'comfy-native-checkpoint-img2img'


def node(class_type, **inputs):
    return {'class_type': class_type, 'inputs': inputs}


def build_graph(mesh, checkpoint, resolution, steps, seed, denoise, backend, *,
                positive_prompt=DEFAULT_PROMPT, material_maps='none'):
    bake = dict(proxy=['1', 0], resolution=resolution, backend=backend, material_maps=material_maps)
    graph = {
        '1': node('BICanonicalizeProxy', filename=mesh),
        '2': node('BIMakeSixGuides', **bake),
        '3': node('CheckpointLoaderSimple', ckpt_name=checkpoint),
        '4': node('CLIPTextEncode', clip=['3', 1], text=positive_prompt),
        '5': node('CLIPTextEncode', clip=['3', 1], text='blurry, low detail, text, watermark'),
        '6': node('BIBakeBox', **bake),
    }
    captured = {}
    for index, (face, socket) in enumerate(zip(FACES, SOCKETS)):
        split, encode, sample, decode, capture = (str(20 + index * 5 + j) for j in range(5))
        graph[split] = node('SplitImageWithAlpha', image=['2', index])
        graph[encode] = node('VAEEncode', pixels=[split, 0], vae=['3', 2])
        graph[sample] = node('KSampler', model=['3', 0], positive=['4', 0], negative=['5', 0],
                             latent_image=[encode, 0], seed=seed + index, steps=steps, cfg=6.0,
                             sampler_name='euler', scheduler='normal', denoise=denoise)
        graph[decode] = node('VAEDecode', samples=[sample, 0], vae=['3', 2])
        provenance = dict(backend='manual_faces', generator=GENERATOR, checkpoint=checkpoint,
                          face=face, seed=seed + index, steps=steps, denoise=denoise, coverage='guide',
                          positive_prompt=positive_prompt)
        graph[capture] = node('BIFaceImage', image=[decode, 0], mask_mode='load_image',
                              mask=[split, 1], provenance=json.dumps(provenance, sort_keys=True))
        captured[socket] = [capture, 0]
    graph['7'] = node('BIFaceInputs', **captured)
    graph['8'] = node('BIAlignAndValidateFaces', faces=['7', 0], guides=['2', 6],
                      min_iou=0.85, max_shift=0.1, max_scale_change=0.15)
    graph['9'] = node('BIBakeBox', **bake, faces=['8', 0])
    graph['10'] = node('BIPreviewBox', asset=['9', 0])
    for asset, render, join, save, label in [('6', '11', '12', '13', 'baseline'), ('9', '14', '15', '16', 'generated')]:
        graph[render] = node('BIRenderBox', asset=[asset, 0], width=resolution, height=resolution,
                             view='diagonal', projection='orthographic', sampling='standard', backend=backend)
        graph[join] = node('JoinImageWithAlpha', image=[render, 0], alpha=[render, 1])
        graph[save] = node('SaveImage', images=[join, 0], filename_prefix='box-smoke/' + label)
    return graph


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def request(origin, path, data=None, *, limit=None):
    payload = json.dumps(data).encode() if data is not None else None
    req = Request(origin.rstrip('/') + path, data=payload, headers={'Content-Type': 'application/json'})
    try:
        with urlopen(req, timeout=60) as response:
            return response.read(limit)
    except HTTPError as exc:
        raise RuntimeError(f'ComfyUI {path} returned HTTP {exc.code}: {exc.read().decode(errors="replace")[:4000]}') from exc


def preflight(origin, graph, checkpoint):
    registry = json.loads(request(origin, '/object_info'))
    missing = sorted({n['class_type'] for n in graph.values()} - registry.keys())
    if missing:
        raise ValueError(f'Missing ComfyUI nodes: {", ".join(missing)}')
    spec = registry['CheckpointLoaderSimple']['input']['required']['ckpt_name']
    choices = spec[0] if isinstance(spec[0], list) else spec[1].get('options', [])
    if checkpoint not in choices:
        raise ValueError(f'Checkpoint {checkpoint!r} is missing. Registered checkpoints: {choices!r}')
    mesh = graph['1']['inputs']['filename']
    path = PurePosixPath(mesh)
    if (path.is_absolute() or '..' in path.parts or ':' in mesh or chr(92) in mesh
            or path.as_posix() != mesh or path.suffix.lower() != '.glb'):
        raise ValueError(f'Expected a ComfyUI input-relative GLB path with forward slashes: {mesh!r}')
    query = urlencode(dict(filename=path.name, subfolder='' if len(path.parts) == 1 else str(path.parent), type='input'))
    header = request(origin, '/view?' + query, limit=12)
    if (len(header) != 12 or header[:8] != b'glTF\x02\x00\x00\x00'
            or int.from_bytes(header[8:12], 'little') < 20):
        raise ValueError('ComfyUI input is not a GLB v2 file: ' + mesh)
    return json.loads(request(origin, '/system_stats')).get('devices', [])


def wait_history(origin, prompt_id, timeout_minutes):
    deadline = time.monotonic() + timeout_minutes * 60
    while time.monotonic() < deadline:
        entry = json.loads(request(origin, '/history/' + prompt_id)).get(prompt_id)
        if entry:
            if entry['status']['status_str'] != 'success':
                raise RuntimeError(json.dumps(entry['status'], ensure_ascii=False)[-8000:])
            return entry
        time.sleep(1)
    raise TimeoutError(f'Prompt {prompt_id} still pending after {timeout_minutes:g} min')


def checked_blob(origin, directory, descriptor):
    data = request(origin, directory + descriptor['path'])
    if hashlib.sha256(data).hexdigest() != descriptor['sha256']:
        raise ValueError('Digest mismatch: ' + descriptor['path'])
    return data


def changed_faces(before, after, resolution):
    shape = (resolution * 2, resolution * 3, 4)
    original = np.frombuffer(before, dtype='<f2').reshape(shape)
    updated = np.frombuffer(after, dtype='<f2').reshape(shape)
    if not np.array_equal(original[..., 3].view('<u2'), updated[..., 3].view('<u2')):
        raise ValueError('Color atlas coverage changed')
    changed = np.any(original[..., :3] != updated[..., :3], axis=-1)
    counts = {}
    for i, name in enumerate(FACES):
        row, col = divmod(i, 3)
        tile = np.s_[row*resolution:(row+1)*resolution, col*resolution:(col+1)*resolution]
        counts[name] = int((changed[tile] & (original[tile][..., 3] > 0)).sum())
        if not counts[name]:
            raise ValueError(f'No generated color reached covered texels on face {name}')
    return counts


def verify(origin, outputs, directory, args):
    preview, = outputs['10']['box_impostor']
    manifest_path, = parse_qs(urlsplit(preview).query)['manifest']
    if not re.fullmatch(r'/box-impostor/assets/[0-9a-f]{64}/manifest\.json', manifest_path):
        raise ValueError('Unexpected preview manifest path')
    asset_dir = manifest_path.rsplit('/', 1)[0] + '/'
    final = json.loads(request(origin, manifest_path))
    parent = final['parentAsset']
    if not re.fullmatch('[0-9a-f]{64}', parent):
        raise ValueError('Invalid baseline asset id')
    base_dir = '/box-impostor/assets/' + parent + '/'
    base = json.loads(request(origin, base_dir + 'manifest.json'))
    for name, manifest in [('baseline', base), ('generated', final)]:
        write_json(directory / (name + '-manifest.json'), manifest)
    if tuple(final['faceOrder']) != FACES or final['resolution'] != args.resolution:
        raise ValueError('Unexpected face order or resolution')
    geometry_keys = ('normalMap', 'intervalMap', 'rangeMap')
    material_keys = MATERIAL_TEXTURES[args.material_maps]
    expected_textures = {'colorMap', *geometry_keys, *material_keys}
    if any(set(m['textures']) != expected_textures for m in (base, final)):
        raise ValueError(f'Expected baseline and generated textures: {sorted(expected_textures)}')
    preserved = {}
    for key in (*geometry_keys, *material_keys):
        descriptor = final['textures'][key]
        if descriptor != base['textures'][key]:
            raise ValueError(f'{key} descriptor differs from baseline')
        checked_blob(origin, asset_dir, descriptor)
        checked_blob(origin, base_dir, descriptor)
        preserved[key] = descriptor['sha256']
    for index, (face, slug) in enumerate(zip(FACES, SOCKETS)):
        edit = final['faces'][face]['edit']
        info = edit['provenance']
        expected = dict(generator=GENERATOR, seed=args.seed + index, face=face, checkpoint=args.checkpoint,
                        positive_prompt=args.positive_prompt)
        if edit['status'] != 'accepted' or any(info.get(k) != v for k, v in expected.items()):
            raise ValueError(f'Generated face {face} was not accepted with the expected provenance')
        (directory / (slug + '.png')).write_bytes(request(origin, asset_dir + edit['view']))
    counts = changed_faces(checked_blob(origin, base_dir, base['textures']['colorMap']),
                           checked_blob(origin, asset_dir, final['textures']['colorMap']), args.resolution)
    frames = []
    for name, node_id in [('baseline', '13'), ('generated', '16')]:
        record, = outputs[node_id]['images']
        path = '/view?' + urlencode({k: record[k] for k in ('filename', 'type', 'subfolder')})
        data = request(origin, path)
        (directory / (name + '-render.png')).write_bytes(data)
        with Image.open(io.BytesIO(data)) as image:
            if image.mode != 'RGBA' or image.size != (args.resolution, args.resolution):
                raise ValueError('Box render did not return an RGBA image of the requested size')
            frames.append(np.asarray(image).copy())
    a, b = frames
    if not np.array_equal(a[..., 3], b[..., 3]):
        raise ValueError('Box render coverage changed')
    visible = a[..., 3] > 0
    pixels = int((visible & np.any(a[..., :3] != b[..., :3], axis=-1)).sum())
    if not pixels:
        raise ValueError('No generated color reached the final Box render')
    return dict(viewer=origin.rstrip('/') + preview, geometrySha256={k: preserved[k] for k in geometry_keys},
                materialSha256={k: preserved[k] for k in material_keys}, colorTexelsChanged=counts,
                renderedPixelsChanged=pixels, renderedPixelsCovered=int(visible.sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mesh', required=True, help='GLB path relative to the ComfyUI input directory')
    parser.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT, help='Installed SD 1.x checkpoint name')
    parser.add_argument('--positive-prompt', default=DEFAULT_PROMPT, help='Appearance prompt used on all six views')
    parser.add_argument('--material-maps', choices=MATERIAL_TEXTURES, default='none',
                        help='Bake these PBR maps and verify their bytes survive six-face color editing')
    parser.add_argument('--origin', default='http://127.0.0.1:8188')
    parser.add_argument('--output', type=Path, default=Path('.test-output/lightweight'))
    parser.add_argument('--resolution', type=int, choices=(128, 256, 512), default=256)
    parser.add_argument('--steps', type=int, default=6)
    parser.add_argument('--denoise', type=float, default=.5)
    parser.add_argument('--seed', type=int, default=19570)
    parser.add_argument('--backend', choices=('auto', 'webgl2', 'webgpu'), default='auto')
    parser.add_argument('--run', action='store_true', help='Submit the graph to ComfyUI after preflight')
    parser.add_argument('--timeout-minutes', type=float, default=30)
    args = parser.parse_args()
    if not (1 <= args.steps <= 100 and 0 < args.denoise <= 1 and 0 <= args.seed <= 2**64-6 and args.timeout_minutes > 0):
        parser.error('Check --steps (1..100), --denoise (0..1, nonzero), --seed, and --timeout-minutes')
    args.output.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix='run-', dir=args.output))
    graph = build_graph(args.mesh, args.checkpoint, args.resolution, args.steps, args.seed, args.denoise, args.backend,
                        positive_prompt=args.positive_prompt, material_maps=args.material_maps)
    write_json(directory / 'api-prompt.json', graph)
    print(directory.resolve(), flush=True)
    proof = dict(checkpoint=args.checkpoint, mesh=args.mesh, positivePrompt=args.positive_prompt,
                 materialMaps=args.material_maps, status='preflight')
    try:
        proof['devices'] = preflight(args.origin, graph, args.checkpoint)
        if args.run:
            receipt = json.loads(request(args.origin, '/prompt', {'prompt': graph}))
            proof.update(promptId=receipt['prompt_id'], status='queued')
            write_json(directory / 'result.json', proof)
            entry = wait_history(args.origin, proof['promptId'], args.timeout_minutes)
            samplers = {node_id for node_id, item in graph.items() if item['class_type'] == 'KSampler'}
            cached = {str(node_id) for name, details in entry['status'].get('messages', [])
                      if name == 'execution_cached' for node_id in details.get('nodes', [])}
            proof['samplerIds'] = sorted(samplers)
            proof['samplerIdsCachedAtStart'] = sorted(cached & samplers)
            proof.update(verify(args.origin, entry['outputs'], directory, args), status='passed')
        else:
            proof['status'] = 'preflight-passed'
    except Exception as exc:
        proof.update(status='failed', error=str(exc))
        raise
    finally:
        write_json(directory / 'result.json', proof)
    print(json.dumps(proof, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
