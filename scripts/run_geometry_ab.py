"""Run fresh six-view H3 comparisons with explicit image-reference structure guides.

The five pictures are RGBA-derived color, source photo, axis depth, world normals,
and foreground mask. They are H3 image-reference conditioning, not a ControlNet.
No material estimation or view reprojection is applied.
"""
import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import shutil
import sys
import time
import urllib.error
import urllib.request
import uuid

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
COMFY = REPO.parents[1]
sys.path[:0] = [str(COMFY), str(REPO)]
import folder_paths
from box_nodes import canvas, faces, store

SLUGS = faces.SLUGS
SEEDS = {fid: 101 + n for n, fid in enumerate(store.FACES)}
MODELS = {
    'model': 'minimax_h3_image_pruned_w4a8.safetensors',
    'clip': 'qwen3vl_32b_minimax_h3_int8_convrot.safetensors',
    'vae': 'minimax_h3_image_vae_fp16.safetensors',
    'matting': 'birefnet.safetensors',
}
PROMPT = (
    "Refine the person in <Picture 1> into a detailed, natural photograph of the "
    "same person shown in <Picture 2>. Preserve the exact orthographic camera, "
    "view direction, pose, silhouette, scale and framing of <Picture 1>. "
    "<Picture 3> is the aligned geometry depth: brighter means nearer the camera. "
    "<Picture 4> is the aligned world-space surface normal map. "
    "<Picture 5> is the aligned foreground mask: white is the person and black is background. "
    "Use Pictures 3, 4 and 5 only as structural references; do not copy their colors. "
    "Use Picture 2 only for identity, skin, hair, dress colors and material details. "
    "Keep the direction of the head and the visibility of facial features consistent with "
    "Picture 1 and the structural maps. Add anatomically plausible eyes, eyelids, nose, lips, "
    "hair strands, fingers, seams and fabric folds where they are visible from this camera. "
    "Keep the body and garment boundaries exactly within the mask. Use neutral even lighting "
    "and a flat mid-gray background. Return only one full-body orthographic image, no panels, "
    "no diagrams, no text, no colored depth or normal-map appearance."
)


VIEW_INSTRUCTIONS = {
    '+X': 'The orthographic camera is on the +X side, perpendicular to the front view. '
          'Return the strict side profile shown in <Picture 1>; preserve its visible side and nose direction. '
          'Do not turn the person or face toward the camera. ',
    '-X': 'The orthographic camera is on the -X side, opposite the +X side and perpendicular to the front view. '
          'Return the strict side profile shown in <Picture 1>; preserve its visible side and nose direction. '
          'Do not turn the person or face toward the camera. ',
    '-Z': 'The orthographic camera is directly behind the person at -Z. Return a strict rear view, '
          'with the back of the head, hair, body and dress as visible in <Picture 1>. '
          'Do not turn the head toward the camera or add a front-facing face. ',
    '+Y': 'The orthographic camera is directly above the person at +Y, looking straight downward. '
          'Preserve the overhead foreshortening, layout and occlusion in <Picture 1>. '
          'Show only surfaces visible from above; do not convert this into an upright front view. ',
    '-Y': 'The orthographic camera is directly below the person at -Y, looking straight upward. '
          'Preserve the underside foreshortening, layout and occlusion in <Picture 1>. '
          'Show only surfaces visible from below; do not convert this into an upright front view. ',
}


def face_prompt(fid):
    if fid == '+Z':
        return PROMPT
    prompt = PROMPT
    if fid in ('+Y', '-Y'):
        prompt = prompt.replace('one full-body orthographic image',
                                'one orthographic image of only the surfaces visible from that direction')
    return VIEW_INSTRUCTIONS[fid] + prompt


def use_cuda_runtime_on_wsl():
    # WSL driver NVML can segfault while the CUDA runtime remains usable.
    if 'microsoft' in platform.release().lower():
        import torch
        torch.cuda._device_count_nvml = lambda: -1


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save_rgb(path, data):
    Image.fromarray(np.rint(np.clip(data, 0, 1) * 255).astype('u1')).save(path)


def content_key(image):
    samples = np.ascontiguousarray(image, dtype='<f4')
    return store.key(samples.shape, hashlib.sha256(samples).hexdigest())


def structure_images(directory, manifest, fid, size=1024):
    """Return five spatially registered conditioning pictures plus canvas metadata."""
    info = manifest['faces'][fid]
    r = manifest['resolution']
    rgba = np.asarray(Image.open(directory / info['guide']).convert('RGBA'), dtype='f4') / 255
    z = np.fromfile(directory / info['depth'], '<f2').reshape(r, r).astype('f4')
    if info.get('depthEncoding') != 'linear-axis-0-near-1-far;nan-background':
        raise ValueError('Axis depth encoding is required.')
    span = np.subtract(manifest['bakeBox']['max'], manifest['bakeBox']['min'])
    width, height = (span[i] for i in canvas.SPAN_AXES[fid[1]])
    rect = canvas.letterbox_rect(width, height, size)
    ref = manifest['reference']
    ref_name = ref.get('camera', {}).get('image', ref['image'])
    reference = np.asarray(Image.open(directory / ref_name).convert('RGBA'), dtype='f4') / 255
    source_rect = canvas.letterbox_rect(reference.shape[1], reference.shape[0], size)
    metadata = {
        'version': 1, 'size': size, 'faceId': fid, 'guideRect': rect,
        'guideInputSha256': content_key(rgba),
        'referenceInputSha256': content_key(reference), 'sourceRect': source_rect,
        'physicalWidthHeight': [float(width), float(height)],
        'imageRowOrder': 'top-to-bottom', 'backgroundSRGB': .5,
    }
    alpha = rgba[..., 3:4]
    valid = np.isfinite(z)
    if not np.array_equal(valid, alpha[..., 0] >= .5):
        raise ValueError('Depth and guide foreground do not share coverage.')
    depth = np.repeat(np.where(valid, 1 - z, .5)[..., None], 3, axis=-1)
    descriptor = manifest['textures']['normalMap']
    normals = np.memmap(directory / descriptor['path'], '<f2', mode='r',
                        shape=(descriptor['height'], descriptor['width'], 4))
    normal = faces.tile_view(normals, fid)[::-1, :, :3].astype('f4')
    if not np.isfinite(normal).all() or normal.min() < -.001 or normal.max() > 1.001:
        raise ValueError('World normal map must be encoded in [0,1].')
    mask = np.zeros((size, size, 3), dtype='f4')
    x, y, w, h = rect
    a = np.asarray(Image.fromarray(alpha[..., 0]).resize((w, h), Image.Resampling.BILINEAR))
    mask[y:y+h, x:x+w] = a[..., None]
    pictures = {
        'guide': canvas.place(rgba, rect, size)[0],
        'source': canvas.place(reference, source_rect, size)[0],
        'axis-depth': canvas.place(np.concatenate((depth, alpha), axis=-1), rect, size)[0],
        'normals': canvas.place(np.concatenate((normal, alpha), axis=-1), rect, size)[0],
        'mask': mask,
    }
    return pictures, metadata


def prepare(run, branch, baseline_path):
    source = Path(baseline_path).resolve()
    manifest = read_json(source / 'manifest.json')
    if manifest['resolution'] != 1024:
        raise ValueError('A/B comparison requires a 1024px baseline bake.')
    target = run / 'output' / 'box_impostor' / manifest['asset_id']
    if not target.exists():
        shutil.copytree(source, target)
    if digest(target / manifest['proxy']) != manifest['proxySha256']:
        raise ValueError('Baseline proxy hash mismatch.')
    case = run / branch
    case.mkdir(exist_ok=True)
    recipe = {
        'branch': branch, 'baseline': manifest['asset_id'],
        'baselinePath': str(target), 'coarseGeometry': 'fixed Pixal3D / TRELLIS2 stages',
        'proxySha256': manifest['proxySha256'], 'bakeBox': manifest['bakeBox'],
        'tightBounds': manifest['tightBounds'], 'transform': manifest['transform'],
        'referenceSha256': manifest['reference']['sha256'],
        'conditioning': 'H3 image references (VLM plus VAE reference latents); no ControlNet',
        'pictureOrder': ['guide', 'source', 'axis-depth', 'normals', 'mask'],
        'pictureSize': [1024, 1024], 'outputSize': [1024, 1024],
        'depthEncoding': '1 minus normalized axis depth, bright-near; gray background',
        'normalEncoding': 'world normal * .5 + .5; RGB xyz; gray background',
        'maskEncoding': 'white foreground, black background',
        'models': MODELS, 'seeds': SEEDS,
        'sampler': {'steps': 6, 'cfg': 1, 'sampler_name': 'euler',
                    'scheduler': 'simple', 'denoise': 1},
        'prompt': PROMPT, 'viewConsistency': 'none', 'materialMode': 'keep',
        'faces': {},
    }
    for fid, slug in SLUGS.items():
        folder = run / 'input' / branch / slug
        folder.mkdir(parents=True, exist_ok=True)
        pictures, meta = structure_images(target, manifest, fid)
        paths = {}
        for role, data in pictures.items():
            path = folder / (role + '.png')
            save_rgb(path, data)
            paths[role] = {'image': str(path.relative_to(run / 'input')),
                           'path': str(path), 'sha256': digest(path)}
        recipe['faces'][fid] = {'canvas': meta, 'pictures': paths, 'prompt': face_prompt(fid),
                               'guideSha256': manifest['faces'][fid]['guideSha256'],
                               'viewMatrix': manifest['faces'][fid]['viewMatrix'],
                               'projectionMatrix': manifest['faces'][fid]['projectionMatrix']}
    write_json(case / 'prepared.json', recipe)
    print(json.dumps({'prepared': str(case / 'prepared.json'), 'baseline': manifest['asset_id']}), flush=True)
    return recipe


def build_prompt(recipe, fid):
    pictures = recipe['faces'][fid]['pictures']
    graph = {}
    models = recipe['models']
    for n, role in enumerate(recipe['pictureOrder'], 1):
        graph[str(n)] = {'class_type': 'LoadImage', 'inputs': {'image': pictures[role]['image']}}
    graph.update({
        '6': {'class_type': 'UNETLoader', 'inputs': {'unet_name': models['model'], 'weight_dtype': 'default'}},
        '7': {'class_type': 'CLIPLoader', 'inputs': {'clip_name': models['clip'], 'type': 'minimax', 'device': 'default'}},
        '8': {'class_type': 'VAELoader', 'inputs': {'vae_name': models['vae']}},
        '10': {'class_type': 'MiniMaxH3ImageConditioning', 'inputs': {
            'clip': ['7', 0], 'vae': ['8', 0], 'prompt': recipe['faces'][fid]['prompt'], 'width': 1024, 'height': 1024,
            **{'images.image_' + str(n): [str(n), 0] for n in range(1, 6)}}},
        '11': {'class_type': 'ConditioningZeroOut', 'inputs': {'conditioning': ['10', 0]}},
        '12': {'class_type': 'KSampler', 'inputs': {**recipe['sampler'], 'seed': recipe['seeds'][fid],
            'model': ['6', 0], 'positive': ['10', 0], 'negative': ['11', 0], 'latent_image': ['10', 1]}},
        '13': {'class_type': 'VAEDecode', 'inputs': {'samples': ['12', 0], 'vae': ['8', 0]}},
        '14': {'class_type': 'SaveImage', 'inputs': {'images': ['13', 0],
            'filename_prefix': recipe['branch'] + '/' + SLUGS[fid] + '/raw'}},
        '15': {'class_type': 'LoadBackgroundRemovalModel', 'inputs': {'bg_removal_name': models['matting']}},
        '16': {'class_type': 'RemoveBackground', 'inputs': {'bg_removal_model': ['15', 0], 'image': ['13', 0]}},
        '17': {'class_type': 'MaskToImage', 'inputs': {'mask': ['16', 0]}},
        '18': {'class_type': 'SaveImage', 'inputs': {'images': ['17', 0],
            'filename_prefix': recipe['branch'] + '/' + SLUGS[fid] + '/mask'}},
    })
    return graph


def request(origin, route, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(origin + route, data=data,
                                 headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(error.read().decode()) from error


def run_face(run, recipe, fid, origin):
    case = run / recipe['branch'] / SLUGS[fid]
    case.mkdir(parents=True, exist_ok=True)
    prompt = build_prompt(recipe, fid)
    for picture in recipe['faces'][fid]['pictures'].values():
        if digest(picture['path']) != picture['sha256']:
            raise ValueError('Conditioning picture changed after preparation: ' + picture['path'])
    if (case / 'result.json').exists():
        previous = read_json(case / 'result.json')
        saved_prompt = case / 'prompt.json'
        if (read_json(saved_prompt) != prompt or digest(saved_prompt) != previous['promptSha256']
                or digest(case / 'generated.png') != previous['rawSha256']
                or digest(case / 'foreground.png') != previous['maskSha256']):
            raise ValueError('Completed face belongs to different inputs or settings; use a new run directory.')
        print(json.dumps({'alreadyCompleted': str(case / 'result.json')}), flush=True)
        return
    write_json(case / 'prompt.json', prompt)
    submitted = request(origin, '/prompt', {'prompt': prompt, 'client_id': str(uuid.uuid4()),
        'extra_data': {'geometryAB': recipe['branch'], 'face': fid}})
    if submitted.get('node_errors'):
        raise RuntimeError(json.dumps(submitted))
    pid = submitted['prompt_id']
    start = time.time()
    write_json(case / 'submitted.json', {'prompt_id': pid, 'startEpoch': start, 'request': submitted})
    print(json.dumps({'submitted': pid, 'face': fid, 'branch': recipe['branch']}), flush=True)
    while True:
        history = request(origin, '/history/' + pid)
        if pid in history:
            result = history[pid]
            write_json(case / 'history.json', result)
            if not result.get('status', {}).get('completed'):
                raise RuntimeError(json.dumps(result.get('status')))
            break
        if time.time() - start > 7200:
            raise TimeoutError('H3 face still running after two hours: ' + pid)
        time.sleep(3)
    def saved(node):
        image = result['outputs'][node]['images'][0]
        if image['type'] != 'output':
            raise ValueError('Expected an output image.')
        return run / 'output' / image['subfolder'] / image['filename']
    raw, matte = saved('14'), saved('18')
    shutil.copyfile(raw, case / 'generated.png')
    shutil.copyfile(matte, case / 'foreground.png')
    rgb = np.asarray(Image.open(raw).convert('RGB'), dtype='f4') / 255
    mask = np.asarray(Image.open(matte).convert('L'), dtype='f4') / 255
    if rgb.shape != (1024, 1024, 3) or mask.shape != (1024, 1024):
        raise ValueError('Unexpected generated image size.')
    save_rgb(case / 'generated-rgba.png', np.concatenate((rgb, mask[..., None]), axis=-1))
    measured = {'prompt_id': pid, 'freshInference': True, 'face': fid,
                'elapsedSeconds': round(time.time() - start, 2),
                'raw': str(case / 'generated.png'), 'mask': str(case / 'foreground.png'),
                'rawSha256': digest(raw), 'maskSha256': digest(matte),
                'promptSha256': digest(case / 'prompt.json')}
    write_json(case / 'result.json', measured)
    print(json.dumps({'completed': measured}), flush=True)


def compose(run, recipe):
    folder_paths.set_output_directory(str(run / 'output'))
    baseline = recipe['baseline']
    original = read_json(Path(recipe['baselinePath']) / 'manifest.json')
    candidates = {}
    for fid, slug in SLUGS.items():
        directory = run / recipe['branch'] / slug
        if not (directory / 'result.json').exists():
            continue
        result = read_json(directory / 'result.json')
        rgb = np.asarray(Image.open(directory / 'generated.png').convert('RGB'), dtype='f4') / 255
        mask = np.asarray(Image.open(directory / 'foreground.png').convert('L'), dtype='f4') / 255
        item = faces.save_view(faces.rgba(rgb[None], 'foreground', mask[None]))
        item['provenance'] = {'backend': 'h3_enhanced', 'faceId': fid,
            'recipe': {'models': recipe['models'], 'conditioning': recipe['conditioning'],
                       'pictures': recipe['faces'][fid]['pictures'], 'prompt': recipe['faces'][fid]['prompt']},
            'recipeSeed': recipe['seeds'][fid], 'samplers': [recipe['sampler']],
            'graphSha256': result['promptSha256'], 'referenceSha256': recipe['referenceSha256'],
            'guideSha256': recipe['faces'][fid]['guideSha256'], 'maskMode': 'foreground',
            'canvas': recipe['faces'][fid]['canvas'], 'freshInference': result['prompt_id']}
        if recipe.get('geometryRefinement'):
            item['provenance']['geometryRefinement'] = recipe['geometryRefinement']
        item['reference'] = {**original['reference'], 'path': str(Path(recipe['baselinePath']) / 'reference')}
        candidates[fid] = item
    aligned = faces.align(candidates, baseline, .85, .10, .15)
    asset = faces.compose(baseline, aligned)
    manifest = store.validate(store.asset_dir(asset), asset, check_bytes=True)
    qa = read_json(store.asset_dir(asset) / manifest['qa'])
    report = {'asset': asset, 'path': str(store.asset_dir(asset)), 'sourceBaseline': baseline,
              'generatedFaces': list(candidates), 'faces': qa['faces'],
              'viewConsistency': 'none', 'materialEstimation': 'none',
              'proxySha256': manifest['proxySha256'],
              'geometryRefinement': recipe.get('geometryRefinement')}
    write_json(run / recipe['branch'] / 'composed.json', report)
    print(json.dumps({'composed': asset, 'generatedFaces': list(candidates),
                     'qa': {fid: data.get('status') for fid, data in qa['faces'].items()}}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--branch', choices=['A', 'B'], required=True)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--face', choices=list(store.FACES))
    parser.add_argument('--origin', default='http://127.0.0.1:8201')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--compose-only', action='store_true')
    args = parser.parse_args()
    use_cuda_runtime_on_wsl()
    run = args.run.resolve()
    if args.baseline:
        recipe = prepare(run, args.branch, args.baseline)
    else:
        recipe = read_json(run / args.branch / 'prepared.json')
    if args.prepare_only:
        return
    if not args.compose_only:
        for fid in ([args.face] if args.face else ['+Z', '+X', '-X', '+Y', '-Y', '-Z']):
            run_face(run, recipe, fid, args.origin)
    compose(run, recipe)


if __name__ == '__main__':
    main()
