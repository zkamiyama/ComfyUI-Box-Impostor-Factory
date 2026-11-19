"""Native Qwen Image 2.1 +X trials over the frozen F Box Impostor."""
import argparse
import copy
import difflib
import hashlib
from pathlib import Path
import re

import run_geometry_ab as ab
import run_geometry_d as replacement

REPO = Path(__file__).resolve().parents[1]
COMFY = REPO.parents[1]
F_ASSET = '4e8090f307b4958de8a94877ad391b3b0e355fa208d7ded8340a229e773b1920'
ORDER = ['guide', 'source', 'axis-depth', 'normals', 'mask']
MODELS = {
    'model': 'qwen_image_2.1_int8_convrot.safetensors',
    'clip': 'qwen3vl_8b_int8_convrot.safetensors',
    'vae': 'qwen_image_2.1_vae_bf16.safetensors',
    'matting': 'birefnet.safetensors',
}
SAMPLER = {'steps': 25, 'cfg': 1, 'sampler_name': 'euler', 'scheduler': 'simple', 'denoise': 1}
ANYANGLE_ORDER = ['source', 'guide']
ANYANGLE_LORA = 'QI2.1_AnyAngle.safetensors'
ANYANGLE_PROMPT = 'Change the camera angle from <image2> to <image1>.'


def qwen_references(prompt):
    return re.sub(r'<Picture ([1-5])>', lambda m: '<image' + m[1] + '>', prompt)


def build_prompt(recipe, fid):
    graph = ab.build_prompt(recipe, fid)
    graph['7']['inputs']['type'] = 'qwen_image'
    anyangle = recipe.get('profile') == 'anyangle'
    if anyangle:
        graph['19'] = {'class_type': 'LoraLoaderModelOnly', 'inputs': {
            'model': ['6', 0], 'lora_name': ANYANGLE_LORA, 'strength_model': 1}}
    graph['9'] = {'class_type': 'QwenImage21Cache', 'inputs': {
        'model': ['19' if anyangle else '6', 0], 'device': 'cpu', 'dtype': 'default'}}
    graph['10'] = {'class_type': 'TextEncodeQwenImage21', 'inputs': {
        'clip': ['7', 0], 'vae': ['8', 0], 'prompt': recipe['faces'][fid]['prompt'],
        'negative_prompt': '', 'resolution': 1024,
        **{'images.image_' + str(n): [str(n), 0]
           for n in range(1, len(recipe['pictureOrder']) + 1)}}}
    del graph['11']
    graph['12']['inputs'].update(model=['9', 0], positive=['10', 0],
                                 negative=['10', 1], latent_image=['10', 2])
    return graph


def inspect_graph(recipe, fid, graph):
    anyangle = recipe.get('profile') == 'anyangle'
    order = ANYANGLE_ORDER if anyangle else ORDER
    sampler = {**SAMPLER, 'cfg': 3} if anyangle else SAMPLER
    assert recipe['pictureOrder'] == order
    assert recipe['seeds'][fid] == 101 and recipe['sampler'] == sampler
    for n, role in enumerate(order, 1):
        assert graph[str(n)] == {'class_type': 'LoadImage', 'inputs': {
            'image': recipe['faces'][fid]['pictures'][role]['image']}}
        assert graph['10']['inputs']['images.image_' + str(n)] == [str(n), 0]
    assert graph['6'] == {'class_type': 'UNETLoader', 'inputs': {
        'unet_name': MODELS['model'], 'weight_dtype': 'default'}}
    assert graph['7']['inputs'] == {'clip_name': MODELS['clip'], 'type': 'qwen_image', 'device': 'default'}
    assert graph['8']['inputs'] == {'vae_name': MODELS['vae']}
    assert graph['9']['inputs'] == {'model': ['19' if anyangle else '6', 0], 'device': 'cpu', 'dtype': 'default'}
    assert graph['10']['class_type'] == 'TextEncodeQwenImage21'
    assert graph['10']['inputs']['negative_prompt'] == '' and graph['10']['inputs']['resolution'] == 1024
    assert graph['12']['inputs'] == {**sampler, 'seed': 101, 'model': ['9', 0],
        'positive': ['10', 0], 'negative': ['10', 1], 'latent_image': ['10', 2]}
    assert graph['14']['inputs']['images'] == ['13', 0]
    assert graph['15']['inputs']['bg_removal_name'] == MODELS['matting']
    assert graph['16']['inputs']['image'] == ['13', 0]
    assert graph['17']['inputs']['mask'] == ['16', 0] and graph['18']['inputs']['images'] == ['17', 0]
    assert not any('MiniMax' in n['class_type'] or
                   n['class_type'] == 'ConditioningZeroOut' for n in graph.values())
    if anyangle:
        assert recipe['models']['lora'] == ANYANGLE_LORA
        assert graph['19'] == {'class_type': 'LoraLoaderModelOnly', 'inputs': {
            'model': ['6', 0], 'lora_name': ANYANGLE_LORA, 'strength_model': 1}}
        assert graph['10']['inputs']['prompt'] == ANYANGLE_PROMPT
        assert set(k for k in graph['10']['inputs'] if k.startswith('images.')) == {
            'images.image_1', 'images.image_2'}
        assert set(recipe['faces'][fid]['pictures']) == set(order)
    else:
        assert not any('Lora' in n['class_type'] for n in graph.values())
    assert recipe['backend'] == 'qwen_enhanced' and recipe['experiment']['sourceBox'] == F_ASSET
    return {'checkedOn': 'CPU', 'gpuInference': False, 'nativeEncoder': 'TextEncodeQwenImage21',
            'referenceCount': len(order), 'referenceOrder': order, 'encoderOutputs': {
                'positive': 0, 'negative': 1, 'emptyLatent': 2},
            'cache': {'device': 'cpu', 'dtype': 'default', 'lossless': True},
            'rawOutputNode': '14', 'foregroundOutputNode': '18',
            'lora': {'name': ANYANGLE_LORA, 'strength_model': 1} if anyangle else False,
            'sourceBox': F_ASSET, 'replacementVerification': 'run_geometry_d.verify_replacement'}


def prepare(run, branch, fid, profile='five'):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', branch):
        raise ValueError('Branch must be a single directory name.')
    if fid != '+X':
        raise ValueError('These trials are the frozen B +X comparison only.')
    if profile not in ('five', 'anyangle'):
        raise ValueError('Unknown Qwen trial profile.')
    anyangle = profile == 'anyangle'
    source = run / 'B/prepared.json'
    b = ab.read_json(source)
    f_path = run / 'F/composed.json'
    f = ab.read_json(f_path)
    if b['pictureOrder'] != ORDER or b['pictureSize'] != [1024, 1024] or b['seeds'][fid] != 101:
        raise ValueError('The frozen B five-picture, 1024px, seed-101 comparison changed.')
    if f['asset'] != F_ASSET or f.get('guideBaseline') != b['baseline'] or not f.get('applied'):
        raise ValueError('The accepted F composition is required as the replacement source.')
    ab.folder_paths.set_output_directory(str(run / 'output'))
    f_directory = ab.store.asset_dir(F_ASSET)
    f_manifest = ab.store.validate(f_directory, F_ASSET, check_bytes=True)
    for picture in b['faces'][fid]['pictures'].values():
        if ab.digest(picture['path']) != picture['sha256']:
            raise ValueError('A frozen B conditioning image changed: ' + picture['path'])
        with ab.Image.open(picture['path']) as image:
            if image.size != (1024, 1024):
                raise ValueError('A B conditioning image is not 1024px square.')
    result_path = run / 'B' / ab.SLUGS[fid] / 'result.json'
    original = ab.read_json(result_path)
    if ab.digest(original['raw']) != original['rawSha256']:
        raise ValueError('The B raw comparison image changed.')
    models = {**MODELS, 'lora': ANYANGLE_LORA} if anyangle else MODELS
    categories = [('model', 'diffusion_models'), ('clip', 'text_encoders'),
                  ('vae', 'vae'), ('matting', 'background_removal')]
    if anyangle:
        categories.append(('lora', 'loras'))
    model_files = {}
    for role, category in categories:
        path = Path(ab.folder_paths.get_full_path_or_raise(category, models[role])).resolve()
        model_files[role] = {'name': models[role], 'path': str(path),
                             'bytes': path.stat().st_size, 'sha256': ab.digest(path)}
    code_paths = [Path(__file__), Path(ab.__file__), Path(replacement.__file__),
                  COMFY / 'comfy_extras/nodes_qwen.py', COMFY / 'comfy_extras/nodes_bg_removal.py',
                  COMFY / 'comfy/text_encoders/qwen_image21.py', COMFY / 'comfy/ldm/qwen_image21/model.py']
    if anyangle:
        code_paths.append(COMFY / 'nodes.py')
    code_hashes = {str(p.resolve()): ab.digest(p) for p in code_paths}
    original_prompt = b['faces'][fid]['prompt']
    adapted_prompt = ANYANGLE_PROMPT if anyangle else qwen_references(original_prompt)
    recipe = copy.deepcopy(b)
    recipe.update(branch=branch, backend='qwen_enhanced', models=models,
                  sampler={**SAMPLER, 'cfg': 3} if anyangle else SAMPLER,
                  seeds={fid: 101}, prompt=ANYANGLE_PROMPT if anyangle else qwen_references(b['prompt']),
                  conditioning='Native Qwen Image 2.1: the same five image references to encoder and VAE; no ControlNet',
                  faces={fid: copy.deepcopy(b['faces'][fid])})
    recipe['faces'][fid]['prompt'] = adapted_prompt
    if anyangle:
        recipe.update(profile=profile, pictureOrder=ANYANGLE_ORDER,
                      conditioning='Native Qwen Image 2.1 + AnyAngle LoRA: image_1 original photo, image_2 +X coarse guide; two encoder and VAE references')
        recipe['faces'][fid]['pictures'] = {
            role: recipe['faces'][fid]['pictures'][role] for role in ANYANGLE_ORDER}
    recipe['experiment'] = {
        'name': 'qwen21-native-five-reference-v1', 'targetFace': fid,
        'sourceBranch': 'F', 'sourceBox': F_ASSET, 'sourceBoxPath': str(f_directory),
        'sourceComposedSha256': ab.digest(f_path),
        'sourceManifestSha256': ab.digest(f_directory / 'manifest.json'),
        'sourceProxySha256': f_manifest['proxySha256'],
        'conditioningSourceBranch': 'B', 'comparisonBranch': 'B',
        'sourceRecipeSha256': ab.digest(source), 'sourceResultSha256': ab.digest(result_path),
        'sourceRaw': original['raw'], 'sourceRawSha256': original['rawSha256'],
        'pictureIndices': [1, 2, 3, 4, 5], 'modelFiles': model_files, 'codeSha256': code_hashes,
        'cache': {'device': 'cpu', 'dtype': 'default', 'purpose': 'lossless memory placement only'},
        'promptAdaptation': {'kind': 'reference-tags-only',
            'from': '<Picture n>', 'to': '<imagen>',
            'original': original_prompt, 'adapted': adapted_prompt,
            'originalSha256': hashlib.sha256(original_prompt.encode()).hexdigest(),
            'adaptedSha256': hashlib.sha256(adapted_prompt.encode()).hexdigest(),
            'replacementCount': len(re.findall(r'<Picture [1-5]>', original_prompt))},
    }
    if anyangle:
        recipe['experiment'].update(
            name='qwen21-anyangle-two-reference-v1', pictureIndices=[1, 2],
            referenceRoles={'image_1': 'original photo (B source)', 'image_2': '+X coarse guide (B guide)'},
            lora={'name': ANYANGLE_LORA, 'strength_model': 1, 'node': 'LoraLoaderModelOnly'},
            promptAdaptation={'kind': 'author-anyangle-workflow', 'prompt': ANYANGLE_PROMPT,
                              'promptSha256': hashlib.sha256(ANYANGLE_PROMPT.encode()).hexdigest()})
    target = run / branch / 'prepared.json'
    if target.exists() and ab.read_json(target) != recipe:
        raise ValueError('Existing preparation has different inputs, models or code; use a new branch.')
    graph = build_prompt(recipe, fid)
    inspection = inspect_graph(recipe, fid, graph)
    ab.write_json(target, recipe)
    graph_path = run / branch / ab.SLUGS[fid] / 'prepared-prompt.json'
    ab.write_json(graph_path, graph)
    diff = ''.join(difflib.unified_diff([original_prompt + '\n'], [adapted_prompt + '\n'],
                                      fromfile='B-px.prompt', tofile='Q2-px.prompt' if anyangle else 'Q1-px.prompt'))
    (run / branch / ('prompt-anyangle.diff' if anyangle else 'prompt-reference.diff')).write_text(diff)
    ab.write_json(run / branch / 'graph-inspection.json', {**inspection,
        'graphPath': str(graph_path), 'graphSha256': ab.digest(graph_path),
        'recipeSha256': ab.digest(target), 'models': model_files, 'codeSha256': code_hashes})
    return recipe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--profile', choices=['five', 'anyangle'], default='five')
    parser.add_argument('--branch')
    parser.add_argument('--face', choices=['+X'], default='+X')
    parser.add_argument('--origin', default='http://127.0.0.1:8201')
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--prepare-only', action='store_true')
    action.add_argument('--compose-only', action='store_true')
    args = parser.parse_args()
    if args.branch is None:
        args.branch = 'Q2-qwen-anyangle-20261119' if args.profile == 'anyangle' else 'Q1-qwen-five-20261119'
    run = args.run.resolve()
    recipe = prepare(run, args.branch, args.face, args.profile)
    if args.prepare_only:
        print({'prepared': str(run / args.branch / 'prepared.json'), 'gpuInference': False}, flush=True)
        return
    if not args.compose_only:
        ab.run_face(run, recipe, args.face, args.origin, prompt_builder=build_prompt)
    composed = replacement.compose_replacement(run, recipe, args.face, prompt_builder=build_prompt)
    result = ab.read_json(run / args.branch / ab.SLUGS[args.face] / 'result.json')
    ab.write_json(run / args.branch / 'comparison-ready.json', {
        'face': args.face, 'B_raw': recipe['experiment']['sourceRaw'],
        'B_rawSha256': recipe['experiment']['sourceRawSha256'],
        'Qwen_raw': result['raw'], 'Qwen_rawSha256': result['rawSha256'],
        'experiment': recipe['experiment'], 'result': result, 'box': composed})
    print({'composed': composed['asset'], 'applied': composed['applied']}, flush=True)


if __name__ == '__main__':
    main()
