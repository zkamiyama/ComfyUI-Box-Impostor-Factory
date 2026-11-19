"""Opt-in normalTexture ablation with unchanged geometry and archived RGB.

Removes only materials[*].normalTexture from the canonical GLB JSON. Binary
chunks and all other JSON values remain unchanged. Native rebaking is required;
no generated RGB, source NORMAL attributes, UVs, or material defaults are edited.

Example (run only after other native-render QA has finished):
  BI_SOFTWARE_GPU=1 BI_CHROMIUM=/path/to/chrome-headless-shell python \
    scripts/run_normal_texture_control.py --baseline /path/to/A/baseline \
    --source-asset /path/to/A/composed --work-dir /path/to/normal-control

Use --prepare-only for the CPU/GLB stage. A later full-six-face source can reuse
the same baseline/work-dir; each composition gets a separate source asset key.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import platform
import struct
import sys

VERSION = 'normal-texture-control-v1'
JSON_CHUNK = 0x4e4f534a
BIN_CHUNK = 0x004e4942
REPO = Path(__file__).resolve().parents[1]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def json_sha(value):
    return sha(json.dumps(value, sort_keys=True, separators=(',', ':'),
                          allow_nan=False).encode())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def read_glb(data):
    if len(data) < 20:
        raise ValueError('Truncated GLB.')
    magic, version, length = struct.unpack_from('<III', data)
    if magic != 0x46546c67 or version != 2 or length != len(data):
        raise ValueError('Expected a complete GLB version 2.')
    chunks = []
    offset = 12
    while offset < length:
        if offset + 8 > length:
            raise ValueError('Truncated GLB chunk header.')
        size, kind = struct.unpack_from('<II', data, offset)
        offset += 8
        if size % 4 or offset + size > length:
            raise ValueError('Invalid GLB chunk length.')
        chunks.append((kind, data[offset:offset + size]))
        offset += size
    if not chunks or chunks[0][0] != JSON_CHUNK:
        raise ValueError('GLB must start with its JSON chunk.')
    if sum(kind == JSON_CHUNK for kind, _ in chunks) != 1:
        raise ValueError('GLB must contain exactly one JSON chunk.')
    return json.loads(chunks[0][1]), chunks


def glb_bytes(document, chunks):
    payload = json.dumps(document, separators=(',', ':'), allow_nan=False).encode()
    payload += b' ' * (-len(payload) % 4)
    body = struct.pack('<II', len(payload), JSON_CHUNK) + payload
    for kind, content in chunks[1:]:
        body += struct.pack('<II', len(content), kind) + content
    return struct.pack('<III', 0x46546c67, 2, 12 + len(body)) + body


def without_normal_texture(document):
    result = copy.deepcopy(document)
    removed = []
    for index, material in enumerate(result.get('materials', [])):
        if 'normalTexture' in material:
            removed.append({'material': index, 'normalTexture': material.pop('normalTexture')})
    return result, removed


def geometry_signature(data):
    document, chunks = read_glb(data)
    keys = ('meshes', 'nodes', 'scenes', 'scene', 'skins', 'animations',
            'accessors', 'bufferViews', 'buffers')
    return json_sha({
        'geometryAndScene': {key: document[key] for key in keys if key in document},
        'nonJsonChunks': [{'type': kind, 'sha256': sha(content)}
                          for kind, content in chunks[1:]],
    })


def glb_comparison(original, control):
    before, chunks_before = read_glb(original)
    after, chunks_after = read_glb(control)
    expected, removed = without_normal_texture(before)
    return {
        'jsonOnlyNormalTextureRemoved': expected == after,
        'nonJsonChunksByteIdentical': chunks_before[1:] == chunks_after[1:],
        'geometrySignatureEqual': geometry_signature(original) == geometry_signature(control),
        'sourceGeometrySignature': geometry_signature(original),
        'controlGeometrySignature': geometry_signature(control),
        'removedNormalTextures': removed,
        'sourceGlbSha256': sha(original),
        'controlGlbSha256': sha(control),
        'binaryChunks': [{'type': kind, 'byteLength': len(content), 'sha256': sha(content)}
                         for kind, content in chunks_before[1:]],
    }


def strip_normal_texture(data):
    document, chunks = read_glb(data)
    changed, removed = without_normal_texture(document)
    if not removed:
        raise ValueError('The source has no material.normalTexture to ablate.')
    result = glb_bytes(changed, chunks)
    proof = glb_comparison(data, result)
    if not all(proof[key] for key in ('jsonOnlyNormalTextureRemoved',
                                     'nonJsonChunksByteIdentical', 'geometrySignatureEqual')):
        raise ValueError('GLB ablation changed data other than material.normalTexture.')
    return result, proof


def relative_file(directory, relative):
    # Use the same containment contract before importing Comfy modules.
    path = Path(relative)
    home = Path(directory).resolve()
    result = (home / path).resolve()
    if path.is_absolute() or not result.is_relative_to(home):
        raise ValueError('Asset path escapes its directory.')
    return result


def normal_coverage_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(65536):
            if len(block) % 8:
                raise ValueError('Expected RGBA16F normal data.')
            digest.update(block[6::8])
            digest.update(block[7::8])
    return digest.hexdigest()


def compare_assets(before_dir, before, after_dir, after):
    checks = {}
    for key in ('resolution', 'tightBounds', 'bakeBox', 'transform',
                'inverseTransform', 'faceOrder', 'backend', 'maps', 'versions'):
        checks[key + 'Equal'] = before.get(key) == after.get(key)
    checks['textureNamesEqual'] = set(before['textures']) == set(after['textures'])
    texture_hashes = {}
    for name in sorted(set(before['textures']) | set(after['textures'])):
        first = before['textures'].get(name)
        second = after['textures'].get(name)
        if first is None or second is None:
            checks[name + 'ExpectedDifference'] = False
            continue
        a = file_sha(relative_file(before_dir, first['path']))
        b = file_sha(relative_file(after_dir, second['path']))
        texture_hashes[name] = {'source': a, 'control': b, 'equal': a == b}
        checks[name + ('Changed' if name == 'normalMap' else 'Equal')] = (
            a != b if name == 'normalMap' else a == b)
        checks[name + 'DescriptorEqual'] = (
            {k: v for k, v in first.items() if k not in ('path', 'sha256')} ==
            {k: v for k, v in second.items() if k not in ('path', 'sha256')})
    first_normal, second_normal = before['textures']['normalMap'], after['textures']['normalMap']
    if first_normal.get('dtype') == second_normal.get('dtype') == 'float16':
        checks['normalCoverageEqual'] = (
            normal_coverage_sha(relative_file(before_dir, first_normal['path'])) ==
            normal_coverage_sha(relative_file(after_dir, second_normal['path'])))
    checks['baseColorEqual'] = (
        file_sha(relative_file(before_dir, before['baseColor']['path'])) ==
        file_sha(relative_file(after_dir, after['baseColor']['path'])))
    guides = {}
    camera_keys = ('up', 'side', 'tile', 'viewMatrix', 'projectionMatrix', 'frustum',
                   'depthEncoding', 'imageRowOrder', 'imageColorSpace', 'foregroundPixels')
    for face_id in before['faceOrder']:
        a, b = before['faces'][face_id], after['faces'][face_id]
        checks[face_id + 'CameraEqual'] = all(a.get(key) == b.get(key) for key in camera_keys)
        guides[face_id] = {}
        for key in ('guide', 'depth'):
            ah = file_sha(relative_file(before_dir, a[key]))
            bh = file_sha(relative_file(after_dir, b[key]))
            guides[face_id][key] = {'source': ah, 'control': bh, 'equal': ah == bh}
            checks[face_id + key + 'Equal'] = ah == bh
    proof = glb_comparison(relative_file(before_dir, before['proxy']).read_bytes(),
                           relative_file(after_dir, after['proxy']).read_bytes())
    for key in ('jsonOnlyNormalTextureRemoved', 'nonJsonChunksByteIdentical',
                'geometrySignatureEqual'):
        checks[key] = proof[key]
    checks['normalTextureRemoved'] = bool(proof['removedNormalTextures'])
    return {'passed': all(checks.values()), 'checks': checks,
            'textures': texture_hashes, 'guides': guides, 'geometry': proof}


def prepare(baseline_dir, work):
    baseline_dir, work = Path(baseline_dir).resolve(), Path(work).resolve()
    manifest = json.loads((baseline_dir / 'manifest.json').read_bytes())
    original = relative_file(baseline_dir, manifest['proxy']).read_bytes()
    if sha(original) != manifest['proxySha256']:
        raise ValueError('Source GLB does not match its manifest hash.')
    data, proof = strip_normal_texture(original)
    key = json_sha([VERSION, manifest['asset_id'], manifest['proxySha256'], sha(data),
                    manifest['resolution'], manifest['backend'], manifest['maps']])
    folder = work / 'baselines' / key
    folder.mkdir(parents=True, exist_ok=True)
    proxy_path = folder / 'canonical.glb'
    if proxy_path.exists() and proxy_path.read_bytes() != data:
        raise ValueError('Existing control GLB differs from this deterministic ablation.')
    if not proxy_path.exists():
        proxy_path.write_bytes(data)
    canonical = {key: manifest[key] for key in (
        'tightBounds', 'bakeBox', 'transform', 'inverseTransform')}
    canonical.update(glbSha256=sha(data), sourceSha256=manifest.get('provenance', {}).get(
        'originalGlbSha256', manifest['proxySha256']))
    metadata = folder / 'canonical.json'
    write_json(metadata, canonical)
    record = {'version': VERSION, 'sourceBaseline': manifest['asset_id'],
              'sourceBaselinePath': str(baseline_dir), 'cacheKey': key,
              'resolution': manifest['resolution'], 'backend': manifest['backend'],
              'proxy': str(proxy_path), 'metadata': str(metadata), 'geometry': proof}
    write_json(folder / 'prepared.json', record)
    return manifest, record, folder


def reuse_aligned_views(source_dir, source, control_id, work):
    """Replay exactly the archived source and aligned samples, with no registration."""
    qa = json.loads(relative_file(source_dir, source['qa']).read_bytes())['faces']
    items, evidence = {}, {}
    for face_id, info in source['faces'].items():
        edit = info.get('edit')
        if not edit:
            continue
        view = relative_file(source_dir, edit['view']).parent
        report = qa[face_id]
        if report['status'] != edit['status']:
            raise ValueError('Archived face status disagrees with the manifest.')
        report_dir = work / 'replays' / source['asset_id'] / face_id.replace('+', 'p').replace('-', 'n')
        write_json(report_dir / 'report.json', report)
        saved = {'path': str(view), 'provenance': edit.get('provenance', {})}
        if source.get('reference'):
            reference = source['reference']
            saved['reference'] = {**reference, 'path': str(
                relative_file(source_dir, reference['image']).parent)}
        hashes = {name: file_sha(view / name) for name in ('source.png', 'source.npy')}
        if edit['status'] == 'accepted':
            digest = file_sha(view / 'aligned.npy')
            if digest != edit['alignedSha256']:
                raise ValueError('Archived aligned samples do not match their hash.')
            hashes['aligned.npy'] = digest
        items[face_id] = {'key': edit['alignmentKey'], 'source': saved,
                          'directory': str(report_dir), 'measured': str(view)}
        evidence[face_id] = {'status': edit['status'], 'alignmentKey': edit['alignmentKey'],
                             'sourceFiles': hashes, 'registration': 'reused-byte-identical'}
    if not items:
        raise ValueError('Source asset has no archived generated RGB to replay.')
    return {'baseline': control_id, 'faces': items}, evidence


def execute(args):
    baseline_dir, source_dir = Path(args.baseline).resolve(), Path(args.source_asset).resolve()
    work = Path(args.work_dir).resolve()
    baseline, record, cache_dir = prepare(baseline_dir, work)
    if args.prepare_only:
        return {'prepared': str(cache_dir / 'prepared.json'), 'nativeBakeStarted': False}
    sys.path[:0] = [str(REPO.parents[1]), str(REPO)]
    # The installed WSL driver can crash during NVML device counting at import.
    if 'microsoft' in platform.release().lower():
        import torch
        torch.cuda._device_count_nvml = lambda: -1
    import folder_paths
    from box_nodes import store, faces
    folder_paths.set_output_directory(str(work / 'output'))
    store.root().mkdir(parents=True, exist_ok=True)
    baseline = store.validate(baseline_dir, baseline['asset_id'])
    source = store.validate(source_dir)
    if source['proxySha256'] != baseline['proxySha256']:
        raise ValueError('Source RGB asset does not use the supplied unchanged A geometry.')
    for key in ('normalMap', 'intervalMap', 'rangeMap'):
        if source['textures'][key]['sha256'] != baseline['textures'][key]['sha256']:
            raise ValueError('Source RGB asset changed the baseline ' + key + '.')
    for key in ('resolution', 'tightBounds', 'bakeBox', 'transform', 'inverseTransform', 'backend'):
        if source[key] != baseline[key]:
            raise ValueError('Source RGB asset changed the baseline ' + key + '.')
    requested_backend = args.backend or baseline['backend']
    if requested_backend != baseline['backend']:
        raise ValueError('The control must use the same native backend as A.')
    proxy = {'path': record['proxy'], 'metadata': record['metadata'],
             'sha256': record['geometry']['controlGlbSha256'], 'stageKey': record['cacheKey'],
             'provenance': {'normalTextureControl': record}}
    if baseline.get('reference'):
        ref = baseline['reference']
        proxy['reference'] = {**ref, 'path': str(relative_file(baseline_dir, ref['image']).parent)}
        if ref.get('camera'):
            proxy['camera'] = ref['camera']
            proxy['croppedImage'] = str(relative_file(baseline_dir, ref['camera']['image']))
    control_id = store.baseline(proxy, baseline['resolution'], requested_backend, tuple(baseline['maps']))
    control_dir = store.asset_dir(control_id)
    control = store.validate(control_dir, control_id)
    verification = compare_assets(baseline_dir, baseline, control_dir, control)
    write_json(cache_dir / 'baseline-verification.json', verification)
    if not verification['passed']:
        raise ValueError('Native baseline differs beyond normalMap; see baseline-verification.json.')
    aligned, replay = reuse_aligned_views(source_dir, source, control_id, work)
    final_id = faces.compose(control_id, aligned)
    final_dir = store.asset_dir(final_id)
    final = store.validate(final_dir, final_id)
    result = compare_assets(source_dir, source, final_dir, final)
    result.update(version=VERSION, sourceAsset=source['asset_id'], sourcePath=str(source_dir),
                  asset=final_id, path=str(final_dir), baselineAsset=control_id,
                  baselinePath=str(control_dir), baselineCache=str(cache_dir),
                  generatedFaces=list(replay), replay=replay,
                  sourceManifestSha256=file_sha(source_dir / 'manifest.json'),
                  scriptSha256=file_sha(__file__),
                  limitation='Normal maps also affect native cross-view incidence weighting; this is not solely a lighting ablation.',
                  noNewInference=True)
    for face_id, item in replay.items():
        view = relative_file(final_dir, final['faces'][face_id]['edit']['view']).parent
        for name, digest in item['sourceFiles'].items():
            result['checks'][face_id + name + 'ReplayEqual'] = file_sha(view / name) == digest
    result['passed'] = all(result['checks'].values())
    output = work / 'comparisons' / source['asset_id'] / 'result.json'
    write_json(output, result)
    if not result['passed']:
        raise ValueError('Composed control differs beyond normalMap; see ' + str(output))
    return {'asset': final_id, 'path': str(final_dir), 'report': str(output),
            'sourceAsset': source['asset_id'], 'generatedFaces': list(replay), 'passed': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True, help='Original geometry baseline asset directory.')
    parser.add_argument('--source-asset', required=True, help='Partial or full A asset with archived RGB.')
    parser.add_argument('--work-dir', required=True, help='Large outputs and reusable baseline cache.')
    parser.add_argument('--backend', choices=('webgl2', 'webgpu'))
    parser.add_argument('--prepare-only', action='store_true', help='Only byte-preserving CPU GLB preparation.')
    args = parser.parse_args()
    print(json.dumps(execute(args), indent=2), flush=True)


if __name__ == '__main__':
    main()
