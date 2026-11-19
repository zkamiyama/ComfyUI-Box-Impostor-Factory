"""Package actual Qwen replacements with official Inspector, then capture fixed views."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import zlib

import numpy as np


LAB = Path('/home/dev/Projects/box-impostor-lab')
REPO = LAB/'ComfyUI/custom_nodes/box_impostor_factory'
RUN = Path('/mnt/f/_dev/box-impostor-lab/experiments/geometry-abc-20261119')
WORK = Path('/mnt/f/_dev/box-impostor-lab/results/qwen-image-20261119')
OLD = Path('/mnt/f/_dev/box-impostor-lab/results/hi3d-depth-cycle-20261119')
VIEW = Path('/home/dev/Projects/3D-Inspector-skills/skills/3d-inspector/view.py')
CAPTURE = REPO/'docs/verification/hi3d-depth-cycle-20261119/capture_inspector_display_yaw.mjs'
PY = LAB/'.venv/bin/python'
F_ID = '4e8090f307b4958de8a94877ad391b3b0e355fa208d7ded8340a229e773b1920'
DISPLAY = {'lighting': 'unlit', 'sampling': 'native', 'colorSampling': 'view'}
CASES = [('K-view', None, 'F-current', 'Baseline / previous H3'),
         ('Q1', 'Q1-qwen-five-20261119', 'Q1-qwen-five', 'Q1 / Qwen Image 2.1'),
         ('Q2', 'Q2-qwen-anyangle-20261119', 'Q2-qwen-anyangle', 'Q2 / Qwen + AnyAngle')]


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def payload(path):
    text = path.read_text()
    encoded = re.search(r'<script id="scene-payload"[^>]*>([\s\S]*?)</script>', text)[1].strip()
    packed = base64.b64decode(encoded)
    return json.loads(zlib.decompress(packed) if packed[0] == 0x78 else packed)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2)+'\n')


def embedded_scripts(path):
    result = []
    for attributes, content in re.findall(r'<script([^>]*)>([\s\S]*?)</script>', path.read_text()):
        if 'scene-payload' not in attributes:
            result.append({'attributes': attributes, 'bytes': len(content.encode()),
                           'sha256': hashlib.sha256(content.encode()).hexdigest()})
    assert result
    return result


def bounded_run(command, log, seconds, cwd=REPO):
    started = time.monotonic()
    with log.open('w') as stream:
        process = subprocess.Popen(command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            code = process.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise RuntimeError(f'Owned process group timed out after {seconds}s; log: {log}')
    assert code == 0, f'Exit {code}; log: {log}'
    return {'command': command, 'log': str(log), 'logSHA256': sha(log),
            'elapsedSeconds': round(time.monotonic()-started, 3), 'exitCode': code}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', action='store_true')
    parser.add_argument('--reuse-json', type=Path)
    args = parser.parse_args()
    reuse = json.loads(args.reuse_json.read_text()) if args.reuse_json else {'files': {}}
    WORK.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(LAB/'ComfyUI'))
    from box_nodes.archive import pack_asset
    base_path = OLD/'K-view.inspector.html'
    base = payload(base_path)
    assert base['assetId'] == F_ID
    assert base['nativeBox']['display'] == DISPLAY
    base_color_spec = base['nativeBox']['manifest']['textures']['colorMap']
    base_color_raw = base64.b64decode(base['nativeBox']['files'][base_color_spec['path']])
    base_color = np.frombuffer(base_color_raw, dtype='<u2').reshape(base_color_spec['height'], base_color_spec['width'], 4)
    reference = json.loads((OLD/'K-view-yaw/capture.json').read_text())['referenceManifest']
    config = {'angles': [0, 30, 45, 60], 'referenceManifest': reference, 'cases': {},
              'baselineNote': 'K-view is an internal QA key for freshly generated F-current, using the current identical renderer and K view/unlit settings. Historical K is used only for camera and source payload checks.'}
    receipt = {'kind': __doc__, 'scriptSHA256': sha(__file__), 'viewPy': str(VIEW),
               'viewPySHA256': sha(VIEW), 'display': DISPLAY, 'baseHtml': str(base_path),
               'baseHtmlSHA256': sha(base_path), 'cases': {}}
    current_scripts = None
    receipt['historicalKEmbeddedScripts'] = embedded_scripts(base_path)
    for key, case_name, slug, label in CASES:
        composed_path = RUN/case_name/'composed.json' if case_name else None
        composed = json.loads(composed_path.read_text()) if composed_path else {
            'asset': F_ID, 'path': str(RUN/'output/box_impostor'/F_ID),
            'preservation': {'baselineAssetUnchanged': True}}
        if composed_path:
            assert composed['sourceBox'] == F_ID and composed['replacedFace'] == '+X'
            assert composed['applied'] and composed['faceQA']['status'] == 'accepted'
            for flag in ['proxyBytesExact', 'geometryMetadataExact', 'nonColorMapsExact',
                         'coverageAlphaExact', 'unchangedFaceSourcesAndMetadata']:
                assert composed['preservation'][flag] is True, (key, flag)
        asset = Path(composed['path'])
        archive = WORK/f'{slug}.box.zip'
        pack_asset(asset, archive)
        html = WORK/f'{slug}.inspector.html'
        command = [str(PY), str(VIEW), str(archive), '--box-lighting', 'unlit',
                   '--box-color-sampling', 'view', '--title', label, '-o', str(html)]
        if html.exists():
            assert str(html) in reuse['files'], f'Existing output requires an explicit SHA receipt: {html}'
            assert sha(html) == reuse['files'][str(html)]['sha256']
            log = WORK/f'{slug}-build.log'
            build = {'command': command, 'log': str(log), 'logSHA256': sha(log), 'exitCode': 0,
                     'reusedExistingOfficialOutput': True, 'reuseReceipt': str(args.reuse_json),
                     'reuseReceiptSHA256': sha(args.reuse_json)}
        else:
            build = bounded_run(command, WORK/f'{slug}-build.log', 240)
        assert html.exists() and str(html) in Path(build['log']).read_text()
        scripts = embedded_scripts(html)
        if current_scripts is None:
            current_scripts = scripts
        assert scripts == current_scripts, (key, 'Embedded renderer/bootstrap differs from F-current')
        data = payload(html)
        geometry = lambda scene: [{k: v for k, v in mesh.items() if k != 'fallbackColors'} for mesh in scene['meshes']]
        checks = {'assetIdMatchesComposed': data['assetId'] == composed['asset'],
                  'boundsEqualToK': data['bounds'] == base['bounds'],
                  'proxyGeometryEqualToK': geometry(data) == geometry(base),
                  'proxyFallbackColorChanges': [i for i, (old, new) in enumerate(zip(base['meshes'], data['meshes'])) if old.get('fallbackColors') != new.get('fallbackColors')],
                  'displayEqualToK': data['nativeBox']['display'] == DISPLAY, 'maps': {}}
        assert all(checks[flag] for flag in ['assetIdMatchesComposed', 'boundsEqualToK',
                                            'proxyGeometryEqualToK', 'displayEqualToK']), checks
        assert set(data['nativeBox']['manifest']['textures']) == set(base['nativeBox']['manifest']['textures'])
        for name, spec in data['nativeBox']['manifest']['textures'].items():
            raw = base64.b64decode(data['nativeBox']['files'][spec['path']])
            digest = hashlib.sha256(raw).hexdigest()
            assert digest == spec['sha256'] and len(raw) == spec['byteLength']
            before_spec = base['nativeBox']['manifest']['textures'][name]
            before_raw = base64.b64decode(base['nativeBox']['files'][before_spec['path']])
            item = {'sha256': digest, 'bytes': len(raw), 'rawEqualToK': raw == before_raw}
            if name == 'colorMap':
                assert spec['dtype'] == 'float16' and spec['endian'] == 'little' and spec['channels'] == 4
                current = np.frombuffer(raw, dtype='<u2').reshape(spec['height'], spec['width'], 4)
                item['coverageAlphaBitsExactToK'] = bool(np.array_equal(current[..., 3], base_color[..., 3]))
                item['changedRGBTexelsToK'] = int(np.count_nonzero(np.any(current[..., :3] != base_color[..., :3], axis=2)))
                assert item['coverageAlphaBitsExactToK']
                assert item['rawEqualToK'] if key == 'K-view' else item['changedRGBTexelsToK'] > 0
            else:
                assert item['rawEqualToK'], (key, name)
            checks['maps'][name] = item
        receipt['cases'][key] = {'assetId': composed['asset'], 'assetPath': str(asset),
            'composed': str(composed_path) if composed_path else None,
            'composedSHA256': sha(composed_path) if composed_path else None,
            'assetManifestSHA256': sha(asset/'manifest.json'),
            'boxArchive': str(archive), 'boxArchiveSHA256': sha(archive),
            'html': str(html), 'htmlSHA256': sha(html), 'htmlBytes': html.stat().st_size,
            'build': build, 'payloadChecks': checks, 'preservation': composed['preservation'],
            'embeddedScripts': scripts, 'embeddedScriptsEqualToFreshF': scripts == current_scripts,
            'embeddedScriptsEqualToHistoricalK': scripts == receipt['historicalKEmbeddedScripts']}
        config['cases'][key] = {'label': label, 'html': str(html), 'assetId': composed['asset'],
                               'composed': str(composed_path) if composed_path else None,
                               'captures': [str(WORK/slug/'capture.json')]}
        write_json(WORK/'viewer-builds.json', receipt)
        print('OFFICIAL_VIEWER_READY', key, str(html), sha(html), flush=True)
        del data
    del base, base_color, base_color_raw
    for key, entry in config['cases'].items():
        control = WORK/f'{key}-capture-control.json'
        write_json(control, {'assetId': entry['assetId'], 'htmlSHA256': sha(entry['html']), 'display': DISPLAY})
        entry['control'] = str(control)
    write_json(WORK/'capture-cases.json', config)
    print('OFFICIAL_BUILD_PAYLOAD_CHECKS_PASSED', flush=True)
    if args.capture:
        captures = {}
        for key, entry in config['cases'].items():
            output = Path(entry['captures'][0]).parent
            command = ['node', str(CAPTURE), entry['html'], str(output), reference, '0,30,45,60', entry['control']]
            print('CAPTURE_START', key, flush=True)
            captures[key] = bounded_run(command, WORK/f'{key}-capture.log', 240)
            assert sha(entry['html']) == json.loads(Path(entry['control']).read_text())['htmlSHA256']
            captures[key]['captureJSON'] = entry['captures'][0]
            captures[key]['captureJSONSHA256'] = sha(entry['captures'][0])
            write_json(WORK/'capture-runs.json', captures)
            print('CAPTURE_FINISHED', key, captures[key]['elapsedSeconds'], flush=True)
        print('ALL_CAPTURE_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
