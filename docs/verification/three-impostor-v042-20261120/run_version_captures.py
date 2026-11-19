"""Capture exact official Inspector HTML for a package-version comparison."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
import zlib


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def scene_info(path):
    text = Path(path).read_text()
    match = re.search(r'<script id="scene-payload"[^>]*>([\s\S]*?)</script>', text)
    packed = base64.b64decode(match[1].strip())
    scene = json.loads(zlib.decompress(packed) if packed[0] == 0x78 else packed)
    box = scene['nativeBox']
    maps = {}
    for name, spec in box['manifest']['textures'].items():
        raw = base64.b64decode(box['files'][spec['path']])
        actual = hashlib.sha256(raw).hexdigest()
        assert actual == spec['sha256'] and len(raw) == spec['byteLength']
        maps[name] = {'sha256': actual, 'bytes': len(raw),
                      'width': spec['width'], 'height': spec['height'],
                      'channels': spec['channels'], 'dtype': spec['dtype']}
    def canonical(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    scripts = [{'attributes': attributes,
                'sha256': hashlib.sha256(contents.encode()).hexdigest()}
               for attributes, contents in re.findall(r'<script([^>]*)>([\s\S]*?)</script>', text)
               if 'scene-payload' not in attributes]
    return {'assetId': scene['assetId'], 'bounds': scene['bounds'],
            'geometryAndFallbackColorSHA256': canonical(scene['meshes']),
            'display': box['display'], 'maps': maps,
            'resolution': box['manifest']['resolution'],
            'embeddedScripts': scripts}


def run_bounded(command, log, cwd, temp_directory, timeout=240):
    start = time.monotonic()
    temp_directory.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update({key: str(temp_directory) for key in ['TMPDIR', 'TMP', 'TEMP']})
    with log.open('w') as stream:
        process = subprocess.Popen(command, cwd=cwd, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True, env=environment)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise RuntimeError(f'Owned capture process group timed out; log={log}')
    assert code == 0, f'Capture failed with exit {code}; log={log}'
    return {'command': command, 'log': str(log), 'logSHA256': digest(log),
            'elapsedSecondsIncludingBrowserStartup': time.monotonic() - start,
            'exitCode': code}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('--phase', choices=['prepare', 'visual', 'benchmark'], default='prepare')
    parser.add_argument('--family', help='Optional family to measure, e.g. Q2')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    work = Path(config['output'])
    work.mkdir(parents=True, exist_ok=True)
    all_cases = config['cases']
    cases = [case for case in all_cases if not args.family or case['family'] == args.family]
    assert cases
    prepared = {}
    baseline = {}
    for case in all_cases:
        name = case['name']
        html = Path(case['html'])
        assert digest(html) == case['htmlSHA256'], name
        info = scene_info(html)
        assert info['assetId'] == case['assetId'], (name, 'asset changed')
        assert info['display'] == config['display'], (name, 'display differs')
        if case['packageVersion'] == '0.4.1':
            baseline[case['family']] = info
        prepared[name] = {**case, 'payload': info, 'control': str(work / f'{name}-control.json')}
    checks = {}
    for name, case in prepared.items():
        base = baseline[case['family']]
        candidate = case['payload']
        for field in ['bounds', 'geometryAndFallbackColorSHA256', 'resolution', 'display']:
            assert candidate[field] == base[field], (name, field)
        for map_name, entry in base['maps'].items():
            assert candidate['maps'][map_name] == entry, (name, 'legacy map changed', map_name)
        additions = sorted(set(candidate['maps']) - set(base['maps']))
        assert set(additions) <= {'distanceMap'}, (name, additions)
        if additions:
            assert case.get('distanceDerivationReceipt'), 'New distance map needs a derivation receipt'
        checks[name] = {'legacyMapsByteIdentical': True, 'geometryAndFallbackColorsIdentical': True,
                        'newMaps': additions,
                        'distanceDerivationReceipt': case.get('distanceDerivationReceipt'),
                        'runtimeDiffersFrom041': candidate['embeddedScripts'] != base['embeddedScripts']}
        if case['packageVersion'] == '0.4.2':
            assert checks[name]['runtimeDiffersFrom041'], (name, '0.4.2 runtime is identical to 0.4.1')
    preparation = {'config': str(args.config), 'configSHA256': digest(args.config),
                   'scriptSHA256': digest(__file__), 'cases': prepared, 'inputChecks': checks,
                   'backend': 'WebGL2 / CPU SwiftShader',
                   'hitDepth': 'Not observed by the existing official snapshot API; output alpha and RGB are measured.'}
    write(work / 'prepared-captures.json', preparation)
    print('PREPARED', str(work / 'prepared-captures.json'), flush=True)
    if args.phase == 'prepare':
        return
    capture_script = Path(config['captureScript'])
    expected_capture_sha = config.get('captureScriptSHA256')
    if expected_capture_sha:
        assert digest(capture_script) == expected_capture_sha
    angles = config['angles'] if args.phase == 'visual' else [45]
    repetitions = 1 if args.phase == 'visual' else 3
    runs = []
    for trial in range(repetitions):
        ordered = cases if trial % 2 == 0 else list(reversed(cases))
        for case in ordered:
            suffix = case['name'] if args.phase == 'visual' else f'trial-{trial + 1}/{case["name"]}'
            output = work / ('captures' if args.phase == 'visual' else 'benchmark') / suffix
            output.mkdir(parents=True, exist_ok=True)
            control = {'assetId': case['assetId'], 'htmlSHA256': case['htmlSHA256'], 'display': config['display']}
            if args.phase == 'benchmark':
                control['benchmark'] = {'angle': 45, 'warmupFrames': 8, 'measuredFrames': 30}
            control_path = output / 'control.json'
            write(control_path, control)
            command = ['node', str(capture_script), case['html'], str(output),
                       config['referenceManifest'], ','.join(map(str, angles)), str(control_path)]
            run = run_bounded(command, output / 'capture.log', config['cwd'], work / 'tmp')
            assert digest(case['html']) == case['htmlSHA256'], 'Official HTML changed during capture'
            capture_path = output / 'capture.json'
            report = json.loads(capture_path.read_text())
            assert not report['errors'] and report['stageState'] == 'Native Box'
            assert 'SwiftShader' in report['graphicsRenderer']
            runs.append({**run, 'case': case['name'], 'family': case['family'],
                         'packageVersion': case['packageVersion'], 'trial': trial + 1,
                         'captureJSON': str(capture_path), 'captureJSONSHA256': digest(capture_path)})
            write(work / f'{args.phase}-runs.json', {'runs': runs})
            print('CAPTURE_COMPLETE', case['name'], trial + 1, flush=True)


if __name__ == '__main__':
    main()
