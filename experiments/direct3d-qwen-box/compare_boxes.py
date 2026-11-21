#!/usr/bin/env python3
"""Verify a selected RGB-only Box using unchanged official Inspector captures.

This command starts captures only when explicitly invoked by the experiment
owner. Existing --before-capture / --after-capture can be supplied instead.
No HTML, materials, meshes, textures, or source captures are modified.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

from compose_faces import compare_assets, file_hashes, manifest_path, sha, write_json

OLD = Path('/mnt/f/_dev/box-impostor-lab/experiments')
CAPTURE = OLD / 'zipsplat-qwen-20261120/capture_inspector.mjs'
ALPHA = OLD / 'zipsplat-qwen-20261120/compare_capture_alpha.py'
ENCODE = OLD / 'idol-qwen-box-20261120/scripts/encode_comparison.py'
PLAYWRIGHT = Path('/home/dev/Projects/box-impostor-lab/ComfyUI/custom_nodes/box_impostor_factory')


def command(argv, log, env=None):
    with log.open('x') as stream:
        subprocess.run([str(value) for value in argv], check=True, env=env, stdout=stream, stderr=subprocess.STDOUT)


def capture(args, side, viewer, reference, output):
    existing = getattr(args, side + '_capture')
    if existing:
        return existing.resolve(strict=True)
    env = dict(os.environ)
    for key in ('DISPLAY', 'WAYLAND_DISPLAY'):
        env.pop(key, None)
    destination = output / 'capture' / side
    if destination.exists():
        raise FileExistsError('Require a new capture directory: ' + str(destination))
    argv = ['node', args.capture_script, '--viewer', viewer, '--reference', reference,
            '--out', destination, '--frames', '240', '--size', '512', '--fps', '24',
            '--backend', 'swiftshader', '--mode', 'default', '--playwright-root', args.playwright_root]
    if args.chromium:
        argv.extend(['--chromium', args.chromium])
    command(argv, output / (side + '-capture.log'), env=env)
    return destination / 'capture.json'


def verify_capture(path, viewer, viewer_sha, reference, reference_sha, asset_id):
    data = json.loads(path.read_text())
    if (not data.get('completed') or data.get('errors') or data.get('framesExpected') != 240
            or len(data['frames']) != 240 or data.get('playbackFPS') != 24 or data.get('durationSeconds') != 10):
        raise ValueError('Capture must be a successful 240-frame, 24fps, 10-second sequence')
    if data['camera']['pixels'] != [512, 512] or data['camera']['projection'] != 'orthographic':
        raise ValueError('Require full 512x512 orthographic captures')
    if data['assetId'] != asset_id or data['sourceFaceResolution'] != 512:
        raise ValueError('Capture does not contain the expected native Box512 asset')
    if Path(data['html']).resolve() != viewer or data['htmlSHA256'] != viewer_sha or sha(viewer) != viewer_sha:
        raise ValueError('Official Inspector HTML path/hash changed or does not match capture')
    if (Path(data['referenceManifest']).resolve() != reference
            or data['referenceManifestSHA256'] != reference_sha or sha(reference) != reference_sha):
        raise ValueError('Capture reference manifest changed or differs')
    display = data['display']
    if display['mode'] != 'default' or display['materialOverride'] or display['toneMappingOverride']:
        raise ValueError('Require the official default display without material/tone overrides')
    for index, frame in enumerate(data['frames']):
        requested = (index * 1.5 + 180) % 360 - 180
        angle_error = lambda x, y: abs((float(x) - float(y) + 180) % 360 - 180)
        if frame['index'] != index or angle_error(frame['requestedYawDegrees'], requested) > 1e-9:
            raise ValueError('Capture frame order/yaw request differs from a complete turntable')
        if (not np.isfinite([frame['actualYawDegrees'], frame['actualPitchDegrees']]).all()
                or angle_error(frame['actualYawDegrees'], requested) >= .03
                or abs(frame['actualPitchDegrees']) >= .01):
            raise ValueError('Actual camera yaw or pitch misses the capture specification')
    return data


def coverage(data, path):
    records = []
    for frame in data['frames']:
        image_path = Path(frame['path'])
        image_path = (image_path if image_path.is_absolute() else path.parent / image_path).resolve(strict=True)
        if sha(image_path) != frame['sha256']:
            raise ValueError('Capture PNG changed during full-body validation')
        with Image.open(image_path) as image:
            if image.mode != 'RGBA' or image.size != (512, 512):
                raise ValueError('Full-body proof requires unscaled RGBA512 images')
            mask = np.asarray(image)[..., 3] > 127
        yy, xx = np.nonzero(mask)
        edges = int(mask[0].sum() + mask[-1].sum() + mask[1:-1, 0].sum() + mask[1:-1, -1].sum())
        if len(xx) <= 1000 or edges:
            raise ValueError('Empty/tiny or edge-clipped foreground at frame ' + str(frame['index']))
        records.append({'index': frame['index'], 'foregroundPixels': len(xx),
                        'boundsXYXY': [int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())],
                        'edgePixels': edges})
    return records


def run(args):
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError('Comparison output must be a fresh directory')
    before = manifest_path(args.baseline_asset)
    receipt_path = args.receipt.resolve(strict=True)
    receipt = json.loads(receipt_path.read_text())
    after = manifest_path(receipt['assetPath'])
    if output.is_relative_to(before.parent) or output.is_relative_to(after.parent):
        raise ValueError('Comparison output must be outside both immutable assets')
    if sha(receipt['archive']) != receipt['archiveSha256']:
        raise ValueError('Selected archive does not match its receipt')
    selected_faces = receipt['qa']['selectedFaces']
    asset_checks = compare_assets(before, after, selected_faces)
    if receipt['qa']['sourceManifestSha256'] != sha(before):
        raise ValueError('Selected variant came from a different baseline')
    if file_hashes(after.parent) != receipt['outputFiles']:
        raise ValueError('Selected output asset changed after composition')
    before_files, after_files = file_hashes(before.parent), file_hashes(after.parent)
    reference = (args.reference or before).resolve(strict=True)
    reference_sha = sha(reference)
    viewers = {side: getattr(args, side + '_viewer').resolve(strict=True) for side in ('before', 'after')}
    viewer_hashes = {side: sha(path) for side, path in viewers.items()}
    assets = {'before': json.loads(before.read_text()), 'after': json.loads(after.read_text())}
    initial_inputs = {str(path): sha(path) for path in
                      (before, after, receipt_path, reference, args.capture_script, args.alpha_script, Path(__file__),
                       Path(__file__).with_name('compose_faces.py'), *viewers.values())}
    output.mkdir(parents=True)
    try:
        paths = {side: capture(args, side, viewers[side], reference, output) for side in ('before', 'after')}
        captures = {side: verify_capture(paths[side], viewers[side], viewer_hashes[side], reference,
                                        reference_sha, assets[side]['asset_id']) for side in paths}
        for key in ('referenceManifestSHA256', 'display', 'backend', 'graphicsRenderer', 'browserVersion', 'scriptSHA256'):
            if captures['before'].get(key) != captures['after'].get(key):
                raise ValueError('Capture environment or script differs: ' + key)
        for side in paths:
            if captures[side]['scriptSHA256'] != sha(args.capture_script):
                raise ValueError('Capture script digest differs from the supplied script')
        alpha_path = output / 'alpha-240.json'
        command([sys.executable, args.alpha_script, '--left', paths['before'], '--right', paths['after'],
                 '--out', alpha_path], output / 'alpha-240.log')
        comparison = json.loads(alpha_path.read_text())
        if not comparison['comparisonValid']:
            raise ValueError('Pixel comparison failed validation')
        if comparison['camera']['verifiedFrames'] != 240 or comparison['camera']['maxAbsDifference'] != 0:
            raise ValueError('Recorded cameras must match exactly in all 240 frames')
        if not comparison['alpha']['all240FramesZero'] or comparison['alpha']['maxUint8'] != 0 or comparison['alpha']['nonzeroPixels'] != 0:
            raise ValueError('Rendered alpha changed; this variant fails the coverage-preservation gate')
        if comparison['rgb']['changedFrames'] <= 0:
            raise ValueError('Rendered RGB did not change')
        full_body = {side: coverage(captures[side], paths[side]) for side in paths}
        media = None
        if args.ffmpeg:
            if not args.encode_script.is_file():
                raise FileNotFoundError('Missing the existing two-column encoder')
            encoding_inputs = output / 'comparison-inputs.json'
            write_json(encoding_inputs, {'items': [
                {'label': args.before_label, 'capture': str(paths['before'])},
                {'label': args.after_label, 'capture': str(paths['after'])}]})
            command([sys.executable, args.encode_script, '--captures', encoding_inputs,
                     '--output', output / 'media', '--ffmpeg', args.ffmpeg], output / 'encode.log')
            media = {'directory': str(output / 'media'), 'files': file_hashes(output / 'media'),
                     'encodeScript': str(args.encode_script), 'encodeScriptSha256': sha(args.encode_script)}
        if file_hashes(before.parent) != before_files or file_hashes(after.parent) != after_files:
            raise RuntimeError('An immutable asset changed during comparison')
        if any(sha(path) != digest for path, digest in initial_inputs.items()):
            raise RuntimeError('An input or official Inspector HTML changed during comparison')
        for side in paths:
            if sha(paths[side]) != comparison['left' if side == 'before' else 'right']['captureSha256']:
                raise RuntimeError('Capture JSON changed after alpha validation')
        summary = {'completed': True, 'comparisonValid': True, 'method': 'official-inspector-rgb-only-box-240-v1',
                   'selectedFaces': selected_faces, 'assetChecks': asset_checks,
                   'receipt': {'path': str(receipt_path), 'sha256': sha(receipt_path)},
                   'inputHashes': initial_inputs,
                   'captures': {side: {'path': str(path), 'sha256': sha(path)} for side, path in paths.items()},
                   'captureEnvironmentEqual': True, 'recordedCamerasExactlyEqual': True,
                   'camera': comparison['camera'], 'alpha': comparison['alpha'], 'rgb': comparison['rgb'],
                   'fullBody240Frames': full_body, 'inspectorHTMLUnmodified': True,
                   'alphaReport': {'path': str(alpha_path), 'sha256': sha(alpha_path)},
                   'media': media, 'geometryDepthNormalIntervalRangeProxyCameraPreserved': True,
                   'limitations': [
                       'Alpha equality proves coverage preservation at these 240 cameras and 8-bit output precision.',
                       'Native depth/normal/interval/range files and metadata are byte-preserved; rendered depth buffers were not captured.',
                       'Mask registration and alpha equality do not prove texture quality, internal alignment, anatomical fidelity or multiview consistency.',
                       'Comparison images show the complete 512x512 frames; no face crops or image resizing are used.']}
        write_json(output / 'summary.json', summary)
        return {'completed': True, 'summary': str(output / 'summary.json'), 'summarySha256': sha(output / 'summary.json'),
                'alphaAll240Zero': True, 'cameraMaxAbsDifference': 0,
                'rgbChangedFrames': comparison['rgb']['changedFrames'], 'output': str(output)}
    except Exception as error:
        write_json(output / 'FAILED.json', {'completed': False, 'error': type(error).__name__ + ': ' + str(error),
                                           'inputHashes': initial_inputs})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-asset', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True, help='compose_faces.py immutable variant receipt')
    parser.add_argument('--before-viewer', type=Path, required=True, help='Unmodified official view.py HTML')
    parser.add_argument('--after-viewer', type=Path, required=True, help='Unmodified official view.py HTML')
    parser.add_argument('--reference', type=Path, help='Shared framing manifest; defaults to baseline manifest')
    parser.add_argument('--output', type=Path, required=True, help='New comparison directory')
    parser.add_argument('--before-capture', type=Path, help='Reuse an already verified-compatible 240-frame capture')
    parser.add_argument('--after-capture', type=Path, help='Reuse an already verified-compatible 240-frame capture')
    parser.add_argument('--capture-script', type=Path, default=CAPTURE)
    parser.add_argument('--alpha-script', type=Path, default=ALPHA)
    parser.add_argument('--playwright-root', type=Path, default=PLAYWRIGHT)
    parser.add_argument('--chromium', type=Path)
    parser.add_argument('--ffmpeg', type=Path, help='When supplied, encode the complete 2-column 24fps video')
    parser.add_argument('--encode-script', type=Path, default=ENCODE)
    parser.add_argument('--before-label', default='Direct3D Box / original')
    parser.add_argument('--after-label', default='Direct3D + Qwen / selected RGB')
    print(json.dumps(run(parser.parse_args()), indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()

