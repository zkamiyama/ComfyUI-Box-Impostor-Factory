#!/usr/bin/env python3
"""Render the three saved models serially, export native Boxes and exact Inspectors."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--factory-root', type=Path, required=True)
    parser.add_argument('--comfy-root', type=Path, required=True)
    parser.add_argument('--metric-renderer', type=Path, required=True)
    parser.add_argument('--inspector', type=Path, required=True)
    parser.add_argument('--ffmpeg', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    scripts, final = root / 'scripts', root / 'final'
    final.mkdir(exist_ok=True)
    template = root / 'common-frame/manifest.json'
    models = json.loads((root / 'models.json').read_text())['models']
    ordered = sorted(models, key=lambda item: {'idol20': 0, 'idol50': 1, 'tripo20': 2}[item['id']])
    result = {'completed': False, 'models': {}, 'commands': [],
              'inspectorSHA256': sha(args.inspector), 'commonFrameSHA256': sha(template)}
    env = dict(os.environ)
    env['PYTHONPATH'] = str(args.factory_root) + os.pathsep + env.get('PYTHONPATH', '')

    def run(name, command):
        command = list(map(str, command))
        started = time.monotonic()
        print(json.dumps({'stage': name, 'started': True}), flush=True)
        with (root / 'logs' / (name + '.log')).open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, env=env)
        result['commands'].append({'name': name, 'command': command,
                                   'seconds': time.monotonic() - started})
        (root / 'render-comparison.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({'stage': name, 'completed': True}), flush=True)

    for item in ordered:
        name, filename = item['id'], item['file']
        base = [sys.executable, scripts / 'render_native_splat.py']
        shared = ['--splat', item['splat'], '--template', template, '--rotate-y', item['rotate_y'],
                  '--comfy-root', args.comfy_root, '--factory-root', args.factory_root]
        native, faces = root / 'render' / (name + '-common-native'), root / 'render' / (name + '-faces512')
        ply = final / (filename + '.ply')
        run(name + '-native', base + ['turntable'] + shared + ['--output', native, '--export-ply', ply])
        run(name + '-faces', base + ['faces'] + shared + ['--output', faces, '--metric-renderer', args.metric_renderer])
        run(name + '-box', [sys.executable, scripts / 'build_metric_box.py', '--template', template,
                           '--captures', faces, '--output', root / 'box' / name])
        manifests = list((root / 'box' / name).glob('*/manifest.json'))
        if len(manifests) != 1:
            raise ValueError('Expected one completed Box manifest')
        asset = manifests[0].parent
        archive = final / (filename + '_Box512.box.zip')
        with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zip_file:
            for path in sorted(asset.rglob('*')):
                if path.is_file():
                    zip_file.write(path, path.relative_to(asset).as_posix())
        with zipfile.ZipFile(archive) as zip_file:
            if zip_file.testzip() is not None:
                raise ValueError('Box archive integrity failed')
        viewers = {}
        for kind, source in [('GS', ply), ('Box512', archive)]:
            viewer = final / (filename + '_' + kind + '.inspector.html')
            run(name + '-' + kind + '-inspector', [sys.executable, args.inspector, source, '-o', viewer])
            viewers[kind] = {'path': str(viewer), 'sha256': sha(viewer)}
        coverage = json.loads((faces / 'status.json').read_text())
        result['models'][name] = {'ply': str(ply), 'plySha256': sha(ply), 'box': str(archive),
            'boxSha256': sha(archive), 'assetPath': str(asset), 'viewers': viewers,
            'nativeStatus': str(native / 'status.json'), 'faceStatus': str(faces / 'status.json'),
            'outOfBoundsAlphaPixels': {k: v['outOfBoundsAlphaPixels'] for k, v in coverage['faces'].items()},
            'faceEdgePixels': {k: v['atlasCoverage']['edgePixels'] for k, v in coverage['faces'].items()}}
        (root / 'render-comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    run('native-comparison', [sys.executable, scripts / 'encode_native_pair.py',
        '--left', result['models']['tripo20']['nativeStatus'],
        '--right', result['models']['idol20']['nativeStatus'],
        '--third', result['models']['idol50']['nativeStatus'],
        '--left-label', 'TripoSplat / 20 steps', '--right-label', 'Idol / 20 steps',
        '--third-label', 'Idol / 50 steps', '--allow-rigid-y-registration',
        '--output', root / 'media/native', '--ffmpeg', args.ffmpeg])
    shutil.copyfile(root / 'media/native/native-gs-comparison.mp4', final / 'Idol_vs_TripoSplat_3DGS.mp4')
    for angle in (0, 90, 180, 270):
        shutil.copyfile(root / ('media/native/poster-yaw%03d.png' % angle),
                        final / ('Idol_vs_TripoSplat_3DGS_%03d.png' % angle))
    result['completed'] = True
    (root / 'render-comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    print('COMPLETE ' + str(root / 'render-comparison.json'), flush=True)


if __name__ == '__main__':
    main()
