#!/usr/bin/env python3
"""Run the foreground -> native Box -> official Inspector stages for one branch."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

from zip_render_faces import sha

HERE = Path(__file__).resolve().parent


def write_json(path, data):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)


def snapshots(paths):
    files = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError('Stage did not create ' + str(path))
        files.extend(sorted(p for p in path.rglob('*') if p.is_file()) if path.is_dir() else [path])
    return {str(path.resolve()): sha(path) for path in files}


def process_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def stage(owner, name, inputs, outputs, command=None, action=None):
    owner.mkdir(parents=True, exist_ok=True)
    ledger_path = owner / 'stages.json'
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
    key_data = {'inputs': inputs, 'command': list(map(str, command)) if command else name}
    key = hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
    previous = ledger.get(name)
    if previous and previous['status'] == 'complete':
        if previous['key'] != key:
            raise ValueError(f'{name}: completed stage has different inputs/code; choose a new output/label')
        if snapshots(outputs) != previous['outputs']:
            raise ValueError(f'{name}: completed output checksum changed')
        print('REUSE', owner.name, name, flush=True)
        return previous
    if previous and previous['status'] == 'running' and process_alive(previous['pid']):
        raise RuntimeError(f'{name}: process {previous["pid"]} still owns this stage')
    leftovers = [path for path in outputs if path.exists()]
    if leftovers and not previous:
        raise FileExistsError(f'{name}: output exists without an owned stage record: {leftovers}')
    backups = []
    for path in leftovers:
        backup = path.with_name(path.name + '.incomplete-' + str(time.time_ns()))
        path.rename(backup)
        backups.append(str(backup))
    record = {'key': key, **key_data, 'status': 'running', 'pid': os.getpid(),
              'startedEpoch': time.time(), 'incompleteOutputsPreserved': backups}
    ledger[name] = record
    write_json(ledger_path, ledger)
    print('START', owner.name, name, flush=True)
    log = owner / (name + '.log')
    try:
        if command:
            with log.open('a', encoding='utf-8') as stream:
                stream.write('\nCOMMAND ' + json.dumps(list(map(str, command))) + '\n')
                stream.flush()
                subprocess.run(list(map(str, command)), check=True, stdout=stream, stderr=subprocess.STDOUT)
        else:
            action()
        record.update(status='complete', outputs=snapshots(outputs), seconds=time.time() - record['startedEpoch'])
    except Exception as error:
        record.update(status='failed', error=str(error), seconds=time.time() - record['startedEpoch'])
        write_json(ledger_path, ledger)
        raise
    write_json(ledger_path, ledger)
    print('DONE', owner.name, name, round(record['seconds'], 2), 'seconds', flush=True)
    return record


def script(name):
    return HERE / name


def script_inputs(name):
    values = {name: sha(script(name))}
    # These helpers import common color/coordinate/hash utilities.
    if name in ('baseline_to_faces.py', 'export_canonical_gs.py'):
        values['zip_render_faces.py'] = sha(script('zip_render_faces.py'))
    return values


def mask_identity(path):
    data = json.loads(path.read_text())
    files = {}
    for item in data['views']:
        mask = Path(item['mask_path'])
        mask = mask if mask.is_absolute() else path.parent / mask
        files[item['id']] = {'path': str(mask.resolve()), 'sha256': sha(mask)}
    return {'manifestSha256': sha(path), 'masks': files}


def pack_box(asset, destination):
    with zipfile.ZipFile(destination, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(asset.rglob('*')):
            if path.is_file():
                archive.write(path, path.relative_to(asset).as_posix())
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise ValueError('Box ZIP integrity check failed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initial-dir', type=Path, help='Required unless all prepared inputs are supplied')
    parser.add_argument('--branch-dir', type=Path, required=True)
    parser.add_argument('--masks', type=Path, help='Required unless all prepared inputs are supplied')
    parser.add_argument('--template', type=Path, help='Required unless all prepared inputs are supplied')
    parser.add_argument('--prepared-foreground', type=Path, help='Existing directory containing selected gaussians.npy')
    parser.add_argument('--prepared-transform', type=Path, help='Existing branch canonical transform JSON')
    parser.add_argument('--prepared-frame', type=Path, help='Existing common frame manifest.json')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--inspector-viewpy', type=Path, required=True)
    parser.add_argument('--inspector-python', type=Path, required=True)
    parser.add_argument('--label', help='Branch folder name; defaults to branch-dir basename')
    args = parser.parse_args()
    prepared_values = (args.prepared_foreground, args.prepared_transform, args.prepared_frame)
    prepared = all(value is not None for value in prepared_values)
    if any(value is not None for value in prepared_values) and not prepared:
        parser.error('--prepared-foreground, --prepared-transform and --prepared-frame must be supplied together')
    if not prepared:
        missing = [flag for flag in ('initial-dir', 'masks', 'template')
                   if getattr(args, flag.replace('-', '_')) is None]
        if missing:
            parser.error('automatic preparation requires ' + ', '.join('--' + flag for flag in missing))
    for name in ('initial_dir', 'branch_dir', 'masks', 'template', 'out', 'inspector_viewpy',
                 'prepared_foreground', 'prepared_transform', 'prepared_frame'):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    args.inspector_python = args.inspector_python.absolute()  # Preserve a virtualenv executable symlink.
    label = args.label or args.branch_dir.name
    if not label or label in ('.', '..') or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in label):
        raise ValueError('Use a plain alphanumeric branch label with optional dash/underscore')
    shared, branch = args.out / 'shared', args.out / 'branches' / label
    branch_gs, branch_cameras = args.branch_dir / 'gaussians.npy', args.branch_dir / 'cameras.json'
    masks = mask_identity(args.masks) if args.masks is not None else None
    prepared_inputs = None
    if prepared:
        foreground, branch_transform, frame_manifest = (
            args.prepared_foreground, args.prepared_transform, args.prepared_frame)
        if not foreground.is_dir():
            parser.error('--prepared-foreground must be a directory')
        if not (foreground / 'gaussians.npy').is_file():
            parser.error('--prepared-foreground must contain gaussians.npy')
        prepared_inputs = {
            'foreground': {'path': str(foreground), 'files': snapshots([foreground])},
            'transform': {'path': str(branch_transform), 'sha256': sha(branch_transform)},
            'frame': {'path': str(frame_manifest), 'sha256': sha(frame_manifest)},
        }
    else:
        initial_gs, initial_cameras = args.initial_dir / 'gaussians.npy', args.initial_dir / 'cameras.json'
        shared_fg = shared / 'foreground'
        initial_inputs = {'gaussians': sha(initial_gs), 'cameras': sha(initial_cameras), 'masks': masks,
                          **script_inputs('zip_render_faces.py')}
        stage(shared, 'select', initial_inputs, [shared_fg], [sys.executable, script('zip_render_faces.py'), 'select',
              '--gaussians', initial_gs, '--cameras', initial_cameras, '--masks', args.masks, '--output', shared_fg])
        canonical, frame = shared / 'canonical.json', shared / 'frame'
        frame_inputs = {'foreground': sha(shared_fg / 'gaussians.npy'), 'cameras': sha(initial_cameras),
                        'template': sha(args.template), **script_inputs('zip_render_faces.py')}
        stage(shared, 'align', frame_inputs, [canonical, frame], [sys.executable, script('zip_render_faces.py'), 'align',
              '--gaussians', shared_fg / 'gaussians.npy', '--cameras', initial_cameras, '--template', args.template,
              '--output', canonical, '--frame-dir', frame])
        foreground = branch / 'foreground'
        stage(branch, 'select', {'gaussians': sha(branch_gs), 'cameras': sha(branch_cameras), 'masks': masks,
              **script_inputs('zip_render_faces.py')}, [foreground], [sys.executable, script('zip_render_faces.py'), 'select',
              '--gaussians', branch_gs, '--cameras', branch_cameras, '--masks', args.masks, '--output', foreground])
        branch_transform = branch / 'canonical.json'
        stage(branch, 'gauge', {'initialCameras': sha(initial_cameras), 'branchCameras': sha(branch_cameras),
              'canonical': sha(canonical), **script_inputs('align_camera_gauge.py')}, [branch_transform],
              [sys.executable, script('align_camera_gauge.py'), '--initial-cameras', initial_cameras,
               '--branch-cameras', branch_cameras, '--canonical', canonical, '--output', branch_transform])
        frame_manifest = frame / 'manifest.json'
    faces, boxes = branch / 'faces512', branch / 'box'
    render_inputs = {'gaussians': sha(foreground / 'gaussians.npy'), 'transform': sha(branch_transform),
                     'frame': sha(frame_manifest), **script_inputs('zip_render_faces.py')}
    if prepared:
        render_inputs['preparedInputs'] = prepared_inputs
    stage(branch, 'render', render_inputs, [faces],
          [sys.executable, script('zip_render_faces.py'), 'render', '--gaussians', foreground / 'gaussians.npy',
           '--template', frame_manifest, '--transform', branch_transform, '--output', faces, '--resolution', '512'])
    stage(branch, 'build', {'captureStatus': sha(faces / 'status.json'), 'frame': sha(frame_manifest),
          **script_inputs('build_metric_box.py')}, [boxes], [sys.executable, script('build_metric_box.py'),
          '--template', frame_manifest, '--captures', faces, '--output', boxes])
    manifests = list(boxes.glob('*/manifest.json'))
    if len(manifests) != 1:
        raise ValueError('Expected exactly one completed Box manifest')
    asset = manifests[0].parent
    box_zip = branch / (label + '.box.zip')
    stage(branch, 'zip', {'assetFiles': snapshots([asset]), 'pipelineSha256': sha(__file__)}, [box_zip],
          action=lambda: pack_box(asset, box_zip))
    ply = branch / (label + '-canonical.ply')
    stage(branch, 'canonical-ply', {'gaussians': sha(foreground / 'gaussians.npy'), 'transform': sha(branch_transform),
          **script_inputs('export_canonical_gs.py')}, [ply, ply.with_suffix('.ply.json')],
          [sys.executable, script('export_canonical_gs.py'), '--input', foreground / 'gaussians.npy',
           '--transform', branch_transform, '--output', ply])
    viewers = {}
    for kind, source in (('box', box_zip), ('gs', ply)):
        html = branch / (label + '-' + kind + '.inspector.html')
        stage(branch, 'inspector-' + kind, {'sourceSha256': sha(source), 'viewPySha256': sha(args.inspector_viewpy)}, [html],
              [args.inspector_python, args.inspector_viewpy, source, '-o', html])
        viewers[kind] = {'path': str(html), 'sha256': sha(html)}
    result = {'label': label, 'initialDir': str(args.initial_dir) if args.initial_dir is not None else None,
              'branchDir': str(args.branch_dir), 'preparationMode': 'prepared' if prepared else 'automatic',
              'preparedInputs': prepared_inputs, 'foregroundDirectory': str(foreground),
              'sourceGaussiansSha256': sha(branch_gs), 'sourceCamerasSha256': sha(branch_cameras),
              'maskIdentity': masks, 'sharedFrameManifest': str(frame_manifest),
              'sharedFrameSha256': sha(frame_manifest), 'canonicalTransform': str(branch_transform),
              'canonicalTransformSha256': sha(branch_transform), 'assetPath': str(asset),
              'assetId': json.loads(manifests[0].read_text())['asset_id'], 'resolution': 512,
              'boxZip': {'path': str(box_zip), 'sha256': sha(box_zip)},
              'canonicalPly': {'path': str(ply), 'sha256': sha(ply)}, 'viewers': viewers,
              'viewerPolicy': 'Official view.py output, unchanged bytes',
              'limitations': ['Gaussian center masks approximate foreground segmentation.',
                              'Expected-depth Box intervals do not recover hidden cavities.',
                              'Retained proxy GLB is schema metadata and does not represent the new GS.']}
    write_json(branch / 'result.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
