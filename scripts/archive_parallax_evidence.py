#!/usr/bin/env python3
"""Inventory, package, and verify explicitly selected RGB-parallax evidence.

The inventory command never creates archives. The pack command requires an exact
frozen inventory and emits deterministic, independently readable tar.gz parts.
Native textures stay external and can be recovered from the unchanged official
Inspector HTML with extract-native; no model weights or caches are collected.
"""
import argparse
import base64
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import tarfile
import zlib


VERSION = 'parallax-evidence-archive-v1'
ALLOWED = {'.png', '.webp', '.webm', '.json', '.mjs', '.js', '.py', '.html', '.glsl', '.txt', '.log', '.md'}
MAX_COMPRESSED_BYTES = 100_000_000
PART_SOURCE_BUDGET = 90_000_000


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def read_json(path):
    return json.loads(Path(path).read_bytes())


def relative_name(value):
    name = PurePosixPath(value)
    if (not name.parts or name.is_absolute() or '..' in name.parts or '\\' in str(name)
            or any(p.startswith('.') for p in name.parts)):
        raise ValueError('Unsafe archive-relative path: ' + str(value))
    return str(name)


def regular_file(path):
    path = Path(path).absolute()
    if path.is_symlink() or not path.is_file():
        raise ValueError('Expected a regular source file: ' + str(path))
    return path


def write_new_or_equal(path, data):
    path = Path(path)
    if path.exists():
        if path.is_symlink() or path.read_bytes() != data:
            raise FileExistsError('Preserve existing output: ' + str(path))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(data)


def decoded_png_hash(path, dimensions=None):
    # Data PNGs must never pass through compositing or color management.
    from PIL import Image
    with Image.open(path) as image:
        if image.mode != 'RGBA' or (dimensions and image.size != tuple(dimensions)):
            raise ValueError('Unexpected packed-data PNG layout: ' + str(path))
        return digest(image.tobytes())


def collect(plan_path):
    plan_path = regular_file(plan_path)
    plan = read_json(plan_path)
    if plan.get('schemaVersion') != 1 or not plan.get('runs'):
        raise ValueError('An explicit nonempty run list is required.')
    base = Path(plan['experimentRoot']).absolute()
    rows, missing, captures, external, code_links = {}, [], [], [], []

    def add(path, archive_path, role, expected=None):
        name = relative_name(archive_path)
        path = regular_file(path)
        if path.suffix.lower() not in ALLOWED or path.name.startswith('.'):
            raise ValueError('Unapproved evidence file type: ' + str(path))
        actual = sha(path)
        if expected is not None and actual != expected:
            raise ValueError('Recorded source checksum differs: ' + str(path))
        row = {'archivePath': name, 'sourcePath': str(path), 'bytes': path.stat().st_size,
               'sha256': actual, 'role': role}
        if name in rows and rows[name] != row:
            raise ValueError('Conflicting archive member: ' + name)
        rows[name] = row
        return name

    def directory(path, prefix, role):
        path = Path(path)
        if path.is_symlink() or not path.is_dir():
            raise ValueError('Expected an explicit regular evidence directory: ' + str(path))
        for item in sorted(path.rglob('*')):
            if item.is_symlink():
                raise ValueError('Symlink in selected evidence directory: ' + str(item))
            if item.is_file():
                add(item, prefix + '/' + item.relative_to(path).as_posix(), role)

    def checked(path, expected):
        if sha(path) != expected:
            raise ValueError('Capture/source bytes differ: ' + str(path))

    def snapshot(relative, expected, run):
        relative = relative_name(relative)
        candidates = [base / 'harness-history' / expected / relative,
                      base / 'harness' / relative]
        candidates.extend(Path(p) for p in plan.get('sourceOverrides', {}).get(expected, []))
        found = next((p for p in candidates if p.is_file() and not p.is_symlink() and sha(p) == expected), None)
        if found is None:
            missing.append({'run': run, 'relativeSource': relative, 'expectedSHA256': expected,
                            'checkedPaths': [str(p) for p in candidates]})
            return
        member = add(found, 'sources/' + expected + '/' + relative, 'exact-capture-source', expected)
        code_links.append({'run': run, 'relativeSource': relative, 'sha256': expected,
                           'archivePath': member, 'origin': plan.get('sourceOrigins', {}).get(expected, 'hash-matched saved source')})

    seen = set()
    for entry in plan['runs']:
        name = relative_name(entry['name'])
        if '/' in name or name in seen:
            raise ValueError('Run names must be unique immediate children.')
        seen.add(name)
        folder = base / name
        sweep_path = folder / 'sweep.json'
        sweep = read_json(sweep_path)
        if sweep.get('status') == 'running':
            raise ValueError('Capture is still running: ' + name)
        directory(folder, 'runs/' + name, 'original-capture-or-diagnostic')
        record = {'name': name, 'purpose': entry['purpose'], 'captureStatus': sweep.get('status'),
                  'sweepSHA256': sha(sweep_path), 'frames': len(sweep.get('frames', [])),
                  'evaluations': entry.get('evaluations', [])}
        captures.append(record)
        for relative, expected in sweep['sourceFilesSHA256'].items():
            snapshot(relative, expected, name)
        add(sweep['cameraReference'], 'cameras/' + sweep['cameraReferenceSHA256'] + '.json',
            'recorded-capture-camera', sweep['cameraReferenceSHA256'])
        checked(sweep['source']['manifestPath'], sweep['source']['manifestSHA256'])
        if sweep.get('frontAtlasReference'):
            ref = sweep['frontAtlasReference']
            checked(ref['path'], ref['sha256'])
        frames = list(sweep.get('frames', []))
        frames += list(sweep.get('nativeGeometryControl', {}).get('images', {}).values())
        data_checks = set()
        for frame in frames:
            capture_paths = [frame[k] for k in ('rgbaPNG', 'facePNG', 'depthPNG') if frame.get(k)]
            if frame.get('rawFloatRGBA'):
                capture_paths.append(frame['rawFloatRGBA']['path'])
            for capture_path in capture_paths:
                if not Path(capture_path).resolve().is_relative_to(folder.resolve()):
                    raise ValueError('Capture data lies outside its selected run: ' + capture_path)
            if 'rgbaPNG' in frame:
                checked(frame['rgbaPNG'], frame['rgbaSHA256'])
            if frame.get('faceSHA256'):
                checked(frame['facePNG'], frame['faceSHA256'])
            if frame.get('depthPNG'):
                key = (frame['depthPNG'], frame['depthRawSHA256'])
                if key not in data_checks:
                    actual = decoded_png_hash(key[0], [sweep['width'], sweep['height']])
                    if actual != key[1]:
                        raise ValueError('Packed Float32 depth bytes differ: ' + key[0])
                    data_checks.add(key)
            if frame.get('rawFloatRGBA'):
                raw = frame['rawFloatRGBA']
                if decoded_png_hash(raw['path'], [raw['pngWidth'], raw['pngHeight']]) != raw['rawSHA256']:
                    raise ValueError('Packed raw RGB Float32 bytes differ: ' + raw['path'])
        for shader in sweep.get('compiledFragmentShaders', []):
            checked(shader['path'], shader['sha256'])
        record['packedDepthFilesDecodedAndVerified'] = len(data_checks)
        for eval_name in entry.get('evaluations', []):
            if '/' in relative_name(eval_name):
                raise ValueError('Evaluation must be an explicit sibling directory.')
            evaluation_dir = base / eval_name
            report = read_json(evaluation_dir / 'evaluation.json')
            if report['sweepSHA256'] != record['sweepSHA256']:
                raise ValueError('Evaluation refers to another sweep: ' + eval_name)
            directory(evaluation_dir, 'evaluations/' + eval_name, 'evaluation-and-exact-source')
            evaluator = evaluation_dir / 'evaluator-source.py'
            checked(evaluator, report['evaluatorSHA256'])
            for p, expected in report.get('inputFrameFileSHA256', {}).items():
                checked(p, expected)
        for file in entry.get('additionalFiles', []):
            add(base / file, 'logs/' + relative_name(file), 'explicit-run-log')

    for asset in plan['assets']:
        name = relative_name(asset['name'])
        manifest_path = regular_file(asset['manifestPath'])
        manifest = read_json(manifest_path)
        add(manifest_path, 'inputs/' + name + '/manifest.json', 'full-original-native-manifest')
        maps = {}
        for key, desc in manifest['textures'].items():
            relative = relative_name(desc['path'])
            path = regular_file(manifest_path.parent / relative)
            if path.stat().st_size != desc['byteLength'] or sha(path) != desc['sha256']:
                raise ValueError('External native map differs: ' + str(path))
            maps[key] = {'relativePath': relative, 'bytes': desc['byteLength'], 'sha256': desc['sha256']}
        recovery = asset['officialInspectorRecovery']
        if not all(recovery.get(k) for k in ('gitCommit', 'gitPath', 'gzipSHA256', 'htmlSHA256')):
            raise ValueError('Pinned official Inspector retrieval information is required.')
        external.append({'name': name, 'assetId': manifest.get('asset_id'),
                         'manifestArchivePath': 'inputs/' + name + '/manifest.json',
                         'manifestSHA256': sha(manifest_path), 'captureManifestSHA256': sha(manifest_path),
                         'manifestKind': 'full original Factory capture manifest', 'maps': maps,
                         'officialInspectorRecovery': recovery,
                         'reasonExternal': 'Exact native texture bytes are already embedded in the archived official Inspector HTML.'})
    for item in plan.get('additionalCaptureReports', []):
        report = read_json(item['path'])
        for relative, expected in report['sourceFilesSHA256'].items():
            snapshot(relative, expected, item['name'])
    asset_hashes = {x['manifestSHA256'] for x in external}
    for entry in plan['runs']:
        sweep = read_json(base / entry['name'] / 'sweep.json')
        if sweep['source']['manifestSHA256'] not in asset_hashes:
            raise ValueError('Captured full source manifest is missing from selected assets.')
    for item in plan.get('files', []):
        add(item['path'], item['archivePath'], item['role'], item.get('sha256'))
    for item in plan.get('evidenceDirectories', []):
        directory(item['path'], relative_name(item['archivePath']), item['role'])
    add(plan_path, 'archive-selection.json', 'explicit-frozen-selection')
    add(__file__, 'tools/archive_parallax_evidence.py', 'archive-and-recovery-helper')
    members = sorted(rows.values(), key=lambda row: row['archivePath'])
    return {'kind': VERSION, 'planPath': str(plan_path), 'planSHA256': sha(plan_path),
            'toolSHA256': sha(__file__), 'runs': captures, 'members': members,
            'externalAssets': external, 'captureSourceMapping': code_links,
            'qualityStatus': plan.get('qualityStatus'),
            'missingExactSources': missing, 'byteInputsComplete': not missing,
            'selectionStatus': plan.get('selectionStatus', 'draft'),
            'readyToArchive': not missing and plan.get('selectionStatus') == 'frozen',
            'sourceMemberCount': len(members), 'sourceMemberBytes': sum(r['bytes'] for r in members),
            'archiveCreationRequested': False,
            'partSourceBudgetBytes': PART_SOURCE_BUDGET, 'maxCompressedPartBytes': MAX_COMPRESSED_BYTES,
            'recovery': {'command': 'python tools/archive_parallax_evidence.py extract-native --inventory archive-inventory.json --asset H --html H.inspector.html.gz --manifest inputs/H/manifest.json --output recovered/H',
                         'retrieval': 'From this project repository, git show <gitCommit>:<gitPath> > <stage>.inspector.html.gz; verify gzipSHA256, then use extract-native. Both exact hashes are pinned per asset.',
                         'manifest': 'Use the included full original manifest; the HTML contains only its native rendering subset.',
                         'dataPNG': 'Packed depth/raw-RGB PNGs contain Float32 bytes. Decode RGBA bytes directly without compositing or color conversion.',
                         'absolutePaths': 'Original JSON absolute paths are preserved as provenance. Use each member sourcePath-to-archivePath mapping to relocate inputs when re-running analysis.',
                         'dependencies': 'Install the included package-lock.json dependencies and a compatible Playwright Chromium runtime; model weights are not required for capture/evaluation.'}}


def tar_size(rows):
    return 10240 + sum(512 + ((r['bytes'] + 511) // 512) * 512 for r in rows)


def partition(rows, budget=PART_SOURCE_BUDGET):
    parts, current = [], []
    for row in rows:
        if tar_size([row]) > budget:
            raise ValueError('Single evidence member exceeds bounded part size: ' + row['archivePath'])
        if current and tar_size(current + [row]) > budget:
            parts.append(current)
            current = []
        current.append(row)
    if current:
        parts.append(current)
    return parts


def verify(index_path):
    index_path = regular_file(index_path)
    index = read_json(index_path)
    expected = {row['archivePath']: row for row in index['members']}
    seen = set()
    for part in index['parts']:
        path = index_path.parent / relative_name(part['file'])
        if sha(path) != part['sha256'] or path.stat().st_size != part['bytes']:
            raise ValueError('Archive part differs: ' + str(path))
        with tarfile.open(path, 'r:gz') as tar:
            for member in tar:
                name = relative_name(member.name)
                if not member.isfile() or name in seen or name not in expected:
                    raise ValueError('Unexpected or duplicate tar member: ' + name)
                row = expected[name]
                if row['part'] != part['file'] or member.size != row['bytes']:
                    raise ValueError('Tar size/part differs: ' + name)
                stream = tar.extractfile(member)
                if hashlib.file_digest(stream, 'sha256').hexdigest() != row['sha256']:
                    raise ValueError('Archived member checksum differs: ' + name)
                seen.add(name)
    if seen != set(expected):
        raise ValueError('Archive members are missing.')
    return {'kind': VERSION + '-independent-byte-verification', 'status': 'verified',
            'indexSHA256': sha(index_path), 'parts': len(index['parts']), 'verifiedMembers': len(seen),
            'sourcePathsRequired': False, 'allMembersReopenedAndHashed': True}


def pack(plan_path, inventory_path, output, stem):
    frozen = read_json(inventory_path)
    actual = collect(plan_path)
    if actual != frozen or not actual['readyToArchive']:
        raise ValueError('Inventory changed or exact source snapshots are missing. Freeze a new explicit inventory first.')
    output = Path(output)
    if '/' in relative_name(stem):
        raise ValueError('Archive stem must be a filename prefix.')
    inventory_bytes = encoded(frozen)
    embedded = {'archivePath': 'archive-inventory.json', 'sourcePath': None,
                'bytes': len(inventory_bytes), 'sha256': digest(inventory_bytes), 'role': 'embedded-inventory'}
    members = sorted(frozen['members'] + [embedded], key=lambda row: row['archivePath'])
    groups = partition(members)
    targets = [output / (stem + '-part%02d.tar.gz' % (i + 1)) for i in range(len(groups))]
    index_path = output / (stem + '-index.json')
    if index_path.exists() or any(p.exists() or p.with_suffix(p.suffix + '.partial').exists() for p in targets):
        raise FileExistsError('Preserve existing archive outputs; choose a new stem.')
    output.mkdir(parents=True, exist_ok=True)
    result = {'kind': VERSION + '-bundle', 'frozenInventorySHA256': sha(inventory_path),
              'runs': frozen['runs'], 'qualityStatus': frozen.get('qualityStatus'), 'parts': [], 'members': []}
    for target, group in zip(targets, groups):
        temporary = target.with_suffix(target.suffix + '.partial')
        with temporary.open('xb') as raw:
            with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0, compresslevel=9) as compressed:
                with tarfile.open(fileobj=compressed, mode='w|', format=tarfile.USTAR_FORMAT) as tar:
                    for row in group:
                        info = tarfile.TarInfo(row['archivePath'])
                        info.size, info.mtime, info.mode = row['bytes'], 0, 0o644
                        info.uid = info.gid = 0
                        info.uname = info.gname = ''
                        if row['sourcePath'] is None:
                            tar.addfile(info, io.BytesIO(inventory_bytes))
                        else:
                            with Path(row['sourcePath']).open('rb') as stream:
                                tar.addfile(info, stream)
                        result['members'].append({**row, 'part': target.name})
        if temporary.stat().st_size >= MAX_COMPRESSED_BYTES:
            raise ValueError('Compressed part exceeds 100 MB ceiling; partial retained: ' + str(temporary))
        temporary.rename(target)
        result['parts'].append({'file': target.name, 'sha256': sha(target), 'bytes': target.stat().st_size})
    for row in frozen['members']:
        if sha(row['sourcePath']) != row['sha256']:
            raise ValueError('Source changed during packaging: ' + row['sourcePath'])
    write_new_or_equal(index_path, encoded(result))
    proof = verify(index_path)
    write_new_or_equal(output / (stem + '-verification.json'), encoded(proof))
    return {'index': str(index_path), 'parts': result['parts'], 'verification': proof}


def extract_native(inventory_path, name, html_path, manifest_path, output):
    asset = next(a for a in read_json(inventory_path)['externalAssets'] if a['name'] == name)
    recovery = asset['officialInspectorRecovery']
    html = Path(html_path).read_bytes()
    if html[:2] == b'\x1f\x8b':
        if digest(html) != recovery['gzipSHA256']:
            raise ValueError('Official Inspector gzip checksum differs.')
        html = gzip.decompress(html)
    if digest(html) != recovery['htmlSHA256']:
        raise ValueError('Official Inspector HTML checksum differs.')
    manifest_bytes = Path(manifest_path).read_bytes()
    if digest(manifest_bytes) != asset['manifestSHA256']:
        raise ValueError('Full original manifest checksum differs.')
    manifest = json.loads(manifest_bytes)
    match = re.search(rb'<script id="scene-payload"[^>]*>(.*?)</script>', html, re.S)
    if not match:
        raise ValueError('Official scene-payload is absent.')
    packed = base64.b64decode(re.sub(rb'\s+', b'', match.group(1)), validate=True)
    payload = zlib.decompress(packed) if packed[:1] == b'\x78' else packed
    scene = json.loads(payload)
    if scene['previewType'] != 'native-box' or scene['assetId'] != asset['assetId']:
        raise ValueError('Official payload is not the recorded native Box asset.')
    native = scene['nativeBox']
    subset_bytes = (json.dumps(native['manifest'], indent=2) + '\n').encode()
    expected_subset = recovery.get('recoveredInspectorManifestSHA256')
    if expected_subset and digest(subset_bytes) != expected_subset:
        raise ValueError('Recorded Inspector rendering-subset manifest checksum differs.')
    for key in ('resolution', 'layout', 'bakeBox'):
        if native['manifest'][key] != manifest[key]:
            raise ValueError('Full manifest and embedded native geometry differ: ' + key)
    output = Path(output)
    recovered = []
    for key, desc in manifest['textures'].items():
        path = relative_name(desc['path'])
        data = base64.b64decode(native['files'][path], validate=True)
        if len(data) != desc['byteLength'] or digest(data) != desc['sha256']:
            raise ValueError('Embedded native map checksum differs: ' + key)
        write_new_or_equal(output / path, data)
        recovered.append({'path': path, 'bytes': len(data), 'sha256': digest(data)})
    write_new_or_equal(output / 'manifest.json', manifest_bytes)
    write_new_or_equal(output / 'native-inspector-subset.json', subset_bytes)
    proof = {'asset': name, 'htmlSHA256': digest(html), 'manifestSHA256': digest(manifest_bytes),
             'captureManifestSHA256': digest(manifest_bytes),
             'recoveredInspectorManifestSHA256': digest(subset_bytes),
             'scenePayloadSHA256': digest(payload), 'textures': recovered, 'officialHTMLModified': False}
    write_new_or_equal(output / 'recovery.json', encoded(proof))
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    inv = sub.add_parser('inventory')
    inv.add_argument('--plan', type=Path, required=True)
    inv.add_argument('--output', type=Path, required=True)
    create = sub.add_parser('pack')
    create.add_argument('--plan', type=Path, required=True)
    create.add_argument('--inventory', type=Path, required=True)
    create.add_argument('--output', type=Path, required=True)
    create.add_argument('--stem', required=True)
    check = sub.add_parser('verify')
    check.add_argument('--index', type=Path, required=True)
    recover = sub.add_parser('extract-native')
    recover.add_argument('--inventory', type=Path, required=True)
    recover.add_argument('--asset', required=True)
    recover.add_argument('--html', type=Path, required=True)
    recover.add_argument('--manifest', type=Path, required=True)
    recover.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'inventory':
        result = collect(args.plan)
        write_new_or_equal(args.output, encoded(result))
        print(json.dumps({k: result[k] for k in ('selectionStatus', 'byteInputsComplete', 'readyToArchive', 'sourceMemberCount', 'sourceMemberBytes', 'runs', 'missingExactSources')}, indent=2))
    elif args.command == 'pack':
        print(json.dumps(pack(args.plan, args.inventory, args.output, args.stem), indent=2))
    elif args.command == 'verify':
        print(json.dumps(verify(args.index), indent=2))
    else:
        print(json.dumps(extract_native(args.inventory, args.asset, args.html, args.manifest, args.output), indent=2))


if __name__ == '__main__':
    main()
