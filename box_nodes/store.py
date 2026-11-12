"""Content-addressed files shared by the node, CLI and same-origin preview."""
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tempfile
import time

REPO = Path(__file__).resolve().parents[1]
FACES = ('+X', '-X', '+Y', '-Y', '+Z', '-Z')
ID = re.compile(r'[0-9a-f]{64}\Z')
REQUIRED_MAPS = ('colorMap', 'normalMap', 'intervalMap', 'rangeMap')


def root():
    import folder_paths
    return Path(folder_paths.get_output_directory()).resolve() / 'box_impostor'


def key(*parts):
    return hashlib.sha256(json.dumps(parts, sort_keys=True, allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def file_hash(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def contained(directory, relative):
    if not isinstance(relative, str) or not relative or '\\' in relative or ':' in relative or '\x00' in relative:
        raise ValueError('Expected a relative asset path')
    posix = PurePosixPath(relative)
    if posix.is_absolute() or '..' in posix.parts or str(posix) != relative or relative == '.':
        raise ValueError(f'Invalid relative asset path: {relative}')
    home = Path(directory).resolve()
    path = (home / relative).resolve()
    if not path.is_relative_to(home):
        raise ValueError('Asset path escapes its directory')
    return path


def asset_dir(asset_id):
    if not ID.fullmatch(str(asset_id)):
        raise ValueError('Expected a 64-character Box asset ID')
    return contained(root(), asset_id)


@contextmanager
def lock(cache_key):
    directory = root() / '.locks'
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / cache_key).open('a+b') as stream:
        if os.name == 'nt':
            import msvcrt
            if not stream.tell():
                stream.write(b'0')
                stream.flush()
            while True:
                stream.seek(0)
                try:
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(.25)
                    check_interrupted()
        else:
            import fcntl
            while True:
                try:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    time.sleep(.25)
                    check_interrupted()
        # Closing the handle releases the operating-system lock, also on crashes.
        yield


def check_interrupted():
    try:
        from comfy.model_management import throw_exception_if_processing_interrupted
    except ImportError:
        return
    throw_exception_if_processing_interrupted()


def worker_version():
    paths = sorted((REPO / 'worker').glob('*.mjs')) + [REPO / 'package-lock.json', REPO / 'worker' / 'package-lock.json']
    return key([(str(path.relative_to(REPO)), file_hash(path)) for path in paths if path.is_file() and 'node_modules' not in path.parts])


def run_worker(*args):
    executable = shutil.which('node')
    if not executable:
        raise RuntimeError('Install Node.js and install the project dependencies before baking.')
    command = [executable, str(REPO / 'worker' / 'cli.mjs'), *map(str, args)]
    with tempfile.TemporaryFile() as log:
        process = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                check_interrupted()
                time.sleep(.2)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        if process.returncode:
            log.seek(max(0, log.tell() - 6000))
            raise RuntimeError('Box worker failed: ' + log.read().decode(errors='replace'))


def validate(directory, expected_id=None, check_bytes=True):
    """Reject invalid manifests before publishing or handing paths to the viewer."""
    directory = Path(directory)
    manifest = json.loads(contained(directory, 'manifest.json').read_bytes())
    if type(manifest.get('schemaVersion')) is not int or manifest['schemaVersion'] != 1 or manifest.get('layout') != 'box':
        raise ValueError('Unsupported Box manifest format')
    if not isinstance(manifest.get('asset_id'), str) or (expected_id is not None and manifest['asset_id'] != expected_id):
        raise ValueError('Box manifest asset ID mismatch')
    r = manifest.get('resolution')
    if type(r) is not int or r < 1 or list(manifest.get('faceOrder', [])) != list(FACES):
        raise ValueError('Invalid resolution or face order')
    for field in ('tightBounds', 'bakeBox'):
        bounds = manifest[field]
        for side in ('min', 'max'):
            if len(bounds[side]) != 3 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in bounds[side]):
                raise ValueError('Invalid bounding box')
        if any(a > b for a, b in zip(bounds['min'], bounds['max'])):
            raise ValueError('Invalid bounding box ordering')
    textures = manifest['textures']
    if any(name not in textures for name in REQUIRED_MAPS):
        raise ValueError('Missing required texture')
    cells = (r + 7) // 8
    material_properties = manifest.get('maps')
    if not isinstance(material_properties, list) or len(set(material_properties)) != len(material_properties) or not set(material_properties) <= {'roughness', 'metalness', 'ao', 'emissive'}:
        raise ValueError('Invalid material map list')
    if ('emissive' in material_properties) != ('emissiveMap' in textures) or bool(set(material_properties) - {'emissive'}) != ('materialMap' in textures):
        raise ValueError('Material map list does not match texture payloads')
    paths = {}
    for name, spec in [*textures.items(), ('colorMap', manifest['baseColor'])]:
        if name not in (*REQUIRED_MAPS, 'materialMap', 'emissiveMap'):
            raise ValueError('Unsupported texture ' + name)
        wh = (3 * r, 2 * r)
        if name == 'intervalMap':
            wh = (3 * r, r)
        if name == 'rangeMap':
            wh = (3 * cells, cells)
        dtype = 'uint8' if name == 'materialMap' else 'float16'
        stride = wh[0] * 4 * (1 if dtype == 'uint8' else 2)
        if (spec['width'], spec['height']) != wh or spec['dtype'] != dtype or spec['channels'] != 4 or spec['rowOrder'] != 'increasingShaderV' or spec['rowBytes'] != stride or spec['byteLength'] != stride * wh[1]:
            raise ValueError('Invalid format or dimensions for ' + name)
        filtering = 'nearest' if name == 'rangeMap' else 'linear'
        if spec.get('endian') != 'little' or spec.get('minFilter') != filtering or spec.get('magFilter') != filtering or spec.get('wrap') != 'clamp-to-edge' or spec.get('mipmaps') is not False:
            raise ValueError('Invalid sampling or endianness for ' + name)
        path = contained(directory, spec['path'])
        digest = spec['sha256']
        if not ID.fullmatch(digest) or path.stat().st_size != spec['byteLength']:
            raise ValueError('Invalid texture byte length or digest for ' + name)
        if check_bytes:
            if path not in paths:
                paths[path] = file_hash(path)
            if paths[path] != digest:
                raise ValueError('Texture digest mismatch for ' + name)
    if not contained(directory, manifest['proxy']).is_file():
        raise ValueError('Missing proxy GLB')
    if set(manifest['faces']) != set(FACES):
        raise ValueError('Missing face metadata')
    for face in manifest['faces'].values():
        if not contained(directory, face['guide']).is_file():
            raise ValueError('Missing face guide')
    return manifest


def publish(asset_id, producer):
    destination = asset_dir(asset_id)
    with lock(asset_id):
        if not destination.exists():
            with tempfile.TemporaryDirectory(prefix='.box-', dir=root()) as temp:
                staging = Path(temp) / 'asset'
                producer(staging)
                validate(staging, asset_id)
                os.rename(staging, destination)
        else:
            validate(destination, asset_id)
    return asset_id


def baseline(proxy, resolution, backend):
    asset_id = key('baseline-v1', worker_version(), proxy['stageKey'], proxy['sha256'], resolution, backend)
    return publish(asset_id, lambda directory: run_worker('bake', '--input', proxy['path'], '--output', directory, '--resolution', resolution, '--backend', backend, '--asset-id', asset_id, '--canonical-meta', proxy['metadata']))
