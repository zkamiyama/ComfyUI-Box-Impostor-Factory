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
import warnings

REPO = Path(__file__).resolve().parents[1]
FACES = ('+X', '-X', '+Y', '-Y', '+Z', '-Z')
ID = re.compile(r'[0-9a-f]{64}\Z')
REQUIRED_MAPS = ('colorMap', 'normalMap', 'intervalMap', 'rangeMap')
MATERIAL_OPTIONS = {'none': (), 'material': ('roughness', 'metalness', 'ao'), 'emissive': ('emissive',), 'material+emissive': ('roughness', 'metalness', 'ao', 'emissive')}


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
    paths = sorted((REPO / 'worker').glob('*.mjs'))
    paths += [REPO / 'web' / 'asset_io.mjs', REPO / 'package-lock.json', REPO / 'worker' / 'package-lock.json']
    return key([(str(path.relative_to(REPO)), file_hash(path)) for path in paths if path.is_file() and 'node_modules' not in path.parts])


def presentation_version():
    paths = ('web/viewer.html', 'web/viewer.mjs', 'web/asset_io.mjs',
             'worker/chromium.mjs', 'worker/cli.mjs', 'worker/server.mjs',
             'worker/runtime.mjs', 'worker/codec.mjs', 'package-lock.json')
    return key([(name, file_hash(REPO / name)) for name in paths])


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
    if type(r) is not int or not 1 <= r <= 8192 or list(manifest.get('faceOrder', [])) != list(FACES):
        raise ValueError('Invalid resolution or face order')
    for field in ('tightBounds', 'bakeBox'):
        bounds = manifest[field]
        for side in ('min', 'max'):
            if len(bounds[side]) != 3 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in bounds[side]):
                raise ValueError('Invalid bounding box')
        if any(a > b or (field == 'bakeBox' and a == b) for a, b in zip(bounds['min'], bounds['max'])):
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
    proxy_path = contained(directory, manifest['proxy'])
    if not proxy_path.is_file() or not ID.fullmatch(str(manifest.get('proxySha256', ''))):
        raise ValueError('Missing proxy GLB or its digest')
    if check_bytes and file_hash(proxy_path) != manifest['proxySha256']:
        raise ValueError('Proxy GLB digest mismatch')
    for name in ('transform', 'inverseTransform'):
        matrix = manifest.get(name)
        if not isinstance(matrix, list) or len(matrix) != 16 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in matrix):
            raise ValueError('Invalid canonical transform')
    if set(manifest['faces']) != set(FACES):
        raise ValueError('Missing face metadata')
    for face in manifest['faces'].values():
        guide = contained(directory, face['guide'])
        if not guide.is_file():
            raise ValueError('Missing face guide')
        if check_bytes and face.get('guideSha256') and file_hash(guide) != face['guideSha256']:
            raise ValueError('Face guide digest mismatch')
        if face.get('depth'):
            depth = contained(directory, face['depth'])
            if depth.stat().st_size != r * r * 2 or (check_bytes and face.get('depthSha256') and file_hash(depth) != face['depthSha256']):
                raise ValueError('Invalid face guide depth')
    return manifest


def publish(asset_id, producer, expected_manifest=None):
    destination = asset_dir(asset_id)
    with lock(asset_id):
        if not destination.exists():
            with tempfile.TemporaryDirectory(prefix='.box-', dir=root()) as temp:
                staging = Path(temp) / 'asset'
                producer(staging)
                validate(staging, asset_id)
                os.rename(staging, destination)
        else:
            existing = validate(destination, asset_id)
            if expected_manifest is not None and existing != expected_manifest:
                raise ValueError('An asset with this ID already exists with a different manifest.')
    return asset_id


def baseline(proxy, resolution, backend, material_maps='none'):
    if material_maps not in MATERIAL_OPTIONS:
        raise ValueError('Unknown Box material map selection.')
    maps = MATERIAL_OPTIONS[material_maps]
    asset_id = key('baseline-v3', worker_version(), proxy['stageKey'], proxy['sha256'], resolution, backend, maps)
    def generate(directory):
        run_worker('bake', '--input', proxy['path'], '--output', directory, '--resolution', resolution, '--backend', backend, '--asset-id', asset_id, '--canonical-meta', proxy['metadata'], '--maps', ','.join(maps))
        manifest_path = directory / 'manifest.json'
        manifest = json.loads(manifest_path.read_bytes())
        canonical = json.loads(Path(proxy['metadata']).read_bytes())
        manifest['provenance']['requestedMaterialMaps'] = list(maps)
        manifest['provenance']['originalGlbSha256'] = canonical['sourceSha256']
        manifest['provenance']['rotationDegreesXYZ'] = proxy.get('rotationDegreesXYZ', (0, 0, 0))
        for face in manifest['faces'].values():
            face['guideSha256'] = file_hash(contained(directory, face['guide']))
            if face.get('depth'):
                face['depthSha256'] = file_hash(contained(directory, face['depth']))
        if proxy.get('provenance'):
            manifest['provenance']['meshGeneration'] = proxy['provenance']
        if proxy.get('reference'):
            ref = dict(proxy['reference'])
            source = Path(ref.pop('path'))
            (directory / 'reference').mkdir(exist_ok=True)
            for name in ('source.npy', 'source.png'):
                shutil.copyfile(source / name, directory / 'reference' / name)
            manifest['reference'] = {**ref, 'image': 'reference/source.png', 'samples': 'reference/source.npy'}
        manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding='utf-8')
    return publish(asset_id, generate)


def copy_thumbnail(image_path, directory):
    target = directory / 'previews' / 'hero.png'
    target.parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as tmp:
        temp = Path(tmp) / 'hero.png'
        shutil.copyfile(image_path, temp)
        os.replace(temp, target)
    return target


def thumbnail(directory, manifest, sampling='standard'):
    from PIL import Image, PngImagePlugin
    if sampling not in ('standard', 'reduce_bleeding'):
        raise ValueError('Unknown Box preview sampling')
    asset_id = manifest['asset_id']
    filename = 'hero.png' if sampling == 'standard' else 'hero-reduce_bleeding.png'
    identity = key('thumbnail-v3', asset_id, sampling, presentation_version())
    with lock(key('thumbnail-v2', asset_id, sampling)):
        target = directory / 'previews' / filename
        saved_preview = False
        if target.is_file():
            try:
                with Image.open(target) as previous:
                    recorded = previous.info.get('BoxPresentation')
                    previous.verify()
                if recorded == identity:
                    return target
                saved_preview = True
            except (OSError, SyntaxError, ValueError):
                pass  # A damaged cache entry can be regenerated from the Box.
        target.parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.thumb-', dir=directory.parent) as tmp:
            rendered = Path(tmp) / 'render.png'
            try:
                run_worker('thumbnail', '--asset', directory, '--output', rendered,
                           '--backend', 'auto', '--sampling', sampling)
            except (OSError, RuntimeError) as exc:
                if not saved_preview:
                    raise
                warnings.warn(f'Using the saved Box preview because thumbnail rendering failed: {exc}',
                              RuntimeWarning)
                return target
            # Store the pixels and their cache identity in one atomically replaced PNG.
            published = Path(tmp) / filename
            metadata = PngImagePlugin.PngInfo()
            metadata.add_text('BoxPresentation', identity)
            with Image.open(rendered) as image:
                image.save(published, format='PNG', pnginfo=metadata)
            os.replace(published, target)
        return target
