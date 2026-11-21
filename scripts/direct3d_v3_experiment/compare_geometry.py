#!/usr/bin/env python3
"""Compare meshes already in the same canonical frame using 3D-Inspector.

This creates neutral-gray copies of the models; it never edits view.py's HTML.
The existing Inspector capture program is reused, with a small, recorded wrapper
that supplies one common scene bounds to the official viewer during capture.
That aligns its camera clipping planes and light placement as well as framing.
The standalone deliverable viewers retain the official, unmodified behavior.

Requires numpy, Pillow and trimesh in this interpreter; Playwright at the given
--playwright-root; the official view.py launcher and the existing capture script.
No 3D inference is performed. --prepare-only builds the models and viewers only.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import zlib

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import trimesh


LAB = Path('/home/dev/Projects/box-impostor-lab')
EXPERIMENTS = Path('/mnt/f/_dev/box-impostor-lab/experiments')
MATERIAL = {
    'baseColorFactor': [180, 180, 180, 255],
    'metallicFactor': 0.0,
    'roughnessFactor': 1.0,
    'doubleSided': True,
    'alphaMode': 'OPAQUE',
}
CAMERA_FIELDS = (
    'position', 'quaternion', 'up', 'matrixWorld', 'matrixWorldInverse',
    'projectionMatrix', 'projectionMatrixInverse', 'near', 'far',
    'top', 'bottom', 'left', 'right', 'zoom', 'canvasPixels',
)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def verify_viewer_face_count(html: Path, source_faces: int) -> dict:
    """Read the official payload and reject any automatic face-stride preview."""
    content = html.read_text()
    match = re.search(r'<script id="scene-payload"[^>]*>([\s\S]*?)</script>', content)
    if match is None:
        raise AssertionError('Official Inspector scene-payload was not found')
    packed = base64.b64decode(match.group(1).strip(), validate=True)
    payload = zlib.decompress(packed) if packed[:1] == b'\x78' else packed
    scene = json.loads(payload)
    meshes = scene.get('meshes', [])
    if not meshes or scene.get('previewType') != 'mesh':
        raise AssertionError('Expected explicit full-mesh payload for the gray GLB')
    counts = []
    vertices = 0
    for mesh in meshes:
        if mesh.get('primitive', 'triangles') == 'points':
            raise AssertionError('A point preview is not a geometry comparison')
        indices = base64.b64decode(mesh['indices'], validate=True)
        positions = base64.b64decode(mesh['positions'], validate=True)
        if len(indices) % 12 or len(positions) % 12:
            raise AssertionError('Unexpected u32-triangle or f32-position payload size')
        counts.append(len(indices) // 12)
        vertices += len(positions) // 12
    renderer_faces = sum(counts)
    metadata_faces = scene.get('original', {}).get('triangles')
    if renderer_faces != source_faces or metadata_faces != source_faces:
        raise AssertionError(
            f'Incomplete Inspector geometry: source={source_faces}, '
            f'metadata={metadata_faces}, rendered={renderer_faces}. '
            'The default per-mesh cap uses face stride and produces holes; '
            'all source faces must be preserved for this comparison.')
    return {'sourceFaceCount': source_faces, 'rendererFaceCount': renderer_faces,
            'originalMetadataFaceCount': metadata_faces,
            'rendererVertexCount': vertices, 'perMeshFaceCounts': counts,
            'allSourceFacesPresent': True, 'faceStridePreview': False,
            'verification': 'Decoded unchanged scene-payload; u32 index byte count / 12'}


def run(command: list[str], log: Path, env: dict) -> dict:
    start = time.perf_counter()
    log.parent.mkdir(parents=True, exist_ok=True)
    print('RUN', log.stem, flush=True)
    with log.open('w') as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                env=env, check=False)
    record = {'command': command, 'log': str(log), 'exitCode': result.returncode,
              'elapsedSeconds': round(time.perf_counter() - start, 3)}
    if result.returncode:
        raise RuntimeError(f'Command failed ({result.returncode}); see {log}')
    return record


def gray_copy(source: Path, target: Path) -> dict:
    loaded = trimesh.load(source, force='mesh', process=False)
    if not isinstance(loaded, trimesh.Trimesh) or not len(loaded.faces):
        raise ValueError(f'Expected a nonempty triangle mesh: {source}')
    if not np.isfinite(loaded.vertices).all():
        raise ValueError(f'Non-finite positions: {source}')
    # Rebuild without texture/vertex-color attributes or imported normal maps.
    # Positions and connectivity are retained; both meshes use the same derived
    # smooth vertex-normal algorithm. This does not normalize, rotate or fit them.
    mesh = trimesh.Trimesh(vertices=np.array(loaded.vertices, copy=True),
                           faces=np.array(loaded.faces, copy=True),
                           process=False, validate=False)
    mesh.visual = trimesh.visual.TextureVisuals(
        material=trimesh.visual.material.PBRMaterial(name='Comparison neutral gray',
                                                     **MATERIAL))
    target.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(target)
    reread = trimesh.load(target, force='mesh', process=False)
    if not np.array_equal(reread.faces, mesh.faces):
        raise AssertionError(f'Triangle connectivity changed during export: {target}')
    if reread.vertices.shape != mesh.vertices.shape:
        raise AssertionError(f'Vertex count changed during export: {target}')
    error = float(np.max(np.abs(reread.vertices - mesh.vertices)))
    tolerance = max(float(np.max(mesh.extents)) * 1e-6, 1e-8)
    if error > tolerance:
        raise AssertionError(f'GLB export position error {error} > {tolerance}')
    return {
        'source': str(source), 'sourceSha256': sha(source),
        'grayMesh': str(target), 'grayMeshSha256': sha(target),
        'vertices': len(mesh.vertices), 'faces': len(mesh.faces),
        'bounds': {'min': mesh.bounds[0].tolist(), 'max': mesh.bounds[1].tolist()},
        'exportedBounds': {'min': reread.bounds[0].tolist(), 'max': reread.bounds[1].tolist()},
        'sourceWatertight': bool(mesh.is_watertight),
        'connectivityPreserved': True,
        'positionMaxAbsExportError': error,
        'positionExportTolerance': tolerance,
        'additionalTransform': 'none; input meshes must already share a frame',
        'material': MATERIAL,
        'normals': 'same trimesh-derived smooth vertex-normal algorithm on both meshes',
    }


def capture_wrapper(source: Path, target: Path) -> dict:
    """Adapt the capture controller, never the official generated HTML."""
    original = source.read_text()
    replacements = [
        ('const target = center(bounds), sourceTarget = center(scene.bounds);',
         'const target = center(bounds), sourceTarget = center(bounds);'),
        ('const sourceDistance = radius(scene.bounds) * 1.05 / Math.sin(20 * Math.PI / 180);',
         'const sourceDistance = referenceDistance;'),
        ('await page.addInitScript(() => {',
         'await page.addInitScript((sharedCaptureBounds) => {'),
        ('const api = await Reflect.apply(target.createThreeViewer, target, args);',
         'const captureOptions = {...args[0], scene: {...args[0].scene, bounds: sharedCaptureBounds}};\n'
         '            const api = await Reflect.apply(target.createThreeViewer, target, [captureOptions, ...args.slice(1)]);'),
        ('  });\n  await page.goto(pathToFileURL(html).href',
         '  }, bounds);\n  await page.goto(pathToFileURL(html).href'),
        ("capture: 'Exact view.py HTML; existing UI and mouse; official Save PNG button',",
         "capture: 'Unmodified view.py HTML; shared scene bounds supplied by capture controller for camera clipping and light placement; UI, mouse and official Save PNG',\n"
         "  sharedBoundsForCameraAndLighting: true,"),
    ]
    text = original
    for before, after in replacements:
        if text.count(before) != 1:
            raise RuntimeError('Capture source changed; review the controller adaptation: ' + before)
        text = text.replace(before, after)
    target.write_text(text)
    return {'upstream': str(source), 'upstreamSha256': sha(source),
            'adaptedController': str(target), 'adaptedControllerSha256': sha(target),
            'purpose': 'common camera near/far, framing and directional-light position',
            'viewerHTMLModified': False,
            'geometryMaterialsShadersModifiedAtCapture': False}


def validate_captures(before: dict, after: dict, args: argparse.Namespace) -> dict:
    for label, report in [('before', before), ('after', after)]:
        if not report.get('completed') or report.get('errors'):
            raise AssertionError(f'{label} capture is incomplete or has errors')
        if len(report['frames']) != args.frames:
            raise AssertionError(f'{label} frame count differs')
        if report.get('sharedBoundsForCameraAndLighting') is not True:
            raise AssertionError('Capture did not use the common camera and lighting bounds')
    for key in ['referenceManifestSHA256', 'backend', 'browserVersion', 'graphicsRenderer', 'display']:
        if before.get(key) != after.get(key):
            raise AssertionError(f'Capture environments differ: {key}')
    by_field = {key: 0.0 for key in CAMERA_FIELDS}
    masks = {'before': [], 'after': []}
    for index, (left, right) in enumerate(zip(before['frames'], after['frames'])):
        if left['index'] != index or right['index'] != index:
            raise AssertionError('Frame ordering differs')
        if left['requestedYawDegrees'] != right['requestedYawDegrees']:
            raise AssertionError('Requested yaw differs')
        for key in CAMERA_FIELDS:
            x = np.asarray(left['camera'][key], dtype=float)
            y = np.asarray(right['camera'][key], dtype=float)
            if x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
                raise AssertionError(f'Invalid camera value: {key}')
            by_field[key] = max(by_field[key], float(np.max(np.abs(x - y))))
        for label, frame in [('before', left), ('after', right)]:
            path = Path(frame['path'])
            if sha(path) != frame['sha256']:
                raise AssertionError(f'Capture PNG changed: {path}')
            rgba = np.asarray(Image.open(path).convert('RGBA'))
            if rgba.shape != (args.size, args.size, 4):
                raise AssertionError(f'Capture dimensions differ: {path}')
            a = rgba[:, :, 3] > 127
            y, x = np.where(a)
            edge = int(a[0].sum() + a[-1].sum() + a[1:-1, 0].sum() + a[1:-1, -1].sum())
            if not len(x) or edge:
                raise AssertionError(f'Empty or clipped full-body frame: {path}')
            masks[label].append({'index': index, 'foregroundPixels': int(a.sum()),
                                 'bboxXYXY': [int(x.min()), int(y.min()), int(x.max()+1), int(y.max()+1)],
                                 'edgePixels': edge})
    largest = max(by_field.values())
    if largest > args.camera_tolerance:
        raise AssertionError(f'Camera maximum difference {largest} exceeds {args.camera_tolerance}')
    return {'framesCompared': args.frames, 'cameraMaxAbsDifference': largest,
            'cameraMaxByField': by_field, 'cameraTolerance': args.camera_tolerance,
            'captureEnvironmentEqual': True, 'allFramesNonempty': True,
            'allFramesEdgePixelsZero': True, 'frameBounds': masks,
            'alphaPolicy': 'Shape is intentionally changed; alpha equality is not an acceptance criterion.'}


def encode(captures: dict, models: dict, out: Path, args: argparse.Namespace) -> dict:
    header, footer = 36, 36
    names = list(captures)
    width, height = args.size * len(names), args.size + header + footer
    if width % 2 or height % 2:
        raise ValueError('Video dimensions must be even for yuv420p')
    font_path = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    font = ImageFont.truetype(str(font_path), 17) if font_path.exists() else ImageFont.load_default()
    movie = out / 'Direct3DV3_Geometry_Comparison.mp4'
    command = [str(args.ffmpeg), '-hide_banner', '-loglevel', 'warning', '-y',
               '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{width}x{height}',
               '-r', str(args.fps), '-i', 'pipe:0', '-an', '-c:v', 'libx264',
               '-preset', 'medium', '-crf', '18', '-pix_fmt', 'yuv420p',
               '-movflags', '+faststart', str(movie)]
    posters = []
    with (out / 'encode.log').open('w') as log:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=log, stderr=log)
        try:
            for index in range(args.frames):
                canvas = Image.new('RGB', (width, height), (25, 30, 35))
                for column, name in enumerate(names):
                    frame = captures[name]['frames'][index]
                    rgba = Image.open(frame['path']).convert('RGBA')
                    background = Image.new('RGBA', rgba.size, (128, 128, 128, 255))
                    full_image = Image.alpha_composite(background, rgba).convert('RGB')
                    canvas.paste(full_image, (column * args.size, header))
                draw = ImageDraw.Draw(canvas)
                for column, name in enumerate(names):
                    draw.text((column*args.size + 10, 8), models[name]['label'], font=font, fill='white')
                caption = f'Neutral gray | Full body, no crop | Same camera | Yaw {360*index/args.frames:05.1f} deg'
                draw.text((10, height - footer + 8), caption, font=font, fill='white')
                assert process.stdin is not None
                process.stdin.write(canvas.tobytes())
                if index in {0, args.frames//4, args.frames//2, 3*args.frames//4}:
                    path = out / f'Direct3DV3_Geometry_{round(360*index/args.frames):03d}.png'
                    canvas.save(path)
                    posters.append({'path': str(path), 'sha256': sha(path), 'frame': index})
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError(f'Video encoder failed: {out / "encode.log"}')
        except BaseException:
            if process.poll() is None:
                process.terminate()
                process.wait()
            raise
    decode = subprocess.run([str(args.ffmpeg), '-hide_banner', '-loglevel', 'error',
                             '-progress', 'pipe:1', '-nostats', '-i', str(movie),
                             '-map', '0:v:0', '-f', 'null', '-'],
                            capture_output=True, text=True, check=False)
    (out / 'decode.log').write_text(decode.stdout + decode.stderr)
    counts = re.findall(r'^frame=(\d+)\s*$', decode.stdout, flags=re.MULTILINE)
    if decode.returncode or not counts or int(counts[-1]) != args.frames:
        raise AssertionError('The final MP4 did not decode to the expected complete frame count')
    return {'path': str(movie), 'sha256': sha(movie), 'pixels': [width, height],
            'columns': [{'key': name, 'label': models[name]['label']} for name in names],
            'frames': args.frames, 'fps': args.fps, 'durationSeconds': args.frames/args.fps,
            'decodedFrames': int(counts[-1]), 'posters': posters,
            'composition': 'Full RGBA frames composited over the same gray; no resize, crop or warp.'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--before-label', default='Pixal3D coarse / neutral gray')
    parser.add_argument('--after-label', default='Direct3D-V3 / neutral gray')
    parser.add_argument('--extra', nargs=2, action='append', default=[],
                        metavar=('LABEL', 'MODEL'),
                        help='Append a comparison column; repeat for further models in the same frame')
    parser.add_argument('--inspector', type=Path, default=LAB/'third_party/3D-Inspector-skills/skills/3d-inspector/view.py')
    parser.add_argument('--capture-script', type=Path, default=EXPERIMENTS/'zipsplat-qwen-20261120/capture_inspector.mjs')
    parser.add_argument('--playwright-root', type=Path, default=LAB/'ComfyUI/custom_nodes/box_impostor_factory')
    parser.add_argument('--ffmpeg', type=Path, default=EXPERIMENTS/'sdf-morph-20261120/tools/ffmpeg-linux-x86_64-v7.0.2')
    parser.add_argument('--python', default=str(LAB/'.venv/bin/python'))
    parser.add_argument('--node', default='node')
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--frames', type=int, default=240)
    parser.add_argument('--fps', type=float, default=24)
    parser.add_argument('--camera-tolerance', type=float, default=1e-5)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.size < 128 or args.size > 2048 or args.size % 2 or args.frames < 4 or args.fps <= 0:
        parser.error('Use an even size in [128,2048], at least 4 frames, and positive fps')
    for key in ['before', 'after', 'inspector', 'capture_script', 'ffmpeg']:
        value = getattr(args, key).resolve()
        if not value.is_file():
            parser.error(f'Missing file --{key.replace("_", "-")}: {value}')
        setattr(args, key, value)
    inputs = [('before', args.before_label, args.before), ('after', args.after_label, args.after)]
    for index, (label, path) in enumerate(args.extra, start=1):
        model = Path(path).resolve()
        if not model.is_file():
            parser.error(f'Missing --extra model: {model}')
        inputs.append((f'extra{index}', label, model))
    args.out = args.out.resolve()
    args.playwright_root = args.playwright_root.resolve()
    plan = {'before': str(args.before), 'beforeSha256': sha(args.before),
            'after': str(args.after), 'afterSha256': sha(args.after),
            'size': args.size, 'frames': args.frames, 'fps': args.fps,
            'beforeLabel': args.before_label, 'afterLabel': args.after_label,
            'extraModels': [{'key': key, 'label': label, 'path': str(path), 'sha256': sha(path)}
                            for key, label, path in inputs[2:]],
            'inspectorSha256': sha(args.inspector), 'captureSourceSha256': sha(args.capture_script),
            'viewerFacePolicy': 'Pass source face count as max-triangles; require decoded payload equality',
            'material': MATERIAL}
    args.out.mkdir(parents=True, exist_ok=True)
    plan_path = args.out/'plan.json'
    if plan_path.exists():
        if not args.resume:
            parser.error('Output already contains a plan; use a new directory or --resume')
        if json.loads(plan_path.read_text()) != plan:
            parser.error('Existing plan differs; use a new output directory')
    write_json(plan_path, plan)
    env = os.environ.copy()
    env.pop('DISPLAY', None)
    env.pop('WAYLAND_DISPLAY', None)
    env['PYTHONUNBUFFERED'] = '1'
    models = {}
    for name, label, source in inputs:
        record = gray_copy(source, args.out/f'{name}.neutral-gray.glb')
        record['label'] = label
        html = args.out/f'{name}.neutral-gray.inspector.html'
        record['viewerCommand'] = run([args.python, str(args.inspector), record['grayMesh'],
                                      '-o', str(html), '--max-triangles', str(record['faces'])],
                                      args.out/'logs'/f'{name}-view.log', env)
        if not html.is_file():
            raise AssertionError(f'Official view.py did not produce {html}')
        record['viewer'] = str(html)
        record['viewerSha256'] = sha(html)
        record['viewerGeometry'] = verify_viewer_face_count(html, record['faces'])
        record['sourceFaceCount'] = record['faces']
        record['rendererFaceCount'] = record['viewerGeometry']['rendererFaceCount']
        models[name] = record
    low = np.min([models[k]['exportedBounds']['min'] for k in models], axis=0)
    high = np.max([models[k]['exportedBounds']['max'] for k in models], axis=0)
    margin = max(float(np.max(high-low))*0.02, 1e-6)
    reference = {'bounds': {'min': (low-margin).tolist(), 'max': (high+margin).tolist()},
                 'method': 'union of both exported full-body bounds plus 2 percent of longest extent',
                 'sourceFramesAlreadyAligned': True, 'additionalModelTransforms': 'none'}
    frame_path = args.out/'common-frame.json'
    write_json(frame_path, reference)
    adapted = args.out/'capture_geometry.mjs'
    adapter = capture_wrapper(args.capture_script, adapted)
    report = {'plan': plan, 'models': models, 'commonFrame': reference,
              'captureController': adapter,
              'viewerFitAndCaptureFit': {
                  'standaloneViewer': 'Exact official HTML; its initial fit and light placement use its own model bounds.',
                  'comparisonVideo': 'Comparison-only sharedBounds override; all models use the union bounds for camera fit, near/far, and light placement.',
                  'viewerHTMLModified': False,
                  'meshTransformModified': False,
                  'rendererOrShaderModified': False,
              },
              'completed': False}
    write_json(args.out/'comparison.json', report)
    if args.prepare_only:
        print(json.dumps({'prepared': True, 'report': str(args.out/'comparison.json')}))
        return
    captures = {}
    for name in models:
        destination = args.out/'capture'/name
        old = destination/'capture.json'
        can_reuse = False
        if args.resume and old.is_file():
            capture = json.loads(old.read_text())
            can_reuse = (capture.get('completed') is True and not capture.get('errors')
                         and capture.get('htmlSHA256') == models[name]['viewerSha256']
                         and capture.get('referenceManifestSHA256') == sha(frame_path)
                         and capture.get('scriptSHA256') == sha(adapted)
                         and capture.get('framesExpected') == args.frames
                         and capture.get('camera', {}).get('pixels') == [args.size, args.size])
        if not can_reuse:
            models[name]['captureCommand'] = run([
                args.node, str(adapted), '--viewer', models[name]['viewer'],
                '--reference', str(frame_path), '--out', str(destination),
                '--frames', str(args.frames), '--size', str(args.size),
                '--fps', str(args.fps), '--backend', 'swiftshader', '--mode', 'default',
                '--playwright-root', str(args.playwright_root)],
                args.out/'logs'/f'{name}-capture.log', env)
        captures[name] = json.loads(old.read_text())
        if sha(Path(models[name]['viewer'])) != models[name]['viewerSha256']:
            raise AssertionError('Official Inspector HTML changed during capture')
    pairs = {name: validate_captures(captures['before'], captures[name], args)
             for name in captures if name != 'before'}
    report['validation'] = {
        'framesCompared': args.frames,
        'cameraMaxAbsDifference': max(p['cameraMaxAbsDifference'] for p in pairs.values()),
        'allFramesNonempty': True, 'allFramesEdgePixelsZero': True,
        'captureEnvironmentEqual': True, 'versusBefore': pairs,
    }
    report['video'] = encode(captures, models, args.out, args)
    report['officialViewerHTMLUnmodified'] = True
    report['completed'] = True
    write_json(args.out/'comparison.json', report)
    print(json.dumps({'completed': True, 'report': str(args.out/'comparison.json'),
                      'video': report['video']['path'],
                      'cameraMaxAbsDifference': report['validation']['cameraMaxAbsDifference']}))


if __name__ == '__main__':
    main()
