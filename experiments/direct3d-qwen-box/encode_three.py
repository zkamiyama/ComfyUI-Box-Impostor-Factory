#!/usr/bin/env python3
"""Encode three complete Inspector captures as a single-row full-image comparison.

Requires Python 3.12, NumPy, Pillow, and the explicit ffmpeg executable passed by CLI.
Input JSON: {"items": [{"label": "A512", "capture": "/path/capture.json"}, ...]}.
"""
import argparse
import hashlib
import io
import json
import re
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

COUNT, SIZE, FPS, BAND, FOOTER = 240, 512, 24, 36, 36
WIDTH, HEIGHT = SIZE * 3, SIZE + BAND + FOOTER


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def resolve(path, parent):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (parent / path).resolve()


def angle_error(a, b):
    return abs((a - b + 180) % 360 - 180)


def require(condition, message):
    if not condition:
        raise ValueError(message)


CAMERA_SHAPES = {
    'matrixWorld': (16,), 'matrixWorldInverse': (16,),
    'projectionMatrix': (16,), 'projectionMatrixInverse': (16,),
    'position': (3,), 'quaternion': (4,), 'up': (3,), 'canvasPixels': (2,),
    'near': (), 'far': (), 'left': (), 'right': (), 'top': (), 'bottom': (), 'zoom': (),
}
CAMERA_TOLERANCE = 0.0


def camera_difference(left, right):
    """Require every recorded camera field to match; no tolerance relaxation."""
    require(set(left) == set(right), 'Recorded camera fields differ')
    for label, camera in [('left', left), ('right', right)]:
        require(camera.get('type') == 'OrthographicCamera', label + ': expected orthographic camera')
        for key, shape in CAMERA_SHAPES.items():
            value = np.asarray(camera[key], dtype='f8')
            require(value.shape == shape and np.isfinite(value).all(), label + ': invalid camera ' + key)
    differences = {}
    for key, value in left.items():
        other = right[key]
        if isinstance(value, (int, float, list)) and not isinstance(value, bool):
            a, b = np.asarray(value, dtype='f8'), np.asarray(other, dtype='f8')
            require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(), 'Invalid camera field: ' + key)
            differences[key] = float(np.max(np.abs(a - b))) if a.size else 0.
        else:
            require(value == other, 'Camera metadata differs: ' + key)
    return differences


def load_rgba(path, expected_sha):
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected_sha, 'Source PNG SHA256 mismatch: ' + str(path))
    require(len(raw) >= 33 and raw[:8] == b'\x89PNG\r\n\x1a\n' and raw[12:16] == b'IHDR'
            and raw[24:26] == bytes((8, 6)), 'Expected an 8-bit RGBA PNG: ' + str(path))
    with Image.open(io.BytesIO(raw)) as image:
        image.verify()
    with Image.open(io.BytesIO(raw)) as image:
        require(image.mode == 'RGBA' and image.size == (SIZE, SIZE), 'Require a complete 512x512 RGBA frame')
        return np.asarray(image, dtype='u1').copy()


def alpha_difference(left, right):
    require(left.shape == right.shape == (SIZE, SIZE), 'Unexpected alpha plane shape')
    delta = np.abs(left.astype('i2') - right.astype('i2'))
    return {'maxUint8': int(delta.max()), 'nonzeroPixels': int(np.count_nonzero(delta))}


def rgb_difference(left, right):
    delta = np.abs(left.astype('i2') - right.astype('i2'))
    return {'maxUint8': int(delta.max()), 'nonzeroPixels': int(np.count_nonzero(np.any(delta != 0, axis=-1))),
            'sumAbsoluteDifference': int(delta.sum(dtype='i8')), 'values': int(delta.size)}


def validate_frames(items):
    pairs = [(0, 1), (1, 2), (0, 2)]
    per_pair = {str(a) + '-' + str(b): [] for a, b in pairs}
    frame_checks = []
    for index in range(COUNT):
        records = [item['capture']['frames'][index] for item in items]
        images = [load_rgba(item['paths'][index], record['sha256']) for item, record in zip(items, records)]
        body_checks = []
        for column, image in enumerate(images):
            mask = image[..., 3] > 127
            yy, xx = np.nonzero(mask)
            edges = int(mask[0].sum() + mask[-1].sum() + mask[1:-1, 0].sum() + mask[1:-1, -1].sum())
            require(len(xx) > 1000 and edges == 0, f'Empty/small/clipped full-body frame: column {column}, frame {index}')
            body_checks.append({'column': column, 'foregroundPixels': len(xx), 'edgePixels': edges,
                                'boundsXYXY': [int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())]})
        checks = []
        for a, b in pairs:
            camera = camera_difference(records[a]['camera'], records[b]['camera'])
            maximum = max(camera.values(), default=0.)
            require(maximum <= CAMERA_TOLERANCE,
                    f'Camera must match exactly: columns {a}/{b}, frame {index}, max difference {maximum:.17g}')
            alpha = alpha_difference(images[a][..., 3], images[b][..., 3])
            require(alpha['nonzeroPixels'] == 0,
                    f'Alpha differs: columns {a}/{b}, frame {index}, pixels {alpha["nonzeroPixels"]}, max {alpha["maxUint8"]}')
            rgb = rgb_difference(images[a][..., :3], images[b][..., :3])
            per_pair[str(a) + '-' + str(b)].append(rgb)
            checks.append({'columns': [a, b], 'cameraMaxAbsDifference': maximum, 'alpha': alpha, 'rgb': rgb})
        frame_checks.append({'index': index, 'pairs': checks, 'fullBody': body_checks})
    summaries = []
    for a, b in pairs:
        values = per_pair[str(a) + '-' + str(b)]
        summaries.append({'columns': [a, b], 'labels': [items[a]['label'], items[b]['label']],
                          'framesCompared': COUNT, 'cameraMaxAbsDifference': 0., 'cameraTolerance': CAMERA_TOLERANCE,
                          'alpha': {'all240FramesZero': True, 'maxUint8': 0, 'nonzeroPixels': 0},
                          'rgb': {'changedFrames': sum(value['nonzeroPixels'] > 0 for value in values),
                                  'maxUint8': max(value['maxUint8'] for value in values),
                                  'nonzeroPixels': sum(value['nonzeroPixels'] for value in values),
                                  'maeUint8': sum(value['sumAbsoluteDifference'] for value in values) / sum(value['values'] for value in values)}})
    return {'completed': True, 'verifiedPngHashes': len(items) * COUNT, 'verifiedFrameOrder': True,
            'all240AlphaFramesEqualAcrossThreeColumns': True, 'recordedCamerasExactlyEqual': True,
            'cameraTolerance': CAMERA_TOLERANCE, 'allFramesNonemptyAndUnclipped': True,
            'pairs': summaries, 'frames': frame_checks,
            'limitations': ['RGB differences measure change, not image quality.',
                           'Rendered alpha equality is limited to these cameras and 8-bit captures; it is not a depth-buffer comparison.']}


def load_captures(path):
    items = json.loads(path.read_text())['items']
    require(len(items) == 3, 'Exactly three captures are required')
    loaded = []
    for item in items:
        label = item['label']
        require(label and label.isascii() and label.isprintable(), 'Use a nonempty single-line English label')
        source = resolve(item['capture'], path.parent)
        capture = json.loads(source.read_text())
        require(capture['completed'] is True and capture['errors'] == [], f'Incomplete capture: {source}')
        require(capture['framesExpected'] == COUNT and len(capture['frames']) == COUNT, f'Expected {COUNT} frames: {source}')
        require(capture['playbackFPS'] == FPS and capture['durationSeconds'] == COUNT / FPS, 'Require 240 frames at 24fps / 10 seconds')
        require(capture['sourceFaceResolution'] == SIZE, 'Expected native Box512')
        require(capture['camera']['pixels'] == [SIZE, SIZE] and capture['camera']['projection'] == 'orthographic', 'Expected 512px orthographic captures')
        display = capture['display']
        require(display['mode'] == 'default' and not display['materialOverride'] and not display['toneMappingOverride'],
                'Require official default material/display without overrides')
        html = resolve(capture['html'], source.parent)
        reference = resolve(capture['referenceManifest'], source.parent)
        require(digest(html) == capture['htmlSHA256'], f'Inspector HTML changed: {html}')
        require(digest(reference) == capture['referenceManifestSHA256'], f'Reference changed: {reference}')
        paths = []
        for index, frame in enumerate(capture['frames']):
            require(type(frame['index']) is int and frame['index'] == index, f'Unexpected frame order: {source}, frame {index}')
            require(np.isfinite([frame['requestedYawDegrees'], frame['actualYawDegrees'], frame['actualPitchDegrees']]).all(), 'Nonfinite camera angle')
            require(angle_error(frame['requestedYawDegrees'], index * 360 / COUNT) < 1e-9, 'Unexpected requested yaw sequence')
            require(angle_error(frame['actualYawDegrees'], frame['requestedYawDegrees']) < .03, 'Actual yaw is outside capture tolerance')
            require(abs(frame['actualPitchDegrees']) < .01, 'Actual pitch is outside capture tolerance')
            require(frame['camera']['canvasPixels'] == [SIZE, SIZE], 'Unexpected recorded canvas dimensions')
            paths.append(resolve(frame['path'], source.parent))
        require(len(set(paths)) == COUNT, 'Capture frames must have 240 distinct paths')
        loaded.append({'label': label, 'source': source, 'sha256': digest(source), 'capture': capture,
                       'html': html, 'reference': reference, 'paths': paths})
    require(len({item['source'] for item in loaded}) == 3, 'Use three distinct capture records')
    baseline = loaded[0]['capture']
    for item in loaded[1:]:
        capture = item['capture']
        for key in ['referenceManifestSHA256', 'display', 'backend', 'graphicsRenderer', 'browserVersion', 'scriptSHA256']:
            require(capture[key] == baseline[key], 'Capture environment/reference differs: ' + key)
        for key in ['projection', 'pixels', 'target', 'halfExtent', 'expectedDistance']:
            require(capture['camera'][key] == baseline['camera'][key], f'Camera framing differs: {key}')
    return loaded, validate_frames(loaded)


def panel(items, index, font):
    canvas = Image.new('RGB', (WIDTH, HEIGHT), '#191d22')
    draw = ImageDraw.Draw(canvas)
    for position, item in enumerate(items):
        x, y = position * SIZE, 0
        label = item['label']
        require(draw.textbbox((0, 0), label, font=font)[2] <= SIZE - 20, f'Label is too long: {label}')
        draw.text((x + 10, y + 8), label, fill='white', font=font)
        raw = item['paths'][index].read_bytes()
        require(hashlib.sha256(raw).hexdigest() == item['capture']['frames'][index]['sha256'], 'Frame changed during encoding')
        with Image.open(io.BytesIO(raw)) as image:
            matte = Image.new('RGBA', (SIZE, SIZE), '#808080')
            image = Image.alpha_composite(matte, image.convert('RGBA')).convert('RGB')
            canvas.paste(image, (x, y + BAND))
    footer = f'Full images, no crop | Exact cameras and alpha | Yaw {index * 360 / COUNT:05.1f} deg | 24 fps'
    draw.text((10, HEIGHT - FOOTER + 8), footer, fill='white', font=font)
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--captures', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ffmpeg', type=Path, required=True)
    args = parser.parse_args()
    specification, output, ffmpeg = args.captures.resolve(), args.output.resolve(), args.ffmpeg.resolve()
    require(ffmpeg.is_file(), f'ffmpeg not found: {ffmpeg}')
    specification_sha = digest(specification)
    require(not output.exists(), 'Use a new immutable output directory')
    items, validation = load_captures(specification)
    output.mkdir(parents=True, exist_ok=False)
    movie, temporary = output / 'comparison.mp4', output / 'comparison.partial.mp4'
    validation_path = output / 'capture-validation.json'
    with validation_path.open('x') as stream:
        stream.write(json.dumps(validation, indent=2, allow_nan=False) + '\n')
    font = ImageFont.load_default(size=18)
    command = [str(ffmpeg), '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo',
               '-pixel_format', 'rgb24', '-video_size', f'{WIDTH}x{HEIGHT}', '-framerate', str(FPS),
               '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-preset', 'medium', '-crf', '19',
               '-pix_fmt', 'yuv420p', '-frames:v', str(COUNT), '-movflags', '+faststart', str(temporary)]
    composite_sha = hashlib.sha256()
    with (output / 'ffmpeg-encode.log').open('wb') as log:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log)
        try:
            for index in range(COUNT):
                image = panel(items, index, font)
                if index in [0, COUNT // 4, COUNT // 2, 3 * COUNT // 4]:
                    image.save(output / f'poster-yaw{int(index * 360 / COUNT):03d}.png')
                raw = image.tobytes()
                composite_sha.update(raw)
                process.stdin.write(raw)
            process.stdin.close()
            require(process.wait() == 0, 'ffmpeg encoding failed; see ffmpeg-encode.log')
        except BaseException:
            process.kill()
            process.wait()
            raise
    decode_command = [str(ffmpeg), '-hide_banner', '-loglevel', 'info', '-nostats', '-xerror',
                      '-i', str(temporary), '-map', '0:v:0', '-an', '-pix_fmt', 'rgb24', '-f', 'rawvideo', 'pipe:1']
    decoded_sha, decoded_bytes = hashlib.sha256(), 0
    with (output / 'ffmpeg-decode.log').open('wb') as log:
        process = subprocess.Popen(decode_command, stdout=subprocess.PIPE, stderr=log)
        while chunk := process.stdout.read(1024 * 1024):
            decoded_sha.update(chunk)
            decoded_bytes += len(chunk)
        require(process.wait() == 0, 'ffmpeg decode failed; see ffmpeg-decode.log')
    require(decoded_bytes == COUNT * WIDTH * HEIGHT * 3, 'Decoded frame count/dimensions differ')
    decode_log = (output / 'ffmpeg-decode.log').read_text(errors='replace')
    video_line = next(line for line in decode_log.splitlines() if 'Video: h264' in line)
    require('yuv420p' in video_line and f'{WIDTH}x{HEIGHT}' in video_line and '24 fps' in video_line, 'Unexpected encoded video format')
    duration = re.search(r'Duration: (\d+):(\d+):(\d+(?:\.\d+)?)', decode_log)
    require(duration is not None, 'Could not verify MP4 duration')
    seconds = int(duration[1]) * 3600 + int(duration[2]) * 60 + float(duration[3])
    require(abs(seconds - COUNT / FPS) < .001, 'MP4 duration is not 10 seconds')
    for item in items:
        require(digest(item['source']) == item['sha256'], 'Capture record changed during encoding')
        require(digest(item['html']) == item['capture']['htmlSHA256'], 'Inspector HTML changed during encoding')
        require(digest(item['reference']) == item['capture']['referenceManifestSHA256'], 'Reference changed during encoding')
    require(digest(specification) == specification_sha, 'Input specification changed during encoding')
    temporary.replace(movie)
    report = {
        'completed': True, 'specification': str(specification), 'specification_sha256': digest(specification),
        'script_sha256': digest(__file__), 'ffmpeg': str(ffmpeg), 'encode_command': command,
        'layout': {'columns': 3, 'rows': 1, 'panel_pixels': [SIZE, SIZE], 'label_band_pixels': BAND,
                   'footer_pixels': FOOTER, 'output_pixels': [WIDTH, HEIGHT], 'background': '#808080',
                   'compositing': 'Pillow straight-sRGB alpha-over; identical for every panel', 'resize': False, 'crop': False},
        'frames': COUNT, 'fps': FPS, 'duration_seconds': seconds, 'codec': 'h264', 'pixel_format': 'yuv420p',
        'crf': 19, 'faststart': True, 'common_reference_sha256': items[0]['capture']['referenceManifestSHA256'],
        'validation': {'path': str(validation_path), 'sha256': digest(validation_path),
                       'all240AlphaFramesEqualAcrossThreeColumns': True, 'recordedCamerasExactlyEqual': True,
                       'cameraTolerance': CAMERA_TOLERANCE, 'pairs': validation['pairs']},
        'inputs': [{'label': item['label'], 'capture': str(item['source']), 'capture_sha256': item['sha256'],
                    'html_sha256': item['capture']['htmlSHA256'], 'verified_frame_hashes': COUNT,
                    'verified_frame_order': True} for item in items],
        'max_cross_capture_actual_yaw_difference_degrees': max(angle_error(a['capture']['frames'][i]['actualYawDegrees'], b['capture']['frames'][i]['actualYawDegrees']) for a in items for b in items for i in range(COUNT)),
        'composite_rgb24_stream_sha256': composite_sha.hexdigest(),
        'decode': {'completed': True, 'frames': decoded_bytes // (WIDTH * HEIGHT * 3),
                   'rgb24_bytes': decoded_bytes, 'rgb24_stream_sha256': decoded_sha.hexdigest(), 'video_stream': video_line.strip()},
        'artifacts': {name: {'path': str(output / name), 'sha256': digest(output / name)} for name in
                      ['comparison.mp4', 'poster-yaw000.png', 'poster-yaw090.png', 'poster-yaw180.png', 'poster-yaw270.png',
                       'ffmpeg-encode.log', 'ffmpeg-decode.log', 'capture-validation.json']},
    }
    with (output / 'encoding-summary.json').open('x') as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output': str(output), 'frames': COUNT, 'duration_seconds': seconds,
                      'report_sha256': digest(output / 'encoding-summary.json'), 'artifacts': report['artifacts']}, indent=2))


if __name__ == '__main__':
    main()

