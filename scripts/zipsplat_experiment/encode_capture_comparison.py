#!/usr/bin/env python3
"""Encode four complete Inspector captures as a 2x2 full-image comparison.

Requires Python 3.12, Pillow, and the explicit ffmpeg executable passed by CLI.
Input JSON: {"items": [{"label": "A512", "capture": "/path/capture.json"}, ...]}.
"""
import argparse
import hashlib
import io
import json
import re
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

COUNT, SIZE, FPS, BAND, FOOTER = 240, 512, 24, 36, 36
WIDTH, HEIGHT = SIZE * 2, (SIZE + BAND) * 2 + FOOTER


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


def load_captures(path):
    items = json.loads(path.read_text())['items']
    require(len(items) == 4, 'Exactly four captures are required')
    loaded = []
    for item in items:
        label = item['label']
        require(label and label.isascii() and label.isprintable(), 'Use a nonempty single-line English label')
        source = resolve(item['capture'], path.parent)
        capture = json.loads(source.read_text())
        require(capture['completed'] is True and capture['errors'] == [], f'Incomplete capture: {source}')
        require(capture['framesExpected'] == COUNT and len(capture['frames']) == COUNT, f'Expected {COUNT} frames: {source}')
        require(capture['camera']['pixels'] == [SIZE, SIZE] and capture['camera']['projection'] == 'orthographic', 'Expected 512px orthographic captures')
        html = resolve(capture['html'], source.parent)
        reference = resolve(capture['referenceManifest'], source.parent)
        require(digest(html) == capture['htmlSHA256'], f'Inspector HTML changed: {html}')
        require(digest(reference) == capture['referenceManifestSHA256'], f'Reference changed: {reference}')
        paths = []
        for index, frame in enumerate(capture['frames']):
            require(frame['index'] == index, f'Unexpected frame order: {source}, frame {index}')
            require(angle_error(frame['requestedYawDegrees'], index * 360 / COUNT) < 1e-7, 'Unexpected requested yaw sequence')
            require(angle_error(frame['actualYawDegrees'], frame['requestedYawDegrees']) < .031, 'Actual yaw is outside the capture tolerance')
            require(abs(frame['actualPitchDegrees']) < .011, 'Actual pitch is outside the capture tolerance')
            require(frame['camera']['canvasPixels'] == [SIZE, SIZE], 'Unexpected recorded canvas dimensions')
            image_path = resolve(frame['path'], source.parent)
            require(digest(image_path) == frame['sha256'], f'Frame hash mismatch: {image_path}')
            with Image.open(image_path) as image:
                require(image.size == (SIZE, SIZE), f'Unexpected image dimensions: {image_path}')
                image.verify()
            paths.append(image_path)
        loaded.append({'label': label, 'source': source, 'sha256': digest(source), 'capture': capture,
                       'html': html, 'reference': reference, 'paths': paths})
    baseline = loaded[0]['capture']
    for item in loaded[1:]:
        capture = item['capture']
        require(capture['referenceManifestSHA256'] == baseline['referenceManifestSHA256'], 'Reference hashes differ')
        for key in ['projection', 'pixels', 'target', 'halfExtent', 'expectedDistance']:
            require(capture['camera'][key] == baseline['camera'][key], f'Camera framing differs: {key}')
    return loaded


def panel(items, index, font):
    canvas = Image.new('RGB', (WIDTH, HEIGHT), '#191d22')
    draw = ImageDraw.Draw(canvas)
    for position, item in enumerate(items):
        x, y = position % 2 * SIZE, position // 2 * (SIZE + BAND)
        label = item['label']
        require(draw.textbbox((0, 0), label, font=font)[2] <= SIZE - 20, f'Label is too long: {label}')
        draw.text((x + 10, y + 8), label, fill='white', font=font)
        raw = item['paths'][index].read_bytes()
        require(hashlib.sha256(raw).hexdigest() == item['capture']['frames'][index]['sha256'], 'Frame changed during encoding')
        with Image.open(io.BytesIO(raw)) as image:
            matte = Image.new('RGBA', (SIZE, SIZE), '#808080')
            image = Image.alpha_composite(matte, image.convert('RGBA')).convert('RGB')
            canvas.paste(image, (x, y + BAND))
    footer = f'Full images, no crop | Same reference frame | Yaw {index * 360 / COUNT:05.1f} deg | 24 fps'
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
    items = load_captures(specification)
    output.mkdir(parents=True, exist_ok=True)
    movie, temporary = output / 'comparison.mp4', output / 'comparison.partial.mp4'
    require(not movie.exists() and not (output / 'comparison.json').exists(), 'Use a new output directory for a completed comparison')
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
                if index in [0, COUNT // 4]:
                    image.save(output / ('poster-yaw000.png' if index == 0 else 'poster-yaw090.png'))
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
    temporary.replace(movie)
    report = {
        'completed': True, 'specification': str(specification), 'specification_sha256': digest(specification),
        'script_sha256': digest(__file__), 'ffmpeg': str(ffmpeg), 'encode_command': command,
        'layout': {'columns': 2, 'rows': 2, 'panel_pixels': [SIZE, SIZE], 'label_band_pixels': BAND,
                   'footer_pixels': FOOTER, 'output_pixels': [WIDTH, HEIGHT], 'background': '#808080',
                   'compositing': 'Pillow straight-sRGB alpha-over; identical for every panel', 'resize': False, 'crop': False},
        'frames': COUNT, 'fps': FPS, 'duration_seconds': seconds, 'codec': 'h264', 'pixel_format': 'yuv420p',
        'crf': 19, 'faststart': True, 'common_reference_sha256': items[0]['capture']['referenceManifestSHA256'],
        'inputs': [{'label': item['label'], 'capture': str(item['source']), 'capture_sha256': item['sha256'],
                    'html_sha256': item['capture']['htmlSHA256'], 'verified_frame_hashes': COUNT,
                    'verified_frame_order': True} for item in items],
        'max_cross_capture_actual_yaw_difference_degrees': max(angle_error(a['capture']['frames'][i]['actualYawDegrees'], b['capture']['frames'][i]['actualYawDegrees']) for a in items for b in items for i in range(COUNT)),
        'composite_rgb24_stream_sha256': composite_sha.hexdigest(),
        'decode': {'completed': True, 'frames': decoded_bytes // (WIDTH * HEIGHT * 3),
                   'rgb24_bytes': decoded_bytes, 'rgb24_stream_sha256': decoded_sha.hexdigest(), 'video_stream': video_line.strip()},
        'artifacts': {name: {'path': str(output / name), 'sha256': digest(output / name)} for name in
                      ['comparison.mp4', 'poster-yaw000.png', 'poster-yaw090.png', 'ffmpeg-encode.log', 'ffmpeg-decode.log']},
    }
    (output / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'output': str(output), 'frames': COUNT, 'duration_seconds': seconds,
                      'report_sha256': digest(output / 'comparison.json'), 'artifacts': report['artifacts']}, indent=2))


if __name__ == '__main__':
    main()
