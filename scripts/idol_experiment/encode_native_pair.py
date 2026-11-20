#!/usr/bin/env python3
"""Encode two or three full native GS turntables with identical render cameras."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    record = json.loads(path.read_text())
    require(record['kind'] == 'native-splat-turntable-v1', 'Expected a native turntable record')
    require(record['completed'] and not record['errors'], 'Capture is incomplete: ' + str(path))
    require(record['resolution'] == 512, 'Expected full 512px frames')
    require(record['framesExpected'] == 240 and len(record['frames']) == 240, 'Expected 240 distinct frames')
    require(record['fps'] == 24, 'Expected 24 fps')
    require(sha(record['template']) == record['templateSha256'], 'Reference manifest changed')
    for index, frame in enumerate(record['frames']):
        require(frame['index'] == index and frame['requestedYawDegrees'] == index * 360. / 240,
                'Unexpected frame or angle order')
        require(frame['coverage']['pixels'] > 0, 'Empty rendered frame')
        require(sha(frame['path']) == frame['sha256'], 'Frame changed: ' + frame['path'])
        with Image.open(frame['path']) as image:
            require(image.size == (512, 512) and image.mode == 'RGB', 'Unexpected frame format')
    return record


def validate_conditions(records, allow_rigid_y_registration=False):
    baseline = records[0]
    for record in records[1:]:
        for key in ('camera', 'templateSha256', 'nativeRendererSha256', 'renderSettings'):
            require(baseline[key] == record[key], 'Capture conditions differ: ' + key)
        require(allow_rigid_y_registration or baseline['splatRotationNativeY'] == record['splatRotationNativeY'],
                'SPLAT Y rotations differ; explicit --allow-rigid-y-registration is required')
        require(len(baseline['frames']) == len(record['frames']), 'Frame counts differ')
        for a, b in zip(baseline['frames'], record['frames']):
            require(a['camera'] == b['camera'], 'Per-frame cameras differ')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--left', type=Path, required=True)
    parser.add_argument('--right', type=Path, required=True)
    parser.add_argument('--third', type=Path, help='Optional third native turntable status.json')
    parser.add_argument('--left-label', default='TripoSplat / native 3DGS')
    parser.add_argument('--right-label', default='Idol / native 3DGS')
    parser.add_argument('--third-label', default='Idol / alternate settings')
    parser.add_argument('--allow-rigid-y-registration', action='store_true',
                        help='Permit differing recorded SPLAT Y rotations; camera and renderer checks remain exact')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ffmpeg', type=Path, required=True)
    args = parser.parse_args()
    left_path, right_path = args.left.resolve(), args.right.resolve()
    left, right = load(left_path), load(right_path)
    items = [('left', left_path, left, args.left_label), ('right', right_path, right, args.right_label)]
    if args.third is not None:
        third_path = args.third.resolve()
        items.append(('third', third_path, load(third_path), args.third_label))
    validate_conditions([item[2] for item in items], args.allow_rigid_y_registration)
    different_rotations = any(item[2]['splatRotationNativeY'] != left['splatRotationNativeY'] for item in items[1:])
    ffmpeg = args.ffmpeg.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    width, height, band = 512 * len(items), 584, 36
    font = ImageFont.load_default(size=18)
    movie = output / 'native-gs-comparison.mp4'
    partial = output / 'native-gs-comparison.partial.mp4'
    command = [str(ffmpeg), '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo',
               '-pixel_format', 'rgb24', '-video_size', f'{width}x{height}', '-framerate', '24',
               '-i', 'pipe:0', '-frames:v', '240', '-an', '-c:v', 'libx264', '-preset', 'medium',
               '-crf', '19', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(partial)]
    raw_sha = hashlib.sha256()
    posters = []
    with (output / 'encode.log').open('wb') as log:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log)
        try:
            for index in range(240):
                panel = Image.new('RGB', (width, height), '#191d22')
                draw = ImageDraw.Draw(panel)
                for column, (_, _, record, label) in enumerate(items):
                    require(label.isascii() and label.isprintable(), 'Labels must be single-line ASCII')
                    require(draw.textbbox((0, 0), label, font=font)[2] <= 492, 'Label is too long')
                    draw.text((column * 512 + 10, 8), label, font=font, fill='white')
                    frame = record['frames'][index]
                    require(sha(frame['path']) == frame['sha256'], 'Frame changed while encoding')
                    with Image.open(frame['path']) as image:
                        panel.paste(image, (column * 512, band))
                registration = ' | Rigid Y registered' if different_rotations else ''
                draw.text((10, height - band + 8),
                          f'Full images | Identical cameras{registration} | Yaw {index * 1.5:05.1f} deg | 24 fps',
                          font=font, fill='white')
                if index in (0, 60, 120, 180):
                    path = output / f'poster-yaw{int(index * 1.5):03d}.png'
                    panel.save(path)
                    posters.append({'path': str(path), 'sha256': sha(path)})
                raw = panel.tobytes()
                raw_sha.update(raw)
                process.stdin.write(raw)
            process.stdin.close()
            require(process.wait() == 0, 'Video encode failed; see encode.log')
        except BaseException:
            process.kill()
            process.wait()
            raise
    # Decode every output frame to validate stream completeness and actual dimensions.
    decoded_bytes = 0
    with (output / 'decode.log').open('wb') as log:
        process = subprocess.Popen([str(ffmpeg), '-hide_banner', '-loglevel', 'error', '-xerror',
                                    '-i', str(partial), '-map', '0:v:0', '-an', '-pix_fmt', 'rgb24',
                                    '-f', 'rawvideo', 'pipe:1'], stdout=subprocess.PIPE, stderr=log)
        while block := process.stdout.read(1024 * 1024):
            decoded_bytes += len(block)
        require(process.wait() == 0, 'Video decode failed; see decode.log')
    require(decoded_bytes == width * height * 3 * 240, 'Decoded frame count or dimensions differ')
    partial.replace(movie)
    record = {'completed': True, 'left': str(left_path), 'leftSha256': sha(left_path),
              'right': str(right_path), 'rightSha256': sha(right_path),
              'scriptSha256': sha(__file__), 'frames': 240, 'fps': 24, 'durationSeconds': 10,
              'panelPixels': [512, 512], 'outputPixels': [width, height], 'columns': len(items), 'crop': 'none',
              'exactCamerasMatch': True, 'renderSettingsMatch': True,
              'rigidYRegistration': {'explicitlyAllowed': args.allow_rigid_y_registration,
                                     'differentInputRotations': different_rotations,
                                     'axis': 'native SPLAT Y',
                                     'operation': 'recorded during rendering; encoder applies no model transform'},
              'inputs': [{'role': role, 'label': label, 'capture': str(path), 'captureSha256': sha(path),
                          'splatRotationNativeY': capture['splatRotationNativeY'],
                          'rotationDeltaDegreesFromLeft': capture['splatRotationNativeY'] - left['splatRotationNativeY'],
                          'edgeContactFrames': sum(frame['coverage']['edgePixels'] > 0 for frame in capture['frames'])}
                         for role, path, capture, label in items],
              'leftEdgeContactFrames': sum(frame['coverage']['edgePixels'] > 0 for frame in left['frames']),
              'rightEdgeContactFrames': sum(frame['coverage']['edgePixels'] > 0 for frame in right['frames']),
              'video': str(movie), 'videoSha256': sha(movie), 'posters': posters,
              'uncompressedCompositeSha256': raw_sha.hexdigest(), 'decodedBytes': decoded_bytes,
              'encodeCommand': command}
    if args.third is not None:
        record.update(third=str(third_path), thirdSha256=sha(third_path))
    (output / 'comparison.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
