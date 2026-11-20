#!/usr/bin/env python3
"""Verify and compare 240-frame official Inspector RGBA captures on the CPU."""
import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

EXPECTED_FRAMES = 240
CAMERA_TOLERANCE = 1e-9
CAMERA_SHAPES = {
    'matrixWorld': (16,), 'matrixWorldInverse': (16,),
    'projectionMatrix': (16,), 'projectionMatrixInverse': (16,),
    'position': (3,), 'quaternion': (4,), 'up': (3,), 'canvasPixels': (2,),
    'near': (), 'far': (), 'left': (), 'right': (), 'top': (), 'bottom': (), 'zoom': (),
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load_capture(path):
    data_bytes = path.read_bytes()
    data = json.loads(data_bytes)
    if data.get('completed') is not True:
        raise ValueError('Capture is not completed')
    if data.get('errors'):
        raise ValueError('Capture contains recorded errors: ' + str(data['errors']))
    frames = data.get('frames')
    if not isinstance(frames, list) or data.get('framesExpected') != EXPECTED_FRAMES or len(frames) != EXPECTED_FRAMES:
        raise ValueError('Require framesExpected=240 and exactly 240 recorded frames')
    paths = []
    for index, frame in enumerate(frames):
        if type(frame.get('index')) is not int or frame['index'] != index:
            raise ValueError('Frame indices must appear exactly once in order 0..239')
        image_path = Path(frame['path'])
        paths.append((image_path if image_path.is_absolute() else path.parent / image_path).resolve())
        value = frame.get('sha256', '')
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('Missing or invalid PNG SHA256 at frame ' + str(index))
    if len(set(paths)) != EXPECTED_FRAMES:
        raise ValueError('Recorded PNG paths must be unique')
    pixels = np.asarray(data['camera']['pixels'])
    if pixels.shape != (2,) or not np.all(np.isfinite(pixels)) or np.any(pixels <= 0) or np.any(pixels != np.floor(pixels)):
        raise ValueError('Invalid declared capture pixel dimensions')
    return data, paths, digest(data_bytes)


def camera_difference(left, right):
    for label, camera in (('left', left), ('right', right)):
        if camera.get('type') != 'OrthographicCamera':
            raise ValueError(label + ': expected recorded OrthographicCamera')
        for key, shape in CAMERA_SHAPES.items():
            value = np.asarray(camera[key], dtype='f8')
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(label + ': invalid camera ' + key)
    if set(left) != set(right):
        raise ValueError('Recorded camera fields differ between captures')
    differences = {}
    for key in left:
        a, b = left[key], right[key]
        if isinstance(a, (int, float, list)) and not isinstance(a, bool):
            a, b = np.asarray(a, dtype='f8'), np.asarray(b, dtype='f8')
            if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
                raise ValueError('Nonfinite or incompatible camera field: ' + key)
            differences[key] = float(np.max(np.abs(a - b))) if a.size else 0.
        elif a != b:
            raise ValueError('Recorded camera metadata differs: ' + key)
    return differences


def load_png(path, expected_sha, pixels):
    raw = path.read_bytes()
    actual_sha = digest(raw)
    if actual_sha != expected_sha:
        raise ValueError('PNG SHA256 mismatch: ' + str(path))
    if len(raw) < 33 or raw[:8] != b'\x89PNG\r\n\x1a\n' or raw[12:16] != b'IHDR' or raw[24:26] != bytes((8, 6)):
        raise ValueError('Expected an 8-bit RGBA PNG: ' + str(path))
    with Image.open(BytesIO(raw)) as image:
        image.verify()
    with Image.open(BytesIO(raw)) as image:
        if image.mode != 'RGBA' or list(image.size) != list(pixels):
            raise ValueError('PNG mode/dimensions differ from recorded camera: ' + str(path))
        rgba = np.asarray(image, dtype='uint8').copy()
    return rgba, actual_sha


def differences(left, right):
    delta = np.abs(left.astype('i2') - right.astype('i2'))
    changed = delta != 0
    pixels = changed if delta.ndim == 2 else changed.any(axis=-1)
    return {'maeUint8': float(delta.mean()), 'maxUint8': int(delta.max()),
            'nonzeroPixels': int(pixels.sum()), 'nonzeroValues': int(changed.sum()),
            'sumAbsoluteDifference': int(delta.sum(dtype='int64')), 'values': int(delta.size),
            'pixels': int(pixels.size)}


def aggregate(frames, field):
    values = [frame[field] for frame in frames]
    total = sum(value['sumAbsoluteDifference'] for value in values)
    size = sum(value['values'] for value in values)
    exact = sum(value['nonzeroPixels'] == 0 for value in values)
    return {'framesCompared': len(values), 'exactFrames': exact, 'changedFrames': len(values) - exact,
            'allFramesZero': exact == len(values), 'all240FramesZero': exact == EXPECTED_FRAMES,
            'maeUint8': total / size, 'maxUint8': max(value['maxUint8'] for value in values),
            'nonzeroPixels': sum(value['nonzeroPixels'] for value in values),
            'nonzeroValues': sum(value['nonzeroValues'] for value in values),
            'sumAbsoluteDifference': total, 'values': size,
            'pixels': sum(value['pixels'] for value in values)}


def compare(left_path, right_path):
    report = {'method': 'official-inspector-240-frame-rgba-comparison-v1',
              'comparisonValid': False, 'status': 'invalid', 'expectedFrames': EXPECTED_FRAMES,
              'cameraTolerance': CAMERA_TOLERANCE,
              'differenceUnits': 'Absolute uint8 differences, range 0..255; MAE divides by channel-value count',
              'left': {'capturePath': str(left_path)}, 'right': {'capturePath': str(right_path)},
              'errors': [], 'camera': {'maxAbsDifference': None, 'maxByField': {}, 'differentFrameIndices': []},
              'pngHashesVerified': {'left': 0, 'right': 0}, 'frames': [], 'alpha': None, 'rgb': None}
    loaded = {}
    for label, path in (('left', left_path), ('right', right_path)):
        try:
            data, paths, capture_sha = load_capture(path)
            loaded[label] = (data, paths)
            report[label].update(captureSha256=capture_sha, completed=True,
                                 framesExpected=data['framesExpected'], frameCount=len(data['frames']),
                                 pixels=data['camera']['pixels'], htmlSha256=data.get('htmlSHA256'),
                                 referenceManifestSha256=data.get('referenceManifestSHA256'),
                                 display=data.get('display'), backend=data.get('backend'),
                                 graphicsRenderer=data.get('graphicsRenderer'), browserVersion=data.get('browserVersion'))
        except (OSError, ValueError, KeyError, TypeError) as error:
            report['errors'].append({'stage': 'capture', 'side': label, 'message': str(error)})
    if len(loaded) != 2:
        return report
    left, left_paths = loaded['left']
    right, right_paths = loaded['right']
    report['recordedReferenceHashesEqual'] = left.get('referenceManifestSHA256') == right.get('referenceManifestSHA256')
    report['recordedDisplaySettingsEqual'] = left.get('display') == right.get('display')
    camera_max, verified_cameras = 0., 0
    for index, (a, b) in enumerate(zip(left['frames'], right['frames'])):
        frame = {'index': index, 'leftPath': str(left_paths[index]), 'rightPath': str(right_paths[index]),
                 'leftPngSha256': a['sha256'], 'rightPngSha256': b['sha256'], 'cameraMaxAbsDifference': None,
                 'cameraDifferences': None, 'alpha': None, 'rgb': None}
        report['frames'].append(frame)
        try:
            expected_yaw = (360 * index / EXPECTED_FRAMES + 180) % 360 - 180
            for label, record in (('left', a), ('right', b)):
                yaw = float(record['requestedYawDegrees'])
                if not np.isfinite(yaw) or abs((yaw - expected_yaw + 180) % 360 - 180) > CAMERA_TOLERANCE:
                    raise ValueError(label + ': requested yaw does not match frame order')
            values = camera_difference(a['camera'], b['camera'])
            maximum = max(values.values(), default=0.)
            frame.update(cameraMaxAbsDifference=maximum, cameraDifferences=values)
            camera_max = max(camera_max, maximum)
            for key, value in values.items():
                report['camera']['maxByField'][key] = max(report['camera']['maxByField'].get(key, 0.), value)
            for label, record, capture in (('left', a, left), ('right', b, right)):
                if record['camera']['canvasPixels'] != capture['camera']['pixels']:
                    raise ValueError(label + ': per-frame canvas differs from declared capture pixels')
            verified_cameras += 1
            if maximum > CAMERA_TOLERANCE:
                report['camera']['differentFrameIndices'].append(index)
        except (ValueError, KeyError, TypeError) as error:
            report['errors'].append({'stage': 'camera', 'index': index, 'message': str(error)})
    report['camera']['maxAbsDifference'] = camera_max if verified_cameras else None
    report['camera']['verifiedFrames'] = verified_cameras
    report['camera']['differentFrames'] = len(report['camera']['differentFrameIndices'])
    if report['camera']['differentFrameIndices']:
        report['errors'].append({'stage': 'camera', 'message': 'Recorded camera difference exceeds 1e-9'})
    cameras_valid = not report['errors']
    for index, frame in enumerate(report['frames']):
        arrays = {}
        for label, data, paths in (('left', left, left_paths), ('right', right, right_paths)):
            try:
                rgba, _ = load_png(paths[index], data['frames'][index]['sha256'], data['camera']['pixels'])
                report['pngHashesVerified'][label] += 1
                arrays[label] = rgba
            except (OSError, ValueError, KeyError, TypeError) as error:
                report['errors'].append({'stage': 'png', 'side': label, 'index': index, 'message': str(error)})
        if cameras_valid and len(arrays) == 2:
            frame['alpha'] = differences(arrays['left'][..., 3], arrays['right'][..., 3])
            frame['rgb'] = differences(arrays['left'][..., :3], arrays['right'][..., :3])
    for label, path in (('left', left_path), ('right', right_path)):
        try:
            if digest(path.read_bytes()) != report[label]['captureSha256']:
                raise ValueError('Capture JSON changed during comparison')
        except (OSError, ValueError) as error:
            report['errors'].append({'stage': 'capture', 'side': label, 'message': str(error)})
    if not report['errors']:
        report.update(comparisonValid=True, status='valid', alpha=aggregate(report['frames'], 'alpha'),
                      rgb=aggregate(report['frames'], 'rgb'))
    else:
        # Partial pixel measurements must not look like a completed alpha comparison.
        for frame in report['frames']:
            frame['alpha'], frame['rgb'] = None, None
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--left', type=Path, required=True)
    parser.add_argument('--right', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    output = args.out.resolve()
    if output.exists():
        raise FileExistsError('Comparison output already exists: ' + str(output))
    report = compare(args.left.resolve(), args.right.resolve())
    report['scriptSha256'] = digest(Path(__file__).read_bytes())
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'result': str(output), 'resultSha256': digest(output.read_bytes()),
                      'comparisonValid': report['comparisonValid'], 'camera': report['camera'],
                      'alpha': report['alpha'], 'rgb': report['rgb'], 'errors': report['errors']}, indent=2), flush=True)
    return 0 if report['comparisonValid'] else 2


if __name__ == '__main__':
    sys.exit(main())
