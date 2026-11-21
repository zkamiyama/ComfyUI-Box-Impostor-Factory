#!/usr/bin/env python3
"""Verify two official top/bottom captures and place complete 512px PNGs side by side."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from encode_three import alpha_difference, camera_difference, digest, load_rgba, rgb_difference


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_capture(path):
    data = json.loads(path.read_text())
    require(data.get('completed') is True and data.get('errors') == [], 'Capture is incomplete: ' + str(path))
    require(data.get('method') == 'official-inspector-top-bottom-axis-ui-v1', 'Unexpected capture method')
    require(data.get('framesExpected') == 2 and data.get('viewsExpected') == ['top', 'bottom'], 'Expected top and bottom')
    require(data.get('sourceFaceResolution') == 512 and data.get('noCropOrResize') is True, 'Require native full Box512')
    require(data['camera']['projection'] == 'orthographic' and data['camera']['pixels'] == [512, 512], 'Require full orthographic512')
    require(data['display']['mode'] == 'default' and not data['display']['materialOverride']
            and not data['display']['toneMappingOverride'], 'Expected unchanged official display')
    require(data['scriptSHA256'] == digest(Path(__file__).with_name('capture_top_bottom.mjs')), 'Capture script changed')
    require(data['frames'] and [frame['view'] for frame in data['frames']] == ['top', 'bottom'], 'View order differs')
    inputs = {str(path): digest(path)}
    for name, hash_name in [('html', 'htmlSHA256'), ('referenceManifest', 'referenceManifestSHA256')]:
        p = Path(data[name]).resolve(strict=True)
        require(digest(p) == data[hash_name], 'Viewer/reference changed: ' + str(p))
        inputs[str(p)] = data[hash_name]
    for index, frame in enumerate(data['frames']):
        sign = 1 if index == 0 else -1
        require(frame['index'] == index and frame['requestedOutward'] == [0, sign, 0], 'Unexpected axis request')
        require(np.isfinite(frame['axisErrorDegrees']) and frame['axisErrorDegrees'] < .03, 'Axis camera missed its requested view')
        require(frame['renderedCallsBeforeReadback'] > 0, 'Save PNG did not render')
        p = Path(frame['path']).resolve(strict=True)
        require(digest(p) == frame['sha256'], 'Source PNG changed')
        inputs[str(p)] = frame['sha256']
    return data, inputs


def coverage(image):
    mask = image[..., 3] > 127
    yy, xx = np.nonzero(mask)
    edges = int(mask[0].sum() + mask[-1].sum() + mask[1:-1, 0].sum() + mask[1:-1, -1].sum())
    require(len(xx) > 1000 and edges == 0, 'Foreground is empty, tiny, or clipped')
    return {'foregroundPixels': len(xx), 'edgePixels': edges,
            'boundsXYXY': [int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())]}


def run(args):
    before_path, after_path = args.before_capture.resolve(strict=True), args.after_capture.resolve(strict=True)
    output = args.output.resolve()
    require(not output.exists(), 'Require a new comparison output directory')
    before, before_inputs = read_capture(before_path)
    after, after_inputs = read_capture(after_path)
    for key in ['referenceManifest', 'referenceManifestSHA256', 'display', 'backend', 'graphicsRenderer',
                'browserVersion', 'scriptSHA256', 'chromiumExecutable', 'camera']:
        require(before[key] == after[key], 'Capture settings differ: ' + key)
    require(before['assetId'] != after['assetId'], 'Use the distinct prior and Qwen assets')
    sheet = Image.new('RGB', (1024, 1096), '#191d22')
    draw = ImageDraw.Draw(sheet)
    font_path = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    font = ImageFont.truetype(str(font_path), 18) if font_path.is_file() else ImageFont.load_default()
    records = []
    for index, (left, right) in enumerate(zip(before['frames'], after['frames'])):
        images = [load_rgba(Path(frame['path']), frame['sha256']) for frame in [left, right]]
        camera = camera_difference(left['camera'], right['camera'])
        maximum = max(camera.values(), default=0.)
        require(maximum == 0, 'Actual top/bottom cameras must be exactly equal')
        alpha = alpha_difference(images[0][..., 3], images[1][..., 3])
        require(alpha['maxUint8'] == 0 and alpha['nonzeroPixels'] == 0, 'Rendered alpha changed')
        body = [coverage(image) for image in images]
        for column, (image, label) in enumerate(zip(images, [args.before_label, args.after_label])):
            y = index * 548
            draw.text((column * 512 + 12, y + 8), left['view'].title() + ' | ' + label, font=font, fill='#f4f5f6')
            frame = Image.alpha_composite(Image.new('RGBA', (512, 512), (128, 128, 128, 255)),
                                          Image.fromarray(image)).convert('RGB')
            sheet.paste(frame, (column * 512, y + 36))
        records.append({'view': left['view'], 'sources': [
            {'path': frame['path'], 'sha256': frame['sha256'], 'camera': frame['camera'],
             'actualOutward': frame['actualOutward'], 'axisErrorDegrees': frame['axisErrorDegrees']}
            for frame in [left, right]], 'cameraMaxAbsDifference': maximum, 'cameraFieldDifferences': camera,
            'alpha': alpha, 'rgb': rgb_difference(images[0][..., :3], images[1][..., :3]), 'coverage': body})
    require(sum(record['rgb']['nonzeroPixels'] for record in records) > 0, 'RGB did not change')
    inputs = {**before_inputs, **after_inputs, str(Path(__file__).resolve()): digest(__file__),
              str(Path(__file__).with_name('encode_three.py').resolve()): digest(Path(__file__).with_name('encode_three.py'))}
    require(all(digest(path) == sha for path, sha in inputs.items()), 'An input changed during comparison')
    output.mkdir(parents=True)
    image_path = output / 'top-bottom-before-after.png'
    sheet.save(image_path)
    report = {'completed': True, 'method': 'official-inspector-top-bottom-rgb-comparison-v1',
              'captures': {'before': str(before_path), 'after': str(after_path)}, 'inputHashes': inputs,
              'cameraMaxAbsDifference': 0, 'all2AlphaFramesZero': True,
              'alphaMaxUint8': 0, 'alphaNonzeroPixels': 0, 'allFramesNonemptyAndUnclipped': True,
              'htmlUnmodified': True, 'sheet': {'path': str(image_path), 'sha256': digest(image_path),
                  'pixels': [1024, 1096], 'sourceFramePixels': [512, 512], 'crop': False, 'resize': False,
                  'backgroundRGB': [128, 128, 128], 'titleHeight': 36}, 'views': records,
              'limitations': ['Alpha equality is limited to these two cameras and 8-bit PNG precision.',
                              'RGB difference proves a change, not internal texture correspondence or quality.']}
    report_path = output / 'comparison.json'
    with report_path.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return {'completed': True, 'report': str(report_path), 'reportSha256': digest(report_path),
            'sheet': report['sheet'], 'cameraMaxAbsDifference': 0, 'all2AlphaFramesZero': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before-capture', required=True, type=Path)
    parser.add_argument('--after-capture', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--before-label', default='Transferred color')
    parser.add_argument('--after-label', default='Transferred + Qwen')
    print(json.dumps(run(parser.parse_args()), indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
