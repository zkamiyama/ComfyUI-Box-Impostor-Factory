#!/usr/bin/env python3
"""Assemble actual immutable camera-depth frames from two complete sweep runs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--first', type=Path, required=True)
    ap.add_argument('--second', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--angles', default='0,30,45')
    args = ap.parse_args()
    angles = list(map(float, args.angles.split(',')))
    paths = [args.first.resolve(), args.second.resolve()]
    reports = [json.loads(p.read_text()) for p in paths]
    if any(r.get('status') != 'complete' or not r.get('geometryEqualityPassed') for r in reports):
        raise ValueError('Only complete, geometry-verified sweeps can be compared.')
    if reports[0]['source']['manifestSHA256'] != reports[1]['source']['manifestSHA256']:
        raise ValueError('The source atlas/geometry manifests differ.')
    selected = [[next(f for f in r['frames'] if f['variant'] == 'camera-depth' and f['angleDegrees'] == angle)
                 for angle in angles] for r in reports]
    for a, b in zip(*selected):
        for key in ('viewMatrix', 'projectionMatrix'):
            if not np.allclose(a[key], b[key], rtol=0, atol=1e-12):
                raise ValueError('Comparison cameras differ.')
    boxes = np.array([f['headROI'] for row in selected for f in row], dtype=int)
    roi = [int(boxes[:, 0].min()), int(boxes[:, 1].min()), int(boxes[:, 2].max()), int(boxes[:, 3].max())]
    scale, gap = 2, 16
    size = ((roi[2] - roi[0]) * scale, (roi[3] - roi[1]) * scale)
    image = Image.new('RGB', (len(angles) * (size[0] + gap) + gap, 2 * (size[1] + 66) + 50), (43, 46, 51))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
    draw.text((gap, 12), 'Actual saved renders | same camera | fixed crop | 2x nearest-neighbor enlargement', font=font, fill='white')
    receipts = []
    for row, entries in enumerate(selected):
        y = 50 + row * (size[1] + 66)
        tolerance = entries[0].get('materialOptions', {}).get('depthToleranceTexels', 'recorded')
        draw.text((gap, y), f'camera-depth | depth tolerance: {tolerance} source texels', font=font, fill='white')
        for column, f in enumerate(entries):
            path = Path(f['rgbaPNG'])
            if not path.is_absolute():
                path = paths[row].parent / path
            if sha(path) != f['rgbaSHA256']:
                raise ValueError('Source capture changed.')
            rgba = Image.open(path).convert('RGBA').crop(roi)
            tile = Image.new('RGBA', rgba.size, (43, 46, 51, 255))
            tile.alpha_composite(rgba)
            x = gap + column * (size[0] + gap)
            draw.text((x, y + 25), f'Yaw {f["angleDegrees"]:+g} degrees', font=font, fill='white')
            image.paste(tile.convert('RGB').resize(size, Image.Resampling.NEAREST), (x, y + 50))
            receipts.append({'row': row, 'angleDegrees': f['angleDegrees'], 'path': str(path),
                             'sha256': sha(path), 'materialOptions': f.get('materialOptions'),
                             'viewMatrix': f['viewMatrix'], 'projectionMatrix': f['projectionMatrix']})
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output / 'depth-tolerance-comparison.webp'
    image.save(output, lossless=True, method=6)
    proof = {'method': 'immutable-two-sweep-fixed-crop-contact-sheet-v1',
             'sweeps': [{'path': str(p), 'sha256': sha(p)} for p in paths],
             'sourceFrames': receipts, 'fixedCropXYXY': roi, 'nearestEnlargement': scale,
             'sameCameras': True, 'sameSourceManifest': True, 'imageTrackingOrRegistration': False,
             'output': str(output.resolve()), 'outputSHA256': sha(output),
             'script': str(Path(__file__).resolve()), 'scriptSHA256': sha(__file__)}
    (args.output / 'depth-tolerance-comparison.json').write_text(json.dumps(proof, indent=2) + '\n')
    print(json.dumps({'image': str(output.resolve()), 'sha256': sha(output), 'crop': roi,
                      'angles': angles, 'size': list(image.size)}, indent=2))


if __name__ == '__main__':
    main()
