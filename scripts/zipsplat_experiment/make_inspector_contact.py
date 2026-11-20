#!/usr/bin/env python3
"""Assemble eight real capture frames; no generated or interpolated views."""
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tile', type=int, default=256)
    args = parser.parse_args()
    record = json.loads(args.capture.read_text())
    if not record['completed'] or record['errors']:
        raise ValueError('A completed successful capture is required')
    frames = record['frames']
    sheet = Image.new('RGB', (args.tile * 4, (args.tile + 24) * 2), '#808080')
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    selected = []
    for index, yaw in enumerate(range(0, 360, 45)):
        frame = min(frames, key=lambda row: abs((row['requestedYawDegrees'] - yaw + 180) % 360 - 180))
        path = Path(frame['path'])
        if digest(path) != frame['sha256']:
            raise ValueError('Capture frame changed: ' + str(path))
        with Image.open(path) as source:
            image = source.convert('RGBA').resize((args.tile, args.tile), Image.Resampling.LANCZOS)
        x, y = index % 4 * args.tile, index // 4 * (args.tile + 24)
        sheet.paste(image, (x, y + 24), image)
        draw.text((x + 8, y + 6), f"yaw {frame['actualYawDegrees']:.2f} deg", fill='white', font=font)
        selected.append({'requested': yaw, 'index': frame['index'], 'sha256': frame['sha256']})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)
    provenance = {'capture': str(args.capture.resolve()), 'capture_sha256': digest(args.capture),
                  'selected_frames': selected, 'background': '#808080',
                  'output': str(args.output.resolve()), 'sha256': digest(args.output)}
    args.output.with_suffix('.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps(provenance, indent=2))


if __name__ == '__main__':
    main()
