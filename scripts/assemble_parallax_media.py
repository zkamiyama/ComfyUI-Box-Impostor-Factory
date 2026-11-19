#!/usr/bin/env python3
"""Assemble saved native/hybrid captures; preserve actual traversal order.

The two animations use independently captured forward and reverse samples when
present. They contain no invented intermediate images or repeated fake reversal.
All panels use a fixed pixel crop and nearest enlargement, with explicit yaw labels.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from evaluate_parallax_sweep import matrix, read_rgba, sha


def assemble(sweep_path, evaluation_path, source_strip, output):
    sweep_path, evaluation_path, output = Path(sweep_path), Path(evaluation_path), Path(output)
    if output.exists():
        raise FileExistsError('Preserve completed media: ' + str(output))
    sweep, evaluation = json.loads(sweep_path.read_text()), json.loads(evaluation_path.read_text())
    if sweep['status'] != 'complete' or evaluation['sweepSHA256'] != sha(sweep_path):
        raise ValueError('Media require a complete sweep and its matching independent evaluation.')
    frames = [f for f in sweep['frames'] if f['variant'] in ('native', 'camera-hybrid')]
    groups = {}
    for frame in frames:
        groups.setdefault(frame['sequenceIndex'], {})[frame['variant']] = frame
    groups = [(i, groups[i]) for i in sorted(groups)]
    if any(set(group) != {'native', 'camera-hybrid'} for _, group in groups):
        raise ValueError('Every camera sample must have both display variants.')
    for _, group in groups:
        for key in ('viewMatrix', 'projectionMatrix'):
            if not np.allclose(matrix(group['native'][key]), matrix(group['camera-hybrid'][key]), rtol=0, atol=1e-12):
                raise ValueError('The display columns use different cameras.')
    rois = np.array([f['headROI'] for f in frames])
    roi = [int(rois[:, 0].min()), int(rois[:, 1].min()), int(rois[:, 2].max()), int(rois[:, 3].max())]
    background, gap = (43, 46, 51), 16
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 17)
        small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 13)
    except OSError:
        font = small = ImageFont.load_default()
    cache, input_hashes = {}, {}

    def image(path, expected):
        path = str(path)
        if path not in cache:
            cache[path] = Image.fromarray(read_rgba(path, expected))
            input_hashes[path] = expected
        return cache[path]

    def panel(im, crop, scale):
        selected = im.crop(crop)
        base = Image.new('RGBA', selected.size, (*background, 255))
        base.alpha_composite(selected)
        return base.convert('RGB').resize((selected.width * scale, selected.height * scale), Image.Resampling.NEAREST)

    output.mkdir(parents=True)
    assets = []
    for name, limit in (('wide-orbit.webp', 60), ('near-front-orbit.webp', 30)):
        selected = [(i, group) for i, group in groups if abs(group['native']['angleDegrees']) <= limit]
        playback = []
        for sequence, group in selected:
            frame = group['native']
            panels = [panel(image(group[mode]['rgbaPNG'], group[mode]['rgbaSHA256']), roi, 2)
                      for mode in ('native', 'camera-hybrid')]
            width, height = panels[0].size
            canvas = Image.new('RGB', (width * 2 + gap * 3, height + 104), background)
            draw = ImageDraw.Draw(canvas)
            traversal = frame.get('traversal', 'recorded')
            draw.text((gap, 10), f'Actual camera captures | yaw {frame["angleDegrees"]:+g} deg | {traversal}', fill='white', font=font)
            caption = ('Wide range +/-60 deg: includes large-angle failures' if limit == 60
                       else 'Near-front range +/-30 deg: outside-range frames omitted; see wide orbit')
            draw.text((gap, 34), caption, fill=(220, 220, 220), font=small)
            for col, (label, im) in enumerate(zip(('Native weights / unlit', 'Selected hybrid / unlit (power 8, tolerance 4)'), panels)):
                x = gap + col * (width + gap)
                draw.text((x, 60), label, fill='white', font=small)
                canvas.paste(im, (x, 84))
            playback.append(canvas)
        path = output / name
        playback[0].save(path, save_all=True, append_images=playback[1:], duration=80, loop=0, lossless=True, method=4)
        with Image.open(path) as encoded:
            encoded_count = encoded.n_frames
        assets.append({'path': str(path), 'sha256': sha(path), 'kind': 'actual capture sequence animation',
                       'absoluteYawLimitDegrees': limit, 'durationMsPerCaptureSample': 80,
                       'sampleSequenceIndices': [i for i, _ in selected],
                       'playbackAnglesDegrees': [g['native']['angleDegrees'] for _, g in selected],
                       'playbackTraversals': [g['native'].get('traversal') for _, g in selected],
                       'allSelectedSamplesMarkedIndependentlyRendered': all(g[m].get('independentlyRendered') is True for _, g in selected for m in g),
                       'omittedSequenceIndices': [i for i, g in groups if abs(g['native']['angleDegrees']) > limit],
                       'capturedSamples': len(selected), 'encodedWebPFrames': encoded_count,
                       'fixedCropXYXY': roi, 'nearestEnlargement': 2,
                       'reversePolicy': 'Uses recorded frame sequence; no synthetic ping-pong or interpolated frames.'})
    angles = [-60, -45, -30, -15, 0, 15, 30, 45, 60]
    unique = {}
    for _, group in groups:
        unique.setdefault(group['native']['angleDegrees'], group)
    if any(a not in unique for a in angles):
        raise ValueError('The contact sheet needs the nine recorded wide yaw angles.')
    width, height = roi[2] - roi[0], roi[3] - roi[1]
    board = Image.new('RGB', (len(angles) * (width + gap) + gap, 2 * (height + 44) + 50), background)
    draw = ImageDraw.Draw(board)
    draw.text((gap, 8), 'Actual wide camera sweep | native weights vs selected hybrid | fixed head crop | unlit', fill='white', font=font)
    for row, mode in enumerate(('native', 'camera-hybrid')):
        for col, angle in enumerate(angles):
            frame = unique[angle][mode]
            x, y = gap + col * (width + gap), 46 + row * (height + 44)
            draw.text((x, y), ('Native' if mode == 'native' else 'Hybrid') + f' | yaw {angle:+g} deg', fill='white', font=small)
            board.paste(panel(image(frame['rgbaPNG'], frame['rgbaSHA256']), roi, 1), (x, y + 24))
    contact = output / 'contact-sheet.webp'
    board.save(contact, lossless=True, method=6)
    assets.append({'path': str(contact), 'sha256': sha(contact), 'kind': 'nine-angle contact sheet',
                   'anglesDegrees': angles, 'fixedCropXYXY': roi, 'nearestEnlargement': 1})
    zero = unique[0]
    front_roi = zero['native']['headROI']
    refs = {a['reference']: a for a in evaluation['assets'] if 'reference' in a}
    originals = [('Original F generated PNG', image(refs['originalGeneratedPNG']['path'], refs['originalGeneratedPNG']['sha256'])),
                 ('H native +Z atlas reference', image(refs['nativeAtlas']['path'], refs['nativeAtlas']['sha256'])),
                 ('Native weights / unlit', image(zero['native']['rgbaPNG'], zero['native']['rgbaSHA256'])),
                 ('Selected hybrid / unlit', image(zero['camera-hybrid']['rgbaPNG'], zero['camera-hybrid']['rgbaSHA256']))]
    front_panels = [(label, panel(im, front_roi, 3)) for label, im in originals]
    width, height = front_panels[0][1].size
    front = Image.new('RGB', (4 * (width + gap) + gap, height + 92), background)
    draw = ImageDraw.Draw(front)
    draw.text((gap, 10), 'Front: independently projected source images and actual saved 3D renders', fill='white', font=font)
    draw.text((gap, 34), 'Same viewport and pixel crop | native H atlas and original generated F PNG are separate references', fill=(220, 220, 220), font=small)
    for col, (label, im) in enumerate(front_panels):
        x = gap + col * (width + gap)
        draw.text((x, 58), label, fill='white', font=small)
        front.paste(im, (x, 80))
    path = output / 'front-comparison.webp'
    front.save(path, lossless=True, method=6)
    assets.append({'path': str(path), 'sha256': sha(path), 'kind': 'front independent-reference comparison',
                   'columns': [label for label, _ in front_panels], 'fixedCropXYXY': front_roi, 'nearestEnlargement': 3})
    strip = Path(source_strip)
    target = output / 'source-heads.webp'
    target.write_bytes(strip.read_bytes())
    assets.append({'path': str(target), 'sha256': sha(target), 'kind': 'unchanged native-source comparison',
                   'copiedFrom': str(strip), 'copiedFromSHA256': sha(strip)})
    helper = Path(__file__)
    (output / 'media-helper.py').write_bytes(helper.read_bytes())
    (output / 'evaluator-source.py').write_bytes(helper.with_name('evaluate_parallax_sweep.py').read_bytes())
    proof = {'kind': 'actual-parallax-review-media-v1', 'sweep': str(sweep_path), 'sweepSHA256': sha(sweep_path),
             'evaluation': str(evaluation_path), 'evaluationSHA256': sha(evaluation_path),
             'helperSHA256': sha(helper), 'evaluatorSourceSHA256': sha(helper.with_name('evaluate_parallax_sweep.py')),
             'capturedReversePolicy': sweep.get('reversePolicy'), 'inputPNGFilesSHA256': input_hashes,
             'assets': assets, 'limitations': ['These images document saved captures, not an interactive viewer.',
               'Animation timing is a chosen playback rate, not measured renderer frame rate.',
               'Large-angle failures remain visible in the wide animation and contact sheet.',
               'Fixed crops and nearest enlargement add no source detail.']}
    (output / 'media-provenance.json').write_text(json.dumps(proof, indent=2, allow_nan=False) + '\n')
    return proof


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sweep', type=Path, required=True)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--source-strip', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    proof = assemble(args.sweep, args.evaluation, args.source_strip, args.output)
    print(json.dumps({'assets': proof['assets']}, indent=2))
