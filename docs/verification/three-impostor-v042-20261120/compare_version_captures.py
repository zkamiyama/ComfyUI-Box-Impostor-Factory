"""Compare unchanged official Inspector snapshots from three-impostor 0.4.1 and 0.4.2."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


VERSIONS = ('0.4.1', '0.4.2')
HEAD = (378, 115, 654, 271)
FACE = (435, 155, 486, 231)
LEGS = (388, 550, 644, 940)
BG = (21, 24, 30)
TEXT = (237, 241, 247)
MUTED = (177, 192, 211)


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def font(size):
    return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', size)


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def backdrop(image):
    background = Image.new('RGBA', image.size, BG + (255,))
    background.alpha_composite(image)
    return background.convert('RGB')


def load(run, case, angles):
    path = Path(run['captureJSON'])
    assert sha(path) == run['captureJSONSHA256']
    report = json.loads(path.read_text())
    assert not report['errors'] and report['stageState'] == 'Native Box'
    assert 'SwiftShader' in report['graphicsRenderer'] and not report['hardwareGPUUsed']
    assert report['htmlSHA256'] == case['htmlSHA256'] == sha(case['html'])
    assert report['assetId'] == case['assetId']
    assert report['display'] == case['payload']['display']
    assert report['canvasPixels'] == [1024, 1024]
    frames, images = {}, {}
    for frame in report['yawFrames']:
        angle = frame['requestedYawDegrees']
        assert angle not in frames
        assert sha(frame['path']) == frame['sha256']
        image = Image.open(frame['path']).convert('RGBA')
        assert image.size == (1024, 1024)
        frames[angle], images[angle] = frame, image
    assert set(frames) == set(angles)
    return {'report': report, 'frames': frames, 'images': images}


def differences(a, b):
    da = np.abs(a[..., 3].astype(np.int16) - b[..., 3].astype(np.int16))
    drgb = np.abs(a[..., :3].astype(np.int16) - b[..., :3].astype(np.int16))
    am, bm = a[..., 3] > 0, b[..., 3] > 0
    union, intersection = am | bm, am & bm
    opaque = (a[..., 3] == 255) & (b[..., 3] == 255)
    assert union.any(), 'Both frames are blank'
    def rgb(mask):
        return {'pixels': int(mask.sum()),
                'changedPixels': int((np.any(drgb != 0, axis=2) & mask).sum()),
                'meanAbsoluteByteDifference': float(drgb[mask].mean()) if mask.any() else None,
                'maxAbsoluteByteDifference': int(drgb[mask].max()) if mask.any() else None}
    return {'rgbaByteIdentical': bool(np.array_equal(a, b)),
            'rgbaChangedPixels': int(np.any(a != b, axis=2).sum()),
            'alpha': {'byteIdentical': bool(np.array_equal(a[..., 3], b[..., 3])),
                      'changedPixels': int((da != 0).sum()),
                      'maxAbsoluteByteDifference': int(da.max()),
                      'meanAbsoluteByteDifference': float(da.mean()),
                      'oldForegroundPixels': int(am.sum()), 'newForegroundPixels': int(bm.sum()),
                      'newlyCoveredPixels': int((bm & ~am).sum()),
                      'newlyEmptyPixels': int((am & ~bm).sum()),
                      'nonzeroCoverageIoU': float(intersection.sum() / union.sum())},
            'rgbOnForegroundUnion': rgb(union), 'rgbOnOpaqueIntersection': rgb(opaque)}


def board(pair, family, angles, crop, scale, title, path):
    width, height = (crop[2] - crop[0]) * scale, (crop[3] - crop[1]) * scale
    cell_width = max(width, 300) + 24
    canvas = Image.new('RGB', (cell_width * 2 + 24, len(angles) * (height + 58) + 116), BG)
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 16), f'{family} / {title}', font=font(23), fill=TEXT)
    draw.text((24, 52), 'Same source maps and camera / official Inspector / WebGL2 SwiftShader', font=font(15), fill=MUTED)
    for row, angle in enumerate(angles):
        for column, version in enumerate(VERSIONS):
            x, y = 24 + cell_width * column, 82 + row * (height + 58)
            actual = pair[version]['frames'][angle]['actualYawDegrees']
            draw.text((x, y), f'three-impostor {version} / yaw {actual:+.4f}', font=font(17), fill=TEXT)
            tile = backdrop(pair[version]['images'][angle].crop(crop))
            canvas.paste(tile.resize((width, height), Image.Resampling.NEAREST), (x, y + 30))
    draw.text((24, canvas.height - 27), f'Crop {list(crop)} / {scale}x nearest / raw 1024px', font=font(14), fill=MUTED)
    canvas.save(path)
    return path


def combined_45_board(grouped, path):
    width, height = (HEAD[2] - HEAD[0]) * 2, (HEAD[3] - HEAD[1]) * 2
    cell_width = width + 24
    canvas = Image.new('RGB', (cell_width * 2 + 24, len(grouped) * (height + 58) + 116), BG)
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 16), 'Actual Box Impostor / +45 degree / 0.4.1 vs 0.4.2', font=font(25), fill=TEXT)
    draw.text((24, 52), 'Same source maps and exact camera / official Inspector / WebGL2 SwiftShader', font=font(17), fill=MUTED)
    for row, (family, pair) in enumerate(grouped.items()):
        for column, version in enumerate(VERSIONS):
            x, y = 24 + cell_width * column, 82 + row * (height + 58)
            yaw = pair[version]['frames'][45]['actualYawDegrees']
            draw.text((x, y), f'{family} / three-impostor {version} / yaw {yaw:+.4f}', font=font(19), fill=TEXT)
            crop = backdrop(pair[version]['images'][45].crop(HEAD))
            canvas.paste(crop.resize((width, height), Image.Resampling.NEAREST), (x, y + 30))
    draw.text((24, canvas.height - 27), f'Crop {list(HEAD)} / 2x nearest / raw 1024px / no retouching', font=font(16), fill=MUTED)
    canvas.save(path)
    return path


def benchmark_summary(work, prepared):
    path = work / 'benchmark-runs.json'
    if not path.exists():
        return {'measured': False, 'reason': 'Benchmark phase has not run.'}
    runs = json.loads(path.read_text())['runs']
    grouped, records, cameras = {}, [], {}
    for run in runs:
        record = load(run, prepared['cases'][run['case']], [45])
        bench = record['report']['benchmark']
        camera = record['frames'][45]['actualCamera']
        anchor = cameras.setdefault(run['family'], camera)
        assert all(np.max(np.abs(np.asarray(camera[key]) - anchor[key])) <= 1e-8
                   for key in ['matrixWorld', 'projectionMatrix'])
        assert bench['warmupFrames'] == 8 and bench['measuredFrames'] == 30
        samples = np.asarray(bench['samplesMilliseconds'], dtype=float)
        assert samples.size == 30 and np.isfinite(samples).all() and (samples >= 0).all()
        row = {'family': run['family'], 'version': run['packageVersion'], 'trial': run['trial'],
               'medianMilliseconds': float(np.median(samples)),
               'p95Milliseconds': float(np.percentile(samples, 95)),
               'zeroDurationSamples': int((samples == 0).sum()),
               'samplesMilliseconds': samples.tolist()}
        records.append(row)
        grouped.setdefault(run['family'], {}).setdefault(run['packageVersion'], []).append(row)
    zero_samples = sum(row['zeroDurationSamples'] for row in records)
    if zero_samples:
        return {'measured': False, 'attempted': True,
                'reason': f'{zero_samples} of {len(records) * 30} per-draw durations were 0 ms; the remaining samples are quantized near 0.1 ms. This probe cannot resolve a defensible frame-rendering speed comparison.',
                'method': '8 warmup draws then 30 official api.draw() + WebGL2 gl.finish() draws per trial; 3 interleaved version pairs',
                'scope': 'Attempted CPU SwiftShader timing; not accepted as frame-rendering or GPU performance',
                'speedupReported': False, 'trials': records}
    totals = {}
    for family, versions in grouped.items():
        if set(versions) != set(VERSIONS) or any(sorted(row['trial'] for row in trials) != [1, 2, 3]
                                                for trials in versions.values()):
            return {'measured': False, 'reason': 'Three paired interleaved trials did not complete.', 'completedTrials': records}
        medians = {}
        for version, trials in versions.items():
            medians[version] = float(np.median([row['medianMilliseconds'] for row in trials]))
        totals[family] = {'medianOfTrialMediansMilliseconds': medians,
                          'newOverOldLatencyRatio': medians['0.4.2'] / medians['0.4.1']}
    return {'measured': True, 'method': '8 warmup draws then 30 official api.draw() + WebGL2 gl.finish() draws, 3 interleaved trials',
            'scope': 'CPU SwiftShader synchronized draw latency, not RTX GPU performance or HTML loading',
            'trials': records, 'families': totals}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('work', type=Path)
    args = parser.parse_args()
    work = args.work
    prepared_path, runs_path = work / 'prepared-captures.json', work / 'visual-runs.json'
    prepared = json.loads(prepared_path.read_text())
    config = json.loads(Path(prepared['config']).read_text())
    assert sha(prepared['config']) == prepared['configSHA256']
    angles = config['angles']
    grouped = {}
    for run in json.loads(runs_path.read_text())['runs']:
        pair = grouped.setdefault(run['family'], {})
        assert run['packageVersion'] not in pair
        pair[run['packageVersion']] = load(run, prepared['cases'][run['case']], angles)
    outputs, metrics = [], {}
    for family, pair in grouped.items():
        assert set(pair) == set(VERSIONS)
        metrics[family] = {}
        for angle in angles:
            old, new = pair['0.4.1'], pair['0.4.2']
            camera_deltas = {key: float(np.max(np.abs(np.asarray(old['frames'][angle]['actualCamera'][key]) - new['frames'][angle]['actualCamera'][key])))
                             for key in ['matrixWorld', 'projectionMatrix']}
            assert max(camera_deltas.values()) <= 1e-8, (family, angle, camera_deltas)
            a, b = np.asarray(old['images'][angle]), np.asarray(new['images'][angle])
            region_metrics = {}
            for region, crop in [('head', HEAD), ('faceDetail', FACE), ('legs', LEGS)]:
                x0, y0, x1, y1 = crop
                region_metrics[region] = differences(a[y0:y1, x0:x1], b[y0:y1, x0:x1])
            metrics[family][angle] = {'maxCameraMatrixDifference': camera_deltas,
                                      'fullFrame': differences(a, b), 'regions': region_metrics}
            alpha_diff = np.abs(a[..., 3].astype(np.int16) - b[..., 3].astype(np.int16)).astype(np.uint8)
            path = work / f'{family}-yaw-{angle}-alpha-absolute-difference.png'
            Image.fromarray(alpha_diff).save(path)
            outputs.append(path)
        outputs.append(board(pair, family, angles, HEAD, 2, 'head / actual version comparison', work / f'{family}-v041-v042-head.png'))
        outputs.append(board(pair, family, [45], FACE, 6, '45 degree face detail', work / f'{family}-v041-v042-face45.png'))
        outputs.append(board(pair, family, [45, 60], LEGS, 1, 'leg coverage / actual version comparison', work / f'{family}-v041-v042-legs.png'))
    outputs.append(combined_45_board(grouped, work / '45-degree-comparison.png'))
    proof = {'kind': __doc__, 'preparedCapturesSHA256': sha(prepared_path), 'visualRunsSHA256': sha(runs_path),
             'scriptSHA256': sha(__file__), 'inputChecks': prepared['inputChecks'],
             'anglesDegrees': angles, 'regions': {'head': HEAD, 'faceDetail': FACE, 'legs': LEGS},
             'backend': 'WebGL2 / CPU SwiftShader', 'metrics': metrics,
             'hitDepth': 'Unavailable through the official snapshot API; alpha coverage is observed, not ray-hit depth.',
             'interpretation': 'Pixel differences quantify renderer-version effects. They are not a perceptual-quality or 3D-accuracy score.',
             'processing': 'Original RGBA snapshots are unchanged. Boards use fixed crops, nearest-neighbor scaling and a solid alpha-composited background only. Difference PNGs contain absolute alpha-byte deltas.',
             'benchmark': benchmark_summary(work, prepared),
             'outputs': [{'path': str(path), 'sha256': sha(path), 'size': list(Image.open(path).size)} for path in outputs]}
    write(work / 'version-comparison.json', proof)
    print(json.dumps({'proof': str(work / 'version-comparison.json'), 'metrics': metrics, 'benchmark': proof['benchmark'], 'outputs': proof['outputs']}))


if __name__ == '__main__':
    main()
