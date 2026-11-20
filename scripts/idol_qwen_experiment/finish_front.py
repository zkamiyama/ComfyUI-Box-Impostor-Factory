#!/usr/bin/env python3
"""Capture and verify the selected front RGB-only Qwen Box without changing Inspector."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--baseline-root', type=Path, required=True)
    p.add_argument('--capture-script', type=Path, required=True)
    p.add_argument('--alpha-script', type=Path, required=True)
    p.add_argument('--playwright-root', type=Path, required=True)
    p.add_argument('--ffmpeg', type=Path, required=True)
    a = p.parse_args()
    root, baseline = a.root.resolve(), a.baseline_root.resolve()
    candidate = json.loads((root / 'candidate-builds.json').read_text())['qwen_rgb_only']
    assert sha(candidate['archive']) == candidate['archiveSha256']
    receipt = json.loads(Path(candidate['receiptPath']).read_text())
    assert receipt['qa']['changedRgbPixels'] > 0
    assert receipt['qa']['geometryDepthNormalIntervalRangeProxyCameraPreserved']
    assert receipt['qa']['colorAlphaBitDifferences'] == 0
    final = root / 'final'
    viewer = final / 'Idol20_Qwen_Front_Box512.inspector.html'
    archive = final / 'Idol20_Qwen_Front_Box512.box.zip'
    if viewer.exists() or archive.exists():
        raise FileExistsError('Selected deliverable already exists')
    shutil.copyfile(candidate['viewer'], viewer)
    shutil.copyfile(candidate['archive'], archive)
    assert sha(viewer) == sha(candidate['viewer'])
    assert sha(archive) == candidate['archiveSha256']
    env = dict(os.environ)
    for name in ('DISPLAY', 'WAYLAND_DISPLAY'):
        env.pop(name, None)
    output = root / 'capture/qwen-final'
    subprocess.run(['node', str(a.capture_script), '--viewer', str(viewer),
                    '--reference', str(baseline / 'common-frame/manifest.json'), '--out', str(output),
                    '--frames', '240', '--size', '512', '--mode', 'default',
                    '--playwright-root', str(a.playwright_root)], check=True, env=env)
    left_path, right_path = baseline / 'capture/idol20/capture.json', output / 'capture.json'
    left, right = [json.loads(path.read_text()) for path in (left_path, right_path)]
    for field in ('referenceManifestSHA256', 'display', 'backend', 'graphicsRenderer', 'browserVersion'):
        assert left[field] == right[field], 'Capture environment differs: ' + field
    alpha_report = root / 'validation/alpha-240.json'
    subprocess.run([sys.executable, str(a.alpha_script), '--left', str(left_path), '--right', str(right_path),
                    '--out', str(alpha_report)], check=True)
    comparison = json.loads(alpha_report.read_text())
    assert comparison['comparisonValid'] and comparison['alpha']['all240FramesZero']
    assert comparison['rgb']['changedFrames'] > 0
    coverage = []
    for frame in right['frames']:
        with Image.open(frame['path']) as image:
            mask = np.asarray(image)[..., 3] > 127
        yy, xx = np.nonzero(mask)
        assert len(xx) > 1000
        edges = int(mask[0].sum() + mask[-1].sum() + mask[1:-1, 0].sum() + mask[1:-1, -1].sum())
        assert edges == 0
        coverage.append({'index': frame['index'], 'foregroundPixels': len(xx),
                         'boundsXYXY': [int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())], 'edgePixels': edges})
    inputs = root / 'comparison-inputs.json'
    inputs.write_text(json.dumps({'items': [{'label': 'Idol Box / original', 'capture': str(left_path)},
                                           {'label': 'Idol + Qwen / front RGB only', 'capture': str(right_path)}]}, indent=2) + '\n')
    subprocess.run([sys.executable, str(Path(__file__).with_name('encode_comparison.py')), '--captures', str(inputs),
                    '--output', str(root / 'media/selected'), '--ffmpeg', str(a.ffmpeg)], check=True)
    shutil.copyfile(root / 'media/selected/comparison.mp4', final / 'Idol_Qwen_Box_Comparison.mp4')
    for angle in (0, 90, 180, 270):
        shutil.copyfile(root / f'media/selected/poster-yaw{angle:03d}.png', final / f'Idol_Qwen_Box_{angle:03d}.png')
    shutil.copyfile(root / 'diagnostics/ablation-front.png', final / 'Idol_Qwen_Normal_Ablation.png')
    summary = {'selectedVariant': candidate, 'refinedFaces': ['+Z'], 'unchangedFaces': ['+X', '-X', '+Y', '-Y', '-Z'],
               'alpha': comparison['alpha'], 'rgb': comparison['rgb'], 'camera': comparison['camera'],
               'comparisonValid': comparison['comparisonValid'], 'captureEnvironmentEqual': True,
               'fullBody240Frames': coverage, 'alphaReport': str(alpha_report), 'alphaReportSha256': sha(alpha_report),
               'inspectorHTMLUnmodified': sha(viewer) == sha(candidate['viewer']),
               'geometryDepthNormalIntervalRangeProxyCameraPreserved': True,
               'limitations': ['Front RGB-only pilot; other five faces use original Idol colors.',
                              'Pixel alpha equality proves rendered coverage preservation, not image quality or accurate 3D geometry.',
                              'Qwen mask registration measures outline overlap, not internal anatomical correspondence.']}
    (root / 'validation/summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({'completed': True, 'alphaAll240Zero': True, 'rgbChangedFrames': comparison['rgb']['changedFrames'],
                      'viewer': str(viewer), 'archive': str(archive), 'movie': str(final / 'Idol_Qwen_Box_Comparison.mp4')}), flush=True)


if __name__ == '__main__':
    main()
