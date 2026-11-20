#!/usr/bin/env python3
"""Build and inspect RGB-only, normal-only, and combined Idol Box ablations."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--baseline-asset', type=Path, required=True)
    p.add_argument('--face-captures', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--qwen-run', type=Path, required=True)
    p.add_argument('--inspector', type=Path, required=True)
    p.add_argument('--capture-script', type=Path, required=True)
    p.add_argument('--playwright-root', type=Path, required=True)
    a = p.parse_args()
    root = a.root.resolve()
    scripts = Path(__file__).resolve().parent
    results = {}
    env = dict(os.environ)
    for name in ('DISPLAY', 'WAYLAND_DISPLAY'):
        env.pop(name, None)
    (root / 'diagnostics').mkdir(exist_ok=True)

    def build(name, command):
        proc = subprocess.run([sys.executable, *map(str, command)], text=True, capture_output=True, check=True)
        result = json.loads(proc.stdout)
        results[name] = result
        (root / 'candidate-builds.json').write_text(json.dumps(results, indent=2) + '\n')
        print(json.dumps({'built': name, **result}), flush=True)
        return result

    normal = build('normal_only', [scripts / 'replace_normals_from_depth.py', '--baseline-asset', a.baseline_asset,
                   '--face-captures', a.face_captures, '--output', root / 'box/normal-only-sigma1', '--smooth-sigma', '1'])
    rgb = build('qwen_rgb_only', [scripts / 'compose_qwen_front.py', '--baseline-asset', a.baseline_asset,
                '--qwen-rgb', a.qwen_run / 'generated.png', '--generated-mask', a.qwen_run / 'mask.png',
                '--square-alpha', root / 'guides1024/front/alpha.npy',
                '--guide-meta', root / 'guides1024/front/guide.json', '--output', root / 'box/qwen-nolora090'])
    both = build('qwen_and_normals', [scripts / 'replace_normals_from_depth.py', '--baseline-asset', rgb['assetPath'],
                 '--face-captures', a.face_captures, '--output', root / 'box/qwen-normals-sigma1', '--smooth-sigma', '1'])
    for name, result in results.items():
        viewer = root / 'diagnostics' / (name + '.inspector.html')
        subprocess.run([sys.executable, str(a.inspector), result['archive'], '-o', str(viewer)], check=True)
        capture = root / 'capture' / name
        subprocess.run(['node', str(a.capture_script), '--viewer', str(viewer), '--reference', str(a.reference),
                        '--out', str(capture), '--frames', '240', '--size', '512', '--mode', 'default', '--sparse',
                        '--playwright-root', str(a.playwright_root)], check=True, env=env)
        result.update(viewer=str(viewer), sparseCapture=str(capture / 'capture.json'))
        (root / 'candidate-builds.json').write_text(json.dumps(results, indent=2) + '\n')
    print('COMPLETE ' + str(root / 'candidate-builds.json'), flush=True)


if __name__ == '__main__':
    main()
