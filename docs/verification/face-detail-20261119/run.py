"""Rebake actual stored H3 faces at 512 and 1024 pixels."""
import argparse, json, os, sys
from pathlib import Path
from unittest.mock import patch
import numpy as np

LAB = Path('/home/dev/Projects/box-impostor-lab')
REPO = LAB / 'ComfyUI/custom_nodes/box_impostor_factory'
ROOT = LAB / 'ComfyUI/output/box_impostor'
OUT = LAB / 'results/face-detail-recovery'
ORIGINAL = '4a8ccaf59bda137a8e34c8eb03833a4c9d91f93b4846fd68ece1e981b9b8c6c0'
sys.path.insert(0, str(REPO))
from box_nodes import faces, store
os.environ['BI_CHROMIUM'] = '/home/dev/.cache/ms-playwright/chromium_headless_shell-1254/chrome-headless-shell-linux64/chrome-headless-shell'
os.environ['BI_SOFTWARE_GPU'] = '1'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['bake', 'compose'])
    args = parser.parse_args()
    original_dir = ROOT / ORIGINAL
    original = json.loads((original_dir / 'manifest.json').read_text())
    report_file = OUT / 'experiments.json'
    report = json.loads(report_file.read_text()) if report_file.exists() else {'original': ORIGINAL, 'baselines': {}, 'aligned': {}}
    meta = {k: original[k] for k in ('tightBounds', 'bakeBox', 'transform', 'inverseTransform')}
    meta.update(glbSha256=original['proxySha256'], sourceSha256=original['provenance']['originalGlbSha256'])
    meta_path = OUT / 'canonical.json'
    meta_path.write_text(json.dumps(meta, indent=2))
    ref = {k: v for k, v in original['reference'].items() if k not in ('image', 'samples', 'camera')}
    ref['path'] = str(original_dir / 'reference')
    proxy = {'stageKey': store.key('face-recovery-original-proxy-v1', original['proxySha256']),
             'sha256': original['proxySha256'], 'path': str(original_dir / original['proxy']),
             'metadata': str(meta_path), 'reference': ref, 'camera': original['reference'].get('camera'),
             'croppedImage': str(original_dir / 'reference/cropped.png'),
             'provenance': original['provenance']['meshGeneration']}
    with patch.object(store, 'root', return_value=ROOT):
        if args.phase == 'bake':
            for resolution in (512, 1024):
                print(json.dumps({'phase': 'bake-start', 'resolution': resolution}), flush=True)
                asset = store.baseline(proxy, resolution, 'webgl2')
                report['baselines'][str(resolution)] = asset
                report_file.write_text(json.dumps(report, indent=2))
                print(json.dumps({'phase': 'bake-done', 'resolution': resolution, 'asset': asset}), flush=True)
        else:
            sources = {}
            for fid in store.FACES:
                edit = original['faces'][fid].get('edit', {})
                if edit.get('view'):
                    data = np.load((original_dir / edit['view']).with_suffix('.npy'), allow_pickle=False)
                    sources[fid] = {**faces.save_view(data), 'provenance': edit.get('provenance', {})}
            for resolution, baseline in report['baselines'].items():
                print(json.dumps({'phase': 'align-start', 'resolution': resolution}), flush=True)
                aligned = faces.align(sources, baseline, .85, .10, .15)
                asset = faces.compose(baseline, aligned)
                measured = {fid: json.loads((Path(entry['directory']) / 'report.json').read_text())
                            for fid, entry in aligned['faces'].items()}
                report['aligned'][resolution] = {'asset': asset, 'faces': measured}
                report_file.write_text(json.dumps(report, indent=2))
                print(json.dumps({'phase': 'align-done', 'resolution': resolution, 'asset': asset,
                                  'faces': {f: {'status': r['status'], 'iou': r.get('after', {}).get('iou')}
                                            for f, r in measured.items()}}), flush=True)
    print('REPORT', str(report_file), flush=True)

if __name__ == '__main__':
    main()
