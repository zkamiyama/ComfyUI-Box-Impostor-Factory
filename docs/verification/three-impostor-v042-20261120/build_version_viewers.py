"""Build untouched official 0.4.2 HTMLs and a fixed-input capture config on 270-PC."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


REPO = Path('/home/dev/Projects/box-impostor-lab/ComfyUI/custom_nodes/box_impostor_factory')
QA = REPO / 'docs/verification/three-impostor-v042-20261120'
WORK = Path('/mnt/f/_dev/box-impostor-lab/results/three-impostor-v042-20261120')
OLD = Path('/mnt/f/_dev/box-impostor-lab/results/qwen-image-20261119')
PYTHON = '/home/dev/Projects/box-impostor-lab/.venv/bin/python'
VIEWER = '/home/dev/Projects/3D-Inspector-skills/skills/3d-inspector/view.py'
REFERENCE = '/mnt/f/_dev/box-impostor-lab/experiments/geometry-abc-20261119/output/box_impostor/d168a94ac6eb0105fc2315a0bdd392846a9edc3ad9e3a19a87ab8eb3c7ef0cf9/manifest.json'
SOURCES = [
    ('F', 'F-current', '4e8090f307b4958de8a94877ad391b3b0e355fa208d7ded8340a229e773b1920',
     '4878c363c4932e76ec9c5035abfa46d2b928ca4c16ab8d073a124912fa1273ce',
     '1a9b97db6662cf726276bb9849d0750281f003c4064d23b25d1c9a00ee6e11e0'),
    ('Q2', 'Q2-qwen-anyangle', '87b95907db2b663184783f1eb7d9c5b594542ce15aa1d1d777a85b1900ab213e',
     '2fa9fb690dc38289dab48fc20df00096a7bb8e35325ed9ab15cf9db741805c56',
     'b1ebcba77a31898b76e108adf796bc9e7c16f07cc0b26a2af8dbf95ff218c80a'),
]


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / 'tmp').mkdir(exist_ok=True)
    environment = os.environ.copy()
    environment.update({key: str(WORK / 'tmp') for key in ['TMPDIR', 'TMP', 'TEMP']})
    assert json.loads((REPO / 'node_modules/three-impostor/package.json').read_text())['version'] == '0.4.2'
    assert json.loads((REPO / 'node_modules/three/package.json').read_text())['version'] == '0.186.0'
    cases, builds = [], []
    for family, stem, asset, zip_sha, old_html_sha in SOURCES:
        source, old_html = OLD / f'{stem}.box.zip', OLD / f'{stem}.inspector.html'
        assert sha(source) == zip_sha and sha(old_html) == old_html_sha, family
        new_html = WORK / f'{family}-v042.inspector.html'
        command = [PYTHON, VIEWER, str(source), '--box-lighting', 'unlit', '--box-color-sampling', 'view',
                   '--title', f'{family} / three-impostor 0.4.2', '-o', str(new_html)]
        start = time.monotonic()
        result = subprocess.run(command, cwd=REPO, env=environment, text=True, capture_output=True, timeout=240)
        log = WORK / f'{family}-viewer-build.log'
        log.write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(f'{family} official view.py failed ({result.returncode}): {log}')
        assert new_html.is_file()
        assert sha(source) == zip_sha and sha(old_html) == old_html_sha, 'Legacy asset/viewer changed'
        builds.append({'family': family, 'command': command, 'elapsedSeconds': time.monotonic() - start,
                       'sourceZIP': str(source), 'sourceZIPSHA256': zip_sha, 'sourceAssetId': asset,
                       'log': str(log), 'logSHA256': sha(log), 'outputHTML': str(new_html),
                       'outputHTMLSHA256': sha(new_html), 'viewPySHA256': sha(VIEWER),
                       'officialHTMLPostprocessed': False})
        for version, html, html_sha in [('0.4.1', old_html, old_html_sha), ('0.4.2', new_html, sha(new_html))]:
            cases.append({'name': f'{family}-v{version.replace(".", "")}', 'family': family,
                          'packageVersion': version, 'html': str(html), 'htmlSHA256': html_sha, 'assetId': asset})
        write(WORK / 'viewer-builds.json', {'builds': builds})
        print('BUILT', family, str(new_html), sha(new_html), flush=True)
    config = {'output': str(WORK), 'captureScript': str(QA / 'capture_inspector_version_yaw.mjs'),
              'captureScriptSHA256': sha(QA / 'capture_inspector_version_yaw.mjs'),
              'referenceManifest': REFERENCE, 'cwd': str(REPO), 'angles': [0, 30, 45, 60],
              'display': {'lighting': 'unlit', 'sampling': 'native', 'colorSampling': 'view'}, 'cases': cases}
    write(WORK / 'capture-config.json', config)
    print('CONFIG', str(WORK / 'capture-config.json'), flush=True)


if __name__ == '__main__':
    main()
