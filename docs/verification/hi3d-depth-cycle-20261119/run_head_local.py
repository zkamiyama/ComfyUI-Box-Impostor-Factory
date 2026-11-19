"""Execute the isolated, prepared head experiment with H's verified GPU runtime."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan', type=Path)
    args = parser.parse_args()
    plan = read(args.plan)
    work = Path(plan['outputDirectory'])
    assert work.resolve().is_relative_to(Path('/mnt/f').resolve())
    assert args.plan.resolve().parent == work.resolve()
    assert plan['fullBodyMergeAllowed'] is False
    assert plan['inference']['resolution'] == 1024 and plan['inference']['steps'] == 25
    assert plan['inference']['seed'] == 0 and plan['inference']['offload']
    h = read(plan['parentHPlan'])
    assert digest(plan['parentHPlan']) == plan['parentHPlanSha256']
    assert digest(h['inferScript']) == plan['inference']['scriptSha256']
    for item in [*plan['artifacts'].values(), plan['sourceCoarse'], plan['sourceReference'],
                 *plan['upstream']['sourceFiles'].values()]:
        assert digest(item['path']) == item['sha256'], item['path']
    status_path = work / 'head-execution.json'
    if status_path.exists() or Path(plan['inference']['rawOutput']).exists():
        raise FileExistsError('Preserve the existing experiment; no automatic rerun.')
    with socket.socket() as sock:
        sock.settimeout(1)
        if sock.connect_ex(('127.0.0.1', 8201)) == 0:
            raise RuntimeError('H3 server is running; do not start a concurrent Hi3D model.')
    previous = read(Path(h['work']) / 'inference-invocation.json')
    environment = dict(previous['environment'])
    for key, folder in [('TMPDIR', 'tmp'), ('XDG_CACHE_HOME', 'cache'), ('TORCH_HOME', 'torch-cache'),
                        ('TORCH_EXTENSIONS_DIR', 'torch-extensions'), ('TORCHINDUCTOR_CACHE_DIR', 'inductor-cache'),
                        ('TRITON_CACHE_DIR', 'triton-cache'), ('CUDA_CACHE_PATH', 'cuda-cache')]:
        environment[key] = str(work / folder)
        (work / folder).mkdir(exist_ok=True)
    if os.environ.get('CUDA_VISIBLE_DEVICES') == '':
        raise RuntimeError('CUDA is disabled in the calling environment.')
    command = plan['inference']['argv']
    record = {'status': 'running', 'command': command, 'environment': environment,
              'cwd': previous['cwd'], 'parentHInvocation': str(Path(h['work']) / 'inference-invocation.json'),
              'planSha256': digest(args.plan), 'runnerSha256': digest(__file__),
              'headPreparationScriptSha256': digest(Path(h['inferScript']).with_name('refine_head_region.py')),
              'startedUTC': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'gpuInference': True, 'fullBodyMerge': False}
    save(status_path, record)
    print(json.dumps({'status': 'running', 'directory': str(work), 'command': command}), flush=True)
    started = time.monotonic()
    with (work / 'head-inference.log').open('w') as log:
        result = subprocess.run(command, cwd=previous['cwd'], env={**os.environ, **environment},
                                stdout=log, stderr=subprocess.STDOUT)
    record.update(status='completed' if result.returncode == 0 else 'failed', exitCode=result.returncode,
                  elapsedSeconds=round(time.monotonic() - started, 3),
                  completedUTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    if result.returncode == 0:
        report = read(plan['inference']['report'])
        assert report['status'] == 'success' and report['resolution'] == 1024 and report['steps'] == 25
        record['rawMeshSha256'] = digest(plan['inference']['rawOutput'])
        record['inferenceReport'] = report
    save(status_path, record)
    print(json.dumps(record), flush=True)
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
