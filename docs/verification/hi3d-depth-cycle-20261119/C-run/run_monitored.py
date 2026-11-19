"""One foreground C run, with stage timing and process-tree RSS observation."""
import datetime
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import time

REPO = Path('/home/dev/Projects/box-impostor-lab/ComfyUI/custom_nodes/box_impostor_factory')
ROOT = Path('/mnt/f/_dev/box-impostor-lab/results/hi3d-depth-cycle-20261119/C')
WORK = ROOT/'actual'
PYTHON = '/home/dev/Projects/box-impostor-lab/.venv/bin/python'
SOURCE = 'ce53eb5429973a519e264d5752fd90348697797cd50f0903a4993076398ba1a2'
COMMAND = [PYTHON,str(REPO/'scripts/refine_full_depth.py'),
    '--asset',SOURCE,
    '--output-directory','/mnt/f/_dev/box-impostor-lab/experiments/geometry-abc-20261119/output',
    '--work-dir',str(WORK),'--infer','--inference-python',PYTHON,
    '--model','/home/dev/Projects/box-impostor-lab/ComfyUI/models/geometry_estimation/moge_2_vitl_normal_fp16.safetensors',
    '--inference-size','1024','--inference-level','8',
    '--max-displacement-height','.015','--iterations','8','--smoothness','.3',
    '--prior','.025','--cg-iterations','120','--backend','webgl2']
ENV = dict(os.environ)
ENV.update({
    'BI_NODE':'/home/dev/.nvm/versions/node/v22.23.3/bin/node',
    'BI_CHROMIUM':'/home/dev/.cache/ms-playwright/chromium_headless_shell-1254/chrome-headless-shell-linux64/chrome-headless-shell',
    'BI_SOFTWARE_GPU':'1','CUDA_VISIBLE_DEVICES':'0',
    'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1',
    'PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1',
    'TMPDIR':str(ROOT/'tmp'),'XDG_CACHE_HOME':str(ROOT/'cache/xdg'),
    'HF_HOME':str(ROOT/'cache/huggingface'),'TORCH_HOME':str(ROOT/'cache/torch'),
    'TORCHINDUCTOR_CACHE_DIR':str(ROOT/'cache/torchinductor'),
    'TORCH_EXTENSIONS_DIR':str(ROOT/'cache/torch_extensions'),
    'TRITON_CACHE_DIR':str(ROOT/'cache/triton')})
for name,value in ENV.items():
    if name in ('TMPDIR','XDG_CACHE_HOME','HF_HOME','TORCH_HOME',
                'TORCHINDUCTOR_CACHE_DIR','TORCH_EXTENSIONS_DIR','TRITON_CACHE_DIR'):
        Path(value).mkdir(parents=True,exist_ok=True)
WORK.mkdir(parents=True,exist_ok=True)
if (WORK/'result.json').exists():
    raise SystemExit('This foreground C run already has a result; refusing accidental rerun.')
start=time.monotonic()
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
record={'command':COMMAND,'sourceAsset':SOURCE,'startedUtc':now(),'stage':'initialization',
        'stages':{},'processes':{},'peakProcessTreeRSSKiB':0,
        'environment':{k:ENV[k] for k in ENV if k.startswith(('BI_','CUDA_'))},
        'rssMethod':'sum /proc VmRSS over descendants every 0.5s; sampled, not exact peak'}
stage='initialization'
stage_start=start
record['stages'][stage]={'startedSeconds':0,'peakProcessTreeRSSKiB':0}
items=queue.Queue()
process=subprocess.Popen(COMMAND,cwd=REPO,env=ENV,stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT,text=True,bufsize=1)
record['pid']=process.pid

def reader():
    for line in process.stdout:
        items.put((time.monotonic(),line))
    items.put((time.monotonic(),None))

threading.Thread(target=reader,daemon=True).start()

def sample_tree(pid):
    seen=set()
    todo=[pid]
    total=0
    while todo:
        current=todo.pop()
        if current in seen:
            continue
        seen.add(current)
        base=Path('/proc')/str(current)
        try:
            todo.extend(int(s) for s in (base/'task'/str(current)/'children').read_text().split())
            status=(base/'status').read_text().splitlines()
            rss=int(next(line for line in status if line.startswith('VmRSS:')).split()[1])
            name=next(line for line in status if line.startswith('Name:')).split(':',1)[1].strip()
            cmd=(base/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
        except (OSError,StopIteration,ValueError):
            continue
        total+=rss
        entry=record['processes'].setdefault(str(current),{'name':name,'command':cmd,'peakRSSKiB':0})
        entry['peakRSSKiB']=max(entry['peakRSSKiB'],rss)
    return total

def save():
    target=WORK/'run-observation.json'
    temp=target.with_suffix('.tmp')
    temp.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    temp.replace(target)

done=False
with (WORK/'process.log').open('w') as log:
    while not done:
        while True:
            try:
                tick,line=items.get_nowait()
            except queue.Empty:
                break
            if line is None:
                done=True
                break
            phase=None
            if line.startswith('MoGe inference starting'):
                phase='mogeInference'
            elif line.startswith('CPU full-depth shared-mesh fit starting'):
                phase='jointGeometryFit'
            elif line.startswith('Native Box rebake and frozen B RGB mapping starting'):
                phase='nativeRebakeAndFixedRGB'
            if phase and phase!=stage:
                record['stages'][stage]['seconds']=round(tick-stage_start,3)
                stage,stage_start=phase,tick
                record['stage']=stage
                record['stages'][stage]={'startedSeconds':round(tick-start,3),'peakProcessTreeRSSKiB':0}
            log.write(line)
            log.flush()
            print(now(),line.rstrip(),flush=True)
        rss=sample_tree(process.pid)
        record['peakProcessTreeRSSKiB']=max(record['peakProcessTreeRSSKiB'],rss)
        record['stages'][stage]['peakProcessTreeRSSKiB']=max(
            record['stages'][stage]['peakProcessTreeRSSKiB'],rss)
        record['elapsedSeconds']=round(time.monotonic()-start,3)
        save()
        if not done:
            time.sleep(.5)
exit_code=process.wait()
record['stages'][stage]['seconds']=round(time.monotonic()-stage_start,3)
record.update(exitCode=exit_code,finishedUtc=now(),elapsedSeconds=round(time.monotonic()-start,3))
save()
print('C RUN EXIT',exit_code,'elapsed',record['elapsedSeconds'],'RSS KiB',record['peakProcessTreeRSSKiB'],flush=True)
raise SystemExit(exit_code)
