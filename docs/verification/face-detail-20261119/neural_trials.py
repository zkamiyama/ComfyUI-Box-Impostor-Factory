"""Compare registered MoGe local relief with identical accepted appearance."""
import json, os, sys
from pathlib import Path
from unittest.mock import patch
LAB=Path('/home/dev/Projects/box-impostor-lab')
REPO=LAB/'ComfyUI/custom_nodes/box_impostor_factory'
ROOT=LAB/'ComfyUI/output/box_impostor'
sys.path.insert(0,str(REPO))
from box_nodes import store
from box_nodes.neural_depth import rebake_with_neural_depth
os.environ['BI_CHROMIUM']='/home/dev/.cache/ms-playwright/chromium_headless_shell-1254/chrome-headless-shell-linux64/chrome-headless-shell'
os.environ['BI_SOFTWARE_GPU']='1'
asset='91eccb2e8a3c9958caecc89fff743fec3e26dc34b1808aadf5f182624180b006'
predictions=LAB/'results/ai-depth-moge-grey'
metadata=json.loads((predictions/'inference.json').read_text())
for slug,info in metadata['faces'].items():
    path=ROOT/asset/'views'/slug/'source.png'
    if store.file_hash(path)!=info['source_sha256']:
        raise ValueError('AI prediction source mismatch: '+slug)
results={}
with patch.object(store,'root',return_value=ROOT):
    for strength,limit in ((.04,.004),(.08,.008)):
        print('START',strength,flush=True)
        result,report=rebake_with_neural_depth(asset,predictions,backend='webgl2',strength=strength,sigma=20.,max_delta=limit)
        results[str(strength)]={'asset':result,'metrics':report}
        (LAB/'results/face-detail-recovery/neural-trials.json').write_text(json.dumps(results,indent=2))
        print(json.dumps({'strength':strength,'asset':result,'mesh':report['mesh']}),flush=True)
