"""Bake a color-space-only Hi3D control in B's exact native capture frame."""
import json
from pathlib import Path
import platform
import sys
import time

REPO=Path('/home/dev/Projects/box-impostor-lab/ComfyUI/custom_nodes/box_impostor_factory')
sys.path[:0]=[str(REPO),str(REPO.parents[1])]
from comfy.cli_args import args
args.cpu=True
if 'microsoft' in platform.release().lower():
    import torch
    torch.cuda._device_count_nvml=lambda:-1
import folder_paths
from box_nodes import store

WORK=Path('/mnt/f/_dev/box-impostor-lab/results/hi3d-depth-cycle-20261119/color-linear')
OUTPUT=Path('/mnt/f/_dev/box-impostor-lab/experiments/geometry-abc-20261119/output')
BASELINE='f6a5d91901f6745980cf73315bdacecc7e9db216b88784018cc7f8093d9dd404'
folder_paths.set_output_directory(str(OUTPUT))
baseline_dir=store.asset_dir(BASELINE)
baseline=store.validate(baseline_dir,BASELINE,check_bytes=False)
metadata=json.loads((WORK/'proxy.json').read_text())
expected_sha='411d924f8fd1170522a08f1998d24d28641a4445834c1a876d5f84b5d3a8a2bd'
assert store.file_hash(WORK/'proxy.glb')==metadata['glbSha256']==expected_sha
for key in ('transform','inverseTransform'):
    assert metadata[key]==baseline[key]
assert baseline['resolution']==1024 and baseline['maps']==[]
metadata['tightBounds']=baseline['tightBounds']
metadata['bakeBox']=baseline['bakeBox']
fixed=WORK/'canonical-fixed.json'
fixed.write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
proxy={'path':str(WORK/'proxy.glb'),'metadata':str(fixed),'sha256':expected_sha,
    'stageKey':store.key('hi3d-linear-color-native-control-v1',BASELINE,
                        expected_sha,store.file_hash(fixed)),
    'rotationDegreesXYZ':baseline['provenance'].get('rotationDegreesXYZ',[0,0,0]),
    'refinement':json.loads((WORK/'prepare-report.json').read_text())}
if baseline['provenance'].get('meshGeneration'):
    proxy['meshGeneration']=baseline['provenance']['meshGeneration']
start=time.monotonic()
print('NATIVE_BAKE_START',flush=True)
asset=store.baseline(proxy,1024,'webgl2',())
result={'sourceBaseline':BASELINE,'assetId':asset,'assetDirectory':str(store.asset_dir(asset)),
    'proxySha256':expected_sha,'canonicalMetadataSha256':store.file_hash(fixed),
    'resolution':1024,'backend':'webgl2','maps':[],
    'elapsedSeconds':time.monotonic()-start,'newInference':False,
    'colorTransferVersion':metadata['refinement']['colorTransferVersion']}
(WORK/'bake-result.json').write_text(json.dumps(result,indent=2)+'\n')
print('NATIVE_BAKE_FINISHED',json.dumps(result),flush=True)
