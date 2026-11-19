"""Collect small H evidence only; model/GLB/atlas/HTML files remain on F."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

WORK = Path(__file__).resolve().parent
REPO = Path('/home/dev/Projects/box-impostor-lab/ComfyUI/custom_nodes/box_impostor_factory')
DEST = REPO/'docs/verification/hi3d-depth-cycle-20261119/H-run'
DEST.mkdir(parents=True,exist_ok=True)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


files = ['plan.json','process.json','resources.txt','inference-invocation.json',
         'inference.json','inference.log','postprocess.json','canonical.json','proxy.json',
         'baseline.json','result.json','process.log','runtime-observations.json','b-replay/proof.json',
         'camera-rebase/camera-rebase-proof.json','collect_evidence.py']
files += [str(p.relative_to(WORK)) for p in (WORK/'camera-rebase').glob('*/report.json')]
inventory = {}
for relative in files:
    source = WORK/relative
    if not source.is_file():
        continue
    if source.stat().st_size > 2_000_000:
        raise ValueError('Evidence unexpectedly large: '+str(source))
    target = DEST/relative
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(source,target)
    inventory[relative] = {'sha256':digest(target),'bytes':target.stat().st_size}

for relative in ['scripts/run_geometry_h_reference.py','box_nodes/camera_rebase.py',
                 'tests/test_camera_rebase.py','tests/test_geometry_h_reference.py',
                 'docs/verification/hi3d-depth-cycle-20261119/H-preparation/hi3d_postprocess_b.py']:
    source = REPO/relative
    target = DEST/'code'/source.name
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(source,target)
    inventory[str(target.relative_to(DEST))] = {
        'source':str(source),'sha256':digest(target),'bytes':target.stat().st_size}

read = lambda name: json.loads((WORK/name).read_text()) if (WORK/name).is_file() else None
plan, result, rebase, replay, inference = [read(name) for name in [
    'plan.json','result.json','camera-rebase/camera-rebase-proof.json',
    'b-replay/proof.json','inference.json']]
summary = {
    'method':plan['method'],'sourceAppearance':plan['sourceAppearance'],
    'asset':result['asset'] if result else None,
    'resultExists':result is not None,
    'BReplayPassed':replay['passed'] if replay else None,
    'BReplayByteIdentical':replay['byteIdentical'] if replay else None,
    'referenceEqualsBFront':plan['referenceEqualsBFront'],
    'referencePngSha256':plan['referencePngSha256'],
    'coarseSha256':plan['coarseSha256'],
    'inference':inference,
    'cameraRebasePassed':rebase['passed'] if rebase else None,
    'sourcePngAndNpyChecks':result['sourcePngAndNpyChecks'] if result else None,
    'faces':{fid:{key:row[key] for key in [
        'status','after','thresholds','alignedResampled','sameAlignedBytes',
        'cameraTransformLimits','newGeometryUncoveredPixels']}
        for fid,row in rebase['faces'].items()} if rebase else None,
    'sourceTightBounds':rebase['sourceTightBounds'] if rebase else None,
    'targetTightBounds':rebase['targetTightBounds'] if rebase else None,
    'sourceBakeBox':rebase['sourceBakeBox'] if rebase else None,
    'targetBakeBox':rebase['targetBakeBox'] if rebase else None,
    'actualCodeCommit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
    'fileInventory':inventory,
    'timingComparisonPolicy':'Do not compare B/H runtime as model performance: H explicitly set OMP_NUM_THREADS=1 and OPENBLAS_NUM_THREADS=1 while B did not.',
    'largeArtifactsPolicy':'GLB/model/atlas/HTML remain on F; this folder contains small proof and code only',
}
(DEST/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
print(json.dumps({'evidence':str(DEST),'summarySha256':digest(DEST/'summary.json'),
                  'files':len(inventory),'bytes':sum(v['bytes'] for v in inventory.values()),
                  'asset':summary['asset'],'cameraRebasePassed':summary['cameraRebasePassed']},indent=2))
