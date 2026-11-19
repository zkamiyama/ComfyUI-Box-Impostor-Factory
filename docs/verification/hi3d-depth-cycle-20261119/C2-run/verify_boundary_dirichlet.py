"""Audit actual published C2 positions, fixed RGB mapping and native depth residuals."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import numpy as np

REPO=Path('/home/dev/Projects/box-impostor-lab/ComfyUI/custom_nodes/box_impostor_factory')
sys.path.insert(0,str(REPO))
from box_nodes import depth_mesh
from box_nodes.full_depth import boundary_axis_pins
ROOT=Path('/mnt/f/_dev/box-impostor-lab/results/hi3d-depth-cycle-20261119/C')
WORK=ROOT/'boundary-dirichlet'
ASSETS=Path('/mnt/f/_dev/box-impostor-lab/experiments/geometry-abc-20261119/output/box_impostor')
EVIDENCE=REPO/'docs/verification/hi3d-depth-cycle-20261119/C2-run'
result=json.loads((WORK/'result.json').read_text())
report=json.loads((WORK/'refinement.json').read_text())
evaluation=json.loads((WORK/'rebake-evaluation.json').read_text())
observation=json.loads((WORK/'run-observation.json').read_text())
before=ASSETS/result['sourceAsset'];after=ASSETS/result['assetId']
bm=json.loads((before/'manifest.json').read_text());am=json.loads((after/'manifest.json').read_text())
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while chunk:=f.read(1024*1024):h.update(chunk)
    return h.hexdigest()
def mesh(path):
    doc,binary=depth_mesh._decode(path.read_bytes())
    parts=list(depth_mesh._primitives(doc,binary))
    assert len(parts)==1
    _,p,tri,t=parts[0]
    return p@t[:3,:3].T+t[:3,3],tri,t
p,tri,t=mesh(before/bm['proxy'])
q,qtri,qt=mesh(after/am['proxy'])
assert p.shape==q.shape and np.array_equal(tri,qtri) and np.array_equal(t,qt)
height=float(np.ptp(p[:,1]));limit=report['solver']['parameters']['maxDisplacementHeight']*height
pins,margin=boundary_axis_pins(p,limit)
delta=q-p
pinned=abs(delta[pins])
plane_counts={}
for axis,name in enumerate('XYZ'):
    for edge,fn in [('min',np.min),('max',np.max)]:
        value=float(fn(p[:,axis]))
        plane_counts[name+edge]={'coordinate':value,
            'B':int(np.count_nonzero(p[:,axis]==value)),
            'C2':int(np.count_nonzero(q[:,axis]==value))}
v2,_,_=mesh(ROOT/'fixed-support/refined.glb')
for axis,name in enumerate('XYZ'):
    for edge,fn in [('min',np.min),('max',np.max)]:
        value=plane_counts[name+edge]['coordinate']
        plane_counts[name+edge]['C_v2']=int(np.count_nonzero(v2[:,axis]==value))
lengths=np.linalg.norm(delta,axis=1)/height
proof={'sourceAsset':result['sourceAsset'],'assetId':result['assetId'],'method':report['method'],
    'sameVertexCountAndIndices':True,'sameMeshTransform':True,
    'fixedCoordinateCountsXYZ':pins.sum(0).astype(int).tolist(),
    'pinnedMovedCoordinateCount':int(np.count_nonzero(pinned)),
    'pinnedMaxAbsWorldDelta':float(pinned.max()) if len(pinned) else 0.,
    'pinnedCoordinatesExactlyPreserved':bool(np.all(pinned==0)),
    'nearTopPlaneCounts':{'toleranceWorld':5e-8,
        'B':int(np.count_nonzero(abs(p[:,1]-p[:,1].max())<=5e-8)),
        'C_v2':int(np.count_nonzero(abs(v2[:,1]-p[:,1].max())<=5e-8)),
        'C2':int(np.count_nonzero(abs(q[:,1]-p[:,1].max())<=5e-8))},
    'boundaryFloat32MarginWorld':margin.tolist(),'boundsPlaneCounts':plane_counts,
    'actualDisplacementHeight':{'p50':float(np.quantile(lengths,.5)),
        'p95':float(np.quantile(lengths,.95)),'max':float(lengths.max())},
    'fixedFrame':evaluation['fixedFrame'],'silhouetteQaPassed':evaluation['silhouetteQaPassed'],
    'allSixSourceAndAlignedRGBEqual':bool(all(f['sameSourceRGB'] and f['sameSourceSamples'] and f['sameAlignedRGB']
                                             and f['samplingMappingUnchanged'] for f in evaluation['faces'].values())),
    'maps':{},'faces':{},'stages':observation['stages'],
    'elapsedSeconds':observation['elapsedSeconds'],
    'sampledPeakProcessTreeRSSKiB':observation['peakProcessTreeRSSKiB'],
    'reusedPredictionDirectory':str(ROOT/'actual/predictions'),'newGpuInference':False,
    'sourceCodeSha256':sha(REPO/'box_nodes/full_depth.py'),
    'sourceProxySha256':sha(before/bm['proxy']),
    'publishedProxySha256':sha(after/am['proxy']),
    'workProxySha256':sha(WORK/'refined.glb')}
for name,spec in am['textures'].items():
    actual=sha(after/spec['path']);assert actual==spec['sha256']
    proof['maps'][name]={'beforeSha256':bm['textures'][name]['sha256'],'afterSha256':actual,
                        'changed':bm['textures'][name]['sha256']!=actual}
for fid,f in evaluation['faces'].items():
    proof['faces'][fid]={'nativeRmsHeightBefore':f['before']['rmsHeight'],
        'nativeRmsHeightAfter':f['after']['rmsHeight'],
        'lowFrequencyRmsHeightBefore':f['before']['lowFrequencyRmsHeight'],
        'lowFrequencyRmsHeightAfter':f['after']['lowFrequencyRmsHeight'],
        'sharedPixels':f['sharedPixels'],'missingAfterPixels':f['missingAfterPixels'],
        'silhouetteBefore':f['silhouetteQA']['sourceSilhouette'],
        'silhouetteAfter':f['silhouetteQA']['revisedSilhouette'],
        'alignmentStatus':f['alignmentStatus'],
        'newGeometryUncoveredPixels':f['silhouetteQA']['newGeometryUncoveredPixels']}
proof['controlChecksPassed']=bool(proof['pinnedCoordinatesExactlyPreserved'] and
    proof['fixedFrame']['passed'] and proof['allSixSourceAndAlignedRGBEqual'] and
    proof['silhouetteQaPassed'] and proof['nearTopPlaneCounts']['B']==proof['nearTopPlaneCounts']['C2'] and all(v['B']==v['C2'] for v in plane_counts.values()) and
    lengths.max()<=limit/height+1e-6)
(WORK/'verification-summary.json').write_text(json.dumps(proof,indent=2,allow_nan=False)+'\n')
EVIDENCE.mkdir(parents=True,exist_ok=True)
for name in ('result.json','refinement.json','rebake-evaluation.json','fixed-rgb-qa.json',
             'process.log','run-observation.json','verification-summary.json'):
    shutil.copyfile(WORK/name,EVIDENCE/name)
for name in ('run_boundary_dirichlet.py','verify_boundary_dirichlet.py'):
    shutil.copyfile(ROOT/name,EVIDENCE/name)
print(json.dumps(proof,indent=2))
if not proof['controlChecksPassed']:raise SystemExit(1)
