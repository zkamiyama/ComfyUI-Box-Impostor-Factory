"""Diagnose the rejected first shared-depth step without relaxing any guard."""
import json,time
from pathlib import Path
import sys
import numpy as np

REPO=Path('/home/dev/Projects/box-impostor-lab/ComfyUI/custom_nodes/box_impostor_factory')
sys.path[:0]=[str(REPO),str(REPO.parents[1])]
from comfy.cli_args import args
args.cpu=True
import folder_paths
folder_paths.set_output_directory('/mnt/f/_dev/box-impostor-lab/experiments/geometry-abc-20261119/output')
from box_nodes import full_depth as fd,store

ROOT=Path('/mnt/f/_dev/box-impostor-lab/results/hi3d-depth-cycle-20261119/C')
WORK=ROOT/'actual'
SOURCE='ce53eb5429973a519e264d5752fd90348697797cd50f0903a4993076398ba1a2'
directory=store.asset_dir(SOURCE)
manifest=store.validate(directory,SOURCE,check_bytes=False)
views,registration,_=fd.prepare_views(directory,manifest,WORK/'predictions',.015)
points,triangles=fd.mesh_arrays(store.contained(directory,manifest['proxy']).read_bytes())
capture={}
original_constraints,original_cg=fd.constraints,fd.cg

def capture_constraints(*args,**kwargs):
    rows,agreement=original_constraints(*args,**kwargs)
    capture['rows']=rows
    capture['agreement']=agreement
    return rows,agreement

def capture_cg(operator,rhs,**kwargs):
    started=time.monotonic()
    result,status=original_cg(operator,rhs,**kwargs)
    residual=operator@result-rhs
    capture['rawStep']=result.copy()
    capture['cg']={'status':int(status),'seconds':time.monotonic()-started,
        'rhsL2':float(np.linalg.norm(rhs)),'residualL2':float(np.linalg.norm(residual)),
        'relativeResidualL2':float(np.linalg.norm(residual)/max(np.linalg.norm(rhs),1e-30)),
        'negativeGradientDotStep':float(np.dot(rhs,result))}
    return result,status

fd.constraints,fd.cg=capture_constraints,capture_cg
_,solver=fd.joint_fit(points,triangles,views,manifest['bakeBox'],
    max_displacement_height=.015,iterations=1,smoothness=.3,prior=.025,cg_iterations=120)
low=np.array(manifest['bakeBox']['min']);span=np.subtract(manifest['bakeBox']['max'],low)
base=((points-low)/span).astype('f4')
height=np.ptp(points[:,1]);max_world=.015*height
step=capture['rawStep'].reshape(-1,3).copy()
length=np.linalg.norm(step*span,axis=1)
step*=np.minimum(1,max_world/3/np.maximum(length,1e-12))[:,None]
pins=np.isclose(base,base.min(0),atol=1e-7)|np.isclose(base,base.max(0),atol=1e-7)
step[pins]=0
rows=capture['rows']
gradient_dot_step=0.
for residual,jac,weight in rows:
    gradient_dot_step+=2*float(np.sum(weight*residual*np.sum(jac*step,axis=1),dtype='f8'))
laplacian=fd.displacement_laplacian(triangles,len(base))
diagonal_prior=np.full_like(base,.025)+pins.astype('f4')*10000

def evaluate(candidate):
    d=candidate-base
    prior=float(np.sum(diagonal_prior*d*d,dtype='f8'))
    smooth=.3*float(np.sum(d*(laplacian@d),dtype='f8'))
    output={'prior':prior,'smoothness':smooth,'data':0.,'validData':0.,'invalidPenalty':0.,'faces':{}}
    for view,(oldresidual,jac,weight) in zip(views,rows):
        residual,_,valid,_=view.sample(candidate)
        positive=weight>0
        invalid=positive & ~valid
        good=positive & valid
        validcost=float(np.sum(weight[good]*residual[good]**2,dtype='f8'))
        invalidcost=float(np.sum(weight[invalid],dtype='f8'))
        rawcost=float(np.sum(weight*residual**2,dtype='f8'))
        output['faces'][view.face]={'supportedRows':int(positive.sum()),
            'invalidRows':int(invalid.sum()),'invalidPenalty':invalidcost,
            'validResidualCost':validcost,'sampledResidualCost':rawcost}
        output['data']+=rawcost
        output['validData']+=validcost
        output['invalidPenalty']+=invalidcost
    output['energy']=prior+smooth+output['validData']+output['invalidPenalty']
    return output

before=evaluate(base)
trials=[]
for backtrack in range(17):
    scale=.5**backtrack
    candidate=base+step*scale
    delta=(candidate-base)*span
    length=np.linalg.norm(delta,axis=1)
    candidate=base+(candidate-base)*np.minimum(1,max_world/np.maximum(length,1e-12))[:,None]
    candidate=np.clip(candidate,base.min(0),base.max(0))
    candidate[pins]=base[pins]
    result=evaluate(candidate)
    result.update(scale=scale,backtrack=backtrack,
        energyChange=result['energy']-before['energy'],
        sampledDataChange=result['data']-before['data'],
        expectedFirstOrderChange=gradient_dot_step*scale)
    trials.append(result)
report={'sourceAsset':SOURCE,'predictionDirectory':str(WORK/'predictions'),
    'solver':solver,'agreement':capture['agreement'],'cg':capture['cg'],
    'gradientDotClippedPinnedStep':gradient_dot_step,'before':before,'backtracking':trials}
(ROOT/'solver-first-step-diagnostic.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({'cg':report['cg'],'agreement':report['agreement'],
    'gradientDotStep':gradient_dot_step,'beforeEnergy':before['energy'],
    'trials':[{k:t[k] for k in ('backtrack','energyChange','invalidPenalty','sampledDataChange','expectedFirstOrderChange')} for t in trials]},indent=2))
