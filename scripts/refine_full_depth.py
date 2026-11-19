#!/usr/bin/env python3
"""Run opt-in full-depth C refinement after B's six generated RGB views are ready."""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
COMFY = REPO.parents[1]
sys.path.insert(0,str(REPO))
sys.path.insert(1,str(COMFY))

from PIL import Image
import folder_paths

from box_nodes import store
from box_nodes.full_depth import SLUGS, refine_to_proxy, rebake_refinement, normal_only_proxy
from scripts.moge_face_depth_infer import image_crop


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset',required=True,help='Completed B asset ID with six accepted generated RGBs.')
    parser.add_argument('--output-directory',type=Path,required=True,help='Comfy output parent containing box_impostor/B_ID.')
    parser.add_argument('--work-dir',type=Path,required=True)
    parser.add_argument('--prediction-dir',type=Path)
    parser.add_argument('--infer',action='store_true',help='Explicitly run MoGe on the GPU first; omit to use existing predictions.')
    parser.add_argument('--model',type=Path)
    parser.add_argument('--inference-python',default=sys.executable)
    parser.add_argument('--inference-size',type=int,default=1024)
    parser.add_argument('--inference-level',type=int,default=8)
    parser.add_argument('--normal-only',action='store_true',help='B normal-only control: preserve all positions/materials; no depth inference or fitting.')
    parser.add_argument('--geometry-only',action='store_true',help='Fit/write geometry without invoking the native baker.')
    parser.add_argument('--backend',choices=['webgl2','webgpu','auto'],default='webgl2')
    parser.add_argument('--max-displacement-height',type=float,default=.015)
    parser.add_argument('--iterations',type=int,default=8)
    parser.add_argument('--smoothness',type=float,default=.3)
    parser.add_argument('--prior',type=float,default=.025)
    parser.add_argument('--cg-iterations',type=int,default=120)
    args=parser.parse_args()
    if args.normal_only and args.infer:
        parser.error('--normal-only cannot run neural inference.')
    if min(args.max_displacement_height,args.smoothness,args.prior)<0:
        parser.error('Displacement bound and quadratic weights must be nonnegative.')
    if args.iterations<0 or args.cg_iterations<1:
        parser.error('Solver iteration counts must be nonnegative, with at least one CG step.')
    # Match the native MoGe runner's WSL/NVML guard before Comfy interruption checks import CUDA state.
    if 'microsoft' in platform.release().lower():
        import torch
        torch.cuda._device_count_nvml=lambda:-1
    # The parent only preprocesses/rebakes on CPU; the independent MoGe child
    # receives its CUDA setting from the environment. Avoid Comfy's interruption
    # check initializing CUDA in a software-only native bake.
    from comfy.cli_args import args as comfy_args
    comfy_args.cpu=True
    folder_paths.set_output_directory(str(args.output_directory.resolve()))
    args.work_dir.mkdir(parents=True,exist_ok=True)
    predictions=args.prediction_dir or args.work_dir/'predictions'
    source=store.asset_dir(args.asset)
    manifest=store.validate(source,args.asset,check_bytes=False)
    if args.infer:
        if args.model is None or not args.model.is_file():
            parser.error('--infer requires an existing local --model checkpoint.')
        capture={}
        for face,slug in SLUGS.items():
            edit=manifest['faces'][face].get('edit',{})
            if edit.get('status')!='accepted':
                parser.error('B is not a completed six-view RGB asset: '+face)
            path=store.contained(source,edit['view'])
            with Image.open(path) as im:
                rgb,crop=image_crop(im,size=args.inference_size)
                capture[face]={'sourceSize':list(im.size),'crop':list(crop),
                               'effectiveInferenceSize':[rgb.shape[1],rgb.shape[0]],
                               'sourceSha256':store.file_hash(path)}
        command=[args.inference_python,str(REPO/'scripts/moge_face_depth_infer.py'),
                 '--asset',str(source),'--model',str(args.model),'--output',str(predictions),
                 '--face','all','--size',str(args.inference_size),'--level',str(args.inference_level)]
        env=dict(os.environ)
        env['PYTHONPATH']=str(COMFY)+(os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
        invocation={'command':command,'inputImages':capture,'outputSize':'source canvas size',
                    'note':'Crop/rescale is recorded; intrinsics are not calibrated to the orthographic guide.'}
        (args.work_dir/'inference-invocation.json').write_text(json.dumps(invocation,indent=2)+'\n')
        log=args.work_dir/'inference-process.log'
        print('MoGe inference starting; log='+str(log),flush=True)
        with log.open('wb') as stream:
            subprocess.run(command,cwd=REPO,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    if args.normal_only:
        print('CPU same-position normal-only control starting',flush=True)
        proxy,report=normal_only_proxy(args.asset,args.work_dir)
    else:
        print('CPU full-depth shared-mesh fit starting',flush=True)
        proxy,report=refine_to_proxy(args.asset,predictions,args.work_dir,
                                   max_displacement_height=args.max_displacement_height,
                                   iterations=args.iterations,smoothness=args.smoothness,
                                   prior=args.prior,cg_iterations=args.cg_iterations)
    result={'sourceAsset':args.asset,'proxy':proxy,'refinementReport':str(args.work_dir/'refinement.json')}
    if not args.geometry_only:
        print('Native Box rebake and frozen B RGB mapping starting',flush=True)
        asset,evaluation=rebake_refinement(args.asset,proxy,args.work_dir,args.backend)
        result.update(assetId=asset,assetDirectory=str(store.asset_dir(asset)),
                      rebakeEvaluation=str(args.work_dir/'rebake-evaluation.json'))
    (args.work_dir/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='proxy'},indent=2),flush=True)


if __name__=='__main__':
    main()
