"""Infer true neural per-face depth from saved generated RGBA views using native MoGe-2.

MoGe-2 predicts a fresh depth map from the actual generated RGB, rather than
smoothing the Pixal3D bake. Outputs carry a source image hash and crop transform.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
from PIL import Image


def image_crop(image, size=768):
    a = np.asarray(image.convert('RGBA'), dtype=np.float32) / 255.
    alpha = a[..., 3]
    yy, xx = np.nonzero(alpha > .2)
    if not len(xx):
        raise ValueError('Empty generated foreground')
    pad = max(24, round(.04 * max(xx.max()-xx.min(), yy.max()-yy.min())))
    x0=max(0,int(xx.min())-pad)
    x1=min(a.shape[1],int(xx.max())+pad+1)
    y0=max(0,int(yy.min())-pad)
    y1=min(a.shape[0],int(yy.max())+pad+1)
    rgb=a[y0:y1,x0:x1,:3] * a[y0:y1,x0:x1,3:4] + .5*(1-a[y0:y1,x0:x1,3:4])
    w,h=x1-x0,y1-y0
    new_w=round(w*min(1,size/max(w,h)))
    new_h=round(h*min(1,size/max(w,h)))
    rgb=np.asarray(Image.fromarray(np.uint8(np.clip(rgb,0,1)*255)).resize((new_w,new_h),Image.Resampling.LANCZOS),dtype=np.float32)/255.
    return rgb,(x0,y0,x1,y1)


def infer(image,model,size=768,level=8):
    import torch
    rgb,crop=image_crop(image,size=size)
    inp=torch.from_numpy(rgb.copy()).permute(2,0,1)[None]
    with torch.inference_mode():
        result=model.infer(inp,resolution_level=level,apply_mask=False,force_projection=False)
    predicted=result['depth'][0].detach().to('cpu',dtype=torch.float32).numpy()
    assert np.isfinite(predicted).all() and np.nanstd(predicted)>1.e-6
    x0,y0,x1,y1=crop
    prediction=np.asarray(Image.fromarray(predicted,mode='F').resize((x1-x0,y1-y0),Image.Resampling.BICUBIC))
    full=np.full((image.height,image.width),np.nan,np.float32)
    full[y0:y1,x0:x1]=prediction
    return full,crop


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--asset',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--face',choices=['px','nx','py','ny','pz','nz','all'],default='all')
    p.add_argument('--size',type=int,default=768)
    p.add_argument('--level',type=int,default=8)
    args=p.parse_args()
    if not 256<=args.size<=1536 or not 0<=args.level<=9:raise ValueError('size or level outside supported range')
    import torch
    if 'microsoft' in platform.release().lower():
        torch.cuda._device_count_nvml=lambda:-1
    from safetensors.torch import load_file
    from comfy.ldm.moge.model import MoGeModel
    start=time.monotonic()
    model=MoGeModel(load_file(str(args.model),device='cpu'))
    print('MODEL',model.version,'GPU',torch.cuda.get_device_name(0),flush=True)
    args.output.mkdir(parents=True,exist_ok=True)
    names=['px','nx','py','ny','pz','nz'] if args.face=='all' else [args.face]
    records={}
    for name in names:
        source=args.asset/'views'/name/'source.png'
        if args.face=='all' and not source.is_file():
            continue
        if not source.is_file():
            raise FileNotFoundError('Missing generated H3/Qwen image: '+str(source))
        im=Image.open(source)
        t=time.monotonic()
        prediction,crop=infer(im,model,size=args.size,level=args.level)
        np.save(args.output/(name+'.npy'),prediction,allow_pickle=False)
        values=prediction[np.isfinite(prediction)]
        rng=np.quantile(values,[.01,.99])
        viz=np.clip((prediction-rng[0])/max(1e-6,rng[1]-rng[0]),0,1)
        viz[~np.isfinite(viz)]=0
        Image.fromarray(np.uint8(viz*255)).save(args.output/(name+'.png'))
        records[name]={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                       'crop':list(crop),'shape':list(prediction.shape),
                       'valid':int(np.isfinite(prediction).sum()),
                       'quantiles':np.quantile(values,[.01,.5,.99]).tolist(),
                       'seconds':round(time.monotonic()-t,2)}
        print('FACE',name,json.dumps(records[name]),flush=True)
    if not records:
        raise ValueError('No stored H3/Qwen-generated faces to estimate depth')
    (args.output/'inference.json').write_text(json.dumps({'model':'moge_2_vitl_normal_fp16','model_sha256':hashlib.sha256(args.model.read_bytes()).hexdigest(),'level':args.level,'size':args.size,'faces':records,'elapsed_s':round(time.monotonic()-start,2)},indent=2)+'\n')
    print('SUCCESS',len(records),flush=True)


if __name__=='__main__':main()
