#!/usr/bin/env python3
"""Verify fixed native Box camera/alpha for all synchronized captures, while RGB changes."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image

def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('base',type=Path)
    p.add_argument('edited',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    original=json.loads((args.base/'capture.json').read_text())
    revised=json.loads((args.edited/'capture.json').read_text())
    for name,j in [('original',original),('revised',revised)]:
        if not j.get('completed') or j.get('errors') or len(j['frames'])!=j['framesExpected']:
            raise ValueError(name+' capture is incomplete')
    if len(original['frames'])!=len(revised['frames']):raise ValueError('Frame counts differ')
    results=[];maxcamera=0
    for a,b in zip(original['frames'],revised['frames']):
        if a['requestedYawDegrees']!=b['requestedYawDegrees']:raise ValueError('Camera yaw mismatch')
        for key in ('matrixWorld','projectionMatrix'):
            error=max(abs(x-y) for x,y in zip(a['camera'][key],b['camera'][key]))
            maxcamera=max(maxcamera,error)
        if maxcamera>1e-6:raise ValueError('Non-identical camera matrices')
        pa,pb=Path(a['path']),Path(b['path'])
        if digest(pa)!=a['sha256'] or digest(pb)!=b['sha256']:
            raise ValueError('Capture frame hash differs')
        with Image.open(pa) as im:aa=np.asarray(im.convert('RGBA'),dtype='u1')
        with Image.open(pb) as im:bb=np.asarray(im.convert('RGBA'),dtype='u1')
        if aa.shape!=bb.shape:raise ValueError('Capture dimensions differ')
        delta=np.abs(aa.astype('i2')-bb.astype('i2'))
        frame={'index':int(a['index']),'yaw':float(a['requestedYawDegrees']),
               'alphaMaxDelta':int(delta[...,3].max()),
               'changedRGBPixels':int(np.count_nonzero(np.any(delta[...,:3]>1,axis=-1))),
               'rgbRms255':float(np.sqrt(np.mean(delta[...,:3].astype('f4')**2)))}
        results.append(frame)
    proof={'schema':'box-identical-shape-rgb-update-v1','baseAsset':original['assetId'],
           'editedAsset':revised['assetId'],'count':len(results),
           'maxCameraMatrixDifference':maxcamera,
           'maxAlphaByteDifference':max(x['alphaMaxDelta'] for x in results),
           'changedRgbPixels':sum(x['changedRGBPixels'] for x in results),
           'meanRgbRms255':float(np.mean([x['rgbRms255'] for x in results])),
           'perFrame':results}
    if proof['maxAlphaByteDifference']!=0 or proof['changedRgbPixels']==0:
        raise ValueError('RGB-only invariant failed')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps({k:v for k,v in proof.items() if k!='perFrame'},indent=2))
if __name__=='__main__':main()
