#!/usr/bin/env python3
"""Subject-agnostic 2D affine registration Qwen->GS from whole-image silhouettes; no face segmentation."""
import argparse,hashlib,json
from pathlib import Path
import cv2,numpy as np
from PIL import Image
from scipy.optimize import differential_evolution

ROOT=Path('/mnt/f/_dev/box-impostor-lab/experiments/triposplat-person-20261120')
TARGET={'front':'R2048-yaw90-s1','yaw045':'R2048-yaw90-s1-camera45','yaw315':'R2048-yaw90-s1-camera-45'}

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def matrix(parameters,size):
    sx,sy,angle,dy,dx=map(float,parameters)
    c=(size-1)/2
    rad=np.deg2rad(angle);co,si=np.cos(rad),np.sin(rad)
    return np.array([[sx*co,-sy*si,c+dx-sx*co*c+sy*si*c],
                     [sx*si,sy*co,c+dy-sx*si*c-sy*co*c]],dtype='f4')

def warp(image,parameters):
    size=image.shape[0]
    return cv2.warpAffine(np.asarray(image),matrix(parameters,size),(size,size),
                          flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)

def mask_iou(a,b):
    return float(np.count_nonzero(a&b)/max(1,np.count_nonzero(a|b)))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--case',choices=TARGET,default='front')
    args=parser.parse_args()
    work=ROOT/'qwen2k'/f'{args.case}-2kopt-10-depth3'
    trial=json.loads((work/'result.json').read_text())
    mask_path=Path(trial['images']['mask']['path'])
    gs_path=ROOT/'render'/TARGET[args.case]/'pZ/alpha.npy'
    with Image.open(mask_path) as im: q=np.asarray(im.convert('L'),dtype='u1')
    g=np.load(gs_path,allow_pickle=False).astype('f4')
    if q.shape!=(2048,2048) or g.shape!=q.shape:raise ValueError('Expected aligned 2048 inputs')
    sz=512
    a=cv2.resize(q,(sz,sz),interpolation=cv2.INTER_AREA)>127
    b=cv2.resize(g,(sz,sz),interpolation=cv2.INTER_AREA)>.5
    def score(x):
        v=warp((a*255).astype('u1'),x)>127
        return mask_iou(v,b)
    opt=differential_evolution(lambda x:-score(x),
       bounds=[(.96,1.06),(.96,1.06),(-3.,3.),(-8,8),(-8,8)],
       maxiter=58,popsize=7,seed=20261120,tol=.0004,polish=False)
    v=np.asarray(opt.x,'f8')
    v[3:]*=4 # 512 -> 2048 pixel translations
    warped=warp(q,v)>127
    dst=g>.5
    fullscore=mask_iou(warped,dst)
    original=mask_iou(q>127,dst)
    doc={'method':'generic-whole-subject-small-affine-2k-v1',
         'case':args.case,'registrationParams':list(map(float,v)),
         'matrixInputToOutput':matrix(v,2048).tolist(),
         'sourceGeneratedSha256':trial['images']['generated']['sha256'],
         'sourceQwenMaskSha256':sha(mask_path),
         'GSAlphaSha256':sha(gs_path),
         'originalIoU':original,'registeredIoU':fullscore,
         'admissionMinIoU':.88,'admissionPassed':bool(fullscore>=.88),
         'optimizationSize':sz,'fullResolution':2048,
         'maxAbsoluteShiftPx2048':32,
         'maxAbsoluteRotationDeg':3,
         'maxRelativeScaleChange':.06,
         'reason':'Whole-foreground similarity only; no facial crops, keypoints or inpainting',
         'shapeChange':'No GS geometry, depth, alpha or 3DGS data changed'}
    path=work/'global-affine-registration.json'
    path.write_text(json.dumps(doc,indent=2)+'\n')
    print(json.dumps(doc,indent=2))
if __name__=='__main__':main()
