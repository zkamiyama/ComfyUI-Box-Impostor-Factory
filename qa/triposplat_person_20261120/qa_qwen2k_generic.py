#!/usr/bin/env python3
"""Generic whole-subject Qwen2K/GS comparisons, no face-only crop or inpainting."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy import ndimage

ROOT=Path('/mnt/f/_dev/box-impostor-lab/experiments/triposplat-person-20261120')
CASES={'front':'R2048-yaw90-s1','yaw045':'R2048-yaw90-s1-camera45','yaw315':'R2048-yaw90-s1-camera-45'}
def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def compare(case):
    k=f'{case}-2kopt-10-depth3'
    qpath=ROOT/'qwen2k'/k/'result.json'
    if not qpath.is_file():raise FileNotFoundError(qpath)
    q=json.loads(qpath.read_text())
    raw=Path(q['images']['generated']['path'])
    mask=Path(q['images']['mask']['path'])
    gs=ROOT/'render'/CASES[case]/'pZ'
    with Image.open(raw) as im:
        size=im.size
        if size!=(2048,2048):raise ValueError('Qwen did not produce genuine 2048x2048 image: '+str(size))
        generated=np.asarray(im.convert('RGB'),dtype='u1')
    with Image.open(gs/'composited.png') as im:
        assert im.size==(2048,2048)
        base=np.asarray(im.convert('RGB'),dtype='u1')
    with Image.open(mask) as im:
        qmask=np.asarray(im.convert('L').resize((2048,2048),Image.Resampling.BILINEAR),dtype='f4')>127
    gmask=np.load(gs/'alpha.npy')>.5
    i=np.count_nonzero(qmask&gmask);u=np.count_nonzero(qmask|gmask)
    overlap=qmask&gmask
    color_rms=float(np.sqrt(np.mean((generated[overlap].astype('f4')-base[overlap].astype('f4'))**2))) if overlap.any() else None
    # Generic unspecialized sharpness: Laplacian over all foreground, not face or semantic region
    ggray=np.mean(generated.astype('f4'),axis=2)
    bgray=np.mean(base.astype('f4'),axis=2)
    mid=ndimage.binary_erosion(overlap,iterations=2)
    gl=ndimage.laplace(ggray)[mid];bl=ndimage.laplace(bgray)[mid]
    out=ROOT/'qwen2k'/k/'downsample-1024.png'
    Image.fromarray(generated).resize((1024,1024),Image.Resampling.LANCZOS).save(out)
    result={'case':case,'imageSize':list(size),'generatedSha256':digest(raw),'maskIoU':float(i/u) if u else None,
            'qwenForegroundPx':int(qmask.sum()),'GSForegroundPx':int(gmask.sum()),
            'sharedRGBRms255':color_rms,'foregroundLaplacianRMS':{
                 'qwen':float(np.sqrt(np.mean(gl**2))) if len(gl) else None,
                 'gs':float(np.sqrt(np.mean(bl**2))) if len(bl) else None},
            'downsampleImage':str(out),'downsampleSha256':digest(out),
            'validationScope':'whole-body silhouette/detail signal; no novel-view ground truth'}
    return result

def make_contact(results):
    w=512; band=45
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
    out=Image.new('RGB',(w*3,(w+band)*len(results)),(22,25,31))
    d=ImageDraw.Draw(out)
    for y,entry in enumerate(results):
        case=entry['case'];k=f'{case}-2kopt-10-depth3'
        p=ROOT/'qwen2k'/k
        gs=ROOT/'render'/CASES[case]/'pZ'
        files=[gs/'composited.png',p/'generated.png',p/'downsample-1024.png']
        for x,file in enumerate(files):
            with Image.open(file) as im:
                frame=im.convert('RGB').resize((w,w),Image.Resampling.LANCZOS)
            out.paste(frame,(x*w,y*(w+band)+band))
            labels=['3DGS 2K','Qwen native 2K','Qwen 2K -> 1024']
            d.text((x*w+10,y*(w+band)+9),case+' | '+labels[x],font=font,fill='white')
    p=ROOT/'media'/'b-qwen-2k-generic-comparison.jpg'
    out.save(p,quality=87,optimize=True)
    return p

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cases',default='front,yaw045,yaw315')
    args=ap.parse_args()
    results=[compare(c) for c in args.cases.split(',')]
    dest=ROOT/'analysis'/'qwen2k-whole-body-qa.json'
    p=make_contact(results)
    report={'cases':results,'comparisonImage':str(p),'comparisonImageSha256':digest(p)}
    dest.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
