#!/usr/bin/env python3
"""GS native geometry/depth unchanged; apply generic full-frame Qwen 2K color after 1024 downsample."""
import argparse, hashlib, json, shutil
from pathlib import Path
import numpy as np
from PIL import Image

ROOT=Path('/mnt/f/_dev/box-impostor-lab/experiments/triposplat-person-20261120')
SRC=Path('/mnt/f/mnt/output/gs-box-20261120/box_impostor/d856345c6c8168a2bd130cf7c1fe5d73fa3b340f227ac6cbc714427155ac4924')
OUT=Path('/mnt/f/mnt/output/gs-box-20261120/box_impostor')
R=1024

def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def linear(x):
    return np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--qwen-result',type=Path,default=ROOT/'qwen2k/front-2k-10-depth3/result.json')
    args=parser.parse_args()
    q=json.loads(args.qwen_result.read_text())
    image=Path(q['images']['generated']['path'])
    if sha(image)!=q['images']['generated']['sha256']:raise ValueError('Qwen generated image digest changed.')
    m=json.loads((SRC/'manifest.json').read_text())
    face=m['faces']['+Z']; r=m['resolution']
    if r!=R or tuple(face['tile'])!=(1,1):raise ValueError('Unexpected frame')
    # The Qwen full-body frame is 2048 square, aligned with the 2048 GS canvas.
    with Image.open(image) as im:
        generated_shape=im.size
        if min(im.size)<1024: raise ValueError('No high-resolution input')
        full=im.convert('RGB').resize((r,r),Image.Resampling.LANCZOS)
    f=face['frustum']
    w=max(1,round(r*abs(f[1]-f[0])/max(abs(f[1]-f[0]),abs(f[3]-f[2]))))
    h=max(1,round(r*abs(f[3]-f[2])/max(abs(f[1]-f[0]),abs(f[3]-f[2]))))
    left=(r-w)//2;top=(r-h)//2
    # Unletterbox once; then align to the native Box atlas tile grid.
    candidate=np.asarray(full.crop((left,top,left+w,top+h)).resize((r,r),Image.Resampling.LANCZOS),dtype='f4')/255.
    candidate=linear(candidate)
    aid=hashlib.sha256(json.dumps({'algorithm':'gs-box-2k-color-front-v1','source':m['asset_id'],
                                   'qwen':sha(image),'bbox':[left,top,w,h]},sort_keys=True).encode()).hexdigest()
    root=OUT/aid
    if root.exists():shutil.rmtree(root)
    shutil.copytree(SRC,root,ignore=shutil.ignore_patterns('previews'))
    atlas_spec=m['textures']['colorMap']
    old_color_sha=atlas_spec['sha256']
    path=root/atlas_spec['path']
    atlas=np.fromfile(path,dtype='<f2').reshape((2*r,3*r,4)).astype('f4')
    before=atlas.copy()
    tile=atlas[r:2*r,r:2*r]
    alpha=tile[...,3].copy()
    tile[:,:,:3]=(candidate[::-1]*alpha[...,None])
    if not np.array_equal(atlas[...,3],before[...,3]):raise RuntimeError('GS alpha changed.')
    if not np.array_equal(atlas[:r],before[:r]) or not np.array_equal(atlas[r:,:r],before[r:,:r]) or not np.array_equal(atlas[r:,2*r:],before[r:,2*r:]):
        raise RuntimeError('Other GS faces changed')
    atlas.astype('<f2').tofile(path)
    digest=sha(path)
    m['asset_id']=aid
    m['parentAsset']=SRC.name
    m['mode']='gs-qwen-2k-front-color'
    m['textures']['colorMap']['sha256']=digest
    qrecord={
      'status':'color-only-pilot','phase':'partial-front-face',
      'sourceAsset':SRC.name,'qwenSource':str(image),'qwenSourceSHA256':sha(image),
      'qwenInputResolution':list(generated_shape),'outputTileResolution':r,
      'unletterbox':[left,top,w,h],
      'unchangedGeometryDepthNormalRange':True,'gsAlphaHashPreserved':True,
      'newColorSha256':digest,'oldColorSha256':old_color_sha,
      'allOtherFiveFacesRetained':True,
      'applied':'front +Z only; 45 and 315 are diagnostic camera views, not native axis faces',
      'limitations':'This is a one-face refinement pilot; not a consistent six-face AI texturing.'
    }
    m.setdefault('provenance',{})['qwen2k']=qrecord
    m['qa']='qa/gs-qwen-2k-front.json'
    (root/'qa/gs-qwen-2k-front.json').write_text(json.dumps(qrecord,indent=2)+'\n')
    (root/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    report={'assetId':aid,'path':str(root),'qwen':qrecord,'colorMapChanged':sha(path)!=old_color_sha}
    (ROOT/'analysis/qwen2k-front-box.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
