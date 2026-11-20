#!/usr/bin/env python3
"""Construct 3DGS-derived native Box maps directly from six camera-axis depth maps and colors."""
import argparse,hashlib,json,shutil,sys
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage

ROOT=Path('/mnt/f/_dev/box-impostor-lab/experiments/triposplat-person-20261120')
SRC=Path('/mnt/f/mnt/output/depth-refinement-20261120/box_impostor/019253e1edc661260f7775735b8bcaabcd9fe2a23a8b6ecd1e8227a7dd8a42d8')
NATIVE=ROOT/'render/GS-native-faces'
OUT=Path('/mnt/f/mnt/output/gs-box-20261120')
FACES={'px':'+X','nx':'-X','py':'+Y','ny':'-Y','pz':'+Z','nz':'-Z'}
R=1024

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def shader_grid(a,slug):
    if slug in ('px','nz'):return a[::-1,::-1]
    if slug=='py':return a
    return a[::-1]
def atlas_desc(path,old):
    return {**old,'sha256':sha(path)}
def write_bin(path,arr):
    np.asarray(arr,dtype='<f2').tofile(path)
    return sha(path)
def srgb(linear):
    return np.where(linear<=.0031308,12.92*linear,1.055*np.maximum(linear,0)**(1/2.4)-.055)
def gen(args):
    src=json.loads((SRC/'manifest.json').read_text())
    status=json.loads((NATIVE/'status.json').read_text())
    if set(status['faces'])!=set(FACES): raise RuntimeError('Six metric source faces are required')
    identity={'base':src['asset_id'],'data':{slug:rec['depthSha256'] for slug,rec in status['faces'].items()},
              'color':'gs-only' if args.qwen is None else str(args.qwen),'method':'native-3dgs-box-v1'}
    aid=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    root=OUT/'box_impostor'/aid
    if root.exists():shutil.rmtree(root)
    root.parent.mkdir(parents=True,exist_ok=True)
    shutil.copytree(SRC,root,ignore=shutil.ignore_patterns('previews'))
    manifest=json.loads((root/'manifest.json').read_text())
    manifest.update(asset_id=aid,parentAsset=src['asset_id'],mode='manual_faces')
    manifest.setdefault('provenance',{})['gsNative']={
       'mode':'GS RGB+metric axis Z directly to Box interval/color/normal/range; no mesh fit',
       'splatSha':status['splatSHA256'],'metricDepthStatus':str(NATIVE/'status.json'),
       'sourceBox':src['asset_id'],'proxyRetainedOnlyForCameraAndMetadata':True,
       'qwenRefine':str(args.qwen) if args.qwen else None,
       'noInteriorGap':'3DGS one depth per view cannot infer reliable hidden cavities',
       'boundsPolicy':'legacy Box physical axis bounds; out-of-bounds GS samples excluded',
       'pairPolicy':'face+opposite first-hit interval; crossovers collapsed to 2/r minimum thickness',
       'emptyTexelPolicy':'native Baker neighboring inside-out interval padding (8-neighbor mean)',
       'colorSpace':'linear sRGB, premultiplied alpha'}
    interval=np.zeros((R,3*R,4),dtype='f4');interval[...,3]=1
    color=np.zeros((2*R,3*R,4),dtype='f4')
    normal=np.zeros((2*R,3*R,4),dtype='f4')
    data={}
    for slug,fid in FACES.items():
        p=NATIVE/slug
        d=np.load(p/'depth.npy').astype('f4')
        a=np.load(p/'alpha.npy').astype('f4')
        c=np.load(p/'premul-color.npy').astype('f4')
        n=np.load(p/'premul-normal.npy').astype('f4')
        if d.shape!=(R,R) or a.shape!=(R,R) or c.shape!=(R,R,3) or not np.isfinite(a).all():raise ValueError(fid)
        valid=(a>.5)&np.isfinite(d)&(d>=0)&(d<=1)
        a=np.where(valid,a,0).astype('f4')
        c=np.where(valid[...,None],c,0).astype('f4')
        n=np.where(valid[...,None],n,0).astype('f4')
        if args.qwen:
            # Per-face accepted Qwen RGB updates only, GS geometry/alpha remain identical.
            case={'pz':'front','px':'yaw045','nz':'yaw315'}.get(slug)
            if case:
                run=args.qwen/case/'result.json'
                if run.exists():
                    gen_path=Path(json.loads(run.read_text())['images']['generated']['path'])
                    with Image.open(gen_path) as im:
                        generated=np.asarray(im.convert('RGB').resize((R,R),Image.Resampling.BILINEAR),dtype='f4')/255
                    rgb_linear=np.where(generated<=.04045,generated/12.92,((generated+.055)/1.055)**2.4)
                    # Qwen output 1024-square in GS reference camera; native atlas uses physical crop.
                    face=manifest['faces'][fid]
                    f=face['frustum'];w=round(R*abs(f[1]-f[0])/max(abs(f[1]-f[0]),abs(f[3]-f[2])))
                    h=round(R*abs(f[3]-f[2])/max(abs(f[1]-f[0]),abs(f[3]-f[2])))
                    left=(R-w)//2;top=(R-h)//2
                    y=rgb_linear[top:top+h,left:left+w]
                    from PIL import Image as I
                    rgb_native=np.stack([np.asarray(I.fromarray(y[...,i]).resize((R,R),I.Resampling.BILINEAR)) for i in range(3)],axis=-1)
                    c=np.where(valid[...,None],rgb_native*a[...,None],0)
        x,y=manifest['faces'][fid]['tile']
        tile=color[y*R:(y+1)*R,x*R:(x+1)*R]
        tile[:,:,:3]=c[::-1]
        tile[:,:,3]=a[::-1]
        ntile=normal[y*R:(y+1)*R,x*R:(x+1)*R]
        ntile[:,:,:3]=n[::-1]
        ntile[:,:,3]=a[::-1]
        data[slug]={'depth':d,'alpha':a,'mask':valid}
        face=manifest['faces'][fid]
        fpath=root/face['depth'];write_bin(fpath,d)
        face['depthSha256']=sha(fpath)
        face['foregroundPixels']=int(valid.sum())
        im=np.uint8(np.clip(np.rint(srgb(c/np.maximum(a[...,None],1e-6))*255),0,255))
        png=Image.fromarray(np.concatenate([im,(a[...,None]*255).astype('uint8')],axis=-1),'RGBA')
        guide=root/face['guide'];png.save(guide)
        face['guideSha256']=sha(guide)
        face.pop('edit',None) # stale mesh-alignment admission must not be misrepresented as GS
    axisstats={}
    for axis,(plus,minus) in enumerate([('px','nx'),('py','ny'),('pz','nz')]):
        dp=shader_grid(data[plus]['depth'],plus)
        dm=shader_grid(data[minus]['depth'],minus)
        ap=shader_grid(data[plus]['mask'],plus)
        am=shader_grid(data[minus]['mask'],minus)
        high=1-dp;low=dm
        both=ap&am
        posonly=ap&~am
        negonly=am&~ap
        minwidth=2/R
        hi=np.zeros((R,R),'f4');lo=np.ones((R,R),'f4')
        hi[both]=np.maximum(high[both],low[both]+minwidth)
        lo[both]=low[both]
        hi[posonly]=high[posonly]
        lo[posonly]=np.maximum(0,high[posonly]-minwidth)
        lo[negonly]=low[negonly]
        hi[negonly]=np.minimum(1,low[negonly]+minwidth)
        mask=both|posonly|negonly
        hi=np.minimum(np.maximum(hi,0),1);lo=np.minimum(np.maximum(lo,0),1)
        valid=mask&(hi>lo)
        # Edges: inside-out inverse interval surrounded by occupied samples, as native Baker does.
        kernel=np.ones((3,3),dtype='f4');kernel[1,1]=0
        count=ndimage.convolve(valid.astype('f4'),kernel,mode='constant',cval=0)
        nbr_hi=ndimage.convolve(np.where(valid,hi,0),kernel,mode='constant',cval=0)
        nbr_lo=ndimage.convolve(np.where(valid,lo,0),kernel,mode='constant',cval=0)
        e=1.5/R
        padded=(~valid)&(count>0)
        hi[padded]=(nbr_lo[padded]/count[padded]) - 2*e
        lo[padded]=(nbr_hi[padded]/count[padded]) + 2*e
        tile=interval[:,axis*R:(axis+1)*R]
        tile[...,0]=np.where(valid|padded,hi,0)
        tile[...,1]=np.where(valid|padded,1-lo,0)
        tile[...,2]=0;tile[...,3]=1
        axisstats['XYZ'[axis]]={'both':int(both.sum()),'plusOnly':int(posonly.sum()),'minusOnly':int(negonly.sum()),
                               'crossed':int(np.count_nonzero(both&(high<=low))),
                               'filled':int(valid.sum()),'edgePad':int(padded.sum()),'noCoverage':int((~mask).sum())}
    old=manifest['textures']
    i=root/old['intervalMap']['path'];write_bin(i,interval);old['intervalMap']=atlas_desc(i,old['intervalMap'])
    c=root/old['colorMap']['path'];write_bin(c,color);old['colorMap']=atlas_desc(c,old['colorMap'])
    n=root/old['normalMap']['path'];write_bin(n,normal);old['normalMap']=atlas_desc(n,old['normalMap'])
    base=root/manifest['baseColor']['path'];write_bin(base,color)
    manifest['baseColor']=atlas_desc(base,manifest['baseColor'])
    cells=(R+7)//8;centers=np.minimum(np.arange(cells)*8+4,R-1)
    range_map=np.zeros((cells,3*cells,4),dtype='f4');range_map[...,3]=1
    for axis in range(3):
        t=interval[:,axis*R:(axis+1)*R]
        lower=ndimage.minimum_filter(1-t[...,1],size=10,mode='nearest')
        upper=ndimage.maximum_filter(t[...,0],size=10,mode='nearest')
        range_map[:,axis*cells:(axis+1)*cells,0]=lower[np.ix_(centers,centers)]
        range_map[:,axis*cells:(axis+1)*cells,1]=upper[np.ix_(centers,centers)]
    rg=root/old['rangeMap']['path'];write_bin(rg,range_map);old['rangeMap']=atlas_desc(rg,old['rangeMap'])
    manifest['qa']='qa/gs-native.json'
    qa={'method':'3DGS-6-view native Box direct','axes':axisstats,'sha256':{k:v['sha256'] for k,v in old.items()},
        'noMeshGeometryUsed':True,'noQwen' : args.qwen is None,'splatSha256':status['splatSHA256']}
    (root/'qa/gs-native.json').write_text(json.dumps(qa,indent=2)+'\n')
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    result={'assetId':aid,'assetPath':str(root),'qa':qa,'outputData':str(NATIVE/'status.json')}
    (OUT/f'gs-{aid[:12]}.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--qwen',type=Path)
    gen(ap.parse_args())
