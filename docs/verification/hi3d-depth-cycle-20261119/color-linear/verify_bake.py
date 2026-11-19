"""Verify the native color-space-only control and make unretouched guide comparisons."""
import gc
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image,ImageDraw,ImageFont

REPO=Path('/home/dev/Projects/box-impostor-lab/ComfyUI/custom_nodes/box_impostor_factory')
sys.path.insert(0,str(REPO))
from box_nodes import depth_mesh
WORK=Path('/mnt/f/_dev/box-impostor-lab/results/hi3d-depth-cycle-20261119/color-linear')
ROOT=Path('/mnt/f/_dev/box-impostor-lab/experiments/geometry-abc-20261119/output/box_impostor')
EVIDENCE=REPO/'docs/verification/hi3d-depth-cycle-20261119/color-linear'
EVIDENCE.mkdir(parents=True,exist_ok=True)
result=json.loads((WORK/'bake-result.json').read_text())
before,after=ROOT/result['sourceBaseline'],ROOT/result['assetId']
bm=json.loads((before/'manifest.json').read_text());am=json.loads((after/'manifest.json').read_text())
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while chunk:=f.read(1024*1024): h.update(chunk)
    return h.hexdigest()
def rgba(root,m,face):
    return Image.open(root/m['faces'][face]['guide']).convert('RGBA')

# Deliver the requested original-pixel full body and nearest-neighbor head comparison first.
images=[rgba(root,m,'+Z') for root,m in ((before,bm),(after,am))]
mask=np.asarray(images[0])[...,3]>250
ys,xs=np.nonzero(mask)
xc=(xs.min()+xs.max())/2
half=.24*(xs.max()-xs.min()+1)
roi=(max(0,int(xc-half)),int(ys.min()),min(1024,int(xc+half)+1),
     min(1024,int(ys.min()+.30*(ys.max()-ys.min()+1))))
font=ImageFont.truetype('DejaVuSans.ttf',24)
def board(pictures,path):
    width=max(im.width for im in pictures);height=max(im.height for im in pictures)
    canvas=Image.new('RGB',(width*2+36,height+70),(32,35,39))
    draw=ImageDraw.Draw(canvas)
    for i,(im,title) in enumerate(zip(pictures,['B: original color transfer','Color-only fix: linear FLOAT32'])):
        x=12+i*(width+12)
        draw.text((x,16),title,font=font,fill=(240,242,245))
        canvas.paste(im,(x,58),im)
    canvas.save(path,lossless=True)
board(images,EVIDENCE/'B-color-linear-front-full.webp')
heads=[im.crop(roi).resize(((roi[2]-roi[0])*2,(roi[3]-roi[1])*2),Image.Resampling.NEAREST) for im in images]
board(heads,EVIDENCE/'B-color-linear-front-face.webp')

checks={}
for key in ('resolution','tightBounds','bakeBox','transform','inverseTransform','faceOrder','coordinateSystem','maps'):
    checks[key+'Equal']=bm.get(key)==am.get(key)
maps={}
for name in bm['textures']:
    bs,as_=bm['textures'][name],am['textures'][name]
    bh,ah=sha(before/bs['path']),sha(after/as_['path'])
    assert bh==bs['sha256'] and ah==as_['sha256']
    maps[name]={'beforeSha256':bh,'afterSha256':ah,'equal':bh==ah,
                'beforeBytes':(before/bs['path']).stat().st_size,'afterBytes':(after/as_['path']).stat().st_size}
    if name!='colorMap':checks[name+'Equal']=bh==ah
    else:
        b=np.fromfile(before/bs['path'],dtype=np.dtype(bs['dtype'])).reshape(-1,bs['channels'])
        a=np.fromfile(after/as_['path'],dtype=np.dtype(as_['dtype'])).reshape(-1,as_['channels'])
        checks['colorAlphaEqual']=np.array_equal(b[:,3],a[:,3])
        maps[name]['alphaBeforeSha256']=hashlib.sha256(b[:,3].copy().tobytes()).hexdigest()
        maps[name]['alphaAfterSha256']=hashlib.sha256(a[:,3].copy().tobytes()).hexdigest()
        del a,b
faces={}
for face in bm['faceOrder']:
    bi,ai=bm['faces'][face],am['faces'][face]
    for key in ('side','up','tile','viewMatrix','projectionMatrix','frustum','imageRowOrder','depthEncoding'):
        checks[face+':'+key+'Equal']=bi.get(key)==ai.get(key)
    bh,ah=sha(before/bi['depth']),sha(after/ai['depth'])
    checks[face+':depthEqual']=bh==ah
    b=np.asarray(rgba(before,bm,face));a=np.asarray(rgba(after,am,face))
    alpha_equal=np.array_equal(b[...,3],a[...,3]);checks[face+':guideAlphaEqual']=alpha_equal
    covered=(b[...,3]>250)&(a[...,3]>250)
    difference=np.abs(a[...,:3].astype('i2')-b[...,:3].astype('i2'))
    faces[face]={'depthBeforeSha256':bh,'depthAfterSha256':ah,'guideAlphaEqual':alpha_equal,
        'guideBeforeSha256':sha(before/bi['guide']),'guideAfterSha256':sha(after/ai['guide']),
        'changedCoveredPixels':int(np.count_nonzero(np.any(difference>0,axis=2)&covered)),
        'coveredPixels':int(covered.sum()),
        'meanRGB8Before':np.mean(b[...,:3][covered],axis=0).tolist(),
        'meanRGB8After':np.mean(a[...,:3][covered],axis=0).tolist()}
    del a,b,difference
bd,bb=depth_mesh._decode((before/bm['proxy']).read_bytes())
ad,ab=depth_mesh._decode((after/am['proxy']).read_bytes())
geometry={}
for key in ('nodes','scenes','scene','materials','textures','images','samplers'):
    geometry[key+'Equal']=bd.get(key)==ad.get(key)
bp=list(depth_mesh._primitives(bd,bb));ap=list(depth_mesh._primitives(ad,ab))
geometry['primitiveCountEqual']=len(bp)==len(ap)
for i,(b,a) in enumerate(zip(bp,ap)):
    geometry[str(i)+'PositionsEqual']=np.array_equal(b[1],a[1])
    geometry[str(i)+'IndicesEqual']=np.array_equal(b[2],a[2])
    geometry[str(i)+'TransformsEqual']=np.array_equal(b[3],a[3])
    geometry[str(i)+'NoNormalsBeforeOrAfter']='NORMAL' not in b[0]['attributes'] and 'NORMAL' not in a[0]['attributes']
    spec=ad['accessors'][a[0]['attributes']['COLOR_0']]
    geometry[str(i)+'Float32Colors']=spec['componentType']==5126 and not spec.get('normalized',False)
checks.update({'geometry:'+key:value for key,value in geometry.items()})
proof={'sourceBaseline':result['sourceBaseline'],'assetId':result['assetId'],
    'passed':bool(all(checks.values())),'checks':checks,'maps':maps,'faces':faces,'geometry':geometry,
    'newInference':False,'frozenGeneratedImagesModified':False,
    'comparison':{'frontFaceCropXYXY':list(roi),'faceScale':2,'faceResampling':'nearest; no retouch or color correction',
        'fullBodyScale':1,'backgroundRGB':[32,35,39],
        'files':[{ 'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size} for p in
            (EVIDENCE/'B-color-linear-front-full.webp',EVIDENCE/'B-color-linear-front-face.webp')]}}
(WORK/'native-proof.json').write_text(json.dumps(proof,indent=2,allow_nan=False)+'\n')
(EVIDENCE/'native-proof.json').write_text(json.dumps(proof,indent=2,allow_nan=False)+'\n')
print(json.dumps({'passed':proof['passed'],'checkCount':len(checks),'failed':[k for k,v in checks.items() if not v],
    'assetId':proof['assetId'],'maps':maps,'faces':faces,'comparison':proof['comparison']},indent=2))
if not proof['passed']:raise SystemExit(1)
