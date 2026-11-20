#!/usr/bin/env python3
"""GS absolute camera-axis depth, not per-view remapped grayscale, aligned to 6 Box atlas faces."""
import hashlib,json,math,sys,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
ROOT=Path('/mnt/f/_dev/box-impostor-lab/experiments/triposplat-person-20261120')
COMFY=Path('/home/dev/Projects/box-impostor-lab/ComfyUI')
sys.path[:0]=[str(ROOT),str(COMFY),str(COMFY/'custom_nodes/box_impostor_factory')]
from box_nodes.splat import camera_plan
from comfy_api.latest import Types
from comfy_extras.nodes_gaussian_splat import TransformSplat, RenderSplat as NativeSplat
from experimental_gaussian_depth import RenderSplat as MetricSplat

CACHE=Path('/mnt/f/mnt/output/box_impostor/.stages/triposplat/ecb8433166915206286d50ab299fe2faa73d7e7b028f8ea19fa36cd3d41fad54')
BOX=Path('/mnt/f/mnt/output/depth-refinement-20261120/box_impostor/019253e1edc661260f7775735b8bcaabcd9fe2a23a8b6ecd1e8227a7dd8a42d8')
FACES={'px':'+X','nx':'-X','py':'+Y','ny':'-Y','pz':'+Z','nz':'-Z'}
R=1024
OUT=ROOT/'render'/'GS-native-faces'
OUT.mkdir(exist_ok=True,parents=True)

def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def resize_f32(data,wh):
    return np.array(Image.fromarray(np.asarray(data,dtype='f4')).resize(wh,Image.Resampling.BILINEAR),dtype='f4',copy=True)
def encode_png(f,values):
    Image.fromarray(np.uint8(np.clip(np.rint(values*255),0,255))).save(f)
def to_linear(x):
    return np.where(x<=.04045,x/12.92,((np.maximum(x,0)+.055)/1.055)**2.4)
def render(splat,camera,style):
    cls=MetricSplat if style=='metric_depth' else NativeSplat
    i,a=cls.execute(splat=splat,width=R,height=R,frames=1,splat_scale=1.,
          sharpen=1.,headlight_shading=0.,opacity_threshold=0.,background='#808080',
          render_style=style,camera_info=camera).result
    return i[0].detach().float().cpu().numpy(),a[0].detach().float().cpu().numpy()
def main():
    began=time.monotonic()
    with np.load(CACHE/'splat.npz',allow_pickle=False) as a:
        splat=Types.SPLAT(*[torch.from_numpy(a[k].copy()) for k in ('positions','scales','rotations','opacities','sh')])
    splat=TransformSplat.execute(splat,0,0,0,0,90,0,1,1,1).result[0]
    m=json.loads((BOX/'manifest.json').read_text())
    record={'kind':'3DGS-physical-camera-depth-native-6-face-v1','splatSHA256':digest(CACHE/'splat.npz'),
            'calibration':'camera-axis metric depth divided by original Box axis span; no affine seed fit',
            'originalBox':str(BOX),'rendererSourceSHA256':digest(ROOT/'experimental_gaussian_depth.py'),
            'faces':{}}
    for slug,fid in FACES.items():
        start=time.monotonic()
        p=OUT/slug;p.mkdir(exist_ok=True)
        face=m['faces'][fid]
        camera,_,_=camera_plan(m,fid,'canonical_box')
        f=face['frustum'];physical=np.array([abs(f[1]-f[0]),abs(f[3]-f[2])])
        camdist=np.linalg.norm(np.array(list(camera['position'].values()))-np.array(list(camera['target'].values())))
        camera['fov']=2*math.degrees(math.atan(max(physical)/(2*camdist)))
        span=np.asarray(m['bakeBox']['max'])-np.asarray(m['bakeBox']['min'])
        axspan=float(np.abs(np.array(face['side'])@span))
        if axspan<=0:raise ValueError(fid)
        crop=np.rint(physical/max(physical)*R).astype(int)
        crop=np.maximum(crop,1)
        x0=(R-crop[0])//2;y0=(R-crop[1])//2;w,h=crop
        metric,alpha=render(splat,camera,'metric_depth')
        color,alpha_c=render(splat,camera,'color')
        encoded,alpha_n=render(splat,camera,'normal')
        if np.max(np.abs(alpha-alpha_c))>3e-5 or np.max(np.abs(alpha-alpha_n))>3e-5:
            raise ValueError('Different splat coverage for same physical camera '+fid)
        axis_depth=(metric[...,0]-(camdist-axspan/2))/axspan
        near=alpha>.0005
        valid=near&(axis_depth>=0)&(axis_depth<=1)
        nearest_weighted=resize_f32(np.where(valid,axis_depth*alpha,0)[y0:y0+h,x0:x0+w],(R,R))
        coverage=resize_f32(np.where(valid,alpha,0)[y0:y0+h,x0:x0+w],(R,R))
        native_z=np.divide(nearest_weighted,coverage,out=np.full((R,R),np.nan,dtype='f4'),where=coverage>.5)
        valid_n=np.isfinite(native_z)
        # Recover premultiplied linear albedo from the renderer's gray background.
        srgb_color=np.clip(color,0,1)
        bg=to_linear(128/255)
        premul=np.maximum(0,to_linear(srgb_color)-(1-alpha[...,None])*bg)
        native_color=np.stack([resize_f32(premul[y0:y0+h,x0:x0+w,i],(R,R)) for i in range(3)],axis=-1)
        native_color[~valid_n]=0
        native_color=np.minimum(native_color,coverage[...,None])
        # RenderSplat normal mode: display RGB = (.5,.5,.5) + .5*(cam_x,-cam_down,-cam_forward).
        raw=np.stack((2*encoded[...,0]-1,1-2*encoded[...,1],1-2*encoded[...,2]),axis=-1)
        side=np.asarray(face['side'],dtype='f4')
        up=np.asarray(face['up'],dtype='f4')
        right=np.cross(up,side)
        world=raw[...,0,None]*right + raw[...,1,None]*(-up) + raw[...,2,None]*(-side)
        world/=np.maximum(np.linalg.norm(world,axis=-1,keepdims=True),1e-6)
        encoding=.5+.5*world
        native_normal=np.stack([resize_f32((encoding[...,i]*alpha)[y0:y0+h,x0:x0+w],(R,R)) for i in range(3)],axis=-1)
        native_normal[~valid_n]=0
        coverage[~valid_n]=0
        np.save(p/'depth.npy',native_z,allow_pickle=False)
        np.save(p/'alpha.npy',coverage,allow_pickle=False)
        np.save(p/'premul-color.npy',native_color.astype('f4'),allow_pickle=False)
        np.save(p/'premul-normal.npy',native_normal.astype('f4'),allow_pickle=False)
        encode_png(p/'depth-preview.png',np.nan_to_num(1-native_z))
        rgb=np.divide(native_color,np.maximum(coverage[...,None],1e-6))
        rgb=np.where(rgb<=.0031308,12.92*rgb,1.055*np.maximum(rgb,0)**(1/2.4)-.055)
        encode_png(p/'color-preview.png',rgb)
        entry={'axisSpan':axspan,'distance':float(camdist),'physicalFrustum':physical.tolist(),
             'rect':[int(x0),int(y0),int(w),int(h)],'pixels':int(valid_n.sum()),
             'depthMin':float(np.nanmin(native_z)) if valid_n.any() else None,
             'depthMax':float(np.nanmax(native_z)) if valid_n.any() else None,
             'outOfBoundsAlphaPixels':int(np.count_nonzero(near&~valid)),
             'runtime':round(time.monotonic()-start,2),
             'depthSha256':digest(p/'depth.npy'),'alphaSha256':digest(p/'alpha.npy')}
        record['faces'][slug]=entry
        (OUT/'status.json').write_text(json.dumps(record,indent=2)+'\n')
        print(fid,entry,flush=True)
    record['totalSeconds']=round(time.monotonic()-began,2)
    (OUT/'status.json').write_text(json.dumps(record,indent=2)+'\n')
    print('COMPLETE',record['totalSeconds'],flush=True)

if __name__=='__main__':main()
