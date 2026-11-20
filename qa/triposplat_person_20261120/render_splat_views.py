#!/usr/bin/env python3
"""Render new TripoSplat model to orthographic RGB/depth/alpha/normal with native ComfyUI renderer."""
import argparse, json, hashlib, math, sys, time
from pathlib import Path
import numpy as np
from PIL import Image

COMFY=Path('/home/dev/Projects/box-impostor-lab/ComfyUI')
sys.path[:0]=[str(COMFY),str(COMFY/'custom_nodes/box_impostor_factory')]
from box_nodes.splat import camera_plan
from comfy_api.latest import Types
from comfy_extras.nodes_gaussian_splat import RenderSplat, TransformSplat
from comfy_extras.nodes_camera import _rotate_camera_info
import torch

ROOT=Path('/mnt/f/_dev/box-impostor-lab/experiments/triposplat-person-20261120')
CACHE=Path('/mnt/f/mnt/output/box_impostor/.stages/triposplat/ecb8433166915206286d50ab299fe2faa73d7e7b028f8ea19fa36cd3d41fad54')
BOX=Path('/mnt/f/mnt/output/depth-refinement-20261120/box_impostor/019253e1edc661260f7775735b8bcaabcd9fe2a23a8b6ecd1e8227a7dd8a42d8')
FACES=['+Z','+X','-X','-Z','+Y','-Y']

def hashfile(path):
    with open(path,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def image(path,data):
    Image.fromarray(np.rint(np.clip(data,0,1)*255).astype('uint8')).save(path)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--size',type=int,default=512)
    ap.add_argument('--faces',default='+Z,+X,-X,-Z,+Y,-Y')
    ap.add_argument('--styles',default='color,depth,normal')
    ap.add_argument('--yaw',type=float,default=0.)
    ap.add_argument('--camera-yaw',type=float,default=0.)
    ap.add_argument('--scale',type=float,default=1.)
    ap.add_argument('--offset',default='0,0,0')
    ap.add_argument('--opacity-threshold',type=float,default=0.)
    a=ap.parse_args()
    out=ROOT/'render'/(f'R{a.size}-yaw{a.yaw:g}-s{a.scale:g}'+(f'-camera{a.camera_yaw:g}' if a.camera_yaw else ''))
    out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((BOX/'manifest.json').read_text())
    with np.load(CACHE/'splat.npz',allow_pickle=False) as f:
        splat=Types.SPLAT(*[torch.from_numpy(f[k].copy()) for k in ('positions','scales','rotations','opacities','sh')])
    tx,ty,tz=map(float,a.offset.split(','))
    splat=TransformSplat.execute(splat=splat,translate_x=tx,translate_y=ty,translate_z=tz,rotate_x=0.,rotate_y=a.yaw,rotate_z=0.,scale_x=a.scale,scale_y=a.scale,scale_z=a.scale).result[0]
    stats={'render':'native-RenderSplat','splatSha':hashfile(CACHE/'splat.npz'),'boxManifestSha':hashfile(BOX/'manifest.json'),
           'size':a.size,'cameraYawDegrees':a.camera_yaw,'yawNativeDegrees':a.yaw,'nativeOffset':[tx,ty,tz], 'scale':a.scale,'faces':{},'styles':a.styles}
    for fid in a.faces.split(','):
        started=time.monotonic()
        path=out/fid.replace('+','p').replace('-','n')
        path.mkdir(exist_ok=True)
        camera, native_size, content=camera_plan(manifest,fid,'canonical_box')
        physical=[abs(manifest['faces'][fid]['frustum'][1]-manifest['faces'][fid]['frustum'][0]),
                  abs(manifest['faces'][fid]['frustum'][3]-manifest['faces'][fid]['frustum'][2])]
        dist=np.linalg.norm(np.array(list(camera['position'].values()))-np.array(list(camera['target'].values())))
        camera['fov']=2*math.degrees(math.atan(max(physical)/(2*dist)))
        camera['zoom']=1.
        if a.camera_yaw: camera=_rotate_camera_info(camera,(0.,1.,0.),a.camera_yaw)
        (path/'camera.json').write_text(json.dumps(camera,indent=2)+'\n')
        entry={'camera':camera,'physicalExtent':physical,'renders':{},'referenceBox':str(BOX)}
        for style in a.styles.split(','):
            result=RenderSplat.execute(splat=splat,width=a.size,height=a.size,frames=1,
                            splat_scale=1.0,sharpen=1.0,headlight_shading=0.,
                            opacity_threshold=a.opacity_threshold,background='#808080',
                            render_style=style,camera_info=camera)
            v,mask=result.result
            v=v[0].detach().float().cpu().numpy()
            m=mask[0].detach().float().cpu().numpy()
            image(path/f'{style}.png',v)
            if style=='color':
                image(path/'alpha.png',m)
                image(path/'rgba.png',np.concatenate([v,m[...,None]],axis=-1))
                image(path/'composited.png',v*m[...,None]+.5*(1-m[...,None]))
                np.save(path/'alpha.npy',m.astype('f4'),allow_pickle=False)
            if style=='depth':
                np.save(path/'depth-relative.npy',v[...,0].astype('f4'),allow_pickle=False)
            entry['renders'][style]={'image':str(path/f'{style}.png'),
                         'meanAlpha':float(m.mean()),'foregroundPixels':int((m>=.5).sum()),
                         'colorRange':[float(v.min()),float(v.max())]}
            print(fid,style,entry['renders'][style],flush=True)
        entry['seconds']=round(time.monotonic()-started,3)
        stats['faces'][fid]=entry
        (out/'status.json').write_text(json.dumps(stats,indent=2)+'\n')
    print('COMPLETE',str(out),flush=True)

if __name__=='__main__':main()
