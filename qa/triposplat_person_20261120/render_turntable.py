#!/usr/bin/env python3
"""True orthographic 3DGS turntable: all frames rendered from the saved Gaussian field."""
import hashlib, json, math, subprocess, sys, time
from pathlib import Path
import numpy as np
from PIL import Image,ImageFont,ImageDraw
import torch
ROOT=Path('/mnt/f/_dev/box-impostor-lab/experiments/triposplat-person-20261120')
COMFY=Path('/home/dev/Projects/box-impostor-lab/ComfyUI')
sys.path[:0]=[str(COMFY),str(COMFY/'custom_nodes/box_impostor_factory')]
from box_nodes.splat import camera_plan
from comfy_api.latest import Types
from comfy_extras.nodes_gaussian_splat import RenderSplat,TransformSplat
from comfy_extras.nodes_camera import _rotate_camera_info

CACHE=Path('/mnt/f/mnt/output/box_impostor/.stages/triposplat/ecb8433166915206286d50ab299fe2faa73d7e7b028f8ea19fa36cd3d41fad54')
BOX=Path('/mnt/f/mnt/output/depth-refinement-20261120/box_impostor/019253e1edc661260f7775735b8bcaabcd9fe2a23a8b6ecd1e8227a7dd8a42d8')
FRAMES=120;SIZE=640;FPS=24
FFMPEG='/mnt/f/_dev/box-impostor-lab/experiments/sdf-morph-20261120/tools/ffmpeg-linux-x86_64-v7.0.2'
OUT=ROOT/'media'/'a-3dgs-turntable'
OUT.mkdir(parents=True,exist_ok=True)
with np.load(CACHE/'splat.npz',allow_pickle=False) as f:
    splat=Types.SPLAT(*[torch.from_numpy(f[k].copy()) for k in ('positions','scales','rotations','opacities','sh')])
splat=TransformSplat.execute(splat,0,0,0,0,90,0,1,1,1).result[0]
manifest=json.loads((BOX/'manifest.json').read_text())
face=manifest['faces']['+Z']
camera,_,_=camera_plan(manifest,'+Z','canonical_box')
phys=[abs(face['frustum'][1]-face['frustum'][0]),abs(face['frustum'][3]-face['frustum'][2])]
dist=np.linalg.norm(np.array(list(camera['position'].values()))-np.array(list(camera['target'].values())))
camera['fov']=2*math.degrees(math.atan(max(phys)/(2*dist)))
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',19)
started=time.monotonic()
for i in range(FRAMES):
    yaw=360.*i/FRAMES
    frame=OUT/f'frame-{i:04d}.png'
    if frame.exists() and frame.stat().st_size>2000:
        continue
    cam=_rotate_camera_info(camera,(0.,1.,0.),yaw)
    color,alpha=RenderSplat.execute(splat,SIZE,SIZE,1,1.,1.,0.,0.,'#202020','color',camera_info=cam).result
    v=color[0].float().cpu().numpy();a=alpha[0].float().cpu().numpy()
    rgb=np.uint8(np.rint(np.clip(v,0,1)*255))
    rendered=Image.fromarray(rgb)
    dst=Image.new('RGB',(SIZE,SIZE+48),(23,27,32))
    dst.paste(rendered,(0,48))
    draw=ImageDraw.Draw(dst)
    draw.text((16,13),f'TripoSplat / native 3DGS / yaw {yaw:.0f} deg',font=font,fill=(235,240,245))
    dst.save(frame)
    if i%12==0: print('FRAME',i,'/',FRAMES,'elapsed',round(time.monotonic()-started,1),flush=True)
video=ROOT/'media'/'a-triposplat-turntable.mp4'
subprocess.run([FFMPEG,'-hide_banner','-loglevel','warning','-y','-framerate',str(FPS),'-i',str(OUT/'frame-%04d.png'),
 '-frames:v',str(FRAMES),'-an','-c:v','libx264','-preset','fast','-crf','19','-pix_fmt','yuv420p',
 '-movflags','+faststart',str(video)],check=True)
digest=hashlib.sha256(video.read_bytes()).hexdigest()
record={'frames':FRAMES,'fps':FPS,'video':str(video),'sha256':digest,'bytes':video.stat().st_size,
        'method':'native RenderSplat, 120 distinct camera positions, no interpolation',
        'cameraRotation':'360 degrees around target, 3 degrees per frame','splatRotationNativeY':90,
        'seconds':round(time.monotonic()-started,2),'sourceSplat':str(CACHE/'splat.npz')}
(ROOT/'analysis/turntable-audit.json').write_text(json.dumps(record,indent=2)+'\n')
print('DONE',json.dumps(record,indent=2),flush=True)
