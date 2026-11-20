#!/usr/bin/env python3
"""Qwen Image 2.1 + AnyAngle with GS target RGB, original identity, and GS Z-depth."""
import argparse, hashlib, json, shutil, time, urllib.request, uuid
from pathlib import Path

ROOT=Path('/mnt/f/_dev/box-impostor-lab/experiments/triposplat-person-20261120')
SERVER='http://127.0.0.1:8198'
INPUT=Path('/home/dev/Projects/box-impostor-lab/ComfyUI/input')
OUTPUT=Path('/mnt/f/mnt/output')
TEMPLATE=Path('/mnt/f/_dev/box-impostor-lab/experiments/qwen-six-depth-20261120/Qwen-six-anyangle/pz/prompt.json')
CASES={
 'front':{'guide':ROOT/'render/R2048-yaw90-s1/pZ','view':'front three-quarter view as in source photo','seed':2026112031},
 'yaw045':{'guide':ROOT/'render/R2048-yaw90-s1-camera45/pZ','view':'left-facing side/profile as seen in the target render','seed':2026112032},
 'yaw315':{'guide':ROOT/'render/R2048-yaw90-s1-camera-45/pZ','view':'three-quarter camera view as in the target render','seed':2026112033},
}
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read_url(url):
    with urllib.request.urlopen(url,timeout=20) as response:return json.load(response)
def main():
    p=argparse.ArgumentParser()
    p.add_argument('case',choices=CASES)
    p.add_argument('--no-depth',action='store_true')
    p.add_argument('--steps',type=int,default=10)
    p.add_argument('--depth-reference-size',type=int,default=1024)
    args=p.parse_args()
    if not 1<=args.steps<=50: p.error('--steps must be 1..50')
    if args.depth_reference_size not in (512,1024,2048): p.error('--depth-reference-size 512, 1024 or 2048')
    case=CASES[args.case]
    dest=INPUT/'tripo-20261120'/('2kopt-'+args.case)
    dest.mkdir(parents=True,exist_ok=True)
    cp={'target-color.png':case['guide']/'composited.png',
        'target-depth.png':case['guide']/'depth.png',
        'original.png':ROOT/'inputs/source-original.png'}
    digest={}
    for destname,src in cp.items():
        if not src.is_file():raise FileNotFoundError(src)
        destination=dest/destname
        if destname=='target-depth.png' and args.depth_reference_size!=2048:
            from PIL import Image
            with Image.open(src) as im:
                im.convert('RGB').resize((args.depth_reference_size, args.depth_reference_size), Image.Resampling.BILINEAR).save(destination)
        elif not destination.exists() or sha(destination)!=sha(src):shutil.copy2(src,destination)
        digest[destname]=sha(destination)
    j=json.loads(TEMPLATE.read_text())
    prefix=f'tripo-20261120/2kopt-{args.case}'
    j['1']['inputs']['image']=f'{prefix}/target-color.png'
    j['2']['inputs']['image']=f'{prefix}/original.png'
    if not args.no_depth:
        j['3']={'class_type':'LoadImage','inputs':{'image':f'{prefix}/target-depth.png'}}
        j['10']['inputs']['images.image_3']=['3',0]
    else:
        j['10']['inputs'].pop('images.image_3',None)
    prompt=(
        'Change the camera angle from <image2> to <image1>. '
        'Image 1 is a 3D Gaussian splatting color render of the REQUIRED target camera viewpoint. '
        'Image 2 is the original reference photograph of the same person, to preserve identity, natural eyes, nose, mouth, '
        'blonde hair, gray sweater dress, skin colors, shoes and details. '
        +('Image 3 is the corresponding black-background grayscale Z-depth image of image 1: brighter means nearer to the camera. '
          'Use this image only to understand silhouette, occlusion and 3D relief; NEVER copy its gray values as material colors. '
          if not args.no_depth else '')
        + 'Keep the exact pose, clothing boundaries, viewing direction, visible facial features, head direction and occlusions '
          'shown in image 1, even when the reference face in image 2 looks toward the camera. '
          'The intended target view is '+case['view']+'. '
          'Do not add a frontal face in a profile or show duplicated eyes, mouths or noses. '
          'Generate a detailed full-resolution 2048 pixel square photographic image. Use original photo texture details for eyes, face, hair, skin and clothing while retaining the target-view camera. Remove the soft 3DGS watercolor blur; generate crisp coherent texture and anatomically plausible features ONLY visible in this camera. '
          'One single photorealistic full-body person on a uniform neutral gray background; exact framing; '
          'no other subjects, panels, text, graphics, depth-map colors or diagrams.'
    )
    j['10']['inputs']['prompt']=prompt
    j['10']['inputs']['resolution']=0  # keep 2048 first/third image and native 1024 original reference; first image determines 2048 latent
    j['12']['inputs']['steps']=args.steps
    j['9']['inputs']['dtype']='int8'  # reduce cache RAM/VRAM for 2K while retaining 2K model output
    j['12']['inputs']['seed']=case['seed']+1000+(300 if args.no_depth else 0)
    trial_tag=f'2kopt-{args.steps}-nodepth' if args.no_depth else f'2kopt-{args.steps}-depth3'
    j['14']['inputs']['filename_prefix']=f'tripo20261120/qwen-{args.case}-{trial_tag}/raw'
    j['18']['inputs']['filename_prefix']=f'tripo20261120/qwen-{args.case}-{trial_tag}/mask'
    work=ROOT/'qwen2k'/f'{args.case}-{trial_tag}'
    work.mkdir(parents=True,exist_ok=True)
    text=work/'prompt.json'
    text.write_text(json.dumps(j,indent=2)+'\n')
    request=urllib.request.Request(SERVER+'/prompt',data=json.dumps({'prompt':j,'client_id':str(uuid.uuid4())}).encode(),
        headers={'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(request,timeout=90) as response: submitted=json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f'ComfyUI HTTP {error.code}: {error.read().decode(errors="replace")[:5000]}') from error
    pid=submitted['prompt_id']
    (work/'submitted.json').write_text(json.dumps({'promptId':pid,'model':'QwenImage2.1+AnyAngle',
                      'targetView':args.case,'inputSHA256':digest,'promptSHA256':sha(text),
                      'test':'three-reference' if not args.no_depth else 'two-reference'},indent=2)+'\n')
    print('QUEUED',args.case,pid,flush=True)
    start=time.monotonic();last=-100
    while time.monotonic()-start<5400:
        history=read_url(SERVER+'/history/'+pid)
        if pid in history:
            entry=history[pid]
            (work/'history.json').write_text(json.dumps(entry,indent=2)+'\n')
            status=entry.get('status',{})
            if not status.get('completed') or status.get('status_str')=='error':
                raise RuntimeError('ComfyUI inference failed: '+str(status)[-1400:])
            files={}
            for key,kind in [('14','generated'),('18','mask')]:
                images=entry.get('outputs',{}).get(key,{}).get('images',[])
                if not images:raise RuntimeError('Missing '+kind+' SaveImage output')
                value=images[-1]
                src=OUTPUT/value.get('subfolder','')/value['filename']
                dst=work/(kind+'.png')
                shutil.copy2(src,dst)
                files[kind]={'path':str(dst),'sha256':sha(dst),'bytes':dst.stat().st_size}
            result={'case':args.case,'promptId':pid,'seconds':time.monotonic()-start,'images':files,
                    'model':'Qwen Image 2.1 int8 + AnyAngle LoRA 1.0','noDepth':args.no_depth,
                    'inputsSHA256':digest,'promptSHA256':sha(text)}
            (work/'result.json').write_text(json.dumps(result,indent=2)+'\n')
            print('COMPLETE',json.dumps(result,indent=2),flush=True)
            return
        elapsed=int(time.monotonic()-start)
        if elapsed-last>=20:
            print('WORKING',args.case,'seconds',elapsed,flush=True)
            last=elapsed
        time.sleep(3)
    raise TimeoutError('Qwen trial timed out')

if __name__=='__main__': main()
