import hashlib,json,pathlib,subprocess,time,urllib.request
root=pathlib.Path('/mnt/f/_dev/box-impostor-lab/experiments/idol-triposplat-20261120')
worktree=pathlib.Path('/home/dev/Projects/box-impostor-lab/worktrees/idol-triposplat-20261120')
if not worktree.exists():
    subprocess.run(['git','-C','/home/dev/Projects/box-impostor-lab/worktrees/zipsplat-qwen-20261120','worktree','add','-b','experiment/idol-triposplat-20261120',str(worktree),'2f142bd8edf1e20d308a32ce1987425619104d91'],check=True)
revision='e907d8dca6c465735eaf4ce9b3a3c2c6401d5705'
models=pathlib.Path('/home/dev/Projects/box-impostor-lab/ComfyUI/models')
files=[('diffusion_models/idol_fp16.safetensors',741106994,'9abf6a99fc6bdad701901b12de8fe1b256f02cae4948f7489d25d86b11b52a93'),('vae/idol_decoder_fp16.safetensors',576148442,'6a146a1e6b2653f74422f7ea835eb8715c547c7de8fc4545051f15c247ae54a9')]
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
records=[]
for rel,size,want in files:
    dst=models/rel
    if dst.exists():
        got=sha(dst)
        if dst.stat().st_size!=size or got!=want:raise RuntimeError('Existing requested model differs: '+rel)
        elapsed=0
    else:
        tmp=dst.with_suffix(dst.suffix+'.idol-download')
        started=time.monotonic();total=0;last=started
        url=f'https://huggingface.co/Comfy-Org/Idol/resolve/{revision}/{rel}'
        print(json.dumps({'stage':'download','file':rel,'bytes':size}),flush=True)
        with urllib.request.urlopen(url,timeout=90) as response,tmp.open('wb') as f:
            while chunk:=response.read(8*1024*1024):
                f.write(chunk);total+=len(chunk)
                if time.monotonic()-last>15:
                    print(json.dumps({'stage':'download-progress','file':rel,'received':total,'total':size}),flush=True);last=time.monotonic()
        got=sha(tmp)
        if tmp.stat().st_size!=size or got!=want:raise RuntimeError('Downloaded checksum mismatch: '+rel)
        tmp.replace(dst);elapsed=time.monotonic()-started
    rec={'file':str(dst),'bytes':size,'sha256':got,'download_seconds':elapsed,'source_repo':'Comfy-Org/Idol','source_revision':revision}
    records.append(rec);print(json.dumps({'stage':'verified','record':rec}),flush=True)
for rel in ['clip_vision/dino_v3_vit_h.safetensors','vae/flux2-vae.safetensors','diffusion_models/triposplat_fp16.safetensors','vae/triposplat_vae_decoder_fp16.safetensors']:
    p=models/rel;records.append({'file':str(p),'bytes':p.stat().st_size,'sha256':sha(p),'existing':True})
(root/'model-files.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps({'stage':'complete','worktree':str(worktree),'models':len(records)}),flush=True)
