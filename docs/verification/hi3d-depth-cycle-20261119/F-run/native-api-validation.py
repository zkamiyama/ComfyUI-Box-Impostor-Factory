"""CPU-only native H3 API test; synthetic CLIP/VAE doubles, no model inference."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

LAB=Path('/home/dev/Projects/box-impostor-lab')
REPO=LAB/'ComfyUI/custom_nodes/box_impostor_factory'
RUN=Path(__file__).resolve().parents[1]
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path[:0]=[str(REPO),str(LAB/'ComfyUI')]
import comfy.options
comfy.options.enable_args_parsing()
sys.argv=['h3_cpu_probe','--cpu']
import torch

def forbidden_gpu():
    raise AssertionError('This validation must never initialize CUDA.')
torch.cuda._lazy_init=forbidden_gpu
from comfy_extras import nodes_minimax_h3 as native
from box_nodes.h3_reference_conditioning import H3ReferenceConditioning
native.IMAGE_REF_SHORT_EDGE=32
native.IMAGE_REF_VAE_MAX_SHORT_EDGE=64

class Clip:
    def tokenize(self,prompt,images,picture_label):
        return {'prompt':prompt,'images':images,'label':picture_label}
    def encode_from_tokens_scheduled(self,tokens):
        values=torch.stack([image.mean() for image in tokens['images']]).reshape(1,-1,1)
        return [[values,{'picture_label':tokens['label'],'prompt':tokens['prompt']}]]

class Vae:
    def __init__(self):self.calls=[]
    def encode(self,image):
        self.calls.append(float(image.mean()))
        return image.permute(0,3,1,2)[:,:1,::16,::16].contiguous()

def exact(a,b):
    if isinstance(a,torch.Tensor):assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a:exact(a[key],b[key])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):exact(x,y)
    else:assert a==b

node=H3ReferenceConditioning();vae=Vae();clip=Clip()
images={f'image_{i}':torch.linspace(i/20,(i+1)/20,64*96*3).reshape(1,64,96,3) for i in range(1,6)}
colors={key:images[key] for key in ('image_1','image_2')}
prompt='Picture 1 is the color guide; Picture 2 supplies appearance.'
direct=native.MiniMaxH3ImageConditioning.execute(clip,prompt,96,64,vae=vae,images=colors).result
direct_calls=list(vae.calls);vae.calls.clear()
wrapped=node.condition(clip,prompt,96,64,**colors,reference_latents='color_only',vae=vae)
wrapped_calls=list(vae.calls);vae.calls.clear()
exact(direct,wrapped)
with_unused=node.condition(clip,prompt,96,64,**images,reference_latents='color_only',vae=vae)
exact(wrapped,with_unused);assert len(vae.calls)==2;vae.calls.clear()
d_condition=node.condition(clip,prompt,96,64,**images,reference_latents='color_first_two',vae=vae)
exact(d_condition[0][0][1]['minimax_refs'],wrapped[0][0][1]['minimax_refs'])
exact(d_condition[1],wrapped[1])
assert len(direct_calls)==2 and direct_calls==wrapped_calls
assert wrapped[0][0][0].shape[1]==2 and d_condition[0][0][0].shape[1]==5
assert wrapped[0][0][1]['picture_label']=='Picture %d: '
assert not torch.cuda.is_initialized()
report={'status':'passed','scope':'Actual installed H3 API with synthetic CPU CLIP/VAE doubles; not a model-quality or GPU inference test.',
        'nativeModule':native.__file__,'nativeModuleSha256':hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
        'wrapperSha256':hashlib.sha256((REPO/'box_nodes/h3_reference_conditioning.py').read_bytes()).hexdigest(),
        'nativeTwoReferenceOutputExact':True,'vlmPictureCount':2,'vaeReferenceCount':2,
        'optionalStructuralInputsUnused':True,'firstTwoReferenceLatentsExactToD':True,
        'outputLatentExactToD':True,'pictureNumberLabelPreserved':True,
        'cudaInitialized':torch.cuda.is_initialized(),'nativeReferenceSizesForCpuTest':[32,64],
        'checkedEpoch':time.time()}
(RUN/'F/native-api-validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
