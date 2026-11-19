"""CPU-only integration probe against the actual installed native H3 node."""
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

node=H3ReferenceConditioning();vae=Vae()
inputs={'clip':Clip(),'vae':vae,'prompt':'Pictures 1 to 5, unchanged.','width':96,'height':64,
        **{f'image_{i}':torch.full((1,64,96,3),i/10) for i in range(1,6)}}
all_cond,all_latent=node.condition(**inputs)
all_calls=list(vae.calls);vae.calls.clear()
split_cond,split_latent=node.condition(**inputs,reference_latents='color_first_two')
assert len(all_calls)==5 and len(vae.calls)==2
assert torch.equal(all_cond[0][0],split_cond[0][0])
assert torch.equal(all_latent['samples'],split_latent['samples'])
a=all_cond[0][1]['minimax_refs'];b=split_cond[0][1]['minimax_refs']
assert len(a)==5 and len(b)==2
for i in range(2):
 assert torch.equal(a[i]['latent'],b[i]['latent'])
 assert a[i]['latent_h']==b[i]['latent_h'] and a[i]['latent_w']==b[i]['latent_w']
assert split_cond[0][1]['picture_label']=='Picture %d: '
assert not torch.cuda.is_initialized()
report={'status':'passed','nativeModule':native.__file__,'allVaeReferenceCount':len(a),
        'colorVaeReferenceCount':len(b),'vlmPictureCount':all_cond[0][0].shape[1],
        'vlmEmbeddingExact':True,'firstTwoReferenceLatentsExact':True,'outputLatentExact':True,
        'pictureNumberLabelPreserved':True,'cudaInitialized':torch.cuda.is_initialized(),
        'nativeReferenceSizesForCpuTest':[32,64],'checkedEpoch':time.time()}
(RUN/'D/native-api-validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
