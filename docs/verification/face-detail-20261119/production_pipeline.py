"""Run the implemented material/consistency node on both newly aligned Boxes."""
import asyncio, json, os, sys, time
from pathlib import Path
from unittest.mock import patch
LAB=Path('/home/dev/Projects/box-impostor-lab');COMFY=LAB/'ComfyUI';REPO=COMFY/'custom_nodes/box_impostor_factory'
sys.path[:0]=[str(REPO),str(COMFY)]
import torch
torch.cuda._device_count_nvml=lambda:-1
import comfy.options
comfy.options.enable_args_parsing()
import nodes
from box_nodes import material_pipeline,store
loop=asyncio.new_event_loop()
assert loop.run_until_complete(nodes.load_custom_node(str(COMFY/'custom_nodes/ComfyUI-SuperPBR')))
assets={'1024':'91eccb2e8a3c9958caecc89fff743fec3e26dc34b1808aadf5f182624180b006','512':'2dd128ae1625a504fc2028fb678a48aea2e41612df33a737c10cd0f422daff3c'}
report={}
with patch.object(store,'root',return_value=COMFY/'output/box_impostor'):
 for resolution,asset in assets.items():
  t=time.monotonic();print('START',resolution,flush=True)
  output,=material_pipeline.GenerateFaceMaterials().generate(asset,source='supermat',face='all',
       edit_resolution=int(resolution),view_consistency='front',detail_strength=.75)
  m=store.validate(store.asset_dir(output),output)
  report[resolution]={'input':asset,'asset':output,'elapsedSeconds':time.monotonic()-t,
       'node':'BIGenerateFaceMaterials','source':'supermat','editResolution':int(resolution),
       'detailStrength':.75,'viewConsistency':m['provenance']['viewConsistency'],
       'textures':m['textures'],'allFacesAccepted':all(f.get('edit',{}).get('status')=='accepted' for f in m['faces'].values())}
  (LAB/'results/face-detail-recovery/production-pipeline.json').write_text(json.dumps(report,indent=2))
  print('DONE',resolution,output,report[resolution]['elapsedSeconds'],flush=True)
