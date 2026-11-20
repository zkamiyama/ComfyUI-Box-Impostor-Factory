import json,hashlib,urllib.request,uuid,time
from pathlib import Path
R=Path('/mnt/f/_dev/box-impostor-lab/experiments/triposplat-person-20261120')
input_file=R/'comfy-evidence/triposplat-UrZVv8/prompt.json'
j=json.loads(input_file.read_text())
keep=set()
def visit(k):
 if k in keep:return
 keep.add(k)
 for v in j[k]['inputs'].values():
  if isinstance(v,list) and len(v)==2 and str(v[0]) in j:visit(str(v[0]))
visit('21')
graph={k:j[k] for k in sorted(keep,key=int)}
assert '16' not in graph and '17' not in graph and '13' in graph and '12' in graph
p=R/'minimal-prompt.json'
p.write_text(json.dumps(graph,indent=2)+'\n')
req=urllib.request.Request('http://127.0.0.1:8198/prompt',data=json.dumps({'prompt':graph,'client_id':'tripo-study-'+str(uuid.uuid4())}).encode(),headers={'Content-Type':'application/json'})
with urllib.request.urlopen(req,timeout=90) as response:answer=json.load(response)
(R/'inference-receipt.json').write_text(json.dumps({'promptId':answer.get('prompt_id'),'prompt':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'serverReply':answer},indent=2)+'\n')
print('SUBMITTED',answer,flush=True)
