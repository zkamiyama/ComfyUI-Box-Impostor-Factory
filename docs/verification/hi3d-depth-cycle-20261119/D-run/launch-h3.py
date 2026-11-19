"""Foreground H3 restart for D, executed only after the C display milestone."""
import hashlib
import json
import os
from pathlib import Path
import socket
import time

RUN=Path(__file__).resolve().parents[1]
LAB=Path('/home/dev/Projects/box-impostor-lab')
REPO=LAB/'ComfyUI/custom_nodes/box_impostor_factory'

def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def read(path):return json.loads(Path(path).read_text())

def main():
    recipe=read(RUN/'D/prepared.json');experiment=recipe['experiment']
    if recipe['branch']!='D' or experiment['vlmPictureIndices']!=[1,2,3,4,5] or experiment['vaePictureIndices']!=[1,2]:
        raise ValueError('Unexpected D conditioning recipe.')
    if digest(REPO/'box_nodes/h3_reference_conditioning.py')!=experiment['wrapperSha256']:
        raise ValueError('The D conditioning implementation changed after preparation.')
    if digest(LAB/'ComfyUI/comfy_extras/nodes_minimax_h3.py')!=experiment['nativeConditioningSha256']:
        raise ValueError('The native H3 conditioning implementation changed.')
    if digest(RUN/'B/prepared.json')!=experiment['sourceRecipeSha256']:
        raise ValueError('The frozen B preparation changed.')
    for picture in recipe['faces'][experiment['targetFace']]['pictures'].values():
        if digest(picture['path'])!=picture['sha256']:
            raise ValueError('A frozen B conditioning PNG changed.')
    sock=socket.socket();sock.settimeout(1)
    try:
        if sock.connect_ex(('127.0.0.1',8201))==0:raise RuntimeError('Port 8201 is already in use.')
    finally:sock.close()
    profile_path=RUN/'server-process-w4a8.json';profile=read(profile_path)
    command=list(profile['args']);index=command.index('-c')+1
    old_log=str(RUN/'comfy-w4a8.log');new_log=RUN/'comfy-w4a8-D.log'
    if command[index].count(old_log)!=1:raise ValueError('Unexpected original H3 profile.')
    command[index]=command[index].replace(old_log,str(new_log))
    if new_log.exists() or (RUN/'server-process-w4a8-D.json').exists():
        raise FileExistsError('Review existing D launch evidence before repeating.')
    metadata={**profile,'pid':os.getpid(),'args':command,'log':str(new_log),'startedEpoch':time.time(),
        'profileSource':str(profile_path),'profileSourceSha256':digest(profile_path),
        'bootstrapSha256':digest(__file__),'D_preparedSha256':digest(RUN/'D/prepared.json'),
        'onlyProfileChange':'Dedicated D log path; H3 flags and recorded environment unchanged'}
    (RUN/'server-process-w4a8-D.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps({'foregroundPid':os.getpid(),'log':str(new_log)}),flush=True)
    os.chdir(LAB/'ComfyUI');os.execve(command[0],command,{**os.environ,**profile['environment']})

if __name__=='__main__':main()
