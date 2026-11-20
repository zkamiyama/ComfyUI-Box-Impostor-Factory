#!/usr/bin/env python3
"""Capture official unchanged Inspector Box viewers serially and encode 3 columns."""
import argparse,json,subprocess,sys,shutil,time,os
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--capture-script',type=Path,required=True)
    p.add_argument('--playwright-root',type=Path,required=True)
    p.add_argument('--ffmpeg',type=Path,required=True)
    a=p.parse_args()
    root=a.root.resolve()
    models=json.loads((root/'models.json').read_text())['models']
    browser_env=dict(os.environ)
    for name in ('DISPLAY','WAYLAND_DISPLAY'):
        browser_env.pop(name,None)
    (root/'capture-environment.json').write_text(json.dumps({'removedForChildOnly':['DISPLAY','WAYLAND_DISPLAY'],
        'reason':'Stale display variables route ANGLE SwiftShader to a nonexistent XCB display.',
        'htmlModified':False,'captureScriptModified':False},indent=2)+'\n')
    entries={}
    for item in sorted(models,key=lambda i:{'idol20':0,'idol50':1,'tripo20':2}[i['id']]):
        name=item['id']; viewer=root/'final'/(item['file']+'_Box512.inspector.html')
        if not viewer.is_file():raise FileNotFoundError(viewer)
        output=root/'capture'/name
        command=['node',str(a.capture_script),'--viewer',str(viewer),'--reference',str(root/'common-frame/manifest.json'),
                 '--out',str(output),'--frames','240','--size','512','--mode','default',
                 '--playwright-root',str(a.playwright_root)]
        print(json.dumps({'capture':name,'started':True}),flush=True)
        with (root/'logs'/('capture-'+name+'.log')).open('w') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,env=browser_env)
        capture=json.loads((output/'capture.json').read_text())
        assert capture['completed'] and not capture['errors']
        entries[name]={'label':item['label'],'capture':str(output/'capture.json')}
        print(json.dumps({'capture':name,'completed':True,'frames':len(capture['frames'])}),flush=True)
    specification=root/'box-comparison-inputs.json'
    specification.write_text(json.dumps({'items':[entries[m['id']] for m in models]},indent=2)+'\n')
    command=[sys.executable,str(root/'scripts/encode_box_comparison.py'),'--captures',str(specification),
             '--output',str(root/'media/box'),'--ffmpeg',str(a.ffmpeg)]
    with (root/'logs/encode-box.log').open('w') as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    shutil.copyfile(root/'media/box/comparison.mp4',root/'final/Idol_vs_TripoSplat_Box512.mp4')
    for angle in (0,90,180,270):
        shutil.copyfile(root/('media/box/poster-yaw%03d.png'%angle),
                        root/('final/Idol_vs_TripoSplat_Box512_%03d.png'%angle))
    print('COMPLETE '+str(root/'media/box/comparison.json'),flush=True)

if __name__=='__main__':main()

