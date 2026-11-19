"""Opt-in Hi3D geometry refinement between canonicalization and six-face baking.

The original BOX_PROXY is returned unchanged unless explicitly enabled.
Hi3D runs in an independent Python/GPU subprocess and never writes over inputs.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

from . import faces, store


class RefineProxyHi3D:
    CATEGORY='Box Impostor'
    FUNCTION='refine'
    RETURN_TYPES=('BOX_PROXY',)
    RETURN_NAMES=('proxy',)

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'proxy':('BOX_PROXY',),
            'mode':(['bypass','hi3d_512','hi3d_1024'],),
            'steps':('INT',{'default':25,'min':1,'max':50,'step':1}),
            'smoothing':(['none','taubin15','taubin40','taubin80'],),
        }, 'optional': {
            'hi3d_python':('STRING',{'default':''}),
            'cpu_offload':('BOOLEAN',{'default':False}),
        }}

    @staticmethod
    def resolve_python(explicit=''):
        if explicit:
            candidate=explicit
        elif os.getenv('BI_HI3D_PYTHON'):
            candidate=os.environ['BI_HI3D_PYTHON']
        else:
            # Convenience for the isolated 270-PC evaluation layout.
            trial=store.REPO.parents[2]/'third_party'/'.venv-hi3d'/'bin'/'python'
            if not trial.is_file():
                raise RuntimeError(
                    'Set BI_HI3D_PYTHON to the Hi3D Python 3.10 virtualenv, '
                    'or enter the path in the node.')
            candidate=str(trial)
        candidate=os.path.expanduser(candidate)
        resolved=shutil.which(candidate) if '/' not in candidate and '\\' not in candidate else candidate
        if not resolved or not Path(resolved).is_file():
            raise FileNotFoundError('Hi3D Python is not an executable file: '+str(candidate))
        # Resolving a venv interpreter symlink bypasses its pyvenv.cfg and packages.
        return os.path.abspath(resolved)

    @staticmethod
    def invoke(cmd, log):
        # Follows ComfyUI cancellation without blocking or spamming CUDA logs.
        with open(log,'wb') as stream:
            process=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,cwd=store.REPO)
            try:
                while process.poll() is None:
                    store.check_interrupted()
                    time.sleep(.25)
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
            if process.returncode:
                with open(log,'rb') as stream:
                    stream.seek(max(0,Path(log).stat().st_size-7000))
                    error=stream.read().decode(errors='replace')
                raise RuntimeError('Hi3D inference failed: '+error)

    def refine(self,proxy,mode='bypass',steps=25,smoothing='none',hi3d_python='',cpu_offload=False):
        if mode=='bypass':
            return (proxy,)
        options={'none':0,'taubin15':15,'taubin40':40,'taubin80':80}
        if mode not in ('hi3d_512','hi3d_1024') or type(steps) is not int or not 1<=steps<=50 or smoothing not in options:
            raise ValueError('Invalid Hi3D mode, steps, or smoothing option')
        resolution=int(mode.split('_')[-1])
        offload=bool(cpu_offload or resolution==1024)
        smoothing=options[smoothing]
        if not isinstance(proxy,dict) or not {'path','sha256','stageKey','metadata','reference'}<=set(proxy):
            raise ValueError('Hi3D needs a canonical proxy connected to the source BOX_REFERENCE')
        reference=proxy['reference']
        source_image=Path(reference['path'])/'source.png'
        source_mesh=Path(proxy['path'])
        source_meta=Path(proxy['metadata'])
        for item in (source_image,source_mesh,source_meta):
            if not item.is_file():
                raise FileNotFoundError('Missing Hi3D input: '+str(item))
        python=self.resolve_python(hi3d_python)
        infer=store.REPO/'scripts'/'hi3d_infer.py'
        post=store.REPO/'scripts'/'hi3d_postprocess.py'
        cache=store.key('hi3d-refine-proxy-v2-linear-color',proxy['stageKey'],proxy['sha256'],
                        store.file_hash(source_mesh),store.file_hash(source_image),
                        reference['sha256'],resolution,steps,smoothing,offload,
                        python,store.file_hash(infer),store.file_hash(post))
        def build(stage):
            raw=stage/'hi3d-raw.glb'
            command=[python,str(infer),'--source',str(source_mesh),'--reference',
                     str(source_image),'--output',str(raw),'--report',str(stage/'inference.json'),
                     '--steps',str(steps),'--resolution',str(resolution)]
            if offload:
                command.append('--offload')
            self.invoke(command,stage/'inference.log')
            # Load from this extension's files; ComfyUI need not put its path on sys.path.
            import importlib.util
            spec=importlib.util.spec_from_file_location('bi_hi3d_postprocess',post)
            module=importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            mesh,report=module.prepare(source_mesh,raw,smooth=smoothing)
            key_file=module.key_file
            output=stage/'canonical.glb'
            module.export_prepared_glb(mesh,output)
            original=json.loads(source_meta.read_text())
            original['glbSha256']=key_file(output)
            original['tightBounds']={
                'min':mesh.bounds[0].tolist(),'max':mesh.bounds[1].tolist()}
            original['refinement']={
                'model':'hi3d-official/Hi3Dv1.0','resolution':resolution,
                'steps':steps,'smoothing':smoothing,
                'hi3dSha256':key_file(raw),'coarseSha256':proxy['sha256'],
                'colorTransferVersion':module.COLOR_TRANSFER_VERSION}
            (stage/'canonical.json').write_text(json.dumps(original,indent=2,allow_nan=False)+'\n')
            (stage/'report.json').write_text(json.dumps({
                **report,'inference':json.loads((stage/'inference.json').read_text())
            },indent=2,allow_nan=False)+'\n')
        stage=faces.cache_dir('hi3d-refine',cache,build)
        output=stage/'canonical.glb'
        metadata=stage/'canonical.json'
        actual_hash=store.file_hash(output)
        if json.loads(metadata.read_text()).get('glbSha256')!=actual_hash:
            raise RuntimeError('Hi3D cache contains a mismatched canonical GLB')
        return ({**proxy,'stageKey':store.key(proxy['stageKey'],cache,actual_hash),
                 'path':str(output),'metadata':str(metadata),'sha256':actual_hash,
                 'refinement':json.loads((stage/'report.json').read_text())},)
