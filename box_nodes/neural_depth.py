"""AI-inferred per-view depth detail anchored to the existing Box geometry.

MoGe-2 derives new depth from the already generated H3/Qwen RGB for every view.
Its smooth/global shape is discarded; only conservative local depth relief moves
visible source-mesh vertices, then the original six-face baker is run again.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image
from scipy import ndimage

from . import depth_mesh, faces, store
from .depth_pipeline import bound_paired_depth, vertex_displacements
from .revision import DEPTH_ENCODING, RebakeProxyRevision, matrix

METHOD = 'moge2_detail_v1'
FACE_SIZES = faces.SLUGS


def _resize2d(array, shape, *, order=1):
    img=Image.fromarray(np.asarray(array,dtype=np.float32), mode='F')
    return np.asarray(img.resize((shape[1],shape[0]),
                      Image.Resampling.BICUBIC if order==3 else Image.Resampling.BILINEAR),
                      dtype=np.float32)


def project_prediction(full, canvas, resolution):
    """Invert the actual H3 guide letterbox and crop to one Box tile."""
    if full.ndim!=2 or not np.isfinite(full[np.isfinite(full)]).all():
        raise ValueError('Neural depth has invalid samples')
    if canvas is None:
        raise ValueError('Neural detail requires recorded generated-face guide canvas')
    x,y,w,h=canvas['guideRect']
    s=canvas['size']
    H,W=full.shape
    if H!=W or min(w,h,s)<1:
        raise ValueError('Invalid generation canvas/depth shape')
    x0,y0,x1,y1=[round(c*W/s) for c in (x,y,x+w,y+h)]
    crop=full[y0:y1,x0:x1]
    if not crop.size:
        raise ValueError('Empty generated depth crop')
    valid=np.isfinite(crop)
    values=np.where(valid,crop,0)
    weighted=_resize2d(values,(resolution,resolution))
    coverage=_resize2d(valid.astype(np.float32),(resolution,resolution))
    return np.divide(weighted,coverage,out=np.full_like(weighted,np.nan),where=coverage>.98)


def neural_detail_delta(coarse, model, alpha, *, max_delta=.025, sigma=10.,
                        strength=.24, margin=4):
    """High-pass neural depth, normalized with the predictor's own depth span.

    The original coarse depth remains the anchor. Silhouette/holes never move.
    Output is signed axial depth in the baker's normalized 0-near/1-far units.
    """
    z=np.asarray(coarse,dtype=np.float32)
    pred=np.asarray(model,dtype=np.float32)
    mask=np.asarray(alpha,dtype=np.float32)
    if pred.shape!=z.shape or mask.shape!=z.shape:
        raise ValueError('Coarse, predicted and coverage arrays must match')
    valid=(mask>.55)&np.isfinite(z)&np.isfinite(pred)
    result=np.zeros_like(z)
    if valid.sum()<128:
        return result, {'validPixels':int(valid.sum()),'changedPixels':0,'reason':'insufficient-coverage'}
    # Do not import model hallucinations from behind the foreground silhouette.
    interior=ndimage.distance_transform_edt(valid).astype(np.float32)
    weight=np.clip(interior/margin,0,1)
    # Normalized convolution protects the high-pass from invisible/background pixels.
    smooth_num=ndimage.gaussian_filter(np.where(valid,pred,0),sigma=sigma)
    smooth_den=ndimage.gaussian_filter(valid.astype(np.float32),sigma=sigma)
    base=np.divide(smooth_num,smooth_den,out=np.zeros_like(z),where=smooth_den>.01)
    q05,q95=np.quantile(pred[valid],[.05,.95])
    spread=float(q95-q05)
    if spread<1e-5:
        return result,{'validPixels':int(valid.sum()),'changedPixels':0,'reason':'uniform-prediction'}
    residual=np.where(valid,pred-base,0)
    # Direction is depth-positive inward for both MoGe and the Box face capture.
    candidate=np.clip(residual*strength/spread,-max_delta,max_delta)
    result[:]=candidate*weight
    result[~valid]=0
    changed=int(np.count_nonzero(np.abs(result)>1e-6))
    return result, {'validPixels':int(valid.sum()),'changedPixels':changed,
        'predictorDepthSpan':round(spread,6),'peakAbsDelta':round(float(np.max(np.abs(result))),6),
        'p95AbsDelta':round(float(np.quantile(np.abs(result[valid]),.95)),6),
        'deltaStd':round(float(np.std(result[valid])),6)}


def deformation_from_predictions(asset, prediction_dir, *, strength=.24, sigma=10.,
                                 max_delta=.025):
    """Deform the source mesh using six AI maps; return GLB bytes and measurements."""
    directory=store.asset_dir(asset)
    original=store.validate(directory,asset,check_bytes=False)
    if not (0 < strength <= 1 and 1 <= sigma <= 40 and 0 < max_delta <= .08):
        raise ValueError('Invalid AI-depth detail strength/sigma/limit')
    r=original['resolution']
    p=Path(prediction_dir)
    pairs=[]
    metrics={}
    for fid in store.FACES:
        info=original['faces'][fid]
        edit=info.get('edit') or {}
        canvas=edit.get('provenance',{}).get('canvas')
        name=FACE_SIZES[fid]
        prediction=p/(name+'.npy')
        if not prediction.is_file() or canvas is None:
            metrics[fid]={'status':'skipped','reason':'missing-ai-face-or-canvas'}
            continue
        if info.get('depthEncoding')!=DEPTH_ENCODING:
            raise ValueError('Requires six first-hit depth captures')
        z=np.fromfile(store.contained(directory,info['depth']),'<f2').reshape(r,r).astype('f4')
        with Image.open(store.contained(directory,info['guide'])) as im:
            alpha=np.asarray(im.getchannel('A'),dtype=np.float32)/255.
        raw=np.load(prediction,allow_pickle=False)
        predicted=project_prediction(raw,canvas,r)
        delta,stat=neural_detail_delta(z,predicted,alpha,strength=strength,sigma=sigma,
                                       max_delta=max_delta)
        other=original['faces'][('+' if fid[0]=='-' else '-')+fid[1]]
        opposition=np.fromfile(store.contained(directory,other['depth']),'<f2').reshape(r,r).astype('f4')
        delta=bound_paired_depth(delta,z,opposition,fid)
        metrics[fid]={**stat,'status':'predicted',
                     'postPairP95':round(float(np.quantile(np.abs(delta),.95)),6)}
        clip=matrix(info['projectionMatrix'])@matrix(info['viewMatrix'])
        pairs.append((clip,np.asarray(info['side']),z,delta))
    if not pairs:
        raise ValueError('No compatible AI depth predictions from generated RGB')
    tight=original['tightBounds']
    def shift(points):
        displacement=vertex_displacements(points,pairs,original['bakeBox'],r)
        for bound in ('min','max'):
            displacement[np.isclose(points,tight[bound],atol=1e-7,rtol=0)]=0
        return np.clip(points+displacement,tight['min'],tight['max'])-points
    mesh,stats=depth_mesh.deform_glb(store.contained(directory,original['proxy']).read_bytes(),shift,local_backtracking=True)
    return mesh,{'method':METHOD,'model':'MoGe-2 ViT-L native','faces':metrics,
                'mesh':stats,'strength':strength,'sigma':sigma,'maxDelta':max_delta}


def predict_faces(asset,stage,model_path,*,inference_python=None):
    from . import store
    directory=store.asset_dir(asset)
    python=inference_python or os.environ.get('BI_MOGE_PYTHON') or sys.executable
    script=store.REPO/'scripts'/'moge_face_depth_infer.py'
    output=stage/'predictions'
    cmd=[python,str(script),'--asset',str(directory),'--model',str(model_path),
         '--output',str(output),'--face','all','--size','768','--level','8']
    env=dict(os.environ)
    comfy_root=str(store.REPO.parents[1])
    env['PYTHONPATH']=comfy_root+(os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    with (stage/'inference.log').open('wb') as log:
        process=subprocess.Popen(cmd,cwd=store.REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                store.check_interrupted()
                time.sleep(.2)
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=8)
        if process.returncode:
            raise RuntimeError('Neural depth inference failed: '+
                 (stage/'inference.log').read_text(errors='replace')[-4500:])
    return output


def freeze_original_appearance(original_id, rebaked_id):
    """Keep source H3/Qwen colors, PBR and normal maps for an A/B-safe Box.

    The new interval/range geometry and six depth captures remain from the AI
    rebake, so only Z/parallax changes. This avoids noisy AI-mesh normals.
    """
    import shutil
    original_dir=store.asset_dir(original_id)
    after_dir=store.asset_dir(rebaked_id)
    original=store.validate(original_dir,original_id)
    rebaked=store.validate(after_dir,rebaked_id)
    if original['resolution']!=rebaked['resolution'] or original['bakeBox']!=rebaked['bakeBox']:
        raise ValueError('Cannot preserve appearance when Box framing changed')
    for fid in store.FACES:
        a,b=original['faces'][fid],rebaked['faces'][fid]
        if a['viewMatrix']!=b['viewMatrix']:
            raise ValueError('Cannot preserve appearance after a camera direction change')
        projectionA=matrix(a['projectionMatrix'])
        projectionB=matrix(b['projectionMatrix'])
        if not np.allclose(projectionA[:2],projectionB[:2],rtol=0,atol=1e-7):
            raise ValueError('Cannot preserve appearance after an XY projection change')
    key=store.key('neural-preserved-appearance-v1',original_id,rebaked_id,
                  original['textures'],original['baseColor'])
    def write(stage):
        shutil.copytree(after_dir,stage,ignore=shutil.ignore_patterns('previews'))
        m=store.validate(stage,rebaked_id,check_bytes=False)
        for name in ('colorMap','normalMap','materialMap','emissiveMap'):
            if name not in original['textures']:
                m['textures'].pop(name,None)
                continue
            old=original['textures'][name]
            path='maps/preserved_'+name+'.bin'
            shutil.copyfile(store.contained(original_dir,old['path']),stage/path)
            m['textures'][name]={**old,'path':path}
        m['maps']=list(original['maps'])
        old=original['baseColor']
        base_path='maps/preserved_baseColor.bin'
        shutil.copyfile(store.contained(original_dir,old['path']),stage/base_path)
        m['baseColor']={**old,'path':base_path}
        m.pop('faceMaterialBase',None)
        m['mode']=original.get('mode','geometry_baseline')
        for fid in store.FACES:
            old_edit=original['faces'][fid].get('edit')
            if old_edit:
                m['faces'][fid]['edit']=old_edit
            else:
                m['faces'][fid].pop('edit',None)
        m['asset_id']=key
        m['parentAsset']=rebaked_id
        m.setdefault('provenance',{})['appearancePreservation']={
            'sourceAsset':original_id,'geometryAsset':rebaked_id,
            'preservedMaps':[k for k in ('colorMap','normalMap','materialMap','emissiveMap') if k in original['textures']],
            'reason':'Compare actual AI-modified depth without color/normal shading changes'}
        (stage/'manifest.json').write_text(json.dumps(m,indent=2,allow_nan=False))
    store.publish(key,write)
    return key


def rebake_with_neural_depth(asset, prediction_dir,backend='auto', *,
                             strength=.08,sigma=10.,max_delta=.008):
    """Rebuild all native Box maps and re-align saved generated face images."""
    directory=store.asset_dir(asset)
    manifest=store.validate(directory,asset)
    data,report=deformation_from_predictions(asset,prediction_dir,strength=strength,
                                             sigma=sigma,max_delta=max_delta)
    if hashlib.sha256(data).hexdigest()==manifest['proxySha256']:
        raise ValueError('Neural depth produced no mesh deformation')
    cache=store.key(METHOD,manifest['proxySha256'],strength,sigma,max_delta,
        [(fid,store.file_hash(Path(prediction_dir)/(FACE_SIZES[fid]+'.npy')))
         for fid in store.FACES if (Path(prediction_dir)/(FACE_SIZES[fid]+'.npy')).is_file()],
        store.file_hash(Path(__file__)),store.file_hash(Path(depth_mesh.__file__)))
    def write(stage):
        output=stage/'proxy.glb'
        output.write_bytes(data)
        metadata={'glbSha256':store.file_hash(output),
          'sourceSha256':manifest.get('provenance',{}).get('originalGlbSha256',manifest['proxySha256']),
          'transform':manifest['transform'],'inverseTransform':manifest['inverseTransform'],
          'tightBounds':manifest['tightBounds'],'refinement':report['mesh']}
        (stage/'canonical.json').write_text(json.dumps(metadata,allow_nan=False))
        (stage/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    stage=faces.cache_dir('neural-depth-mesh',cache,write)
    proxy={'sha256':store.file_hash(stage/'proxy.glb'),'stageKey':cache,
        'path':str(stage/'proxy.glb'),'metadata':str(stage/'canonical.json'),
        'depthRefinement':{**report,'sourceAsset':asset},
        'rotationDegreesXYZ':manifest.get('provenance',{}).get('rotationDegreesXYZ',[0,0,0])}
    ref=manifest.get('reference')
    if ref:
        folder=store.contained(directory,ref['image']).parent
        if (folder/'source.npy').is_file():
            proxy['reference']={**ref,'path':str(folder)}
            if ref.get('camera'):
                proxy['camera']=ref['camera']
                proxy['croppedImage']=str(store.contained(directory,ref['camera']['image']))
    if manifest.get('provenance',{}).get('meshGeneration'):
        proxy['provenance']=manifest['provenance']['meshGeneration']
    revised,_=RebakeProxyRevision().revise(asset,proxy,backend)
    qa=json.loads(store.contained(directory,manifest['qa']).read_bytes()).get('faces',{}) if manifest.get('qa') else {}
    items={}
    for fid,info in manifest['faces'].items():
        edit=info.get('edit')
        if not edit or 'view' not in edit:
            continue
        folder=store.contained(directory,edit['view']).parent
        source=faces.save_view(np.load(folder/'source.npy',allow_pickle=False))
        source['provenance']={**edit.get('provenance',{}),'depthRefinementMethod':METHOD}
        if edit.get('reference'):
            refpath=store.contained(directory,edit['reference']).parent
            source['reference']={'path':str(refpath),'sha256':source['provenance'].get('referenceSha256',manifest.get('reference',{}).get('sha256'))}
        threshold=qa.get(fid,{}).get('thresholds',{})
        aligned=faces.align({fid:source},revised,
                            threshold.get('minIoU',.85),
                            threshold.get('maxShiftFraction',.10),
                            threshold.get('maxScaleChange',.15))
        items.update(aligned['faces'])
    if items:
        revised=faces.compose(revised,{'baseline':revised,'faces':items})
    return freeze_original_appearance(asset,revised),report
def infer_and_rebake(asset, backend='auto', *, model_path=None, inference_python=None):
    """Opt-in ComfyUI entrypoint: infer six neural depth maps, then rebake a real Box."""
    directory=store.asset_dir(asset)
    manifest=store.validate(directory,asset,check_bytes=False)
    generated=[store.contained(directory,info['edit']['view']) for info in manifest['faces'].values()
               if info.get('edit') and info['edit'].get('view')]
    if not generated or not all(p.is_file() for p in generated):
        raise ValueError('MoGe detail needs archived H3/Qwen-generated RGBA face images')
    if model_path is None:
        import folder_paths
        model_path=os.environ.get('BI_MOGE_MODEL') or folder_paths.get_full_path(
            'geometry_estimation','moge_2_vitl_normal_fp16.safetensors')
    if not model_path or not Path(model_path).is_file():
        raise FileNotFoundError('Missing MoGe-2 model: place moge_2_vitl_normal_fp16.safetensors in ComfyUI/models/geometry_estimation')
    model=Path(model_path)
    source_hashes=[(str(p.relative_to(directory)),store.file_hash(p)) for p in generated]
    dep=store.key('neural-depth-inference-v1',asset,store.file_hash(model),
                  source_hashes,store.file_hash(store.REPO/'scripts'/'moge_face_depth_infer.py'),
                  'size768','level8',inference_python or os.getenv('BI_MOGE_PYTHON',''))
    stage=faces.cache_dir('moge-neural-inference',dep,
                          lambda s:predict_faces(asset,s,model,inference_python=inference_python))
    prediction_dir=stage/'predictions'
    if not (prediction_dir/'inference.json').is_file():
        raise RuntimeError('MoGe depth inference cache lacks an inference report')
    return rebake_with_neural_depth(asset,prediction_dir,backend=backend)[0]


class NeuralDepthDetail:
    """ComfyUI opt-in node; AI depth inference needs no new downloadable weights."""
    CATEGORY='Box Impostor'
    FUNCTION='enhance'
    RETURN_TYPES=('BOX_IMPOSTOR',)
    RETURN_NAMES=('box',)
    DESCRIPTION='MoGe-2 ViT-L predicts per-view depth from generated H3/Qwen RGB, adds bounded local relief to the Pixal3D Z, then rebakes the six-face Box. Original RGB and normals are preserved for stability.'

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'asset':('BOX_IMPOSTOR',),
            'mode':(['bypass','moge2_detail'],),
            'backend':(['auto','webgpu','webgl2'],),
        }}

    def enhance(self,asset,mode='bypass',backend='auto'):
        if mode=='bypass':
            return (asset,)
        if mode!='moge2_detail':
            raise ValueError('Unsupported AI-depth mode')
        return (infer_and_rebake(asset,backend),)
