#!/usr/bin/env python3
"""Isolated H: original coarse mesh + generated +Z Hi3D reference + frozen F RGB.

Defaults to a metadata-only plan. GPU inference/native baking require explicit
stages. B's legacy postprocessor is a hashed snapshot, never the production file.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
COMFY = REPO.parents[1]
LAB = COMFY.parent
B_ASSET = 'ce53eb5429973a519e264d5752fd90348697797cd50f0903a4993076398ba1a2'
COARSE_ASSET = 'd168a94ac6eb0105fc2315a0bdd392846a9edc3ad9e3a19a87ab8eb3c7ef0cf9'
METHOD = 'hi3d-generated-front-reference-with-frozen-f-appearance-v1'
SNAPSHOT = REPO/'docs/verification/hi3d-depth-cycle-20261119/H-preparation/hi3d_postprocess_b.py'
SNAPSHOT_SHA = 'f7daf0272926d51acbedcf6c188317eeafcafb6a977269c9a1f38d6828b119a4'
SNAPSHOT_COMMIT = '873ba9d90bfd794498a3211d0ceeaf8b003f7c56'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def runtime(run):
    sys.path.insert(0, str(REPO));sys.path.insert(1, str(COMFY))
    from comfy.cli_args import args
    args.cpu = True  # CPU parent; only the explicit Hi3D child may initialize CUDA
    import folder_paths
    folder_paths.set_output_directory(str(run/'output'))
    from box_nodes import store, faces, depth_mesh
    return store, faces, depth_mesh


def legacy_postprocess():
    if digest(SNAPSHOT) != SNAPSHOT_SHA:
        raise ValueError('The frozen B-time postprocessor changed.')
    spec = importlib.util.spec_from_file_location('h_legacy_hi3d_postprocess', SNAPSHOT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def glb_semantics(path, depth_mesh):
    """Hash decoded attribute values; distinguish storage metadata from model data."""
    import copy
    import numpy as np
    doc, binary = depth_mesh._decode(Path(path).read_bytes())
    raw_doc = copy.deepcopy(doc)
    for accessor in raw_doc['accessors']:
        accessor.pop('normalized', None)
    meshes = []
    for mesh in doc['meshes']:
        primitives = []
        for prim in mesh['primitives']:
            attributes = {}
            for name, index in {**prim['attributes'], **({'indices':prim['indices']} if 'indices' in prim else {})}.items():
                info = doc['accessors'][index]
                values = depth_mesh._accessor(raw_doc, binary, index, info['type'], {5121,5123,5125,5126})
                attributes[name] = {
                    'componentType': info['componentType'], 'type': info['type'],
                    'normalized': info.get('normalized',False), 'count': info['count'],
                    'decodedBytesSha256': hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()}
            primitives.append({'mode':prim.get('mode',4),'material':prim.get('material'),
                               'attributes':attributes})
        meshes.append(primitives)
    return {
        'meshes':meshes,
        'nodes':[{k:v for k,v in node.items() if k not in ('name','extras')} for node in doc['nodes']],
        'scenes':[{k:v for k,v in scene.items() if k not in ('name','extras')} for scene in doc['scenes']],
        'scene':doc.get('scene',0), 'materials':doc.get('materials',[]),
        'textures':doc.get('textures',[]), 'images':doc.get('images',[]),
    }


def execution_environment_proof(run):
    success = read(run/'B/hi3d-process.json')
    env = success['environment']
    materialization = read(run/'B/model-materialization.json')
    rows = []
    for item in materialization['files']:
        path, blob = Path(item['regularFile']), Path(item['sourceBlob'])
        current = path.stat()
        row = {'path':str(path), 'regularFile':path.is_file() and not path.is_symlink(),
               'bytes':current.st_size, 'expectedBytes':item['bytes'],
               'sameHardlinkedFile':os.path.samefile(path,blob),
               'bVerifiedSha256':item['sha256']}
        if not row['regularFile'] or row['bytes'] != row['expectedBytes'] or not row['sameHardlinkedFile']:
            raise ValueError('Hi3D model materialization differs from successful B: '+str(path))
        rows.append(row)
    if env['HF_HUB_CACHE'] != materialization['materializedHub']:
        raise ValueError('B model cache provenance differs.')
    code = (REPO/'scripts/hi3d_infer.py').read_text()
    guard = code.index('torch.cuda._device_count_nvml=lambda:-1')
    cuda = code.index('torch.cuda.is_available()')
    pipeline_import = code.index('from sparc3d import Sparc3DPipeline')
    if not guard < min(cuda,pipeline_import):
        raise ValueError('WSL NVML guard must precede pipeline import and CUDA initialization.')
    return {
        'successfulBProcessEvidence':str(run/'B/hi3d-process.json'),
        'successfulBProcessEvidenceSha256':digest(run/'B/hi3d-process.json'),
        'pythonExecutable':env['BI_HI3D_PYTHON'], 'executablePathResolved':False,
        'sameRegularModelHub':env['HF_HUB_CACHE'], 'modelFiles':rows,
        'modelHashPolicy':'B verified SHA256 retained; current regular-file/size/hardlink stat only',
        'cpuOffload':True, 'hfOffline':env['HF_HUB_OFFLINE'],
        'wslNvmlGuardBeforeCudaAndPipelineImport':True,
    }


def build_plan(args):
    from PIL import Image
    run, work = args.run.resolve(), args.work.resolve()
    if run == work or not str(work).startswith('/mnt/f/') or not str(run).startswith('/mnt/f/'):
        raise ValueError('H output and cache roots must be separate F-drive directories.')
    assets = run/'output/box_impostor'
    b_dir, appearance_dir, coarse_dir = assets/B_ASSET, assets/args.appearance, assets/COARSE_ASSET
    b, appearance, coarse = [read(p/'manifest.json') for p in (b_dir,appearance_dir,coarse_dir)]
    if appearance['asset_id'] != args.appearance or b['asset_id'] != B_ASSET:
        raise ValueError('Unexpected source manifest identity.')
    frame_keys = ('proxySha256','tightBounds','bakeBox','transform','inverseTransform','resolution',
                  'faceOrder','coordinateSystem')
    if any(b[k] != appearance[k] for k in frame_keys):
        raise ValueError('F appearance must retain the actual B geometry and capture frame.')
    for name, spec in b['textures'].items():
        if name != 'colorMap' and spec['sha256'] != appearance['textures'][name]['sha256']:
            raise ValueError('F changed a non-color native map: '+name)
    for fid, info in b['faces'].items():
        other = appearance['faces'][fid]
        if any(info[k] != other[k] for k in ('viewMatrix','projectionMatrix','frustum','guideSha256','depthSha256')):
            raise ValueError('F changed a B capture: '+fid)
        if other.get('edit',{}).get('status') != 'accepted':
            raise ValueError('F requires six accepted source RGB views.')
    b_front = b_dir/b['faces']['+Z']['edit']['view']
    reference = appearance_dir/appearance['faces']['+Z']['edit']['view']
    if digest(reference) != digest(b_front):
        raise ValueError('B and F must have identical +Z reference PNGs.')
    with Image.open(reference) as image:
        if image.mode != 'RGBA' or image.size != (1024,1024) or image.getchannel('A').getextrema() != (0,255):
            raise ValueError('Reference must be the physical 1024px RGBA source canvas.')
    if appearance['faces']['+Z']['edit']['provenance']['canvas'] != b['faces']['+Z']['edit']['provenance']['canvas']:
        raise ValueError('B/F reference canvases differ.')
    b_run = read(run/'B/hi3d-completed.json')
    if coarse['proxySha256'] != b_run['coarseSha256']:
        raise ValueError('Original coarse mesh identity differs from B.')
    if digest(SNAPSHOT) != SNAPSHOT_SHA:
        raise ValueError('B-time snapshot hash mismatch.')
    plan = {
        'method':METHOD, 'sourceAppearance':args.appearance, 'referenceSourceAsset':B_ASSET,
        'comparison':'F to H; F six source RGB retained, Hi3D reference/geometry changed',
        'sourceB':B_ASSET, 'coarseAsset':COARSE_ASSET,
        'coarseMesh':str(coarse_dir/coarse['proxy']), 'coarseSha256':b_run['coarseSha256'],
        'reference':str(reference), 'referencePngSha256':digest(reference),
        'referenceEqualsBFront':True, 'referenceCanvas':appearance['faces']['+Z']['edit']['provenance']['canvas'],
        'sourceTransform':coarse['transform'], 'sourceInverseTransform':coarse['inverseTransform'],
        'BTransform':b['transform'], 'BInverseTransform':b['inverseTransform'],
        'legacyPostprocess':{'path':str(SNAPSHOT),'sha256':SNAPSHOT_SHA,'commit':SNAPSHOT_COMMIT,
                             'policy':'B-time uint8 color transfer/export retained only for this control'},
        'inference':{'model':'hi3d-official/Hi3Dv1.0','resolution':1024,'steps':25,'seed':0,
                     'smoothing':0,'cpuOffload':True},
        'inferScript':str(REPO/'scripts/hi3d_infer.py'),
        'inferScriptSha256':digest(REPO/'scripts/hi3d_infer.py'),
        'bRawMesh':str(Path(b_run['outputProxy']).parent/'hi3d-raw.glb'),
        'bRawSha256':b_run['report']['hi3d_sha256'],
        'bCanonicalMesh':str(b_dir/b['proxy']), 'bCanonicalSha256':b['proxySha256'],
        'appearanceDirectory':str(appearance_dir), 'coarseDirectory':str(coarse_dir),
        'run':str(run), 'case':str(run/'H'), 'work':str(work),
        'cameraPolicy':'native true bounds/cameras; deterministic parallel-orthographic RGB affine',
        'nativeResolution':1024, 'backend':'webgl2', 'materialMaps':[],
        'executionEnvironment':execution_environment_proof(run),
        'newH3Inference':False, 'freeRegistration':False,
    }
    if plan['sourceTransform'] != plan['BTransform'] or plan['sourceInverseTransform'] != plan['BInverseTransform']:
        raise ValueError('Coarse/B source transforms differ.')
    destination = run/'H/plan.json'
    if destination.exists() and read(destination) != plan:
        raise ValueError('Existing H plan differs; keep this experiment immutable.')
    save(destination,plan)
    save(work/'plan.json',plan)
    return plan


def check_inputs(plan):
    for path, expected in [(plan['coarseMesh'],plan['coarseSha256']),
                           (plan['reference'],plan['referencePngSha256']),
                           (plan['inferScript'],plan['inferScriptSha256']),
                           (SNAPSHOT,SNAPSHOT_SHA)]:
        if digest(path) != expected:
            raise ValueError('Frozen H input changed: '+str(path))


def verify_b(plan, depth_mesh):
    check_inputs(plan)
    work = Path(plan['work'])/'b-replay'
    work.mkdir(parents=True,exist_ok=True)
    if digest(plan['bRawMesh']) != plan['bRawSha256']:
        raise ValueError('The archived B raw Hi3D mesh changed.')
    old = legacy_postprocess()
    mesh, report = old.prepare(plan['coarseMesh'],plan['bRawMesh'],smooth=0)
    output = work/'canonical.glb'
    mesh.export(output)  # exact B-time export path, with no implicit new normal computation
    expected = glb_semantics(plan['bCanonicalMesh'],depth_mesh)
    actual = glb_semantics(output,depth_mesh)
    result = {
        'snapshotSha256':SNAPSHOT_SHA,'sourceSha256':plan['coarseSha256'],
        'rawSha256':plan['bRawSha256'],'expectedSha256':plan['bCanonicalSha256'],
        'replayedSha256':digest(output), 'byteIdentical':digest(output)==plan['bCanonicalSha256'],
        'decodedSemanticsEqual':expected==actual, 'expectedSemantics':expected,'actualSemantics':actual,
        'sourceTransformEqual':plan['sourceTransform']==plan['BTransform'],
        'sourceInverseTransformEqual':plan['sourceInverseTransform']==plan['BInverseTransform'],
        'postprocessReport':report,
    }
    result['passed'] = (result['decodedSemanticsEqual'] and result['sourceTransformEqual']
                        and result['sourceInverseTransformEqual'])
    save(work/'proof.json',result)
    if not result['passed']:
        raise ValueError('B-time prepare/export replay changed geometry, colors, normals, materials or transforms.')
    print(json.dumps({'stage':'verify-b','passed':True,'byteIdentical':result['byteIdentical']}),flush=True)


def environment(plan):
    original = read(Path(plan['run'])/'B/hi3d-process.json')['environment']
    selected = dict(original)
    work = Path(plan['work'])
    for key, folder in [('TMPDIR','tmp'),('XDG_CACHE_HOME','cache'),('TORCH_HOME','torch-cache'),
                        ('TORCH_EXTENSIONS_DIR','torch-extensions'),('TORCHINDUCTOR_CACHE_DIR','inductor-cache'),
                        ('TRITON_CACHE_DIR','triton-cache'),('CUDA_CACHE_PATH','cuda-cache')]:
        selected[key] = str(work/folder)
        Path(selected[key]).mkdir(parents=True,exist_ok=True)
    selected.update(PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',
                    OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    return selected


def infer(plan, store, depth_mesh):
    replay = read(Path(plan['work'])/'b-replay/proof.json')
    if not replay['passed'] or replay['snapshotSha256'] != SNAPSHOT_SHA:
        raise ValueError('A successful B-time replay is required before H.')
    check_inputs(plan)
    work = Path(plan['work'])
    raw = work/'hi3d-raw.glb'
    if raw.exists() or (work/'inference.json').exists():
        raise FileExistsError('H inference output already exists; do not overwrite an attempt.')
    with socket.socket() as sock:
        sock.settimeout(1)
        if sock.connect_ex(('127.0.0.1',8201)) == 0:
            raise RuntimeError('Stop the H3 server before Hi3D inference.')
    selected = environment(plan)
    if os.environ.get('CUDA_VISIBLE_DEVICES') == '':
        raise ValueError('H inference needs CUDA; remove the CPU-test-only CUDA_VISIBLE_DEVICES override.')
    command = [selected['BI_HI3D_PYTHON'],plan['inferScript'],'--source',plan['coarseMesh'],
               '--reference',plan['reference'],'--output',str(raw),'--report',str(work/'inference.json'),
               '--resolution','1024','--steps','25','--offload']
    save(work/'inference-invocation.json',{'command':command,'environment':selected,'settings':plan['inference'],
                                         'sourceAppearance':plan['sourceAppearance'],
                                         'coarseSha256':plan['coarseSha256'],
                                         'referencePngSha256':plan['referencePngSha256'],'cwd':str(REPO)})
    print(json.dumps({'stage':'hi3d-inference','command':command}),flush=True)
    started = time.monotonic()
    with (work/'inference.log').open('w') as log:
        subprocess.run(command,cwd=REPO,env={**os.environ,**selected},stdout=log,stderr=subprocess.STDOUT,check=True)
    inference = read(work/'inference.json')
    if inference['resolution']!=1024 or inference['steps']!=25 or inference['status']!='success':
        raise ValueError('H inference result does not match the requested settings.')
    check_inputs(plan)
    old = legacy_postprocess()
    mesh, post = old.prepare(plan['coarseMesh'],raw,smooth=0)
    output = work/'canonical.glb'
    mesh.export(output)
    semantics = glb_semantics(output,depth_mesh)
    b_semantics = replay['expectedSemantics']
    for current, previous in zip(semantics['meshes'][0],b_semantics['meshes'][0],strict=True):
        if ('NORMAL' in current['attributes']) != ('NORMAL' in previous['attributes']):
            raise ValueError('H changed B-time NORMAL export policy.')
        for key in ('componentType','type','normalized'):
            if current['attributes']['COLOR_0'][key] != previous['attributes']['COLOR_0'][key]:
                raise ValueError('H changed B-time vertex color storage.')
    coarse = read(Path(plan['coarseDirectory'])/'manifest.json')
    metadata = {
        'glbSha256':digest(output),'sourceSha256':coarse['provenance']['originalGlbSha256'],
        'transform':plan['sourceTransform'],'inverseTransform':plan['sourceInverseTransform'],
        'tightBounds':{'min':mesh.bounds[0].tolist(),'max':mesh.bounds[1].tolist()},
        'refinement':{**plan['inference'],'method':METHOD,'coarseSha256':plan['coarseSha256'],
                      'referencePngSha256':plan['referencePngSha256'],'legacyPostprocessSha256':SNAPSHOT_SHA}}
    save(work/'canonical.json',metadata)
    import numpy as np
    samples = np.load(Path(plan['reference']).with_suffix('.npy'),allow_pickle=False)
    ref_key = store.key('face-v1',samples.shape,hashlib.sha256(samples.tobytes()).hexdigest())
    proxy = {
        'path':str(output),'metadata':str(work/'canonical.json'),'sha256':digest(output),
        'stageKey':store.key(METHOD,plan,digest(raw),digest(output)),
        'reference':{'path':str(Path(plan['reference']).parent),'sha256':ref_key,'sourceAsset':plan['sourceAppearance'],
                     'sourceFace':'+Z','role':'Hi3D-generated-image-condition'},
        'provenance':{'method':METHOD,'sourceAppearance':plan['sourceAppearance'],
                      'referencePngSha256':plan['referencePngSha256'],'coarseSha256':plan['coarseSha256'],
                      'legacyPostprocessSha256':SNAPSHOT_SHA},
    }
    save(work/'proxy.json',proxy)
    save(work/'postprocess.json',{'report':post,'semantics':semantics,'inference':inference,
                                 'elapsedSeconds':time.monotonic()-started})
    print(json.dumps({'stage':'hi3d-complete','proxy':str(output),'sha256':digest(output)}),flush=True)


def bake(plan, store):
    check_inputs(plan)
    for key,value in environment(plan).items():
        os.environ[key] = value
    work = Path(plan['work'])
    proxy = read(work/'proxy.json')
    if digest(proxy['path']) != proxy['sha256']:
        raise ValueError('H proxy changed.')
    print(json.dumps({'stage':'native-bake','resolution':1024}),flush=True)
    started = time.monotonic()
    asset = store.baseline(proxy,1024,'webgl2','none')
    manifest = store.validate(store.asset_dir(asset),asset,check_bytes=True)
    save(work/'baseline.json',{'asset':asset,'path':str(store.asset_dir(asset)),
                              'elapsedSeconds':time.monotonic()-started,
                              'tightBounds':manifest['tightBounds'],'bakeBox':manifest['bakeBox'],
                              'actualFaces':manifest['faces']})
    print(json.dumps({'stage':'native-complete','asset':asset}),flush=True)


def compose(plan, store, faces):
    from box_nodes.camera_rebase import rebase_views
    check_inputs(plan)
    work = Path(plan['work'])
    baseline = read(work/'baseline.json')
    source_dir = Path(plan['appearanceDirectory'])
    source = store.validate(source_dir,plan['sourceAppearance'],check_bytes=True)
    target_dir = store.asset_dir(baseline['asset'])
    target = store.validate(target_dir,baseline['asset'],check_bytes=True)
    aligned, evidence = rebase_views(source_dir,source,target_dir,target,work/'camera-rebase')
    asset = faces.compose(baseline['asset'],aligned)
    final_dir = store.asset_dir(asset)
    final = store.validate(final_dir,asset,check_bytes=True)
    checks = {}
    for fid, info in source['faces'].items():
        old_view = store.contained(source_dir,info['edit']['view']).parent
        new_view = store.contained(final_dir,final['faces'][fid]['edit']['view']).parent
        checks[fid] = {name:digest(old_view/name)==digest(new_view/name) for name in ('source.png','source.npy')}
    for name, spec in target['textures'].items():
        if name != 'colorMap' and spec['sha256'] != final['textures'][name]['sha256']:
            raise ValueError('Composition changed H geometry/material map: '+name)
    if not all(all(row.values()) for row in checks.values()):
        raise ValueError('H composition changed frozen F RGB source bytes.')
    result = {
        'method':METHOD,'asset':asset,'path':str(final_dir),'sourceAppearance':plan['sourceAppearance'],
        'geometryBaseline':baseline['asset'],'proxySha256':final['proxySha256'],
        'sourcePngAndNpyChecks':checks,'cameraRebaseProof':str(work/'camera-rebase/camera-rebase-proof.json'),
        'trueTightBounds':final['tightBounds'],'nativeBakeBox':final['bakeBox'],
        'textures':{k:v['sha256'] for k,v in final['textures'].items()},
        'viewConsistency':'none','materialEstimation':'none','registrationOptimized':False,
        'acceptedFaces':list(evidence['faces']),'postprocessSnapshotSha256':SNAPSHOT_SHA,
        'codeSha256':{'runner':digest(__file__),'cameraRebase':digest(REPO/'box_nodes/camera_rebase.py')}}
    save(work/'result.json',result)
    save(Path(plan['case'])/'composed.json',result)
    print(json.dumps({'stage':'complete','asset':asset,'sourceAppearance':plan['sourceAppearance']}),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--work',required=True,type=Path)
    parser.add_argument('--appearance',required=True,help='Frozen F Box asset ID')
    parser.add_argument('--stage',choices=['plan','verify-b','infer','bake','compose','all'],default='plan')
    args = parser.parse_args()
    plan = build_plan(args)
    if args.stage=='plan':
        print(json.dumps(plan,indent=2),flush=True)
        return
    store,faces,depth_mesh = runtime(Path(plan['run']))
    steps = ['verify-b','infer','bake','compose'] if args.stage=='all' else [args.stage]
    for step in steps:
        if step=='verify-b':verify_b(plan,depth_mesh)
        elif step=='infer':infer(plan,store,depth_mesh)
        elif step=='bake':bake(plan,store)
        else:compose(plan,store,faces)


if __name__=='__main__':
    main()
