"""D: frozen B pictures, all five VLM references, only two color VAE references.

This replaces only the selected B face; the other five RGB views and all geometry stay fixed.
"""
import argparse
import copy
import hashlib
from pathlib import Path

import run_geometry_ab as ab

REPO=Path(__file__).resolve().parents[1]


def build_prompt(recipe,fid):
    graph=ab.build_prompt(recipe,fid)
    inputs=graph['10']['inputs']
    for index in range(1,6):
        inputs[f'image_{index}']=inputs.pop(f'images.image_{index}')
    inputs['reference_latents']='color_first_two'
    graph['10']['class_type']='BIH3ReferenceConditioning'
    return graph


def prepare(run,fid):
    source=run/'B/prepared.json'
    b=ab.read_json(source)
    completed=ab.read_json(run/'B/composed.json')
    if len(completed['generatedFaces'])!=6 or completed['sourceBaseline']!=b['baseline']:
        raise ValueError('The completed six-face B result is required.')
    if b['pictureOrder']!=['guide','source','axis-depth','normals','mask']:
        raise ValueError('The fixed five-picture order changed.')
    for picture in b['faces'][fid]['pictures'].values():
        if ab.digest(picture['path'])!=picture['sha256']:
            raise ValueError('A frozen B conditioning image changed: '+picture['path'])
    result_path=run/'B'/ab.SLUGS[fid]/'result.json'
    original=ab.read_json(result_path)
    if ab.digest(original['raw'])!=original['rawSha256']:
        raise ValueError('The B raw comparison image changed.')
    recipe=copy.deepcopy(b)
    recipe.update(branch='D',conditioning='H3: five VLM pictures; VAE reference pictures 1 and 2 only; no ControlNet',
        experiment={'name':'h3-color-reference-latents-v1','targetFace':fid,
          'sourceBranch':'B','sourceRecipeSha256':ab.digest(source),'sourceBox':completed['asset'],
          'sourceResultSha256':ab.digest(result_path),'sourceRaw':original['raw'],
          'sourceRawSha256':original['rawSha256'],'vlmPictureIndices':[1,2,3,4,5],
          'vaePictureIndices':[1,2],
          'nativeConditioningSha256':ab.digest(REPO.parents[1]/'comfy_extras/nodes_minimax_h3.py'),
          'wrapperSha256':ab.digest(REPO/'box_nodes/h3_reference_conditioning.py')})
    target=run/'D/prepared.json'
    if target.exists() and ab.read_json(target)!=recipe:
        raise ValueError('Existing D preparation belongs to different inputs or implementation.')
    ab.write_json(target,recipe)
    ab.write_json(run/'D'/ab.SLUGS[fid]/'prepared-prompt.json',build_prompt(recipe,fid))
    return recipe



def verify_replacement(source,output,before,after,fid):
    # The experiment replaces RGB only. Other accepted views and geometry stay exact.
    for key in ['proxy','proxySha256','transform','inverseTransform','tightBounds','bakeBox',
                'resolution','baseColor','geometryAsset']:
        if before.get(key)!=after.get(key):
            raise ValueError('D changed frozen B geometry metadata: '+key)
    if ab.digest(source/before['proxy'])!=ab.digest(output/after['proxy']):
        raise ValueError('D changed the frozen B proxy bytes.')
    if set(before['textures'])!=set(after['textures']):
        raise ValueError('D changed the B map channel set.')
    for name,texture in before['textures'].items():
        if name!='colorMap' and (after['textures'].get(name)!=texture or
            ab.digest(source/texture['path'])!=ab.digest(output/texture['path'])):
            raise ValueError('D changed a non-color map: '+name)
    def atlas(directory,manifest):
        desc=manifest['textures']['colorMap']
        return ab.np.fromfile(directory/desc['path'],dtype='<f2').reshape(desc['height'],desc['width'],4)
    old,new=atlas(source,before),atlas(output,after)
    if not ab.np.array_equal(old[...,3].view('u2'),new[...,3].view('u2')):
        raise ValueError('D changed B coverage alpha.')
    untouched={}
    for other in ab.store.FACES:
        if other==fid:continue
        x=ab.faces.tile_view(old,other).tobytes();y=ab.faces.tile_view(new,other).tobytes()
        if x!=y or before['faces'][other]!=after['faces'][other]:
            raise ValueError('D changed another B view: '+other)
        view=before['faces'][other].get('edit',{}).get('view')
        if view:
            for name in ['source.png','source.npy','aligned.npy']:
                relative=Path(view).parent/name
                if (source/relative).exists() and ab.digest(source/relative)!=ab.digest(output/relative):
                    raise ValueError('D changed an untouched source view: '+other)
        untouched[other]=hashlib.sha256(x).hexdigest()
    return {'proxyBytesExact':True,'geometryMetadataExact':True,'nonColorMapsExact':True,
            'coverageAlphaExact':True,'unchangedFaceRGBAHashes':untouched,
            'unchangedFaceSourcesAndMetadata':True}


def compose_replacement(run,recipe,fid,*,prompt_builder=None):
    ab.folder_paths.set_output_directory(str(run/'output'))
    baseline=recipe['experiment']['sourceBox']
    source=ab.store.asset_dir(baseline)
    original=ab.store.validate(source,baseline,check_bytes=True)
    branch=recipe['branch']
    case=run/branch/ab.SLUGS[fid]
    result=ab.read_json(case/'result.json')
    if (ab.read_json(case/'prompt.json')!=(prompt_builder or build_prompt)(recipe,fid) or
        ab.digest(case/'prompt.json')!=result['promptSha256'] or
        ab.digest(case/'generated.png')!=result['rawSha256'] or
        ab.digest(case/'foreground.png')!=result['maskSha256']):
        raise ValueError('D result no longer matches its frozen prompt or image files.')
    rgb=ab.np.asarray(ab.Image.open(case/'generated.png').convert('RGB'),dtype='f4')/255
    mask=ab.np.asarray(ab.Image.open(case/'foreground.png').convert('L'),dtype='f4')/255
    item=ab.faces.save_view(ab.faces.rgba(rgb[None],'foreground',mask[None]))
    face=recipe['faces'][fid]
    item['provenance']={'backend':'h3_enhanced','faceId':fid,
        'recipe':{'models':recipe['models'],'conditioning':recipe['conditioning'],
                  'pictures':face['pictures'],'prompt':face['prompt'],'experiment':recipe['experiment']},
        'recipeSeed':recipe['seeds'][fid],'samplers':[recipe['sampler']],
        'graphSha256':result['promptSha256'],'referenceSha256':recipe.get('referenceSha256'),
        'guideSha256':face.get('guideSha256'),'maskMode':'foreground',
        'canvas':face.get('canvas'),'freshInference':result['prompt_id'],
        'geometryRefinement':recipe.get('geometryRefinement')}
    if original.get('reference'):
        item['reference']={**original['reference'],'path':str(source/'reference')}
    aligned=ab.faces.align({fid:item},baseline,.85,.10,.15)
    report=ab.read_json(Path(aligned['faces'][fid]['directory'])/'report.json')
    applied=report['status']=='accepted'
    # A rejected replacement keeps the previous B view, not the coarse guide color.
    asset=ab.faces.compose(baseline,aligned) if applied else baseline
    output=ab.store.asset_dir(asset)
    manifest=ab.store.validate(output,asset,check_bytes=True)
    preservation=verify_replacement(source,output,original,manifest,fid)
    proof={'asset':asset,'path':str(output),'sourceBox':baseline,'guideBaseline':recipe['baseline'],
        'replacedFace':fid,'applied':applied,'faceQA':report,'preservation':preservation,
        'proxySha256':manifest['proxySha256'],'viewConsistency':'none','materialEstimation':'none'}
    ab.write_json(run/branch/'composed.json',proof)
    return proof


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--face',choices=ab.SLUGS,default='-X')
    parser.add_argument('--origin',default='http://127.0.0.1:8201')
    action=parser.add_mutually_exclusive_group()
    action.add_argument('--prepare-only',action='store_true')
    action.add_argument('--compose-only',action='store_true')
    args=parser.parse_args()
    ab.use_cuda_runtime_on_wsl()
    run=args.run.resolve()
    recipe=prepare(run,args.face)
    if args.prepare_only:
        print({'prepared':str(run/'D/prepared.json'),'gpuInference':False},flush=True)
        return
    if not args.compose_only:
        ab.run_face(run,recipe,args.face,args.origin,prompt_builder=build_prompt)
    composed=compose_replacement(run,recipe,args.face)
    result=ab.read_json(run/'D'/ab.SLUGS[args.face]/'result.json')
    ab.write_json(run/'D/comparison-ready.json',{
        'face':args.face,'rawOnly':False,'B_raw':recipe['experiment']['sourceRaw'],
        'B_rawSha256':recipe['experiment']['sourceRawSha256'],
        'D_raw':result['raw'],'D_rawSha256':result['rawSha256'],
        'experiment':recipe['experiment'],'result':result,'box':composed})


if __name__=='__main__':
    main()
