"""F: frozen B color guide and source photo, two references for both H3 encoders.

Compared with D, omit the three structural VLM pictures and their prompt instructions.
Generate only -X and replace it in the original complete B asset.
"""
import argparse
import copy
import difflib
from pathlib import Path

import run_geometry_ab as ab
import run_geometry_d as d

PROMPT_CHANGES=[
    ('<Picture 3> is the aligned geometry depth: brighter means nearer the camera. ',''),
    ('<Picture 4> is the aligned world-space surface normal map. ',''),
    ('<Picture 5> is the aligned foreground mask: white is the person and black is background. ',''),
    ('Use Pictures 3, 4 and 5 only as structural references; do not copy their colors. ',''),
    ('Picture 1 and the structural maps.','Picture 1.'),
    ('Keep the body and garment boundaries exactly within the mask.',
     'Keep the body and garment boundaries exactly as shown in Picture 1.'),
    (', no colored depth or normal-map appearance.','.'),
]


def color_prompt(prompt):
    for before,after in PROMPT_CHANGES:
        if prompt.count(before)!=1:
            raise ValueError('The frozen structural prompt changed: '+before)
        prompt=prompt.replace(before,after)
    return prompt


def build_prompt(recipe,fid):
    if recipe['pictureOrder']!=['guide','source']:
        raise ValueError('F requires exactly the frozen color guide and source photo.')
    graph=ab.build_prompt(recipe,fid)
    inputs=graph['10']['inputs']
    for index in range(1,6):
        link=inputs.pop(f'images.image_{index}')
        if index<=2:inputs[f'image_{index}']=link
    inputs['reference_latents']='color_only'
    graph['10']['class_type']='BIH3ReferenceConditioning'
    return graph


def prepare(run,fid):
    if fid!='-X':
        raise ValueError('F is limited to the approved -X experiment.')
    source=run/'B/prepared.json'
    b=ab.read_json(source)
    completed=ab.read_json(run/'B/composed.json')
    previous=ab.read_json(run/'D/prepared.json')
    d_box=ab.read_json(run/'D/composed.json')
    d_result_path=run/'D/nx/result.json'
    d_result=ab.read_json(d_result_path)
    if (len(completed['generatedFaces'])!=6 or completed['sourceBaseline']!=b['baseline'] or
        d_box['sourceBox']!=completed['asset'] or
        previous['experiment']['sourceRecipeSha256']!=ab.digest(source)):
        raise ValueError('Completed B and D must share the frozen B inputs.')
    if (b['pictureOrder']!=['guide','source','axis-depth','normals','mask'] or
        previous['experiment']['vlmPictureIndices']!=[1,2,3,4,5] or
        previous['experiment']['vaePictureIndices']!=[1,2]):
        raise ValueError('The B/D reference policy changed.')
    for picture in b['faces'][fid]['pictures'].values():
        if ab.digest(picture['path'])!=picture['sha256']:
            raise ValueError('A frozen B conditioning image changed: '+picture['path'])
    if ab.digest(d_result['raw'])!=d_result['rawSha256']:
        raise ValueError('The D raw comparison image changed.')
    original_prompt=b['faces'][fid]['prompt']
    if previous['faces'][fid]['prompt']!=original_prompt:
        raise ValueError('B and D no longer share the original prompt.')
    result_path=run/'B/nx/result.json'
    original=ab.read_json(result_path)
    if ab.digest(original['raw'])!=original['rawSha256']:
        raise ValueError('The B raw comparison image changed.')
    recipe=copy.deepcopy(b)
    recipe['faces'][fid]['prompt']=color_prompt(original_prompt)
    recipe['faces'][fid]['pictures']={role:b['faces'][fid]['pictures'][role] for role in ['guide','source']}
    recipe.update(branch='F',pictureOrder=['guide','source'],
        conditioning='H3: color guide and source photo only for VLM and VAE; no structural pictures; no ControlNet',
        experiment={'name':'h3-two-color-pictures-v1','targetFace':fid,'sourceBranch':'B',
          'sourceRecipeSha256':ab.digest(source),'sourceBox':completed['asset'],
          'sourceResultSha256':ab.digest(result_path),'sourceRaw':original['raw'],
          'sourceRawSha256':original['rawSha256'],'comparisonBranch':'D',
          'comparisonRecipeSha256':ab.digest(run/'D/prepared.json'),
          'comparisonResultSha256':ab.digest(d_result_path),'comparisonRaw':d_result['raw'],
          'comparisonRawSha256':d_result['rawSha256'],'comparisonBox':d_box['asset'],
          'vlmPictureIndices':[1,2],'vaePictureIndices':[1,2],
          'omittedPictureRoles':['axis-depth','normals','mask'],
          'promptChanges':[{'before':before,'after':after} for before,after in PROMPT_CHANGES],
          'promptIdenticalToD':False,
          'differenceFromD':'Removed structural VLM pictures and their corresponding prompt instructions; the two VAE color references are unchanged.',
          'nativeConditioningSha256':ab.digest(d.REPO.parents[1]/'comfy_extras/nodes_minimax_h3.py'),
          'wrapperSha256':ab.digest(d.REPO/'box_nodes/h3_reference_conditioning.py')})
    target=run/'F/prepared.json'
    if target.exists() and ab.read_json(target)!=recipe:
        raise ValueError('Existing F preparation belongs to different inputs or implementation.')
    ab.write_json(target,recipe)
    ab.write_json(run/'F/nx/prepared-prompt.json',build_prompt(recipe,fid))
    revised=recipe['faces'][fid]['prompt']
    ab.write_json(run/'F/prompt-diff.json',{'before':original_prompt,'after':revised,
        'changes':recipe['experiment']['promptChanges'],'identical':False})
    (run/'F/prompt-diff.patch').write_text(''.join(difflib.unified_diff(
        [original_prompt+'\n'],[revised+'\n'],fromfile='B/D -X prompt',tofile='F -X prompt')))
    return recipe


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--face',choices=['-X'],default='-X')
    parser.add_argument('--origin',default='http://127.0.0.1:8201')
    action=parser.add_mutually_exclusive_group()
    action.add_argument('--prepare-only',action='store_true')
    action.add_argument('--compose-only',action='store_true')
    args=parser.parse_args()
    ab.use_cuda_runtime_on_wsl()
    run=args.run.resolve()
    recipe=prepare(run,args.face)
    if args.prepare_only:
        print({'prepared':str(run/'F/prepared.json'),'gpuInference':False},flush=True)
        return
    if not args.compose_only:
        ab.run_face(run,recipe,args.face,args.origin,prompt_builder=build_prompt)
    composed=d.compose_replacement(run,recipe,args.face,prompt_builder=build_prompt)
    result=ab.read_json(run/'F/nx/result.json')
    ab.write_json(run/'F/comparison-ready.json',{
        'face':args.face,'rawOnly':False,'B_raw':recipe['experiment']['sourceRaw'],
        'D_raw':recipe['experiment']['comparisonRaw'],'F_raw':result['raw'],
        'experiment':recipe['experiment'],'result':result,'box':composed})


if __name__=='__main__':main()
