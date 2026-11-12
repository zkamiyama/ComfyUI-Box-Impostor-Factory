// Load the distributed JSON as a file in the real Comfy frontend, then execute its
// graphToPrompt result unchanged. Model workflows are expanded without inference.
import assert from 'node:assert/strict';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';
import { chromium } from 'playwright';
import { chromiumOptions } from '../worker/chromium.mjs';
import { preflight } from '../scripts/run_model_workflow.mjs';

const [origin,inputDir,artifacts]=process.argv.slice(2);
if(!origin||!inputDir||!artifacts)throw new Error('Usage: node tests/comfy_frontend_smoke.mjs ORIGIN INPUT_DIR OUTPUT_DIR');
const workflowNames=['01_box_from_glb','04_manual_six_faces','02_box_from_reference_qwen21',
  '03_box_from_reference_h3','05_pixal3d_baseline','06_replace_saved_face','07_render_saved_box','08_triposplat_guide'];
const sides=[['+X','px'],['-X','nx'],['+Y','py'],['-Y','ny'],['+Z','pz'],['-Z','nz']];
const resources=new URL('../workflows/',import.meta.url);
const browser=await chromium.launch(chromiumOptions('webgl2'));
const page=await browser.newPage();
page.on('pageerror',e=>console.error('Frontend page error:',e.message));
await mkdir(artifacts,{recursive:true});

async function freshPage(){
  await page.goto(origin,{waitUntil:'domcontentloaded',timeout:90000});
  // Same ready flag used by the pinned upstream frontend's own ComfyPage fixture.
  await page.waitForFunction(()=>window.app?.extensionManager && window.app?.rootGraph && window.app?.canvas,null,{timeout:120000});
}

async function compile(name){
  const filename=name+'.json',contents=await readFile(new URL(filename,resources),'utf8');
  const prompt=await page.evaluate(async ({name,contents})=>{
    const app=window.app;
    // The same file-reader entry point used for a workflow dropped onto the canvas.
    await app.handleFile(new File([contents],name+'.json',{type:'application/json'}),'file_drop',{deferWarnings:true});
    const compiled=await app.graphToPrompt();
    return compiled.output;
  },{name,contents});
  assert.ok(prompt&&Object.keys(prompt).length,'No API prompt for '+name);
  await writeFile(join(artifacts,name+'.api.json'),JSON.stringify(prompt,null,2)+'\n');
  const byType=t=>Object.entries(prompt).filter(([,n])=>n.class_type===t);
  if(name.startsWith('08')){
    const [[ref]]=byType('BIPrepareReference');
    const [[decode,decoder]]=byType('VAEDecodeTripoSplat');
    const [[cache,cached]]=byType('BICachedTripoSplat');
    const [[align,transform]]=byType('TransformSplat');
    const [[guide,render]]=byType('BITripoSplatGuide');
    assert.equal(decoder.inputs.num_gaussians,262144);
    assert.deepEqual(cached.inputs,{reference:[ref,2],splat:[decode,0]});
    assert.deepEqual(transform.inputs.splat,[cache,0]);
    assert.deepEqual(render.inputs.splat,[align,0]);
    assert.deepEqual(render.inputs.guides,[byType('BIMakeSixGuides')[0][0],6]);
    assert.equal(render.inputs.face_id,'+Z');
    assert.equal(render.inputs.coordinate_space,'canonical_box');
    assert.deepEqual(byType('PreviewImage')[0][1].inputs.images,[guide,0]);
    assert.deepEqual(byType('PreviewAny')[0][1].inputs.source,[guide,2]);
    return prompt;
  }
  if(name.startsWith('07')){
    const [[id,render]]=byType('BIRenderBox');
    const [[join,rgba]]=byType('JoinImageWithAlpha');
    assert.equal(render.inputs.sampling,'standard');
    assert.deepEqual(rgba.inputs,{image:[id,0],alpha:[id,1]});
    assert.deepEqual(byType('SaveImage')[0][1].inputs.images,[join,0]);
    return prompt;
  }
  assert.equal(byType('BIPreviewBox').length,1,'Missing preview output in '+name);
  assert.equal(byType('BISaveBox').length,1,'Missing export output in '+name);
  assert.equal(byType('BIBakeBox').length,name.startsWith('06')?0:1);
  if(name.startsWith('01')||name.startsWith('04')){
    assert.equal(byType('BICanonicalizeProxy')[0][1].inputs.filename,'asymmetric.glb');
    if(name.startsWith('04')){
      assert.deepEqual(byType('LoadImage').map(([,n])=>n.inputs.image).sort(),sides.map(([,s])=>s+'.png').sort());
      assert.equal(byType('BIFaceImage').length,6);
    }
  }
  if(name.startsWith('02')||name.startsWith('03')){
    const encoder=name.startsWith('02')?'TextEncodeQwenImage21':'MiniMaxH3ImageConditioning';
    assert.equal(byType(encoder).length,6,'Six native view encoders must survive subgraph expansion');
    for(const [,n] of byType(encoder)){
      for(const port of ['images.image_1','images.image_2','vae','clip']){
        assert(Array.isArray(n.inputs[port]),name+' missing '+encoder+'.'+port);
      }
    }
    assert.equal(byType('BICachedGeneratedFace').length,6);
    for(const [,n] of byType('BICachedGeneratedFace')){
      assert.equal(n.inputs.mask_mode,'foreground',name+' '+n.inputs.face_id);
      const matte=prompt[n.inputs.mask[0]];
      assert.equal(matte.class_type,'RemoveBackground');
      assert.deepEqual(matte.inputs.image,n.inputs.image);
    }
    const registry=await (await page.request.get(origin+'/object_info')).json();
    const inventory=preflight(JSON.parse(contents),registry,'reference.png');
    assert.equal(inventory.inputAvailable,true);
    assert.deepEqual(inventory.missingNodes,[]);
    assert.equal(inventory.ready,false,'CPU CI should report absent model weights');
    await writeFile(join(artifacts,name+'.preflight.json'),JSON.stringify(inventory,null,2));
  }
  if(name.startsWith('02')||name.startsWith('03')||name.startsWith('05')){
    assert.equal(byType('Pixal3DConditioning').length,1,'Pixal geometry subgraph did not expand');
    assert.equal(byType('MeshToFile3D').length,1);
    assert.equal(byType('LoadImage').length,1);
  }
  return prompt;
}

async function queue(prompt){
  // Request matches the queue payload consumed by Comfy's API; do not rewrite any
  // node inputs, link IDs, file selectors or execution-mode settings here.
  const res=await page.request.post(origin+'/prompt',{data:{prompt,client_id:'box-frontend-smoke'}});
  if(!res.ok())throw new Error('POST /prompt '+res.status()+' '+(await res.text()).slice(0,6000));
  const {prompt_id}=await res.json();
  const deadline=Date.now()+360000;
  while(Date.now()<deadline){
    const result=await page.request.get(origin+'/history/'+prompt_id);
    assert(result.ok());
    const history=(await result.json())[prompt_id];
    if(history){
      if(history.status.status_str!=='success')throw new Error(JSON.stringify(history.status).slice(-6500));
      const save=Object.keys(prompt).find(k=>prompt[k].class_type==='SaveImage');
      if(save){
        const [image]=history.outputs[save].images;
        const url=new URL('/view',origin);
        url.search=new URLSearchParams(image).toString();
        const png=await page.request.get(url.href);
        assert(png.ok(),'Missing saved transparent PNG');
        await writeFile(join(artifacts,'rendered-box.png'),await png.body());
        return {prompt_id};
      }
      const id=Object.keys(prompt).find(k=>prompt[k].class_type==='BIPreviewBox');
      const viewer=history.outputs[id]?.box_impostor?.[0];
      assert.ok(viewer,'Queued workflow produced no saved Box preview URL');
      const url=new URL(new URL(viewer,origin).searchParams.get('manifest'),origin);
      const response=await page.request.get(url.href);assert(response.ok());
      return {manifest:await response.json(),url,prompt_id};
    }
    await delay(500);
  }
  throw new Error('Frontend workflow queue timed out '+prompt_id);
}

try{
  // The input GLB already exists, with the exact filename selected by 01 and 04.
  await freshPage();
  const one=await compile(workflowNames[0]);
  const baseline=await queue(one);
  for(const [id,slug] of sides){
    const path=baseline.manifest.faces[id].guide;
    const response=await page.request.get(new URL(path,baseline.url).href);
    assert(response.ok(),'Missing render for '+id);
    await writeFile(join(inputDir,slug+'.png'),await response.body());
  }
  // Populate the input dropdown from the new six files. No GUI JSON is changed.
  await freshPage();
  const four=await compile(workflowNames[1]);
  const manual=await queue(four);
  assert.equal(manual.manifest.parentAsset,baseline.manifest.asset_id);
  for(const [id] of sides)assert.equal(manual.manifest.faces[id].edit.status,'accepted','Manual '+id);
  for(const key of ['normalMap','rangeMap','intervalMap'])
    assert.deepEqual(manual.manifest.textures[key],baseline.manifest.textures[key]);
  const preview=await page.context().newPage();
  try {
    const url=new URL('/box-impostor/web/viewer.html',origin);
    url.searchParams.set('manifest',manual.url.pathname);
    url.searchParams.set('backend','webgl2');
    await preview.goto(url.href);
    await preview.locator('#faceQA summary').click();
    const decisions=await preview.locator('#faceQA tbody tr').allTextContents();
    assert.equal(decisions.length,6);
    assert(decisions.every(row=>row.includes('手動画像') && row.includes('輪郭の基準を満たした')));
    const qa=await page.request.get(await preview.locator('#faceQA a').getAttribute('href'));
    assert(qa.ok());
  } finally { await preview.close(); }
  const proof={
    frontendBrowser:browser.version(),
    executed:[{workflow:workflowNames[0],prompt_id:baseline.prompt_id,asset:baseline.manifest.asset_id},
      {workflow:workflowNames[1],prompt_id:manual.prompt_id,asset:manual.manifest.asset_id,acceptedFaces:6}],
    compiledOnly:[],
  };
  for(const name of workflowNames.slice(2)){
    // Supply an actual input image to the standard input combo. No model is loaded.
    if(name===workflowNames[2]){
      await writeFile(join(inputDir,'reference.png'),await readFile(join(inputDir,'px.png')));
      await freshPage();
    }
    const prompt=await compile(name);
    if(name.startsWith('07')){
      const {prompt_id}=await queue(prompt);
      proof.executed.push({workflow:name,prompt_id,image:'rendered-box.png'});
    }else proof.compiledOnly.push({workflow:name,expandedNodes:Object.keys(prompt).length});
  }
  await writeFile(join(artifacts,'frontend-workflows.json'),JSON.stringify(proof,null,2)+'\n');
  console.log('Frontend workflow proof',JSON.stringify(proof));
}catch(e){
  console.error('Comfy frontend at',page.url(),'title',await page.title().catch(()=>null));
  await page.screenshot({path:join(artifacts,'frontend-failure.png'),timeout:6000}).catch(()=>{});
  throw e;
}finally{await browser.close();}
