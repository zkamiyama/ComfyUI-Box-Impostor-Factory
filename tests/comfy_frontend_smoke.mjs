// Load the distributed JSON as a file in the real Comfy frontend, then execute its
// graphToPrompt result unchanged. Model workflows are expanded without inference.
import assert from 'node:assert/strict';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';
import { chromium } from 'playwright';
import { chromiumOptions } from '../worker/chromium.mjs';
import { checkCompiled, configureCompactProxy, configureEditGraph, preflight } from '../scripts/run_model_workflow.mjs';

const [origin,inputDir,artifacts]=process.argv.slice(2);
if(!origin||!inputDir||!artifacts)throw new Error('Usage: node tests/comfy_frontend_smoke.mjs ORIGIN INPUT_DIR OUTPUT_DIR');
const workflowNames=['01_box_from_glb','04_manual_six_faces','02_box_from_reference_qwen21',
  '03_box_from_reference_h3','05_pixal3d_baseline','06_replace_saved_face','07_render_saved_box','08_triposplat_guide','09_rebake_proxy_revision',
  '10_material_from_saved_box'];
const sides=[['+X','px'],['-X','nx'],['+Y','py'],['-Y','ny'],['+Z','pz'],['-Z','nz']];
const resources=new URL('../workflows/',import.meta.url);
const browser=await chromium.launch(chromiumOptions('webgl2'));
const context=await browser.newContext();
const page=await context.newPage();
page.on('pageerror',e=>console.error('Frontend page error:',e.message));
await mkdir(artifacts,{recursive:true});

async function freshPage(){
  await page.goto(origin,{waitUntil:'domcontentloaded',timeout:90000});
  // Same ready flag used by the pinned upstream frontend's own ComfyPage fixture.
  await page.waitForFunction(()=>window.app?.extensionManager && window.app?.rootGraph && window.app?.canvas,null,{timeout:120000});
}

async function compile(name, edit, savedAsset){
  const graph=JSON.parse(await readFile(new URL(name+'.json',resources),'utf8'));
  const model=name.startsWith('05')?'pixal3d':name.startsWith('02')?'qwen':'h3';
  if(edit?.compactProxy){
    configureCompactProxy(graph);
    name+='-compact';
  }else if(edit){
    configureEditGraph(graph,model,edit);
    name+='-supplied-glb';
  }
  if(savedAsset)graph.nodes.find(n=>n.type==='BILoadBox').widgets_values[0]=savedAsset;
  const contents=JSON.stringify(graph);
  const prompt=await page.evaluate(async ({name,contents})=>{
    const app=window.app;
    // The same file-reader entry point used for a workflow dropped onto the canvas.
    await app.handleFile(new File([contents],name+'.json',{type:'application/json'}),'file_drop',{deferWarnings:true});
    const compiled=await app.graphToPrompt();
    return compiled.output;
  },{name,contents});
  assert.ok(prompt&&Object.keys(prompt).length,'No API prompt for '+name);
  await writeFile(join(artifacts,name+'.api.json'),JSON.stringify(prompt,null,2)+'\n');
  if(edit){
    // CPU CI has no model weights. Check the native frontend's expanded edges and
    // actual encoder dimensions independently of loader filenames.
    checkCompiled(prompt,model,{...edit,input:'reference.png',models:[]});
    return prompt;
  }
  const byType=t=>Object.entries(prompt).filter(([,n])=>n.class_type===t);
  for(const [,n] of [...byType('BIAlignAndValidateFaces'),...byType('BIReplaceFaces')]){
    assert.deepEqual([n.inputs.min_iou,n.inputs.max_shift,n.inputs.max_scale_change],
      [0.85,0.10,0.15],name+' must preserve the shipped alignment thresholds');
  }
  if(['02','03','10'].includes(name.slice(0,2))){
    const [[id,material]]=byType('BIGenerateFaceMaterials');
    const [[source]]=byType(name.startsWith('10')?'BILoadBox':'BIBakeBox');
    assert.deepEqual(material.inputs,{asset:[source,0],source:'keep',face:'all',material_seed:0,edit_resolution:2048});
    for(const type of ['BIPreviewBox','BISaveBox'])assert.deepEqual(byType(type)[0][1].inputs.asset,[id,0]);
    if(name.startsWith('10'))return prompt;
  }
  if(name.startsWith('09')){
    const [[id,revision]]=byType('BIRebakeProxyRevision');
    assert.deepEqual(revision.inputs.asset,[byType('BILoadBox')[0][0],0]);
    assert.deepEqual(revision.inputs.corrected_proxy,[byType('BICanonicalizeProxy')[0][0],0]);
    assert.deepEqual(byType('BIPreviewBox')[0][1].inputs.asset,[id,0]);
    return prompt;
  }
  if(name.startsWith('08')){
    const [[ref]]=byType('BIPrepareReference');
    const [[decode,decoder]]=byType('VAEDecodeTripoSplat');
    const [[cache,cached]]=byType('BICachedTripoSplat');
    const [[align,transform]]=byType('TransformSplat');
    const [[guide,render]]=byType('BITripoSplatGuide');
    const [[file,serialized]]=byType('SplatToFile3D');
    assert.deepEqual(serialized.inputs,{splat:[align,0],format:'ply'});
    assert.deepEqual(byType('PreviewGaussianSplat')[0][1].inputs.model_3d,[file,0]);
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
    const [[secondary]]=byType('BISecondaryReferences');
    for(const [,n] of byType(encoder)){
      for(const port of ['images.image_1','images.image_2','vae','clip']){
        assert(Array.isArray(n.inputs[port]),name+' missing '+encoder+'.'+port);
      }
      for(let slot=0;slot<8;slot++){
        const [split,output]=n.inputs['images.image_'+(slot+3)];
        assert.equal(output,slot);
        assert.equal(prompt[split].class_type,'BISecondaryReferenceSlots');
        assert.deepEqual(prompt[split].inputs.references,[secondary,0]);
      }
    }
    assert.equal(byType('BICachedGeneratedFace').length,6);
    for(const [,n] of byType('BICachedGeneratedFace')){
      assert.deepEqual(n.inputs.secondary,[secondary,0]);
      assert.equal(n.inputs.mask_mode,'foreground',name+' '+n.inputs.face_id);
      const matte=prompt[n.inputs.mask[0]];
      assert.equal(matte.class_type,'RemoveBackground');
      assert.deepEqual(matte.inputs.image,n.inputs.image);
    }
    const registry=await (await page.request.get(origin+'/object_info')).json();
    const inventory=preflight(JSON.parse(contents),registry,'reference.png');
    assert.equal(inventory.inputAvailable,true);
    assert.deepEqual(inventory.missingNodes,[]);
    // Native CPU CI has no model weights, but GPU-equipped developer machines
    // may have them installed. Never queue any inference in this smoke test.
    assert.equal(inventory.ready,inventory.models.every(m=>m.available),
      'preflight must reflect installed model inventory');
    await writeFile(join(artifacts,name+'.preflight.json'),JSON.stringify(inventory,null,2));
  }
  if(name.startsWith('02')||name.startsWith('03')||name.startsWith('05')){
    checkCompiled(prompt,name.startsWith('05')?'pixal3d':model,{input:'reference.png',models:[]});
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
  const preview=await context.newPage();
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
    const prompt=await compile(name,null,name.startsWith('10')?manual.manifest.asset_id:undefined);
    if (name.startsWith('02') || name.startsWith('03')) {
      const edits = await page.evaluate(async () => {
        const tab = window.app.extensionManager.getSidebarTabs().find(t => t.id === 'box-impostor-workflow');
        const host = document.createElement('div');
        document.body.append(host);
        try {
          tab.render(host);
          const compiled = () => window.app.graphToPrompt().then(p => p.output);
          const before = await compiled();
          [...host.querySelectorAll('button')].find(b => b.textContent.startsWith('+Z のみ')).click();
          const after = await compiled();
          const dirty = [...host.querySelectorAll('section')].filter(s => s.textContent.includes('Dirty'));
          return {dirty: dirty.map(s => s.querySelector('strong').textContent),
            changed: Object.keys(before).filter(id => JSON.stringify(before[id].inputs) !== JSON.stringify(after[id].inputs))
              .map(id => after[id].class_type)};
        } finally { tab.destroy(); host.remove(); }
      });
      assert.deepEqual(edits, {dirty: ['+Z','Box Impostor'], changed: ['PrimitiveInt']});
    }
    if(name.startsWith('10')){
      const {manifest,prompt_id}=await queue(prompt);
      assert.equal(manifest.asset_id,manual.manifest.asset_id);
      proof.executed.push({workflow:name,prompt_id,asset:manifest.asset_id});
    }else if(name.startsWith('07')){
      const {prompt_id}=await queue(prompt);
      proof.executed.push({workflow:name,prompt_id,image:'rendered-box.png'});
    }else if(name.startsWith('09')){
      const {manifest,url,prompt_id}=await queue(prompt);
      const qaResponse=await page.request.get(new URL(manifest.qa,url).href);
      assert(qaResponse.ok());
      const qa=await qaResponse.json();
      assert.equal(qa.status,'rebaked');
      assert.equal(qa.originalAsset,manifest.parentAsset);
      assert.equal(qa.revisedAsset,manifest.asset_id);
      assert.notEqual(qa.originalProxySha256,qa.correctedProxySha256);
      assert(Object.values(qa.guideComparisons).every(face=>face.status==='compared'));
      await writeFile(join(artifacts,'geometry-revision.json'),JSON.stringify(qa,null,2));
      proof.executed.push({workflow:name,prompt_id,asset:manifest.asset_id});
    }else proof.compiledOnly.push({workflow:name,expandedNodes:Object.keys(prompt).length});
  }
  for(const name of workflowNames.slice(2,4)){
    const prompt=await compile(name,{mesh:'asymmetric.glb',face:'+Z',faceSeed:{face:'+Z',seed:321},size:512,h3ReferenceFirst:name.startsWith('03')});
    proof.compiledOnly.push({workflow:name+'-supplied-glb',expandedNodes:Object.keys(prompt).length});
  }
  const compactName='05_pixal3d_baseline';
  const compact=await compile(compactName,{compactProxy:true});
  proof.compiledOnly.push({workflow:compactName+'-compact',expandedNodes:Object.keys(compact).length});
  await writeFile(join(artifacts,'frontend-workflows.json'),JSON.stringify(proof,null,2)+'\n');
  console.log('Frontend workflow proof',JSON.stringify(proof));
}catch(e){
  console.error('Comfy frontend at',page.url(),'title',await page.title().catch(()=>null));
  await page.screenshot({path:join(artifacts,'frontend-failure.png'),timeout:6000}).catch(()=>{});
  throw e;
}finally{await browser.close();}
