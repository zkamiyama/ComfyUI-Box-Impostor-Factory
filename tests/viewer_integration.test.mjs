import assert from 'node:assert/strict';
import test from 'node:test';
import { resolve } from 'node:path';
import { chromium } from 'playwright';
import { chromiumOptions, presentedForeground } from '../worker/chromium.mjs';
import { startServer } from '../worker/server.mjs';

// Sample opaque fragments only. A screenshot also includes the CSS checkerboard.
function inspectDiagnostic(mode) {
  const canvas=document.createElement('canvas');
  canvas.width=192; canvas.height=96;
  const ctx=canvas.getContext('2d');
  ctx.drawImage(document.querySelector('#canvas'),0,0,canvas.width,canvas.height);
  const pixels=ctx.getImageData(0,0,canvas.width,canvas.height).data;
  const panes=[0,1].map(()=>({covered:0,chromatic:0,min:255,max:0,hash:0,bounds:[192,96,0,0]}));
  for(let i=0;i<pixels.length;i+=4) {
    if(pixels[i+3]<250) continue;
    const pane=panes[Math.floor((i/4 % canvas.width)/96)];
    const [r,g,b]=pixels.subarray(i,i+3);
    pane.covered++;
    const x=i/4%canvas.width, y=Math.floor(i/(4*canvas.width));
    pane.bounds=[Math.min(pane.bounds[0],x),Math.min(pane.bounds[1],y),
      Math.max(pane.bounds[2],x),Math.max(pane.bounds[3],y)];
    if(Math.max(r,g,b)-Math.min(r,g,b)>3) pane.chromatic++;
    pane.min=Math.min(pane.min,r); pane.max=Math.max(pane.max,r);
    pane.hash=(pane.hash*31+((r*256+g)*256+b))|0;
  }
  return panes.every(p=>p.covered>10 && (mode!=='depth' ||
    (p.chromatic===0 && p.min<p.max && p.max>0 && p.min<255))) && panes;
}

test('saved Box preview switches proxy, Box and synchronized split view, rotates and shows captured maps', {
  skip: process.env.BI_TEST_ASSET ? false : 'set BI_TEST_ASSET to a baked asset directory'
}, async () => {
  const server = await startServer({assetDir:resolve(process.env.BI_TEST_ASSET)});
  let browser;
  try {
    const backend = process.env.BI_TEST_BACKEND || 'auto';
    browser = await chromium.launch(chromiumOptions(backend));
    const page = await browser.newPage({viewport:{width:800,height:580}});
    const errors = [];
    page.on('pageerror',e => errors.push(e.message));
    page.on('console',m => {
      if (m.type()==='error' && /THREE|shader error|invalid shader|validation error|framebuffer is incomplete/i.test(m.text())) errors.push(m.text());
    });
    await page.goto(server.origin+'/web/viewer.html?manifest=/asset/manifest.json&backend='+encodeURIComponent(backend));
    await page.waitForFunction(() => /webgpu|webgl2|表示エラー/.test(document.querySelector('#status').textContent));
    assert.equal(await page.locator('#status.error').count(),0,await page.locator('#status').innerText());
    const manifest=await (await page.request.get(server.origin+'/asset/manifest.json')).json();
    const img2imgFaces=manifest.faceOrder.filter(id=>{
      const edit=manifest.faces[id].edit;
      return edit?.status==='accepted' && edit.provenance?.generator==='comfy-native-checkpoint-img2img';
    });
    for(const id of img2imgFaces){
      const row=page.locator('#faceQA tbody tr').nth(manifest.faceOrder.indexOf(id));
      assert.match(await row.textContent(),/img2img/);
    }
    const canvas = page.locator('#canvas');
    await page.selectOption('#mode','proxy');
    await page.selectOption('#mode','both');
    assert.equal(await page.locator('#labels').isVisible(),true);
    // Check the presented canvas itself. Renderer initialization and offscreen readback can
    // both succeed when a headless WebGPU device cannot allocate a swapchain image.
    await page.waitForFunction(presentedForeground,2);
    const diagnostics={};
    for(const mode of ['albedo','normal','depth']) {
      await page.selectOption('#shading',mode);
      assert.equal(await page.locator('#lightLabel').isVisible(),false);
      diagnostics[mode]=await (await page.waitForFunction(inspectDiagnostic,mode)).jsonValue();
    }
    for(const i of [0,1]) assert.notEqual(diagnostics.normal[i].hash,diagnostics.albedo[i].hash);
    await page.locator('#light').evaluate(input=>{
      input.value='-84';input.dispatchEvent(new Event('input'));
    });
    await page.selectOption('#shading','albedo');
    assert.deepEqual(await (await page.waitForFunction(inspectDiagnostic,'albedo')).jsonValue(),diagnostics.albedo);
    await page.selectOption('#shading','lit');
    await page.selectOption('#view','+Y');
    await page.selectOption('#projection','orthographic');
    const bounds = await canvas.boundingBox();
    const x=bounds.x+bounds.width*.2, y=bounds.y+bounds.height*.4;
    await page.mouse.move(x,y); await page.mouse.down();
    await page.mouse.move(x+30,y+20,{steps:4}); await page.mouse.up();
    await page.mouse.wheel(0,-180);
    await page.waitForFunction(presentedForeground,2);
    const savedPose=await canvas.screenshot();
    const coverage=await (await page.waitForFunction(inspectDiagnostic,'lit')).jsonValue();
    await page.selectOption('#mode','proxy');
    await page.selectOption('#mode','both');
    await page.waitForFunction(presentedForeground,2);
    assert.ok(savedPose.equals(await canvas.screenshot()),'preserve orbit and zoom while comparing');
    await page.selectOption('#projection','perspective');
    await page.selectOption('#projection','orthographic');
    await page.waitForFunction(presentedForeground,2);
    const projected=await (await page.waitForFunction(inspectDiagnostic,'lit')).jsonValue();
    for(const i of [0,1]) for(const j of [0,1,2,3])
      assert.ok(Math.abs(projected[i].bounds[j]-coverage[i].bounds[j])<=1,'preserve zoom on projection round trip');
    await page.selectOption('#mode','box');
    assert.equal(await page.locator('#sampling').inputValue(),'standard');
    await page.selectOption('#sampling','reduce_bleeding');
    await page.waitForFunction(() => document.querySelector('#status').textContent.includes('色にじみ低減'));
    await page.waitForFunction(presentedForeground,1);
    await page.selectOption('#sampling','standard');
    await page.waitForFunction(() => !document.querySelector('#status').textContent.includes('色にじみ低減'));
    await page.waitForFunction(presentedForeground,1);
    await page.selectOption('#mode','atlas');
    for(const [i,property] of ['roughness','metalness','ao'].entries()){
      const value='materialMap:'+i;
      assert.equal(await page.locator('#map option[value="'+value+'"]').count(),manifest.maps.includes(property)?1:0);
      if(!manifest.maps.includes(property)) continue;
      await page.selectOption('#map',value);
      await page.waitForFunction(label=>{
        const img=document.querySelector('#atlas');
        return img.alt.endsWith(label) && img.complete && img.naturalWidth>0;
      },['ラフネス','メタルネス','AO'][i]);
      assert.equal(await page.evaluate(async ({descriptor,channel})=>{
        const raw=new Uint8Array(await (await fetch('/asset/'+descriptor.path)).arrayBuffer());
        const canvas=document.createElement('canvas');
        canvas.width=descriptor.width;canvas.height=descriptor.height;
        const ctx=canvas.getContext('2d');ctx.drawImage(document.querySelector('#atlas'),0,0);
        const {data}=ctx.getImageData(0,0,canvas.width,canvas.height);
        for(let y=0;y<canvas.height;y++) for(let x=0;x<canvas.width;x++){
          const offset=4*(y*canvas.width+x);
          const expected=raw[4*((canvas.height-1-y)*canvas.width+x)+channel];
          if(data[offset]!==expected || data[offset+1]!==expected || data[offset+2]!==expected) return false;
        }
        return true;
      },{descriptor:manifest.textures.materialMap,channel:i}),true);
    }
    await page.selectOption('#map','intervalMap');
    await page.waitForFunction(() => document.querySelector('#atlas').naturalWidth > 0);
    assert.equal(await page.locator('#atlas').isVisible(),true);
    await page.selectOption('#map','intervalMap:3');
    await page.waitForFunction(()=>{
      const img=document.querySelector('#atlas');
      return img.alt.includes('A 空隙の終点') && img.complete && img.naturalWidth>0;
    });
    assert.equal(await page.evaluate(async()=>{
      const {textures}=await (await fetch('/asset/manifest.json')).json();
      const {path,width,height}=textures.intervalMap;
      const raw=new DataView(await (await fetch('/asset/'+path)).arrayBuffer());
      const canvas=document.createElement('canvas');
      canvas.width=width;canvas.height=height;
      const ctx=canvas.getContext('2d');ctx.drawImage(document.querySelector('#atlas'),0,0);
      const pixels=ctx.getImageData(0,0,width,height).data;
      for(let y=0;y<height;y++) for(let x=0;x<width;x++){
        const i=4*(y*width+x), source=4*((height-1-y)*width+x)+3;
        const value=Math.round(255*Math.max(0,Math.min(1,raw.getFloat16(2*source,true))));
        if(pixels[i+3]!==255 || [0,1,2].some(c=>Math.abs(pixels[i+c]-value)>1)) return false;
      }
      return true;
    }),true);
    await page.selectOption('#map','depth:+Z');
    await page.waitForFunction(()=>{
      const img=document.querySelector('#atlas');
      return img.alt.includes('+Z ガイド深度') && img.complete && img.naturalWidth>0;
    });
    // The preview uses the saved 16-bit guide data and preserves the proxy silhouette.
    const guide=await page.evaluate(async()=>{
      const manifest=await (await fetch('/asset/manifest.json')).json();
      const face=manifest.faces['+Z'];
      const [depth,guide]=await Promise.all([
        fetch('/asset/'+face.depth).then(r=>r.arrayBuffer()),
        fetch('/asset/'+face.guide).then(r=>r.blob()).then(createImageBitmap)
      ]);
      const raw=new DataView(depth);
      const canvas=document.createElement('canvas');
      canvas.width=guide.width;canvas.height=guide.height;
      const ctx=canvas.getContext('2d',{willReadFrequently:true});
      const read=image=>{
        ctx.clearRect(0,0,canvas.width,canvas.height);ctx.drawImage(image,0,0);
        return ctx.getImageData(0,0,canvas.width,canvas.height).data;
      };
      const mask=read(guide), preview=read(document.querySelector('#atlas'));
      let covered=0, differences=0;
      for(let i=0;i<raw.byteLength/2;i++) {
        const alpha=mask[4*i+3], z=raw.getFloat16(2*i,true);
        if(preview[4*i+3] !== (Number.isFinite(z)?alpha:0)) differences++;
        if(alpha!==255 || !Number.isFinite(z)) continue;
        covered++;
        const v=Math.round(Math.max(0,Math.min(1,1-z))*255);
        if([0,1,2].some(c=>Math.abs(preview[4*i+c]-v)>1)) differences++;
      }
      guide.close();return {covered,differences};
    });
    assert.ok(guide.covered>0);
    assert.equal(guide.differences,0);
    await page.selectOption('#mode','box');
    await page.getByRole('button',{name:'全体表示'}).click();
    assert.deepEqual(errors,[]);
  } finally { await browser?.close(); await server.close(); }
});
