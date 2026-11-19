// Test-only instrumentation of the current colour resolver. Production shader
// source and generated Inspector HTML remain untouched.
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {mkdir,readFile,writeFile} from 'node:fs/promises';
import {resolve,join} from 'node:path';
import {chromium} from 'playwright';
import {chromiumOptions} from '../worker/chromium.mjs';
import {startServer} from '../worker/server.mjs';

const [assetArg,cameraArg,probesArg,outputArg]=process.argv.slice(2);
assert(assetArg&&cameraArg&&probesArg&&outputArg);
const output=resolve(outputArg);await mkdir(output,{recursive:true});
const capture=JSON.parse(await readFile(cameraArg,'utf8'));
const camera=capture.yawFrames.find(v=>v.requestedYawDegrees===-30).actualCamera;
const probes=JSON.parse(await readFile(probesArg,'utf8'));
const source=await readFile(new URL('../worker/box_display_material.mjs',import.meta.url),'utf8');
let debugSource=source;
function replace(from,to){assert.equal(debugSource.split(from).length,2,from);debugSource=debugSource.replace(from,to);}
replace("colorSampling='native'}={}","colorSampling='native',debugChannel='color'}={}");
replace('return material;',"material.boxDebug=debugChannel;return material;");
replace("':box-color-'+this.boxColorSampling;","':box-color-'+this.boxColorSampling+':debug-'+this.boxDebug;");
replace('const selected=original.rgb.toVar();','const selected=original.rgb.toVar(); const probes={weights:[],visible:[],distances:[],rgb:[],coverage:[],valid:[]};');
replace('const rgb=c.rgb.div(max(c.a,1e-3)).mul(materialColor);',`const rgb=c.rgb.div(max(c.a,1e-3)).mul(materialColor);
        probes.rgb.push(rgb);probes.visible.push(visible);probes.distances.push(distance.mul(resolution));
        probes.coverage.push(min(c.a,n.a));probes.valid.push(select(valid,1,0));`);
replace('accumulated.addAssign(rgb.mul(weight));total.addAssign(weight);','accumulated.addAssign(rgb.mul(weight));total.addAssign(weight);probes.weights.push(weight);');
replace('return vec4(selected,original.a);',`const pack=(v,offset=0)=>vec4(vec3(...v.slice(offset,offset+3)),original.a);
      if(this.boxDebug==='hit')return vec4(p,original.a);
      if(this.boxDebug==='weights0')return pack(probes.weights.map(v=>v.div(max(total,1e-12))));
      if(this.boxDebug==='weights1')return pack(probes.weights.map(v=>v.div(max(total,1e-12))),3);
      if(this.boxDebug==='visible0')return pack(probes.visible);
      if(this.boxDebug==='visible1')return pack(probes.visible,3);
      if(this.boxDebug==='distances0')return pack(probes.distances);
      if(this.boxDebug==='distances1')return pack(probes.distances,3);
      if(this.boxDebug==='coverage')return vec4(vec3(probes.coverage[1],probes.coverage[4],total),original.a);
      if(this.boxDebug==='valid')return vec4(vec3(probes.valid[1],probes.valid[4],select(total.greaterThan(1e-8),0,1)),original.a);
      if(this.boxDebug==='native')return vec4(original.rgb,original.a);
      if(this.boxDebug==='minusX')return vec4(probes.rgb[1],original.a);
      if(this.boxDebug==='plusZ')return vec4(probes.rgb[4],original.a);
      return vec4(selected,original.a);`);
const hash=s=>createHash('sha256').update(s).digest('hex');
await writeFile(join(output,'debug-helper.mjs'),debugSource);
const server=await startServer({assetDir:resolve(assetArg)});
let browser;const errors=[];
try {
  browser=await chromium.launch(chromiumOptions('webgl2'));
  const page=await browser.newPage();
  page.on('pageerror',e=>errors.push(String(e)));
  page.on('console',m=>{if(m.type()==='error')errors.push(m.text().slice(0,2000));});
  await page.goto(server.origin+'/bridge');
  const result=await page.evaluate(async({camera,probes,debugSource})=>{
    const T=await import('three/webgpu');
    const {restoreBox}=await import('/worker/runtime.mjs');
    const {makeRenderer}=await import('/web/asset_io.mjs');
    const url=URL.createObjectURL(new Blob([debugSource],{type:'text/javascript'}));
    const {createBoxDisplayMaterial}=await import(url);URL.revokeObjectURL(url);
    const renderer=await makeRenderer('webgl2');
    renderer.setSize(1024,1024,false);renderer.setClearColor(0,0);
    renderer.toneMapping=T.NoToneMapping;renderer.outputColorSpace=T.SRGBColorSpace;
    const manifest=await (await fetch('/asset/manifest.json')).json();
    const files=new Map(await Promise.all(Object.values(manifest.textures).map(async d=>[d.path,new Uint8Array(await (await fetch('/asset/'+d.path)).arrayBuffer())])));
    const restored=await restoreBox(manifest,files,renderer,{lighting:'unlit',colorSampling:'view'});
    const nativeMaterial=restored.mesh.material,scene=new T.Scene();scene.add(restored.mesh);
    const cam=new T.OrthographicCamera(camera.left,camera.right,camera.top,camera.bottom,camera.near,camera.far);
    cam.position.fromArray(camera.position);cam.quaternion.fromArray(camera.quaternion);cam.up.fromArray(camera.up);cam.zoom=camera.zoom;
    cam.updateMatrixWorld();cam.updateProjectionMatrix();
    const colorTarget=new T.RenderTarget(1024,1024);colorTarget.texture.colorSpace=T.SRGBColorSpace;
    const floatTarget=new T.RenderTarget(1024,1024,{type:T.FloatType});floatTarget.texture.colorSpace=T.NoColorSpace;
    const rows=probes.map(p=>({...p,channels:{}}));
    const materialData=nativeMaterial.impostor;
    const channels=['color','hit','weights0','weights1','visible0','visible1','distances0','distances1','coverage','valid','native','minusX','plusZ'];
    try {
      for(const channel of channels) {
        const material=channel==='color'?nativeMaterial:createBoxDisplayMaterial(materialData,{lighting:'unlit',colorSampling:'view',debugChannel:channel});
        material.steps=nativeMaterial.steps;restored.mesh.material=material;
        const target=channel==='color'?colorTarget:floatTarget;
        renderer.setRenderTarget(target);renderer.render(scene,cam);
        for(const row of rows) {
          const values=await renderer.readRenderTargetPixelsAsync(target,row.xy[0],1023-row.xy[1],1,1);
          row.channels[channel]=Array.from(values);
        }
        if(material!==nativeMaterial)material.dispose();
      }
      return {assetId:manifest.asset_id,manifestTextures:manifest.textures,camera:{position:cam.position.toArray(),quaternion:cam.quaternion.toArray(),matrixWorld:cam.matrixWorld.toArray(),projectionMatrix:cam.projectionMatrix.toArray()},
        backend:'webgl2',size:[1024,1024],rowOrder:'top-to-bottom probes; readback uses height-1-y',
        channelOrder:{weights0:['+X','-X','+Y'],weights1:['-Y','+Z','-Z'],coverage:['-X min alpha','+Z min alpha','total raw weight'],valid:['-X valid','+Z valid','native fallback fraction']},rows};
    }finally{
      restored.mesh.material=nativeMaterial;restored.dispose();colorTarget.dispose();floatTarget.dispose();renderer.dispose();
    }
  },{camera,probes,debugSource});
  result.errors=errors;result.productionHelperSha256=hash(source);result.debugHelperSha256=hash(debugSource);
  result.referenceCapture=cameraArg;
  result.maxColorByteDelta=Math.max(...result.rows.flatMap(row=>row.expected.map((v,c)=>Math.abs(v-row.channels.color[c]))));
  await writeFile(join(output,'boundary-probes.json'),JSON.stringify(result,null,2)+'\n');
  assert.deepEqual(errors,[]);assert.ok(result.maxColorByteDelta<=1,'Diagnostic camera/renderer does not reproduce Inspector probe pixels');
  console.log(JSON.stringify({passed:true,maxColorByteDelta:result.maxColorByteDelta,rows:result.rows},null,2));
}finally{await browser?.close();await server.close();}
