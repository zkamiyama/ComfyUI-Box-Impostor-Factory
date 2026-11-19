// Compile the production source-depth feather function against controlled
// flat, steep linear-ramp and discontinuous-step interval textures.
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {mkdir,readFile,writeFile} from 'node:fs/promises';
import {resolve,join} from 'node:path';
import {chromium} from 'playwright';
import {chromiumOptions} from '../worker/chromium.mjs';
import {startServer} from '../worker/server.mjs';

const output=resolve(process.argv[2]||'.test-output/box-source-feather');
await mkdir(output,{recursive:true});
const source=await readFile(new URL('../worker/box_display_material.mjs',import.meta.url),'utf8');
const server=await startServer();let browser;const errors=[];
try {
  browser=await chromium.launch(chromiumOptions('webgl2'));
  const page=await browser.newPage();
  page.on('pageerror',e=>errors.push(String(e)));
  page.on('console',m=>{if(m.type()==='error')errors.push(m.text().slice(0,2000));});
  await page.goto(server.origin+'/bridge');
  const report=await page.evaluate(async source=>{
    const T=await import('three/webgpu'),N=await import('three/tsl');
    const {makeRenderer}=await import('/web/asset_io.mjs');
    const url=URL.createObjectURL(new Blob([source+'\nexport {sourceDepthEdgeWeight};\n'],{type:'text/javascript'}));
    const {sourceDepthEdgeWeight}=await import(url);URL.revokeObjectURL(url);
    const renderer=await makeRenderer('webgl2');renderer.setSize(1,1,false);renderer.toneMapping=T.NoToneMapping;
    const resolution=128,bytes=new Float32Array(3*resolution*resolution*4);
    const intervals=new T.DataTexture(bytes,3*resolution,resolution,T.RGBAFormat,T.FloatType);
    const target=new T.RenderTarget(1,1,{type:T.FloatType,depthBuffer:false});target.texture.colorSpace=T.NoColorSpace;
    const uv=N.uniform(new T.Vector2(.5,.5)),quad=new T.QuadMesh();
    const positions=[-7,-6,-5.75,-5.5,-5.25,-5,-4,-3,-2,-1,-.5,-.25,0,.25,.5,1,2,3,4,5,5.25,5.5,5.75,6,7];
    const rows=[];
    try {
      for(const gradientAxis of [0,1])for(const kind of ['flat','linear','step']) {
        for(let y=0;y<resolution;y++)for(let x=0;x<3*resolution;x++) {
          const at=4*(y*3*resolution+x),s=gradientAxis===0?x%resolution:y;
          // The steep ramp changes by19.2depthtexels per texel, above the
          // absolute threshold, but has no isolated jump in its valid region.
          const value=kind==='flat'?.5:kind==='linear'?.5+.15*(s-64):s<64?.3:.7;
          bytes.set([value,1-value,0,0],at);
        }
        intervals.needsUpdate=true;
        for(let axis=0;axis<3;axis++)for(const positive of [true,false]) {
          const material=new T.MeshBasicNodeMaterial();
          material.fragmentNode=N.Fn(()=>N.vec4(sourceDepthEdgeWeight(uv,axis,positive,resolution,intervals),0,0,1))();
          quad.material=material;const samples=[];
          for(const offset of positions) {
            uv.value.set(.5,.5);uv.value.setComponent(gradientAxis,(64+offset)/resolution);
            renderer.setRenderTarget(target);quad.render(renderer);
            const pixel=await renderer.readRenderTargetPixelsAsync(target,0,0,1,1);
            samples.push({offset,weight:pixel[0]});
          }
          rows.push({gradientAxis,kind,axis,positive,samples});material.dispose();
        }
      }
      return {backend:'webgl2',resolution,description:'Production sourceDepthEdgeWeight compiled unchanged; synthetic interval data only',
        parameters:{radiusTexels:6,minJumpTexels:16,outerSlopeMultiplier:4,outerSlopeAllowanceTexels:8,featherPower:8,weightFloor:.001},rows};
    }finally{intervals.dispose();target.dispose();renderer.dispose();}
  },source);
  report.errors=errors;report.helperSha256=createHash('sha256').update(source).digest('hex');
  await writeFile(join(output,'source-feather-report.json'),JSON.stringify(report,null,2)+'\n');
  assert.deepEqual(errors,[]);
  for(const row of report.rows) {
    if(row.kind!=='step')for(const sample of row.samples)assert.ok(Math.abs(sample.weight-1)<1e-5,row.kind+' falsely marked as a seam');
    else {
      const at=offset=>row.samples.find(s=>s.offset===offset).weight;
      assert.ok(at(0)<=.00101&&at(0)>=.00099,'Isolated step was not feathered');
      assert.ok(at(6)>=.99999&&at(-6)>=.99999,'Feather extends beyond its radius');
      for(const sample of row.samples)assert.ok(sample.weight>=.00099&&sample.weight<=1.00001,'Invalid relative weight');
      for(const sample of row.samples.filter(s=>s.offset>0))assert.ok(Math.abs(sample.weight-at(-sample.offset))<1e-5,'Feather has a direction or atlas parity bias');
      const positive=row.samples.filter(s=>s.offset>=0);
      for(let i=1;i<positive.length;i++)assert.ok(positive[i].weight+1e-5>=positive[i-1].weight,'Feather is not monotonic');
    }
  }
  console.log(JSON.stringify({passed:true,helperSha256:report.helperSha256,fixtures:report.rows.length,
    flatAndRampMaxError:Math.max(...report.rows.filter(r=>r.kind!=='step').flatMap(r=>r.samples.map(s=>Math.abs(1-s.weight)))),
    representativeStep:report.rows.find(r=>r.kind==='step').samples},null,2));
}finally{await browser?.close();await server.close();}
