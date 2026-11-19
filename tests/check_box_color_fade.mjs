// Isolate the production colour resolver with controlled hits and a known
// native fallback. check_box_color_sampling.mjs separately tests the real
// interval raymarch, six capture axes, normal, alpha and fragment depth.
// node tests/check_box_color_fade.mjs OUTPUT_DIR [webgl2|webgpu]
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {mkdir,readFile,writeFile} from 'node:fs/promises';
import {resolve,join} from 'node:path';
import {chromium} from 'playwright';
import {chromiumOptions} from '../worker/chromium.mjs';
import {startServer} from '../worker/server.mjs';

const outputDir=resolve(process.argv[2]||'.test-output/box-color-fade');
const backend=process.argv[3]||'webgl2';
assert.ok(['webgl2','webgpu'].includes(backend));
await mkdir(outputDir,{recursive:true});
const server=await startServer();
let browser;
const diagnostics=[];
try {
  browser=await chromium.launch(chromiumOptions(backend));
  const page=await browser.newPage();
  page.on('pageerror',e=>diagnostics.push(String(e)));
  page.on('console',m=>{if(m.type()==='error')diagnostics.push(m.text().slice(0,1600));});
  await page.goto(server.origin+'/bridge');
  const report=await page.evaluate(async backend=>{
    const T=await import('three/webgpu'),N=await import('three/tsl');
    const {ImpostorNodeMaterial}=await import('three-impostor');
    const {createBoxDisplayMaterial}=await import('/worker/box_display_material.mjs');
    const {makeRenderer}=await import('/web/asset_io.mjs');
    const renderer=await makeRenderer(backend);
    const size=33,resolution=64;
    renderer.setSize(size,size,false);renderer.setClearColor(0,0);
    renderer.toneMapping=T.NoToneMapping;renderer.outputColorSpace=T.SRGBColorSpace;
    const red=[.7,.04,.02],green=[.03,.55,.04],fallback=[.08,.15,.2];
    const colourData=new Float32Array(resolution*3*resolution*2*4);
    const normalData=new Float32Array(colourData.length);
    const intervalData=new Float32Array(resolution*3*resolution*4);
    const colourMap=new T.DataTexture(colourData,3*resolution,2*resolution,T.RGBAFormat,T.FloatType);
    colourMap.colorSpace=T.LinearSRGBColorSpace;
    const normalMap=new T.DataTexture(normalData,3*resolution,2*resolution,T.RGBAFormat,T.FloatType);
    const intervalMap=new T.DataTexture(intervalData,3*resolution,resolution,T.RGBAFormat,T.FloatType);
    const data={layout:'box',maps:[],resolution,colorMap:colourMap,normalMap,intervalMap,
      box:new T.Box3(new T.Vector3(-1,-1,-1),new T.Vector3(1,1,1))};
    function fill({zDistance=0,zCoverage=1,xVisible=false}={}) {
      for(let y=0;y<2*resolution;y++)for(let x=0;x<3*resolution;x++) {
        const at=4*(y*3*resolution+x),face=Math.floor(y/resolution)*3+Math.floor(x/resolution);
        colourData.set([...(face===4?red:green),1],at);
        normalData.set([.5,.5,1,face===4?zCoverage:1],at);
      }
      for(let y=0;y<resolution;y++)for(let x=0;x<3*resolution;x++) {
        const at=4*(y*3*resolution+x),axis=Math.floor(x/resolution);
        const first=axis===2?.5+zDistance/resolution:(axis===0&&xVisible?.5:1);
        intervalData.set([first,1,0,0],at);
      }
      for(const texture of [colourMap,normalMap,intervalMap])texture.needsUpdate=true;
    }
    const nativeBox=ImpostorNodeMaterial.prototype._setupBox;
    ImpostorNodeMaterial.prototype._setupBox=function(space) {
      return N.Fn(()=>{
        space.ray();
        space.setNormal(N.vec3(0,0,1));
        space.setDepth(N.positionLocal);
        return N.vec4(N.vec3(...fallback),1);
      })();
    };
    const modes=['view','view-fade','view-soft'];
    const materials=Object.fromEntries(modes.map(colorSampling=>{
      const material=createBoxDisplayMaterial(data,{lighting:'unlit',colorSampling});
      material.side=T.DoubleSide;
      return [colorSampling,material];
    }));
    const geometry=new T.PlaneGeometry(1,1),mesh=new T.Mesh(geometry,materials.view),scene=new T.Scene();
    scene.add(mesh);
    const camera=new T.OrthographicCamera(-.65,.65,.65,-.65,.1,10);
    const target=new T.RenderTarget(size,size);target.texture.colorSpace=T.SRGBColorSpace;
    const srgb=x=>255*(x<=.0031308?12.92*x:1.055*x**(1/2.4)-.055);
    const error=(rgba,linear)=>Math.max(...linear.map((v,c)=>Math.abs(srgb(v)-rgba[c])));
    function setYaw(degrees) {
      const a=degrees*Math.PI/180;camera.position.set(3*Math.sin(a),0,3*Math.cos(a));
      camera.lookAt(0,0,0);camera.updateMatrixWorld();camera.updateProjectionMatrix();
    }
    async function capture(mode) {
      mesh.material=materials[mode];renderer.setRenderTarget(target);renderer.render(scene,camera);
      const bytes=await renderer.readRenderTargetPixelsAsync(target,0,0,size,size);
      const at=4*(Math.floor(size/2)*size+Math.floor(size/2));
      return Array.from(bytes.subarray(at,at+4));
    }
    async function sweep(values,parameters,yaw) {
      setYaw(yaw);const rows=[];
      for(const value of values) {
        fill(parameters(value));const rgba={};
        for(const mode of modes)rgba[mode]=await capture(mode);
        rows.push({value,rgba});
      }
      const maxStep=Object.fromEntries(modes.map(mode=>[mode,
        Math.max(...rows.slice(1).map((row,i)=>Math.max(...row.rgba[mode].slice(0,3).map((n,c)=>Math.abs(n-rows[i].rgba[mode][c])))))
      ]));
      return {yaw,rows,maxAdjacentRGBByteStep:maxStep};
    }
    try {
      const valid=[];
      for(const yaw of [0,45]) {
        setYaw(yaw);fill();const rgba={};
        for(const mode of modes)rgba[mode]=await capture(mode);
        valid.push({yaw,rgba,maxError:Object.fromEntries(modes.map(mode=>[mode,error(rgba[mode],red)]))});
      }
      const single=await sweep(Array.from({length:89},(_,i)=>1.8+i*.05),zDistance=>({zDistance}),45);
      const equal=await sweep(Array.from({length:61},(_,i)=>i*.001),zCoverage=>({zCoverage,xVisible:true}),45);
      const unequal=await sweep(Array.from({length:121},(_,i)=>.0005+i*.000025),zCoverage=>({zCoverage,xVisible:true}),Math.atan(.2)*180/Math.PI);
      return {backend,size,resolution,fixture:'Production colour resolver; controlled plane hit, native fallback constant; native raymarch stubbed only in this isolated test',
        colors:{red,green,fallback},valid,singleDepthDropout:single,twoCandidateCoverageDropout:equal,
        unequalAngleCoverageDropout:unequal,
        endpointErrors:Object.fromEntries(modes.map(mode=>[mode,error(single.rows.at(-1).rgba[mode],fallback)]))};
    } finally {
      ImpostorNodeMaterial.prototype._setupBox=nativeBox;
      Object.values(materials).forEach(m=>m.dispose());geometry.dispose();target.dispose();
      for(const texture of [colourMap,normalMap,intervalMap])texture.dispose();renderer.dispose();
    }
  },backend);
  report.diagnostics=diagnostics;
  report.helperSha256=createHash('sha256').update(await readFile(new URL('../worker/box_display_material.mjs',import.meta.url))).digest('hex');
  await writeFile(join(outputDir,'fade-report.json'),JSON.stringify(report,null,2)+'\n');
  assert.deepEqual(diagnostics,[]);
  for(const pose of report.valid)for(const [mode,error] of Object.entries(pose.maxError))assert.ok(error<=1.1,mode+' sole valid source changed at '+pose.yaw);
  for(const error of Object.values(report.endpointErrors))assert.ok(error<=1.1,'No-source fallback mismatch');
  const single=report.singleDepthDropout.maxAdjacentRGBByteStep,two=report.twoCandidateCoverageDropout.maxAdjacentRGBByteStep;
  assert.ok(single.view>30,'Single-source fixture did not expose the original abrupt fallback');
  assert.ok(two.view>30,'Two-source fixture did not expose the original coverage boundary');
  for(const mode of ['view-fade','view-soft']) {
    assert.ok(single[mode]<single.view*.3,mode+' single-source transition stayed abrupt');
    assert.ok(two[mode]<two.view*.3,mode+' two-source transition stayed abrupt');
  }
  for(const sweep of [report.singleDepthDropout,report.twoCandidateCoverageDropout,report.unequalAngleCoverageDropout])
    for(const row of sweep.rows)for(const rgba of Object.values(row.rgba))assert.equal(rgba[3],255,'Fade changed object alpha');
  console.log(JSON.stringify({passed:true,helperSha256:report.helperSha256,valid:report.valid.map(({yaw,maxError})=>({yaw,maxError})),
    singleDepthDropout:single,twoCandidateCoverageDropout:two,
    unequalAngleCoverageDropout:report.unequalAngleCoverageDropout.maxAdjacentRGBByteStep,
    endpointErrors:report.endpointErrors},null,2));
}finally{
  await browser?.close();await server.close();
}
