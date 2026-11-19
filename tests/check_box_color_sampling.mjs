// One opt-in browser run: six capture axes and the +X/+Z angular boundary.
// node tests/check_box_color_sampling.mjs OUTPUT_DIR [webgl2|webgpu]
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {mkdir,readFile,writeFile} from 'node:fs/promises';
import {resolve,join} from 'node:path';
import {chromium} from 'playwright';
import {chromiumOptions} from '../worker/chromium.mjs';
import {startServer} from '../worker/server.mjs';

const outputDir=resolve(process.argv[2]||'.test-output/box-color-sampling');
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
  const {sheet,...report}=await page.evaluate(async backend=>{
    const T=await import('three/webgpu'),N=await import('three/tsl');
    const {ImpostorNodeMaterial}=await import('three-impostor');
    const {GLTFExporter}=await import('three/addons/exporters/GLTFExporter.js');
    const {bakeGLB,restoreBox}=await import('/worker/runtime.mjs');
    const {decodeRaw,encodeRaw,sha256}=await import('/worker/codec.mjs');
    const {makeRenderer}=await import('/web/asset_io.mjs');
    const size=160,half=.65,resolution=64;
    const renderer=await makeRenderer(backend);
    renderer.setSize(size,size,false);renderer.setClearColor(0,0);
    renderer.toneMapping=T.NoToneMapping;renderer.outputColorSpace=T.SRGBColorSpace;
    const source=new T.Mesh(new T.SphereGeometry(.4,32,20).scale(1,1.2,.8),
      new T.MeshStandardMaterial({color:0xffffff,roughness:.8,metalness:0}));
    const input=await new GLTFExporter().parseAsync(source,{binary:true});
    const baked=await bakeGLB(input,{renderer,resolution,maps:[],guides:false});
    source.geometry.dispose();source.material.dispose();
    // Deliberately distinct captures expose wrong face selection and UV flips.
    // Alpha and every geometry/normal/depth map remain the same for all modes.
    const palette=[[.65,.06,.04],[.04,.6,.08],[.07,.08,.65],[.55,.4,.04],[.05,.45,.5],[.45,.04,.45]];
    const desc=baked.manifest.textures.colorMap;
    const color=decodeRaw(baked.files.get(desc.path),desc.dtype);
    for(let y=0;y<desc.height;y++)for(let x=0;x<desc.width;x++){
      const i=4*(y*desc.width+x),face=Math.floor(y/resolution)*3+Math.floor(x/resolution);
      const a=T.DataUtils.fromHalfFloat(color[i+3]),u=((x%resolution)+.5)/resolution,v=((y%resolution)+.5)/resolution;
      const gradient=[.12*u,.15*v,.05*u+.03*v];
      for(let c=0;c<3;c++)color[i+c]=T.DataUtils.toHalfFloat((palette[face][c]+gradient[c])*a);
    }
    const bytes=encodeRaw(color),manifest=structuredClone(baked.manifest),files=new Map(baked.files);
    files.set(desc.path,bytes);manifest.textures.colorMap.sha256=await sha256(bytes);
    const objects={};
    for(const mode of ['native','view','dominant','view-fade','view-soft','view-feather'])
      objects[mode]=await restoreBox(manifest,files,renderer,{lighting:'unlit',colorSampling:mode});
    const oldNative=new ImpostorNodeMaterial(objects.native.mesh.material.impostor);
    oldNative.steps=objects.native.mesh.material.steps;
    oldNative.outputNode=N.vec4(N.diffuseColor.rgb,N.diffuseColor.a);
    const mesh=new T.Mesh(objects.native.mesh.geometry,oldNative),scene=new T.Scene();
    scene.add(mesh);
    const camera=new T.OrthographicCamera(-half,half,half,-half,.1,10);
    const target=new T.RenderTarget(size,size);
    target.texture.colorSpace=T.SRGBColorSpace;target.depthTexture=new T.DepthTexture(size,size,T.UnsignedIntType);
    const normalTarget=new T.RenderTarget(size,size);
    normalTarget.texture.name='output';normalTarget.texture.colorSpace=T.NoColorSpace;
    const normalOutput=N.mrt({output:N.vec4(N.normalWorld.mul(.5).add(.5),1)});
    const depthTarget=new T.RenderTarget(size,size,{depthBuffer:false});
    depthTarget.texture.colorSpace=T.NoColorSpace;
    const depthMaterial=new T.MeshBasicNodeMaterial();
    const encoded=N.min(N.floor(N.clamp(N.texture(target.depthTexture,N.screenUV).r,0,1).mul(16777215).add(.5)),16777215);
    depthMaterial.fragmentNode=N.vec4(N.floor(encoded.div(65536)),N.mod(N.floor(encoded.div(256)),256),N.mod(encoded,256),255).div(255);
    const quad=new T.QuadMesh(depthMaterial);
    async function capture(material){
      mesh.material=material;renderer.setRenderTarget(target);renderer.setMRT(null);renderer.render(scene,camera);
      const rgba=await renderer.readRenderTargetPixelsAsync(target,0,0,size,size);
      renderer.setRenderTarget(depthTarget);quad.render(renderer);
      const z=await renderer.readRenderTargetPixelsAsync(depthTarget,0,0,size,size);
      renderer.setRenderTarget(normalTarget);renderer.setMRT(normalOutput);renderer.render(scene,camera);
      const normal=await renderer.readRenderTargetPixelsAsync(normalTarget,0,0,size,size);
      renderer.setMRT(null);return {rgba,z,normal};
    }
    const maxDelta=(a,b)=>a.reduce((m,n,i)=>Math.max(m,Math.abs(n-b[i])),0);
    const alphaDelta=(a,b)=>a.reduce((m,n,i)=>i%4===3?Math.max(m,Math.abs(n-b[i])):m,0);
    const srgb=x=>255*(x<=.0031308?12.92*x:1.055*x**(1/2.4)-.055);
    const directions=[[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]];
    const names=['+X','-X','+Y','-Y','+Z','-Z'];
    const poses=directions.map((direction,axis)=>({name:names[axis],direction,axis}));
    for(const degrees of [44,45,46]){
      const a=degrees*Math.PI/180;poses.push({name:'yaw '+degrees,direction:[Math.sin(a),0,Math.cos(a)]});
    }
    const spans=objects.native.mesh.material.impostor.box.getSize(new T.Vector3()).toArray();
    const widthAxis=[2,2,0,0,0,0],heightAxis=[1,1,2,2,1,1];
    const probePixels=[[80,80],[64,80],[96,80],[80,64],[80,96]];
    const sheet=document.createElement('canvas'),ctx=sheet.getContext('2d');
    sheet.width=size*6;sheet.height=(size+22)*poses.length;
    ctx.fillStyle='#20252b';ctx.fillRect(0,0,sheet.width,sheet.height);
    function draw(rgba,col,row,label){
      const pixels=new Uint8ClampedArray(rgba.length);
      for(let y=0;y<size;y++)pixels.set(rgba.subarray((size-1-y)*size*4,(size-y)*size*4),y*size*4);
      const c=document.createElement('canvas');c.width=c.height=size;
      c.getContext('2d').putImageData(new ImageData(pixels,size,size),0,0);
      ctx.drawImage(c,col*size,row*(size+22)+22);
      ctx.fillStyle='white';ctx.font='12px sans-serif';ctx.fillText(label,col*size+4,row*(size+22)+15);
    }
    const views=[];
    for(const [row,pose] of poses.entries()){
      camera.position.set(...pose.direction).multiplyScalar(3);camera.up.set(0,1,0);
      if(pose.axis===2)camera.up.set(0,0,-1);if(pose.axis===3)camera.up.set(0,0,1);
      camera.lookAt(0,0,0);camera.updateMatrixWorld();camera.updateProjectionMatrix();
      const reference=await capture(oldNative),base=await capture(objects.native.mesh.material);
      const result={name:pose.name,nativeDefaultMaxByteDelta:maxDelta(reference.rgba,base.rgba),modes:{}};
      const rendered={native:base};
      for(const mode of ['view','dominant','view-fade','view-soft','view-feather']){
        const current=await capture(objects[mode].mesh.material);rendered[mode]=current;
        const data={alphaMaxByteDelta:alphaDelta(base.rgba,current.rgba),normalMaxByteDelta:maxDelta(base.normal,current.normal),
          depthMaxByteDelta:maxDelta(base.z,current.z),rgbMaxByteDeltaFromNative:maxDelta(base.rgba,current.rgba)};
        if(pose.axis!==undefined){
          data.probes=probePixels.map(([x,y])=>{
            const u=.5+(((x+.5)/size)*2-1)*half/spans[widthAxis[pose.axis]];
            const v=.5+(((y+.5)/size)*2-1)*half/spans[heightAxis[pose.axis]];
            const gradient=[.12*u,.15*v,.05*u+.03*v];
            const expected=palette[pose.axis].map((n,c)=>srgb(n+gradient[c]));
            const observed=Array.from(current.rgba.subarray(4*(y*size+x),4*(y*size+x)+4));
            return {xy:[x,y],uv:[u,v],expected,observed,maxRGBError:Math.max(...expected.map((n,c)=>Math.abs(n-observed[c])))};
          });
          data.maxCaptureRGBError=Math.max(...data.probes.map(p=>p.maxRGBError));
        }
        result.modes[mode]=data;
      }
      result.viewVsDominantMaxByteDelta=maxDelta(rendered.view.rgba,rendered.dominant.rgba);
      views.push(result);
      ['native','view','dominant','view-fade','view-soft','view-feather'].forEach((mode,col)=>draw(rendered[mode].rgba,col,row,pose.name+' / '+mode));
    }
    oldNative.dispose();Object.values(objects).forEach(o=>o.dispose());baked.live.dispose();
    target.dispose();normalTarget.dispose();depthTarget.dispose();depthMaterial.dispose();renderer.dispose();
    return {backend,resolution,size,lighting:'unlit',palette,gradient:['.12*u','.15*v','.05*u+.03*v'],
      visibility:'view/view-fade/dominant/view-feather: full through2texels, zero at3; view-soft: full through2, zero at6; invalid endpoints excluded',
      angularWeight:'view/view-fade cosine^8; view-soft cosine^4; dominant with fixed face-order ties',
      fade:'view-fade/view-soft separately fade to native using max reliable angular candidate; normal/depth/alpha untouched',
      views,sheet:sheet.toDataURL('image/png')};
  },backend);
  report.diagnostics=diagnostics;
  report.helperSha256=createHash('sha256').update(await readFile(new URL('../worker/box_display_material.mjs',import.meta.url))).digest('hex');
  await writeFile(join(outputDir,'synthetic-report.json'),JSON.stringify(report,null,2)+'\n');
  await writeFile(join(outputDir,'synthetic.png'),Buffer.from(sheet.split(',')[1],'base64'));
  assert.deepEqual(diagnostics,[]);
  let selectorChanged=false,boundaryDiff=false;
  for(const view of report.views){
    assert.equal(view.nativeDefaultMaxByteDelta,0,'Native default changed');
    for(const data of Object.values(view.modes)){
      for(const key of ['alphaMaxByteDelta','normalMaxByteDelta','depthMaxByteDelta'])assert.equal(data[key],0,view.name+' '+key);
      selectorChanged ||= data.rgbMaxByteDeltaFromNative>3;
      if(data.probes){
        assert.ok(data.probes.every(p=>p.observed[3]===255),'Missing axis probe');
        assert.ok(data.maxCaptureRGBError<=1.1,view.name+' capture RGB/UV mismatch: '+data.maxCaptureRGBError);
      }
    }
    if(view.name.startsWith('yaw'))boundaryDiff ||= view.viewVsDominantMaxByteDelta>3;
  }
  assert.ok(selectorChanged,'Selector had no measurable colour effect');
  assert.ok(boundaryDiff,'Continuous and dominant controls were indistinguishable');
  console.log(JSON.stringify({passed:true,helperSha256:report.helperSha256,
    views:report.views.map(v=>({name:v.name,nativeDefaultMaxByteDelta:v.nativeDefaultMaxByteDelta,viewVsDominantMaxByteDelta:v.viewVsDominantMaxByteDelta,
      modes:Object.fromEntries(Object.entries(v.modes).map(([key,value])=>[key,Object.fromEntries(Object.entries(value).filter(([k])=>k!=='probes'))]))}))},null,2));
}finally{
  await browser?.close();await server.close();
}
