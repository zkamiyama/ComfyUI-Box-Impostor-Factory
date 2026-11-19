// Explicit browser check: native resolution/hit/depth stay intact in unlit display.
// node tests/check_box_display.mjs OUTPUT_DIR [webgl2|webgpu]
// This script launches a browser; run separately from the CPU unit tests.
import assert from 'node:assert/strict';
import {mkdir,mkdtemp,rm,writeFile} from 'node:fs/promises';
import {join,resolve} from 'node:path';
import {chromium} from 'playwright';
import {chromiumOptions} from '../worker/chromium.mjs';
import {startServer} from '../worker/server.mjs';
import {makeFixtureGeometry} from './make_fixture.mjs';

const outputDir=resolve(process.argv[2] || '.test-output/box-display');
const backend=process.argv[3] || 'webgl2';
assert.ok(['webgl2','webgpu'].includes(backend));
await mkdir(outputDir,{recursive:true});
const temp=await mkdtemp(join(outputDir,'fixture-'));
const input=join(temp,'input.glb');
const knownLinear=[.18,.33,.72];
await writeFile(input,makeFixtureGeometry([
  [[0,0,0],[.8,1,.3],knownLinear],
  [[.12,.1,.27],[.3,.34,.2],knownLinear],
  [[.47,-.2,0],[.2,.25,.3],knownLinear],
]));
const server=await startServer({input});
let browser;
const diagnostics=[];
try {
  browser=await chromium.launch(chromiumOptions(backend));
  const page=await browser.newPage();
  page.on('pageerror',e=>diagnostics.push(String(e)));
  page.on('console',m=>{if(m.type()==='error')diagnostics.push(m.text().slice(0,1600));});
  await page.goto(server.origin+'/bridge');
  const {sheet,...report}=await page.evaluate(async({backend,knownLinear})=>{
    const T=await import('three/webgpu');
    const N=await import('three/tsl');
    const {ImpostorNodeMaterial}=await import('three-impostor');
    const {bakeGLB,restoreBox}=await import('/worker/runtime.mjs');
    const {makeRenderer}=await import('/web/asset_io.mjs');
    const size=160, renderer=await makeRenderer(backend);
    renderer.setSize(size,size,false);
    renderer.setClearColor(0,0);
    renderer.toneMapping=T.NoToneMapping;
    renderer.outputColorSpace=T.SRGBColorSpace;
    const data=new Uint8Array(await(await fetch('/job/input.glb')).arrayBuffer());
    const baked=await bakeGLB(data,{renderer,resolution:64,maps:['roughness','metalness','ao','emissive'],guides:false});
    const lit=await restoreBox(baked.manifest,baked.files,renderer);
    const unlit=await restoreBox(baked.manifest,baked.files,renderer,{lighting:'unlit'});
    const native=new ImpostorNodeMaterial(lit.mesh.material.impostor);
    native.steps=lit.mesh.material.steps;
    const mesh=new T.Mesh(lit.mesh.geometry,native);
    const scene=new T.Scene();
    const sky=new T.HemisphereLight(0xffffff,0x777777,1.4);
    const sun=new T.DirectionalLight(0xffffff,2);
    sun.position.set(3,4,5);scene.add(sky,sun,mesh);
    const camera=new T.OrthographicCamera(-.7,.7,.7,-.7,.1,10);
    // Measure the ordinary colour output; linear normal diagnostics use their own pass.
    const target=new T.RenderTarget(size,size);
    target.texture.colorSpace=T.SRGBColorSpace;
    target.depthTexture=new T.DepthTexture(size,size,T.UnsignedIntType);
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
      mesh.material=material;
      renderer.setRenderTarget(target);renderer.setMRT(null);
      renderer.render(scene,camera);
      const rgba=await renderer.readRenderTargetPixelsAsync(target,0,0,size,size);
      renderer.setRenderTarget(depthTarget);quad.render(renderer);
      const z=await renderer.readRenderTargetPixelsAsync(depthTarget,0,0,size,size);
      renderer.setRenderTarget(normalTarget);renderer.setMRT(normalOutput);renderer.render(scene,camera);
      const normal=await renderer.readRenderTargetPixelsAsync(normalTarget,0,0,size,size);
      renderer.setMRT(null);
      return {rgba,normal,z};
    }
    const maxDelta=(a,b)=>a.reduce((max,n,i)=>Math.max(max,Math.abs(n-b[i])),0);
    const alphaDelta=(a,b)=>a.reduce((max,n,i)=>i%4===3?Math.max(max,Math.abs(n-b[i])):max,0);
    const srgb=x=>255*(x<=.0031308?12.92*x:1.055*Math.pow(x,1/2.4)-.055);
    const expectedSRGB=knownLinear.map(srgb);
    const sheet=document.createElement('canvas'), ctx=sheet.getContext('2d');
    sheet.width=size*4;sheet.height=(size+24)*2;
    ctx.fillStyle='#20252b';ctx.fillRect(0,0,sheet.width,sheet.height);
    function draw(rgba,col,row,label){
      const pixels=new Uint8ClampedArray(rgba.length);
      for(let y=0;y<size;y++)pixels.set(rgba.subarray((size-1-y)*size*4,(size-y)*size*4),y*size*4);
      const canvas=document.createElement('canvas');canvas.width=canvas.height=size;
      canvas.getContext('2d').putImageData(new ImageData(pixels,size,size),0,0);
      ctx.drawImage(canvas,col*size,row*(size+24)+24);
      ctx.fillStyle='white';ctx.font='12px sans-serif';ctx.fillText(label,col*size+5,row*(size+24)+16);
    }
    const views=[];
    for(const [index,degrees] of [0,30].entries()){
      const a=degrees*Math.PI/180;camera.position.set(3*Math.sin(a),0,3*Math.cos(a));
      camera.lookAt(0,0,0);camera.updateMatrixWorld();camera.updateProjectionMatrix();
      const plain=await capture(native), relit=await capture(lit.mesh.material), rgb=await capture(unlit.mesh.material);
      const channels=[[],[],[]], depths=[];
      let foreground=0,background=0;
      for(let p=0;p<size*size;p++){
        const i=p*4,x=p%size,y=Math.floor(p/size);
        if(rgb.rgba[i+3]===255){
          foreground++;
          depths.push((65536*rgb.z[i]+256*rgb.z[i+1]+rgb.z[i+2])/16777215);
          if(x>0&&x<size-1&&y>0&&y<size-1&&[-1,1,-size,size].every(d=>rgb.rgba[(p+d)*4+3]===255))
            for(let c=0;c<3;c++)channels[c].push(rgb.rgba[i+c]);
        } else if(rgb.rgba[i+3]===0) background++;
      }
      const medianSRGB=channels.map(v=>v.sort((a,b)=>a-b)[Math.floor(v.length/2)]);
      const nativeDepthRangeWorld=(Math.max(...depths)-Math.min(...depths))*(camera.far-camera.near);
      sky.intensity=0;sun.intensity=19;sun.color.setRGB(.2,1,.3);sun.position.set(-4,1,-3);
      unlit.mesh.material.emissive.setRGB(9,4,2);
      const perturbed=await capture(unlit.mesh.material);
      sky.intensity=1.4;sun.intensity=2;sun.color.setRGB(1,1,1);sun.position.set(3,4,5);
      unlit.mesh.material.emissive.setRGB(0,0,0);
      views.push({degrees,foreground,background,interior:channels[0].length,
        nativeVsDefaultMaxByteDelta:maxDelta(plain.rgba,relit.rgba),
        unlitSilhouetteMaxByteDelta:alphaDelta(relit.rgba,rgb.rgba),
        unlitNormalMaxByteDelta:maxDelta(relit.normal,rgb.normal),
        unlitDepthMaxByteDelta:maxDelta(relit.z,rgb.z),
        changedLightingUnlitMaxByteDelta:maxDelta(rgb.rgba,perturbed.rgba),
        litVsUnlitMaxByteDelta:maxDelta(relit.rgba,rgb.rgba),
        expectedSRGB,medianSRGB,maxSRGBError:Math.max(...medianSRGB.map((v,i)=>Math.abs(v-expectedSRGB[i]))),
        nativeDepthRangeWorld});
      draw(plain.rgba,0,index,'Native '+degrees+' deg');
      draw(relit.rgba,1,index,'Default helper');
      draw(rgb.rgba,2,index,'Unlit / NoToneMapping');
      draw(rgb.normal,3,index,'Native normal');
    }
    native.dispose();lit.dispose();unlit.dispose();baked.live.dispose();
    target.dispose();normalTarget.dispose();depthTarget.dispose();depthMaterial.dispose();renderer.dispose();
    return {backend,resolution:64,size,lighting:'unlit',sampling:'native',views,sheet:sheet.toDataURL('image/png')};
  },{backend,knownLinear});
  report.diagnostics=diagnostics;
  await writeFile(join(outputDir,'synthetic-report.json'),JSON.stringify(report,null,2)+'\n');
  await writeFile(join(outputDir,'synthetic.png'),Buffer.from(sheet.split(',')[1],'base64'));
  assert.deepEqual(diagnostics,[]);
  for(const v of report.views){
    assert.ok(v.foreground>500 && v.background>500 && v.interior>300,'Missing synthetic silhouette');
    for(const key of ['nativeVsDefaultMaxByteDelta','unlitSilhouetteMaxByteDelta','unlitNormalMaxByteDelta','unlitDepthMaxByteDelta','changedLightingUnlitMaxByteDelta'])
      assert.equal(v[key],0,key);
    assert.ok(v.litVsUnlitMaxByteDelta>10,'Unlit did not remove lighting');
    assert.ok(v.maxSRGBError<=1.1,'Linear-sRGB transfer mismatch');
    assert.ok(v.nativeDepthRangeWorld>.05,'Native raycast depth was replaced by box depth');
  }
  console.log(JSON.stringify(report,null,2));
}finally{
  await browser?.close();await server.close();await rm(temp,{recursive:true,force:true});
}
