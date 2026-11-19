// Run from Factory qa/: node qa/check_interval_morph_depth.mjs CONFIG.json
// One fixed 45-degree camera. Reads the actual fragment-depth attachment;
// differences do not depend on RGB, normals, lights, or an encoded fake depth.
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {mkdir,readFile,writeFile} from 'node:fs/promises';
import {join,resolve} from 'node:path';
import {chromium} from 'playwright';
import {chromiumOptions} from '../worker/chromium.mjs';
import {startServer} from '../worker/server.mjs';

const config=JSON.parse(await readFile(resolve(process.argv[2]),'utf8'));
const output=resolve(config.output),assetDir=resolve(config.assetDir);
const backend=config.backend??'webgl2',pixels=config.pixels??320;
const bands=config.bands??[8];
assert(['webgl2','webgpu'].includes(backend));
await mkdir(output,{recursive:true});
const manifestBytes=await readFile(join(assetDir,'manifest.json'));
const report={status:'running',startedAt:new Date().toISOString(),config,
  manifestSHA256:createHash('sha256').update(manifestBytes).digest('hex'),diagnostics:[],
  materialSHA256:createHash('sha256').update(await readFile(new URL('../worker/box_morph_material.mjs',import.meta.url))).digest('hex'),
  contract:'Same immutable restored Q2 maps, same camera and interval-morph solver; change morphStrength 0 to 1 only.',
  depthReadback:'Actual RenderTarget DepthTexture sampled to linear RGB24; common foreground excludes silhouette/background changes.'};
let browser,server,timer,error;
try {
  server=await startServer({assetDir});
  browser=await chromium.launch(chromiumOptions(backend));
  report.browserVersion=await browser.version();
  const page=await browser.newPage();
  await page.exposeFunction('qaProgress',message=>console.log('DEPTH_QA '+message));
  page.on('pageerror',e=>report.diagnostics.push('page: '+String(e)));
  page.on('console',m=>{if(m.type()==='error')report.diagnostics.push('console: '+m.text().slice(0,1800));});
  await page.goto(server.origin+'/bridge');
  const evaluation=page.evaluate(async({backend,pixels,bands,framing,compareSkip})=>{
    const T=await import('three/webgpu'),N=await import('three/tsl');
    const {restoreBox,createBoxMaterial}=await import('/worker/runtime.mjs');
    const {compactPixels}=await import('/worker/codec.mjs');
    const {makeRenderer}=await import('/web/asset_io.mjs');
    const renderer=await makeRenderer(backend);
    renderer.setSize(pixels,pixels,false);renderer.setClearColor(0,0);
    renderer.toneMapping=T.NoToneMapping;renderer.outputColorSpace=T.SRGBColorSpace;
    const actualBackend=renderer.backend.isWebGPUBackend?'webgpu':'webgl2';
    if(actualBackend!==backend)throw new Error('Unexpected backend '+actualBackend);
    const gl=renderer.domElement.getContext('webgl2');
    const debug=gl?.getExtension('WEBGL_debug_renderer_info');
    const gpu=debug?{vendor:gl.getParameter(debug.UNMASKED_VENDOR_WEBGL),renderer:gl.getParameter(debug.UNMASKED_RENDERER_WEBGL)}:null;
    const shaderSources=[];
    let shaderLabel='restore';
    if(gl){
      const original=gl.shaderSource;
      gl.shaderSource=function(shader,source){
        shaderSources.push({label:shaderLabel,type:gl.getShaderParameter(shader,gl.SHADER_TYPE)===gl.FRAGMENT_SHADER?'fragment':'vertex',
          characters:source.length,lines:source.split('\n').length,morphFieldMentions:(source.match(/boxMorphField/g)??[]).length});
        return original.call(this,shader,source);
      };
    }
    const manifest=await(await fetch('/asset/manifest.json')).json();
    const files=new Map(await Promise.all(Object.values(manifest.textures).map(async d=>
      [d.path,new Uint8Array(await(await fetch('/asset/'+d.path)).arrayBuffer())])));
    await globalThis.qaProgress('asset bytes loaded');
    const restored=await restoreBox(manifest,files,renderer,{lighting:'unlit',colorSampling:'view',
      shapeSampling:'interval-morph',morphStrength:0,morphBandTexels:bands[0],morphStepTexels:.75});
    await globalThis.qaProgress('textures restored');
    const mesh=restored.mesh,originalMaterial=mesh.material,data=originalMaterial.impostor;
    if(compareSkip&&!data.morphDistanceMap)throw new Error('Dedicated morph skip cache is missing');
    const size=data.box.getSize(new T.Vector3()),center=data.box.getCenter(new T.Vector3());
    const target=framing?.target?new T.Vector3(...framing.target):center;
    const diagonal=size.length(),halfExtent=framing?.halfExtent??diagonal*.55;
    const distance=halfExtent/Math.tan(20*Math.PI/180);
    const camera=new T.OrthographicCamera(-halfExtent,halfExtent,halfExtent,-halfExtent,
      Math.max(diagonal*.001,.0001),distance+diagonal*3);
    camera.position.copy(target).add(new T.Vector3(Math.SQRT1_2,0,Math.SQRT1_2).multiplyScalar(distance));
    camera.lookAt(target);camera.updateMatrixWorld();camera.updateProjectionMatrix();
    const scene=new T.Scene();scene.add(mesh);
    const colorTarget=new T.RenderTarget(pixels,pixels,{type:T.UnsignedByteType,format:T.RGBAFormat});
    colorTarget.texture.colorSpace=T.SRGBColorSpace;
    colorTarget.depthTexture=new T.DepthTexture(pixels,pixels,T.UnsignedIntType);
    const depthTarget=new T.RenderTarget(pixels,pixels,{type:T.UnsignedByteType,format:T.RGBAFormat,depthBuffer:false});
    depthTarget.texture.colorSpace=T.NoColorSpace;
    const depthMaterial=new T.MeshBasicNodeMaterial();depthMaterial.toneMapped=false;
    const code=N.min(N.floor(N.clamp(N.texture(colorTarget.depthTexture,N.screenUV).r,0,1).mul(16777215).add(.5)),16777215);
    depthMaterial.fragmentNode=N.vec4(N.floor(code.div(65536)),N.mod(N.floor(code.div(256)),256),N.mod(code,256),255).div(255);
    const quad=new T.QuadMesh(depthMaterial),cases=[],images=[],failures=[];
    const read=async target=>compactPixels(await renderer.readRenderTargetPixelsAsync(target,0,0,pixels,pixels),pixels,pixels,'uint8',backend);
    const pack=(a,i)=>65536*a[i]+256*a[i+1]+a[i+2];
    async function capture(material,label){
      mesh.material=material;
      shaderLabel=label;
      renderer.setMRT(null);renderer.setRenderTarget(colorTarget);
      const compileStart=performance.now();
      await renderer.compileAsync(scene,camera);
      const compileMilliseconds=performance.now()-compileStart,start=performance.now();
      renderer.render(scene,camera);
      const rgba=await read(colorTarget);
      renderer.setRenderTarget(depthTarget);quad.render(renderer);
      const depth=await read(depthTarget);
      return {rgba,depth,milliseconds:performance.now()-start,compileMilliseconds};
    }
    function png(rgba){
      const displayed=new Uint8ClampedArray(rgba.length);
      // compactPixels removes row padding; WebGL readback has a bottom origin.
      for(let y=0;y<pixels;y++){
        const sy=backend==='webgl2'?pixels-1-y:y;
        displayed.set(rgba.subarray(sy*pixels*4,(sy+1)*pixels*4),y*pixels*4);
      }
      const c=document.createElement('canvas');c.width=c.height=pixels;
      c.getContext('2d').putImageData(new ImageData(displayed,pixels,pixels),0,0);
      return c.toDataURL('image/png');
    }
    try {
      for(const band of bands){
        const baseline=createBoxMaterial(data,{lighting:'unlit',colorSampling:'view',shapeSampling:'interval-morph',morphStrength:0,morphBandTexels:band,morphStepTexels:.75});
        const changed=createBoxMaterial(data,{lighting:'unlit',colorSampling:'view',shapeSampling:'interval-morph',morphStrength:1,morphBandTexels:band,morphStepTexels:.75});
        try {
          await globalThis.qaProgress('band '+band+' strength0 render start');
          const a=await capture(baseline,'band'+band+'-strength0');
          await globalThis.qaProgress('band '+band+' strength0 complete '+a.milliseconds.toFixed(1)+'ms; strength1 render start');
          const b=await capture(changed,'band'+band+'-strength1');
          await globalThis.qaProgress('band '+band+' strength1 complete '+b.milliseconds.toFixed(1)+'ms');
          const deltas=[],heat=new Uint8Array(a.rgba.length);
          let foreground0=0,foreground1=0,common=0,added=0,removed=0,alphaChanged=0,rgbChanged=0,changedDepth=0,aboveFourCodes=0;
          let maxCodeDelta=0,minSignedCodeDelta=0,maxSignedCodeDelta=0,minCode=16777215,maxCode=0;
          for(let i=0;i<a.rgba.length;i+=4){
            const fa=a.rgba[i+3]>0,fb=b.rgba[i+3]>0;
            foreground0+=fa;foreground1+=fb;added+=!fa&&fb;removed+=fa&&!fb;
            alphaChanged+=a.rgba[i+3]!==b.rgba[i+3];
            rgbChanged+=a.rgba[i]!==b.rgba[i]||a.rgba[i+1]!==b.rgba[i+1]||a.rgba[i+2]!==b.rgba[i+2];
            if(fa){const x=pack(a.depth,i);minCode=Math.min(minCode,x);maxCode=Math.max(maxCode,x);}
            if(fa&&fb){
              common++;const d=pack(b.depth,i)-pack(a.depth,i),ad=Math.abs(d);
              if(ad){changedDepth++;deltas.push(ad);}
              aboveFourCodes+=ad>4;maxCodeDelta=Math.max(maxCodeDelta,ad);
              minSignedCodeDelta=Math.min(minSignedCodeDelta,d);maxSignedCodeDelta=Math.max(maxSignedCodeDelta,d);
              const v=ad>4?Math.min(255,Math.round(48+32*Math.log2(ad/4))):0;
              heat.set([v,Math.round(v*.2),0,255],i);
            } else heat.set(!fa&&fb?[0,255,120,255]:[0,0,0,255],i);
          }
          deltas.sort((x,y)=>x-y);
          const quantile=q=>deltas.length?deltas[Math.min(deltas.length-1,Math.floor(q*(deltas.length-1)))]:0;
          const worldPerCode=(camera.far-camera.near)/16777215;
          const row={bandTexels:band,bandLocal:band*Math.min(size.x,size.y,size.z)/data.resolution,
            morphCache:{present:Boolean(data.morphDistanceMap),bandTexels:data.morphDistanceBandTexels??null},
            steps:baseline.steps,foreground0,foreground1,commonForeground:common,addedPixels:added,removedPixels:removed,
            alphaChangedPixels:alphaChanged,rgbChangedPixels:rgbChanged,
            depth:{commonChangedPixels:changedDepth,commonChangedMoreThanFourCodes:aboveFourCodes,
              maxCodeDelta,p50ChangedCodeDelta:quantile(.5),p95ChangedCodeDelta:quantile(.95),
              minSignedCodeDelta,maxSignedCodeDelta,maxWorldDelta:maxCodeDelta*worldPerCode,
              worldPerCode,nativeDepthRangeWorld:(maxCode-minCode)*worldPerCode},
            milliseconds:{strength0:a.milliseconds,strength1:b.milliseconds},
            compileMilliseconds:{strength0:a.compileMilliseconds,strength1:b.compileMilliseconds}};
          cases.push(row);
          if(common<100)failures.push('Insufficient common foreground at band '+band);
          if(aboveFourCodes===0)failures.push('No depth change above four RGB24 codes at band '+band);
          if((maxCode-minCode)*worldPerCode<diagonal*.005)failures.push('No meaningful surface depth range at band '+band);
          images.push({name:'depth45-band'+band+'-strength0.png',dataURL:png(a.rgba)},
            {name:'depth45-band'+band+'-strength1.png',dataURL:png(b.rgba)},
            {name:'depth45-band'+band+'-difference.png',dataURL:png(heat)});
          if(compareSkip){
            const fineData={...data,morphDistanceMap:null,morphDistanceBandTexels:undefined};
            const fine=createBoxMaterial(fineData,{lighting:'unlit',colorSampling:'view',shapeSampling:'interval-morph',morphStrength:1,morphBandTexels:band,morphStepTexels:.75});
            try {
              await globalThis.qaProgress('band '+band+' fine-only equivalence render start');
              const f=await capture(fine,'band'+band+'-fine-only');
              let alpha=0,rgb=0,depth=0,maxDepth=0;
              for(let i=0;i<f.rgba.length;i+=4){
                alpha+=f.rgba[i+3]!==b.rgba[i+3];
                rgb+=f.rgba[i]!==b.rgba[i]||f.rgba[i+1]!==b.rgba[i+1]||f.rgba[i+2]!==b.rgba[i+2];
                const d=Math.abs(pack(f.depth,i)-pack(b.depth,i));depth+=d>0;maxDepth=Math.max(maxDepth,d);
              }
              row.skipVsFine={pixels:pixels*pixels,alphaChangedPixels:alpha,rgbChangedPixels:rgb,
                depthChangedPixels:depth,maxDepthCodeDelta:maxDepth,exactAlphaAndDepth:alpha===0&&depth===0,
                fineMilliseconds:f.milliseconds,fineCompileMilliseconds:f.compileMilliseconds};
              if(alpha||depth)failures.push('Skip/fine depth or alpha differs at band '+band);
              await globalThis.qaProgress('band '+band+' fine-only equivalence '+JSON.stringify(row.skipVsFine));
            } finally {fine.dispose();mesh.material=changed;}
          }
        } finally {baseline.dispose();changed.dispose();mesh.material=originalMaterial;}
      }
    } finally {
      renderer.setRenderTarget(null);restored.dispose();colorTarget.dispose();depthTarget.dispose();depthMaterial.dispose();await renderer.dispose();
    }
    return {actualBackend,gpu,shaderSources,threeRevision:T.REVISION,assetId:manifest.assetId??null,resolution:manifest.resolution,
      camera:{yawDegrees:45,pitchDegrees:0,projection:'orthographic',pixels:[pixels,pixels],target:target.toArray(),
        position:camera.position.toArray(),halfExtent,near:camera.near,far:camera.far},cases,failures,images};
  },{backend,pixels,bands,framing:config.framing,compareSkip:config.compareSkip??false});
  const {images,...result}=await Promise.race([evaluation,new Promise((_,reject)=>{
    timer=setTimeout(()=>reject(new Error('Single-angle depth QA exceeded 300 seconds')),300000);
  })]);
  clearTimeout(timer);Object.assign(report,result);
  for(const image of images)await writeFile(join(output,image.name),Buffer.from(image.dataURL.split(',')[1],'base64'));
  if(report.diagnostics.length)report.failures.push('Browser reported '+report.diagnostics.length+' errors');
  report.status=report.failures.length?'fail':'pass';
  if(report.status==='fail')error=new Error(report.failures.join('\n'));
} catch(e){error=e;report.status='fail';report.error=String(e);report.stack=e?.stack??null;}
finally {
  clearTimeout(timer);report.finishedAt=new Date().toISOString();
  await writeFile(join(output,'depth45-report.json'),JSON.stringify(report,null,2)+'\n');
  await browser?.close();await server?.close();
}
console.log(JSON.stringify(report,null,2));
if(error)throw error;
