// Explicit GPU compatibility check for three-impostor 0.4.2.
// Place in the Factory tests directory and run separately from CPU unit tests:
// node tests/check_box_distance_map.mjs OUTPUT_DIR [webgl2|webgpu] [41,64,96]
// Uses the project's server and Chromium policy; installs no dependencies.
import assert from 'node:assert/strict';
import {mkdir,mkdtemp,rm,writeFile} from 'node:fs/promises';
import {join,resolve} from 'node:path';
import {chromium} from 'playwright';
import {chromiumOptions} from '../worker/chromium.mjs';
import {startServer} from '../worker/server.mjs';
import {makeFixtureGeometry} from './make_fixture.mjs';

const outputDir=resolve(process.argv[2] || '.test-output/box-distance-map');
const backend=process.argv[3] || 'webgl2';
const resolutions=(process.argv[4] || '41,64,96').split(',').map(Number);
assert.ok(['webgl2','webgpu'].includes(backend),'Expected webgl2 or webgpu');
assert.ok(resolutions.length && resolutions.every(n=>Number.isSafeInteger(n)&&n>=8&&n<=1024),
  'Expected comma-separated resolutions between 8 and 1024');
await mkdir(outputDir,{recursive:true});
const temp=await mkdtemp(join(outputDir,'fixture-'));
const input=join(temp,'input.glb');
// Disconnected, asymmetric marks leave both occupied and empty grid cells.
// Face colours and distinct top/bottom geometry expose texture-V mistakes.
await writeFile(input,makeFixtureGeometry([
  [[0,0,0],[.50,.90,.35]],
  [[.45,.18,.12],[.32,.22,.26],[.92,.88,.85]],
  [[-.43,.58,-.21],[.12,.18,.08],[.10,.16,.90]],
  [[.31,-.58,.25],[.14,.12,.15],[.05,.07,.10]],
  [[-.30,-.19,-.35],[.10,.12,.10],[.65,.07,.65]],
]));
const diagnostics=[];
const report={status:'running',backend,resolutions,
  startedAt:new Date().toISOString(),cases:[],diagnostics,
  contract:'native 0.4.2 bake -> schema 1 bytes -> CPU distance cache -> restore',
  readback:'R8 sampled through the same TSL quad into RGBA8; valid grid cells only',
  camera:{yawDegrees:45,pitchDegrees:0,projection:'orthographic'},
};
let browser,server,error,evaluationTimer;
try {
  server=await startServer({input});
  browser=await chromium.launch(chromiumOptions(backend));
  report.browserVersion=await browser.version();
  const page=await browser.newPage();
  page.setDefaultTimeout(120000);
  page.on('pageerror',e=>diagnostics.push('page: '+String(e)));
  page.on('console',m=>{if(m.type()==='error')diagnostics.push('console: '+m.text().slice(0,2000));});
  await page.goto(server.origin+'/bridge');
  const evaluation=page.evaluate(async({backend,resolutions})=>{
    const T=await import('three/webgpu');
    const N=await import('three/tsl');
    const {bakeGLB,restoreBox}=await import('/worker/runtime.mjs');
    const {compactPixels,decodeRaw}=await import('/worker/codec.mjs');
    const {buildDistanceMap}=await import('/worker/distance_map.mjs');
    const {makeRenderer}=await import('/web/asset_io.mjs');
    const source=new Uint8Array(await(await fetch('/job/input.glb')).arrayBuffer());
    const renderer=await makeRenderer(backend);
    const actualBackend=renderer.backend.isWebGPUBackend?'webgpu':'webgl2';
    if(actualBackend!==backend)throw new Error('Unexpected backend '+actualBackend);
    renderer.setSize(192,192,false);
    renderer.setClearColor(0,0);
    renderer.toneMapping=T.NoToneMapping;
    renderer.outputColorSpace=T.SRGBColorSpace;
    const failures=[],cases=[],images=[];
    const requireCase=(condition,label)=>{if(!condition)failures.push(label);};
    function byteDelta(a,b,channels=[0,1,2,3]) {
      if(a.length!==b.length)throw new Error('Readback length mismatch');
      let differingPixels=0,differingChannels=0,maxByteDelta=0,sum=0;
      for(let i=0;i<a.length;i+=4){
        let changed=false;
        for(const c of channels){
          const d=Math.abs(a[i+c]-b[i+c]);sum+=d;
          if(d){changed=true;differingChannels++;maxByteDelta=Math.max(maxByteDelta,d);}
        }
        if(changed)differingPixels++;
      }
      return {pixels:a.length/4,differingPixels,differingChannels,maxByteDelta,
        meanAbsoluteByteDelta:sum/((a.length/4)*channels.length)};
    }
    function depthDelta(a,b) {
      let differingPixels=0,maxCodeDelta=0;
      for(let i=0;i<a.length;i+=4){
        const x=65536*a[i]+256*a[i+1]+a[i+2],y=65536*b[i]+256*b[i+1]+b[i+2];
        const d=Math.abs(x-y);if(d)differingPixels++;
        maxCodeDelta=Math.max(maxCodeDelta,d);
      }
      return {pixels:a.length/4,differingPixels,maxCodeDelta,maxNormalizedDelta:maxCodeDelta/16777215};
    }
    async function readRGBA8(target) {
      return compactPixels(await renderer.readRenderTargetPixelsAsync(target,0,0,target.width,target.height),
        target.width,target.height,'uint8',backend);
    }
    function drawPixels(context,rgba,width,height,x,y,displayWidth=width,displayHeight=height) {
      const canvas=document.createElement('canvas');canvas.width=width;canvas.height=height;
      canvas.getContext('2d').putImageData(new ImageData(new Uint8ClampedArray(rgba),width,height),0,0);
      context.imageSmoothingEnabled=false;
      context.drawImage(canvas,x,y,displayWidth,displayHeight);
    }
    function gridRGBA(data) {
      const rgba=new Uint8ClampedArray(data.length*4);
      for(let i=0;i<data.length;i++){
        const value=Math.round(data[i]*255/7);
        rgba.set([value,value,value,255],i*4);
      }
      return rgba;
    }
    try {
      for(const resolution of resolutions){
        let baked,restored,colorTarget,depthTarget,copyTarget,copyMaterial,depthMaterial;
        const label='R'+resolution;
        try {
          baked=await bakeGLB(source,{renderer,resolution,maps:['roughness','metalness','ao','emissive'],guides:false});
          // Copy the actual persisted representation; do not pass the live bake's distance map.
          const manifest=JSON.parse(JSON.stringify(baked.manifest));
          const files=new Map([...baked.files].map(([name,bytes])=>[name,new Uint8Array(bytes)]));
          requireCase(manifest.versions.baker==='0.4.2',label+': wrong baker version');
          requireCase(manifest.schemaVersion===1,label+': schema 1 required');
          requireCase(!Object.hasOwn(manifest.textures,'distanceMap'),label+': distance cache leaked into saved schema');
          restored=await restoreBox(manifest,files,renderer);
          const nativeData=baked.live.mesh.material.impostor;
          const restoredData=restored.mesh.material.impostor;
          if(!nativeData.distanceMap||!restoredData.distanceMap)throw new Error(label+': distance map is missing');
          const rangeDescriptor=manifest.textures.rangeMap;
          const cpu=buildDistanceMap(decodeRaw(files.get(rangeDescriptor.path),rangeDescriptor.dtype),resolution);
          const cells=Math.ceil(resolution/8),columns=Math.ceil(Math.sqrt(cells)),rows=Math.ceil(cells/columns);
          const width=columns*cells,height=rows*cells;
          requireCase(cpu.width===width&&cpu.height===height,label+': CPU atlas dimensions');
          for(const [name,texture] of [['native',nativeData.distanceMap],['restored',restoredData.distanceMap]]){
            requireCase(texture.image.width===width&&texture.image.height===height,label+': '+name+' atlas dimensions');
            requireCase(texture.format===T.RedFormat&&texture.type===T.UnsignedByteType,label+': '+name+' R8 format');
            requireCase(texture.minFilter===T.NearestFilter&&texture.magFilter===T.NearestFilter,label+': '+name+' nearest filter');
            requireCase(!texture.generateMipmaps,label+': '+name+' mipmaps');
          }
          copyTarget=new T.RenderTarget(width,height,{type:T.UnsignedByteType,format:T.RGBAFormat,depthBuffer:false});
          copyTarget.texture.colorSpace=T.NoColorSpace;
          copyMaterial=new T.MeshBasicNodeMaterial();
          copyMaterial.toneMapped=false;
          const copyQuad=new T.QuadMesh(copyMaterial);
          // The same quad and UV expression sample both textures. This avoids relying on
          // backend-specific direct readback of a single-channel render target.
          async function sampleDistance(texture){
            copyMaterial.fragmentNode=N.vec4(N.vec3(N.texture(texture,N.uv()).r),1);
            copyMaterial.needsUpdate=true;
            renderer.setMRT(null);renderer.setRenderTarget(copyTarget);copyQuad.render(renderer);
            const rgba=await readRGBA8(copyTarget);
            const data=new Uint8Array(width*height);
            for(let i=0;i<data.length;i++)data[i]=rgba[i*4];
            return {rgba,data};
          }
          const native=await sampleDistance(nativeData.distanceMap);
          const loaded=await sampleDistance(restoredData.distanceMap);
          function compareCells(a,b){
            let differingCells=0,maxDistanceDelta=0;const firstDifferences=[];
            for(let z=0;z<cells;z++)for(let y=0;y<cells;y++)for(let x=0;x<cells;x++){
              const px=(z%columns)*cells+x,py=Math.floor(z/columns)*cells+y,index=py*width+px;
              const delta=Math.abs(a[index]-b[index]);
              if(delta){
                differingCells++;maxDistanceDelta=Math.max(maxDistanceDelta,delta);
                if(firstDifferences.length<24)firstDifferences.push({cell:[x,y,z],pixel:[px,py],a:a[index],b:b[index]});
              }
            }
            return {comparedCells:cells**3,differingCells,maxDistanceDelta,firstDifferences};
          }
          const nativeVsRestored=compareCells(native.data,loaded.data);
          const nativeVsCPUBuffer=compareCells(native.data,cpu.data);
          const restoredVsCPUBuffer=compareCells(loaded.data,cpu.data);
          const histogram=new Array(256).fill(0);
          for(let z=0;z<cells;z++)for(let y=0;y<cells;y++)for(let x=0;x<cells;x++){
            const index=(Math.floor(z/columns)*cells+y)*width+(z%columns)*cells+x;
            histogram[native.data[index]]++;
          }
          const occupiedCells=histogram[0],emptyCells=cells**3-occupiedCells;
          requireCase(occupiedCells>0&&emptyCells>0,label+': fixture must exercise occupied and empty cells');
          requireCase(histogram.slice(8).every(n=>n===0),label+': distance outside [0,7]');
          for(const [name,comparison] of Object.entries({nativeVsRestored,nativeVsCPUBuffer,restoredVsCPUBuffer}))
            requireCase(comparison.differingCells===0,label+': '+name+' differs in '+comparison.differingCells+' cells');

          const size=192;
          colorTarget=new T.RenderTarget(size,size,{type:T.UnsignedByteType,format:T.RGBAFormat});
          colorTarget.texture.colorSpace=T.SRGBColorSpace;
          colorTarget.depthTexture=new T.DepthTexture(size,size,T.UnsignedIntType);
          depthTarget=new T.RenderTarget(size,size,{type:T.UnsignedByteType,format:T.RGBAFormat,depthBuffer:false});
          depthTarget.texture.colorSpace=T.NoColorSpace;
          depthMaterial=new T.MeshBasicNodeMaterial();depthMaterial.toneMapped=false;
          const code=N.min(N.floor(N.clamp(N.texture(colorTarget.depthTexture,N.screenUV).r,0,1).mul(16777215).add(.5)),16777215);
          depthMaterial.fragmentNode=N.vec4(N.floor(code.div(65536)),N.mod(N.floor(code.div(256)),256),N.mod(code,256),255).div(255);
          const depthQuad=new T.QuadMesh(depthMaterial);
          const scene=new T.Scene();
          scene.add(new T.HemisphereLight(0xffffff,0x777777,1.4));
          const light=new T.DirectionalLight(0xffffff,2);light.position.set(3,4,5);scene.add(light);
          const center=nativeData.box.getCenter(new T.Vector3());
          const camera=new T.OrthographicCamera(-.65,.65,.65,-.65,.1,10);
          camera.position.copy(center).add(new T.Vector3(3/Math.sqrt(2),0,3/Math.sqrt(2)));
          camera.lookAt(center);camera.updateMatrixWorld();camera.updateProjectionMatrix();
          // Preserve the production step budget in both paths.
          requireCase(baked.live.mesh.material.steps===restored.mesh.material.steps,label+': live/restore steps differ');
          async function capture(mesh){
            scene.add(mesh);
            try {
              renderer.setMRT(null);renderer.setRenderTarget(colorTarget);renderer.render(scene,camera);
              const rgba=await readRGBA8(colorTarget);
              renderer.setRenderTarget(depthTarget);depthQuad.render(renderer);
              const depth=await readRGBA8(depthTarget);
              return {rgba,depth};
            } finally {scene.remove(mesh);}
          }
          const live=await capture(baked.live.mesh),roundtrip=await capture(restored.mesh);
          const rgb=byteDelta(live.rgba,roundtrip.rgba,[0,1,2]);
          const alpha=byteDelta(live.rgba,roundtrip.rgba,[3]);
          const depth=depthDelta(live.depth,roundtrip.depth);
          let foreground=0,background=0,minDepth=16777215,maxDepth=0;
          for(let i=0;i<live.rgba.length;i+=4){
            if(live.rgba[i+3]>0){
              foreground++;const d=65536*live.depth[i]+256*live.depth[i+1]+live.depth[i+2];
              minDepth=Math.min(minDepth,d);maxDepth=Math.max(maxDepth,d);
            } else background++;
          }
          const nativeDepthRangeWorld=(maxDepth-minDepth)/16777215*(camera.far-camera.near);
          requireCase(foreground>300&&background>300,label+': insufficient foreground/background');
          requireCase(nativeDepthRangeWorld>.03,label+': no meaningful ray-hit depth range');
          requireCase(rgb.differingPixels===0,label+': live/restore RGB differs in '+rgb.differingPixels+' pixels');
          requireCase(alpha.differingPixels===0,label+': live/restore alpha differs in '+alpha.differingPixels+' pixels');
          requireCase(depth.differingPixels===0,label+': live/restore depth differs in '+depth.differingPixels+' pixels');
          const row={resolution,threeRevision:T.REVISION,versions:manifest.versions,
            schemaVersion:manifest.schemaVersion,savedTextureNames:Object.keys(manifest.textures),
            distance:{cells,columns,rows,width,height,paddingSlices:columns*rows-cells,
              occupiedCells,emptyCells,histogram:histogram.slice(0,8),nativeVsRestored,nativeVsCPUBuffer,restoredVsCPUBuffer},
            render:{size,steps:baked.live.mesh.material.steps,foreground,background,nativeDepthRangeWorld,rgb,alpha,depth},
          };
          cases.push(row);
          const sheet=document.createElement('canvas');sheet.width=size*4;sheet.height=size+30;
          const ctx=sheet.getContext('2d');ctx.fillStyle='#20252b';ctx.fillRect(0,0,sheet.width,sheet.height);
          const labels=['Live 0.4.2 / 45 deg','Restored / 45 deg','Native distance x36','Restored distance x36'];
          ctx.fillStyle='white';ctx.font='12px sans-serif';
          labels.forEach((s,i)=>ctx.fillText(s,i*size+5,19));
          drawPixels(ctx,live.rgba,size,size,0,30);
          drawPixels(ctx,roundtrip.rgba,size,size,size,30);
          drawPixels(ctx,gridRGBA(native.data),width,height,size*2,30,size,size);
          drawPixels(ctx,gridRGBA(loaded.data),width,height,size*3,30,size,size);
          images.push({name:'distance-map-r'+resolution+'.png',dataURL:sheet.toDataURL('image/png')});
        } catch(e){
          failures.push(label+': '+String(e));
          cases.push({resolution,error:String(e),stack:e?.stack??null});
        } finally {
          renderer.setRenderTarget(null);renderer.setMRT(null);
          restored?.dispose();baked?.live?.dispose();
          colorTarget?.dispose();depthTarget?.dispose();copyTarget?.dispose();copyMaterial?.dispose();depthMaterial?.dispose();
        }
      }
    } finally {await renderer.dispose();}
    return {actualBackend,threeRevision:T.REVISION,cases,failures,images};
  },{backend,resolutions});
  const result=await Promise.race([evaluation,new Promise((_,reject)=>{
    evaluationTimer=setTimeout(()=>reject(new Error('GPU fixture exceeded 85 seconds')),85000);
  })]);
  clearTimeout(evaluationTimer);
  const {images,...numbers}=result;
  Object.assign(report,numbers);
  for(const image of images)await writeFile(join(outputDir,image.name),Buffer.from(image.dataURL.split(',')[1],'base64'));
  if(diagnostics.length)report.failures.push('Browser reported '+diagnostics.length+' errors');
  report.status=report.failures.length?'fail':'pass';
  if(report.status==='fail')error=new Error(report.failures.join('\n'));
} catch(e) {
  error=e;report.status='fail';report.error=String(e);report.stack=e?.stack??null;
} finally {
  clearTimeout(evaluationTimer);
  report.finishedAt=new Date().toISOString();
  await writeFile(join(outputDir,'distance-map-report.json'),JSON.stringify(report,null,2)+'\n');
  await browser?.close();await server?.close();await rm(temp,{recursive:true,force:true});
}
console.log(JSON.stringify(report,null,2));
if(error)throw error;
