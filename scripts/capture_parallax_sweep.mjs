// Internal native-Box RGB experiment. Outputs PNGs/provenance, never a viewer.
// node scripts/capture_parallax_sweep.mjs ASSET_DIR OUT --camera PROOF.json
//   [--yaws=0,30,45] [--size=1024] [--reverse] [--recovery=RECOVERY.json]
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {mkdir,readFile,writeFile,access} from 'node:fs/promises';
import {resolve,join} from 'node:path';
import {parseArgs} from 'node:util';
import {deflateSync} from 'node:zlib';
import {chromium} from 'playwright';
import {chromiumOptions} from '../worker/chromium.mjs';
import {startServer} from '../worker/server.mjs';

const {positionals,values}=parseArgs({allowPositionals:true,options:{
  camera:{type:'string'},recovery:{type:'string'},size:{type:'string',default:'1024'},
  yaws:{type:'string',default:'-60,-45,-30,-15,0,15,30,45,60'},reverse:{type:'boolean',default:false},
  modes:{type:'string',default:'native,camera-single,camera-blend,camera-depth'},
  'camera-power':{type:'string',default:'8'},'depth-tolerance':{type:'string',default:'4'},
  'allow-geometry-mismatch':{type:'boolean',default:false},
}});
assert(positionals.length===2&&values.camera,'Expected ASSET_DIR OUT --camera PROOF.json');
const [assetRoot,out]=positionals.map(p=>resolve(p));
const size=Number(values.size),yaws=values.yaws.split(',').map(Number),modes=values.modes.split(',');
assert(Number.isInteger(size)&&size>=64&&size<=4096&&yaws.length&&yaws.every(Number.isFinite));
assert(new Set(modes).size===modes.length&&modes.every(x=>['native','camera-single','camera-blend','camera-depth'].includes(x)));
const sequence=values.reverse?[...yaws,...yaws.slice().reverse()]:yaws;
const sha=bytes=>createHash('sha256').update(bytes).digest('hex');
const cameraBytes=await readFile(resolve(values.camera)),proof=JSON.parse(cameraBytes);
const manifestPath=join(assetRoot,'manifest.json'),manifestBytes=await readFile(manifestPath),manifest=JSON.parse(manifestBytes);
const recovery=values.recovery?JSON.parse(await readFile(resolve(values.recovery))):null;
if(recovery)assert.equal(recovery.manifestSHA256,sha(manifestBytes));
assert.equal(proof.camera.projection,'orthographic');
assert(Array.isArray(proof.headWorldBounds)&&proof.headWorldBounds.length===2);
try{await access(join(out,'sweep.json'));throw Error('Preserve existing sweep; choose a new output directory');}catch(e){if(e.code!=='ENOENT')throw e;}
await mkdir(out,{recursive:true});
const sourceFiles={};
for(const relative of ['scripts/capture_parallax_sweep.mjs','worker/parallax_material.mjs','worker/runtime.mjs','worker/codec.mjs','worker/server.mjs','web/asset_io.mjs']){
  const path=new URL('../'+relative,import.meta.url);sourceFiles[relative]=sha(await readFile(path));
}
const textures={};
for(const [name,d] of Object.entries(manifest.textures)){
  const path=resolve(assetRoot,d.path),bytes=await readFile(path);
  assert.equal(bytes.length,d.byteLength);assert.equal(sha(bytes),d.sha256);
  textures[name]={...d,path};
}
// PNG encoder deliberately bypasses Canvas: Float32 bytes packed into RGBA must
// survive even when their fourth byte would be interpreted as a translucent alpha.
const crcTable=Uint32Array.from({length:256},(_,n)=>{for(let k=0;k<8;k++)n=n&1?0xedb88320^(n>>>1):n>>>1;return n>>>0;});
function crc32(bytes){let c=0xffffffff;for(const b of bytes)c=crcTable[(c^b)&255]^(c>>>8);return(c^0xffffffff)>>>0;}
function chunk(kind,data){const type=Buffer.from(kind),n=Buffer.alloc(4),c=Buffer.alloc(4);n.writeUInt32BE(data.length);c.writeUInt32BE(crc32(Buffer.concat([type,data])));return Buffer.concat([n,type,data,c]);}
function pngRGBA(bytes,width,height){
  assert.equal(bytes.length,width*height*4);const header=Buffer.alloc(13);header.writeUInt32BE(width);header.writeUInt32BE(height,4);header[8]=8;header[9]=6;
  const rows=Buffer.alloc(height*(width*4+1));for(let y=0;y<height;y++)bytes.copy(rows,y*(width*4+1)+1,y*width*4,(y+1)*width*4);
  return Buffer.concat([Buffer.from([137,80,78,71,13,10,26,10]),chunk('IHDR',header),chunk('IDAT',deflateSync(rows)),chunk('IEND',Buffer.alloc(0))]);
}
function cropRGBA(bytes,roi){const [x0,y0,x1,y1]=roi,w=x1-x0,h=y1-y0,b=Buffer.alloc(w*h*4);for(let y=0;y<h;y++)bytes.copy(b,y*w*4,((y+y0)*size+x0)*4,((y+y0)*size+x1)*4);return b;}
const report={schemaVersion:1,method:'native-box-unlit-parallax-sweep-v1',status:'running',width:size,height:size,
  source:{assetRoot,manifestPath,manifestSHA256:sha(manifestBytes),assetId:manifest.asset_id??recovery?.assetId??null,
    htmlPath:recovery?.htmlPath??null,htmlSHA256:recovery?.htmlSHA256??null,recoveryProof:values.recovery?resolve(values.recovery):null,textures},
  cameraReference:resolve(values.camera),cameraReferenceSHA256:sha(cameraBytes),headWorldBounds:proof.headWorldBounds,
  outputColorSpace:'srgb',renderBufferColorSpace:'linear-srgb',toneMapping:'NoToneMapping',exposure:1,
  outputEncoding:'Piecewise linear-sRGB to sRGB, rounded to nearest8bit, no lighting/tone curve/sharpening; straightRGBA.',
  depthEncoding:'WebGL window0..1; one IEEE754 Float32 little-endian value per pixel, packed byte-for-byte into rawRGBA8 PNG; top-to-bottom.',
  depthPNGWarning:'Data image: decode RGBA bytes directly and reinterpret as little-endian Float32; do not composite or color-convert.',
  sourceFilesSHA256:sourceFiles,anglesDegrees:sequence,modes,frames:[],errors:[],originalInspectorHTMLModified:false,
  argv:process.argv.slice(2),scriptSHA256:sourceFiles['scripts/capture_parallax_sweep.mjs']};
const saveReport=()=>writeFile(join(out,'sweep.json'),JSON.stringify(report,null,2)+'\n');
await saveReport();
const server=await startServer({assetDir:assetRoot});
let browser;const depths=new Map();
try{
  browser=await chromium.launch({...chromiumOptions('webgl2'),chromiumSandbox:true,
    args:[...chromiumOptions('webgl2').args,'--disable-dev-shm-usage']});
  report.browserVersion=browser.version();
  const page=await browser.newPage({viewport:{width:size,height:size},deviceScaleFactor:1});
  page.on('pageerror',e=>report.errors.push('page: '+e.message));
  page.on('console',m=>{if(m.type()==='error')report.errors.push('console: '+m.text());});
  await page.exposeFunction('saveSweepFrame',async(entry,rgba64,depth64)=>{
    const rgba=Buffer.from(rgba64,'base64'),depth=Buffer.from(depth64,'base64');
    assert.equal(rgba.length,size*size*4);assert.equal(depth.length,size*size*4);
    const alpha=Buffer.alloc(size*size);for(let i=0;i<alpha.length;i++)alpha[i]=rgba[4*i+3];
    const stem=String(entry.sequenceIndex).padStart(3,'0')+'-'+entry.variant+'-yaw'+String(entry.angleDegrees).replace('-','m').replace('.','p');
    const rgbaPNG=join(out,stem+'.png'),facePNG=join(out,stem+'-head.png');
    const encoded=pngRGBA(rgba,size,size);await writeFile(rgbaPNG,encoded);
    const roi=entry.headROI,face=pngRGBA(cropRGBA(rgba,roi),roi[2]-roi[0],roi[3]-roi[1]);await writeFile(facePNG,face);
    const depthRawSHA256=sha(depth);let depthPNG=depths.get(depthRawSHA256);
    if(!depthPNG){depthPNG=join(out,stem+'-depth-f32.png');await writeFile(depthPNG,pngRGBA(depth,size,size));depths.set(depthRawSHA256,depthPNG);}
    report.frames.push({...entry,rgbaPNG,rgbaSHA256:sha(encoded),alphaSHA256:sha(alpha),facePNG,faceSHA256:sha(face),depthPNG,depthRawSHA256});
    await saveReport();console.log('FRAME',entry.sequenceIndex,entry.variant,entry.angleDegrees,facePNG);
  });
  await page.exposeFunction('saveSweepControl',async(data,original64,originalDepth64,native64,nativeDepth64)=>{
    const images={};
    for(const [label,color64,depth64] of [['original-native',original64,originalDepth64],['factory-native',native64,nativeDepth64]]){
      const rgba=Buffer.from(color64,'base64'),depth=Buffer.from(depth64,'base64');
      const rgbaPNG=join(out,'control-'+label+'-front.png'),depthPNG=join(out,'control-'+label+'-front-depth-f32.png');
      const encoded=pngRGBA(rgba,size,size);await writeFile(rgbaPNG,encoded);await writeFile(depthPNG,pngRGBA(depth,size,size));
      let nonzeroAlphaPixels=0;const alpha=Buffer.alloc(size*size);for(let i=0;i<alpha.length;i++){alpha[i]=rgba[4*i+3];if(alpha[i])nonzeroAlphaPixels++;}
      images[label]={rgbaPNG,rgbaSHA256:sha(encoded),depthPNG,depthRawSHA256:sha(depth),alphaSHA256:sha(alpha),nonzeroAlphaPixels};
    }
    report.nativeGeometryControl={...data,images};await saveReport();console.log('CONTROL',JSON.stringify(report.nativeGeometryControl));
  });
  await page.exposeFunction('saveAtlasReference',async(rgba64,resolution)=>{
    const bytes=pngRGBA(Buffer.from(rgba64,'base64'),resolution,resolution),path=join(out,'native-atlas-plusZ.png');await writeFile(path,bytes);
    report.frontAtlasReference={path,sha256:sha(bytes),resolution,face:'+Z',tile:[1,1],sourceTextureSHA256:manifest.textures.colorMap.sha256,
      encoding:'Stored linear premultiplied atlas unpremultiplied, rows flipped to PNG, same piecewise sRGB output curve; no camera reprojection.'};await saveReport();
  });
  await page.goto(server.origin+'/bridge');
  const result=await page.evaluate(async({proof,size,sequence,modes,options,allowMismatch})=>{
    const T=await import('three/webgpu'),N=await import('three/tsl');
    const {restoreBox}=await import('/worker/runtime.mjs');
    const {FACES}=await import('/worker/codec.mjs');
    const {createParallaxMaterial}=await import('/worker/parallax_material.mjs');
    const {makeRenderer,backendName}=await import('/web/asset_io.mjs');
    const source=await(await fetch('/asset/manifest.json')).json();
    const files=new Map(await Promise.all(Object.values(source.textures).map(async d=>[d.path,new Uint8Array(await(await fetch('/asset/'+d.path)).arrayBuffer())])));
    const manifest={...source,proxy:source.proxy??'unused-inspection-proxy.glb',tightBounds:source.tightBounds??source.bakeBox,
      faceOrder:source.faceOrder??FACES.map(f=>f.id),faces:source.faces??Object.fromEntries(FACES.map(f=>[f.id,{}]))};
    const renderer=await makeRenderer('webgl2',{antialias:false,preserveDrawingBuffer:true});
    renderer.setPixelRatio(1);renderer.setSize(size,size,false);renderer.setClearColor(0,0);
    renderer.outputColorSpace=T.LinearSRGBColorSpace;renderer.toneMapping=T.NoToneMapping;renderer.toneMappingExposure=1;
    const gl=renderer.getContext(),debug=gl.getExtension('WEBGL_debug_renderer_info');
    const adapter=debug?gl.getParameter(debug.UNMASKED_RENDERER_WEBGL):null;
    const restored=await restoreBox(manifest,files,renderer),original=restored.mesh.material;
    const materials=Object.fromEntries([...new Set(['native',...modes])].map(mode=>{
      const mat=createParallaxMaterial(original.impostor,{...options,mode});mat.steps=original.steps;return[mode,mat];}));
    const scene=new T.Scene();scene.add(restored.mesh);
    const c=proof.camera,target=new T.Vector3(...c.target),radius=c.sourceRadius??c.referenceRadius;
    const near=Math.max(1e-5,radius*.001),far=Math.max(50,radius*120),span=c.halfExtent;
    const camera=new T.OrthographicCamera(-span,span,span,-span,near,far);
    const colorTarget=new T.RenderTarget(size,size,{type:T.FloatType,depthBuffer:true});
    colorTarget.texture.name='output';colorTarget.texture.colorSpace=T.LinearSRGBColorSpace;colorTarget.depthTexture=new T.DepthTexture(size,size,T.UnsignedIntType);
    const depthTarget=new T.RenderTarget(size,size,{type:T.FloatType,depthBuffer:false});depthTarget.texture.colorSpace=T.LinearSRGBColorSpace;
    const depthMaterial=new T.MeshBasicNodeMaterial();depthMaterial.toneMapped=false;depthMaterial.depthTest=false;depthMaterial.depthWrite=false;
    depthMaterial.fragmentNode=N.vec4(N.texture(colorTarget.depthTexture,N.screenUV).r,0,0,1);
    const quad=new T.QuadMesh(depthMaterial);
    const b64=typed=>{const bytes=new Uint8Array(typed.buffer,typed.byteOffset,typed.byteLength);let s='';for(let i=0;i<bytes.length;i+=16384)s+=String.fromCharCode(...bytes.subarray(i,i+16384));return btoa(s);};
    const encode=x=>{x=Math.max(0,x);return Math.round(255*Math.max(0,Math.min(1,x<=.0031308?12.92*x:1.055*x**(1/2.4)-.055)));};
    const frameRGBA=data=>{const out=new Uint8Array(size*size*4);for(let i=0;i<data.length;i+=4){out[i]=encode(data[i]);out[i+1]=encode(data[i+1]);out[i+2]=encode(data[i+2]);out[i+3]=Math.round(255*Math.max(0,Math.min(1,data[i+3])));}return out;};
    const setYaw=degrees=>{const angle=degrees*Math.PI/180,direction=new T.Vector3(Math.sin(angle),0,Math.cos(angle));camera.position.copy(target).addScaledVector(direction,c.distance);camera.up.set(0,1,0);camera.lookAt(target);camera.updateMatrixWorld();camera.updateProjectionMatrix();return direction.toArray();};
    const roi=()=>{const points=[];for(let x=0;x<2;x++)for(let y=0;y<2;y++)for(let z=0;z<2;z++)points.push(new T.Vector3(proof.headWorldBounds[x][0],proof.headWorldBounds[y][1],proof.headWorldBounds[z][2]).project(camera));
      const xs=points.map(p=>(p.x+1)*size/2),ys=points.map(p=>(1-p.y)*size/2);return[Math.max(0,Math.floor(Math.min(...xs))),Math.max(0,Math.floor(Math.min(...ys))),Math.min(size,Math.ceil(Math.max(...xs))),Math.min(size,Math.ceil(Math.max(...ys)))];};
    const capture=async(material,mrt=null)=>{
      restored.mesh.material=material;renderer.setMRT(mrt);renderer.setRenderTarget(colorTarget);renderer.clear();renderer.render(scene,camera);
      const raw=await renderer.readRenderTargetPixelsAsync(colorTarget,0,0,size,size);if(!(raw instanceof Float32Array))throw Error('Expected Float32 color readback');
      const color=new Float32Array(raw.length);for(let y=0;y<size;y++)color.set(raw.subarray(y*size*4,(y+1)*size*4),(size-1-y)*size*4);
      renderer.setMRT(null);renderer.setRenderTarget(depthTarget);quad.render(renderer);
      const d=await renderer.readRenderTargetPixelsAsync(depthTarget,0,0,size,size),depth=new Float32Array(size*size);
      for(let y=0;y<size;y++)for(let x=0;x<size;x++)depth[(size-1-y)*size+x]=d[(y*size+x)*4];
      return{color,depth,rgba:frameRGBA(color)};
    };
    const equal=(a,b)=>{let alphaChangedPixels=0,depthChangedPixels=0,maxAbsDepthDelta=0;const aa=new Uint32Array(a.depth.buffer),bb=new Uint32Array(b.depth.buffer);
      for(let i=0;i<aa.length;i++){if(a.rgba[4*i+3]!==b.rgba[4*i+3])alphaChangedPixels++;if(aa[i]!==bb[i])depthChangedPixels++;maxAbsDepthDelta=Math.max(maxAbsDepthDelta,Math.abs(a.depth[i]-b.depth[i]));}
      return{alphaEqual:alphaChangedPixels===0,depthBitEqual:depthChangedPixels===0,alphaChangedPixels,depthChangedPixels,maxAbsDepthDelta};};
    try{
      const resolution=source.resolution,raw=files.get(source.textures.colorMap.path),halves=new Uint16Array(raw.buffer,raw.byteOffset,raw.byteLength/2),atlas=new Uint8Array(resolution*resolution*4);
      for(let y=0;y<resolution;y++)for(let x=0;x<resolution;x++){
        const i=((2*resolution-1-y)*3*resolution+resolution+x)*4,j=(y*resolution+x)*4,a=T.DataUtils.fromHalfFloat(halves[i+3]);
        for(let k=0;k<3;k++)atlas[j+k]=encode(a>0?T.DataUtils.fromHalfFloat(halves[i+k])/a:0);atlas[j+3]=Math.round(255*Math.max(0,Math.min(1,a)));
      }
      await window.saveAtlasReference(b64(atlas),resolution);
      setYaw(0);const control=await capture(original,N.mrt({output:N.diffuseColor})),nativeControl=await capture(materials.native);
      const nativeGeometryControl={angleDegrees:0,against:'original restoreBox material with native diffuseColor MRT',...equal(control,nativeControl)};
      await window.saveSweepControl(nativeGeometryControl,b64(control.rgba),b64(control.depth),b64(nativeControl.rgba),b64(nativeControl.depth));
      for(let index=0;index<sequence.length;index++){
        const angleDegrees=sequence[index],direction=setYaw(angleDegrees),headROI=roi();
        const baseline=angleDegrees===0?nativeControl:await capture(materials.native);
        let mismatched=!nativeGeometryControl.alphaEqual||!nativeGeometryControl.depthBitEqual;
        for(const variant of modes){
          const frame=variant==='native'?baseline:await capture(materials[variant]);
          const geometryEquality=equal(baseline,frame);mismatched ||= !geometryEquality.alphaEqual||!geometryEquality.depthBitEqual;
          if(!frame.rgba.some((v,i)=>i%4===3&&v>0))throw Error('Empty rendered frame: '+variant);
          await window.saveSweepFrame({variant,angleDegrees,sequenceIndex:index,direction,headROI,near,far,
            viewMatrix:camera.matrixWorldInverse.toArray(),projectionMatrix:camera.projectionMatrix.toArray(),cameraWorldMatrix:camera.matrixWorld.toArray(),
            cameraPosition:camera.position.toArray(),cameraTarget:target.toArray(),materialOptions:materials[variant].parallaxOptions,geometryEquality},b64(frame.rgba),b64(frame.depth));
        }
        if(mismatched&&!allowMismatch)throw Error('Native geometry equality failed; saved first-angle diagnostic frames');
      }
      return{backend:backendName(renderer),adapter,hardwareGPUUsed:adapter?!/SwiftShader|llvmpipe/i.test(adapter):null,
        rayStepLimit:original.steps,camera:{projection:'orthographic',target:target.toArray(),distance:c.distance,halfExtent:span,near,far,pixels:[size,size]},
        geometryEqualityPassed:nativeGeometryControl.alphaEqual&&nativeGeometryControl.depthBitEqual};
    }finally{restored.mesh.material=original;for(const m of Object.values(materials))m.dispose();restored.dispose();colorTarget.dispose();depthTarget.dispose();depthMaterial.dispose();await renderer.dispose();}
  },{proof,size,sequence,modes,options:{cameraPower:Number(values['camera-power']),depthToleranceTexels:Number(values['depth-tolerance'])},allowMismatch:values['allow-geometry-mismatch']});
  Object.assign(report,result);
  assert.equal(report.errors.length,0,report.errors.join('\n'));
  for(const [relative,digest] of Object.entries(sourceFiles))assert.equal(sha(await readFile(new URL('../'+relative,import.meta.url))),digest,'Source changed during capture: '+relative);
  report.geometryEqualityPassed &&= report.frames.every(f=>f.geometryEquality.alphaEqual&&f.geometryEquality.depthBitEqual);
  report.status='complete';await saveReport();console.log('DONE',join(out,'sweep.json'));
}catch(error){report.status='failed';report.error=String(error.stack??error);await saveReport();throw error;}
finally{await browser?.close();await server.close();}
