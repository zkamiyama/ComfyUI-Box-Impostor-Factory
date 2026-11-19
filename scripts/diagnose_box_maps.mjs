// Inspect archived native Box maps without editing their Inspector HTML or any map bytes.
// node scripts/diagnose_box_maps.mjs ROOT OUT [--views front,left,right,iso]
// ROOT contains F/manifest.json and B-clay/scene.json recovered from official viewers.
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { resolve, join } from 'node:path';
import { parseArgs } from 'node:util';
import { chromium } from 'playwright';
import { chromiumOptions } from '../worker/chromium.mjs';
import { startServer } from '../worker/server.mjs';

const {positionals, values} = parseArgs({allowPositionals:true, options:{
  views:{type:'string',default:'front'},
  manifest:{type:'string',default:'F/manifest.json'},
  mesh:{type:'string',default:'B-clay/scene.json'},
  'box-label':{type:'string',default:'F-box'},
  'mesh-label':{type:'string',default:'B-mesh'},
  'mesh-html':{type:'string'},
  camera:{type:'string',default:'docs/verification/hi3d-depth-cycle-20261119/H-front-provenance.json'},
}});
if (positionals.length!==2) throw Error('Expected archive ROOT and output OUT');
const [root,out] = positionals.map(p=>resolve(p));
const proof = JSON.parse(await readFile(resolve(values.camera),'utf8'));
if (proof.camera.target.some(x=>x!==0)) throw Error('Only the recorded origin-centered camera is supported');
for(const key of ['box-label','mesh-label'])if(!/^[a-zA-Z0-9_-]+$/.test(values[key]))throw Error('Invalid stage label');
const views=values.views.split(',');
if(views.some(v=>!['front','left','right','iso'].includes(v))) throw Error('Unsupported view');
const manifestBytes=await readFile(join(root,values.manifest));
const meshBytes=await readFile(join(root,values.mesh));
const sha=x=>createHash('sha256').update(x).digest('hex');
await mkdir(out,{recursive:true});
const server=await startServer({assetDir:root});
let browser;
const written=[];
try {
  browser=await chromium.launch({...chromiumOptions('webgl2'),chromiumSandbox:true});
  const page=await browser.newPage({viewport:{width:1024,height:1024},deviceScaleFactor:1});
  const errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  page.on('console',m=>{if(m.type()==='error')errors.push(m.text());});
  await page.exposeFunction('saveDiagnostic',async(name,base64)=>{
    if(!/^[a-zA-Z0-9_.-]+$/.test(name))throw Error('Invalid output filename');
    const bytes=Buffer.from(base64,'base64');
    await writeFile(join(out,name),bytes);
    const entry={name,bytes:bytes.length,sha256:sha(bytes)};
    written.push(entry); return entry;
  });
  await page.goto(server.origin+'/bridge');
  const result=await page.evaluate(async({manifestPath,meshPath,proof,views,boxLabel,meshLabel})=>{
    const T=await import('three/webgpu');
    const N=await import('three/tsl');
    const {restoreBox}=await import('/worker/runtime.mjs');
    const {FACES}=await import('/worker/codec.mjs');
    const {makeRenderer,backendName}=await import('/web/asset_io.mjs');
    const base='/asset/'+manifestPath;
    const sourceManifest=await (await fetch(base)).json();
    const meshScene=await (await fetch('/asset/'+meshPath)).json();
    const files=new Map(await Promise.all(Object.values(sourceManifest.textures).map(async d=>[
      d.path,new Uint8Array(await(await fetch(new URL(d.path,new URL(base,location.href)))).arrayBuffer())
    ])));
    // The Inspector embeds only the render subset of the manifest. These validator-only
    // fields do not change any rendering input; no proxy file is loaded through this path.
    const manifest={...sourceManifest,proxy:'inspection-unused-proxy.glb',tightBounds:meshScene.bounds,
      faceOrder:FACES.map(f=>f.id),faces:Object.fromEntries(FACES.map(f=>[f.id,{}]))};
    const renderer=await makeRenderer('webgl2',{antialias:true,preserveDrawingBuffer:true});
    const gl=renderer.getContext();
    const debug=gl.getExtension('WEBGL_debug_renderer_info');
    const adapter=debug?gl.getParameter(debug.UNMASKED_RENDERER_WEBGL):null;
    renderer.setPixelRatio(1);renderer.setSize(1024,1024,false);
    document.body.append(renderer.domElement);
    renderer.outputColorSpace=T.SRGBColorSpace;
    renderer.toneMapping=T.ACESFilmicToneMapping;renderer.toneMappingExposure=1;
    renderer.setClearColor(0,0);
    const restored=await restoreBox(manifest,files,renderer);
    const meshRoot=new T.Group();
    const decode=(s,Type)=>{const b=atob(s),a=new Uint8Array(b.length);for(let i=0;i<b.length;i++)a[i]=b.charCodeAt(i);return new Type(a.buffer);};
    let triangles=0;
    for(const m of meshScene.meshes){
      const geometry=new T.BufferGeometry();
      geometry.setAttribute('position',new T.BufferAttribute(decode(m.positions,Float32Array),3));
      if(m.normals)geometry.setAttribute('normal',new T.BufferAttribute(decode(m.normals,Float32Array),3));
      const indices=decode(m.indices,Uint32Array);geometry.setIndex(new T.BufferAttribute(indices,1));
      triangles+=indices.length/3;
      const src=meshScene.materials[m.material]??meshScene.materials[0];
      const material=new T.MeshStandardNodeMaterial({color:new T.Color(...src.baseColor.slice(0,3)),
        roughness:src.roughness,metalness:src.metallic,side:src.doubleSided?T.DoubleSide:T.FrontSide});
      const object=new T.Mesh(geometry,material);object.frustumCulled=false;meshRoot.add(object);
    }
    if(triangles!==meshScene.original.triangles)throw Error('Mesh triangle count changed');
    const scene=new T.Scene();
    const radius=proof.camera.sourceRadius;
    const light=new T.DirectionalLight(0xffffff,2.1);
    light.position.set(radius*2,radius*3,radius*2);
    scene.add(new T.HemisphereLight(16250349,8947855,1.55),light);
    const span=proof.camera.halfExtent,distance=proof.camera.distance;
    const near=Math.max(1e-5,radius*.001),far=Math.max(50,radius*120);
    const camera=new T.OrthographicCamera(-span,span,span,-span,near,far);
    const target=new T.RenderTarget(1024,1024,{type:T.FloatType});
    target.texture.name='output';
    target.depthTexture=new T.DepthTexture(1024,1024,T.UnsignedIntType);
    const depthTarget=new T.RenderTarget(1024,1024,{type:T.FloatType,depthBuffer:false});
    const depthMaterial=new T.MeshBasicNodeMaterial();
    depthMaterial.toneMapped=false;depthMaterial.depthTest=false;depthMaterial.depthWrite=false;
    // Read the depth buffer written by the native ray marcher (or by the mesh rasterizer).
    // No replacement shape or inferred depth is used.
    depthMaterial.fragmentNode=N.vec4(N.texture(target.depthTexture,N.screenUV).r,0,0,1);
    const quad=new T.QuadMesh(depthMaterial);
    const mrtAlbedo=N.mrt({output:N.diffuseColor});
    const mrtNormal=N.mrt({output:N.vec4(N.normalWorld.mul(.5).add(.5),N.diffuseColor.a)});
    const b64=(array)=>{const bytes=new Uint8Array(array.buffer,array.byteOffset,array.byteLength);let s='';for(let i=0;i<bytes.length;i+=16384)s+=String.fromCharCode(...bytes.subarray(i,i+16384));return btoa(s);};
    const topDown=(a)=>{const b=new a.constructor(a.length),row=1024*4;for(let y=0;y<1024;y++)b.set(a.subarray(y*row,(y+1)*row),(1023-y)*row);return b;};
    const srgb=x=>x<=.0031308?12.92*x:1.055*Math.pow(x,1/2.4)-.055;
    const png=(data,mode,mask)=>{
      const canvas=document.createElement('canvas');canvas.width=canvas.height=1024;
      const ctx=canvas.getContext('2d'),im=ctx.createImageData(1024,1024);
      for(let p=0;p<1024*1024;p++){
        const i=p*4,a=mask?mask[i+3]:data[i+3];
        for(let c=0;c<3;c++){
          let v=data[i+c];
          if(mode==='albedo')v=srgb(Math.max(0,v));
          if(mode==='depth'){
            const viewDistance=near+data[i]*(far-near);
            v=.5+(distance-viewDistance)/(2*radius);
          }
          im.data[i+c]=Math.round(255*Math.max(0,Math.min(1,v)));
        }
        im.data[i+3]=a>.5?255:0;
      }
      ctx.putImageData(im,0,0);return canvas.toDataURL('image/png').split(',')[1];
    };
    const frameMetadata=[];
    try{
      for(const view of views){
        const direction=new T.Vector3(...({front:[0,0,1],left:[-1,0,0],right:[1,0,0],iso:[1,1,1]}[view])).normalize();
        camera.position.copy(direction).multiplyScalar(distance);camera.lookAt(0,0,0);camera.updateMatrixWorld();
        for(const [name,object] of [[meshLabel,meshRoot],[boxLabel,restored.mesh]]){
          scene.add(object);
          renderer.setMRT(null);renderer.setRenderTarget(null);renderer.render(scene,camera);
          await window.saveDiagnostic(name+'-'+view+'-lit.png',renderer.domElement.toDataURL('image/png').split(',')[1]);
          const dataByMode={};
          for(const [mode,mrtNode] of [['albedo',mrtAlbedo],['normal',mrtNormal]]){
            renderer.setMRT(mrtNode);renderer.setRenderTarget(target);renderer.render(scene,camera);
            const data=topDown(await renderer.readRenderTargetPixelsAsync(target,0,0,1024,1024));
            if(!(data instanceof Float32Array))throw Error('Expected Float32 diagnostic readback');
            dataByMode[mode]=data;
            await window.saveDiagnostic(name+'-'+view+'-'+mode+'.rgba32f.bin',b64(data));
            await window.saveDiagnostic(name+'-'+view+'-'+mode+'.png',png(data,mode));
          }
          renderer.setMRT(null);renderer.setRenderTarget(depthTarget);quad.render(renderer);
          const depth=topDown(await renderer.readRenderTargetPixelsAsync(depthTarget,0,0,1024,1024));
          await window.saveDiagnostic(name+'-'+view+'-depth.rgba32f.bin',b64(depth));
          await window.saveDiagnostic(name+'-'+view+'-depth.png',png(depth,'depth',dataByMode.albedo));
          frameMetadata.push({name,view,cameraPosition:camera.position.toArray(),normalEncoding:'world normal * 0.5 + 0.5',
            depthEncoding:'normalized actual orthographic depth-buffer value',
            albedo:'native diffuseColor before lights; native ray intersection, normal-dependent view weights and cross-view texture blending retained'});
          scene.remove(object);
        }
      }
      return {backend:backendName(renderer),adapter,triangles,
        camera:{projection:'orthographic',target:[0,0,0],distance,halfExtent:span,near,far,
          sourceRadius:radius,pixels:[1024,1024],cssPixels:[1024,1024],deviceScaleFactor:1,
          matching:'Numerically equal camera target, position, and half-extent from recorded H-front-provenance.json'},
        headCrop:proof.headCrops.front,views:frameMetadata,rayStepLimit:restored.mesh.material.steps,
        lighting:{directional:{color:0xffffff,intensity:2.1,position:light.position.toArray()},
          hemisphere:{sky:16250349,ground:8947855,intensity:1.55},toneMapping:'ACESFilmicToneMapping',exposure:1},
        limitations:['Mesh normals are the full-triangle official Inspector payload normals; these can differ from the baker normals.',
          'Float depth readback preserves the depth buffer value, which is still subject to depth-buffer precision.',
          'Albedo rendering retains normal-dependent view weighting; it is not a normal-independence experiment.',
          'This is an internal diagnostic harness using the production renderer; it is not a replacement Inspector viewer.']};
    }finally{
      restored.dispose();meshRoot.traverse(o=>{o.geometry?.dispose();o.material?.dispose();});
      target.dispose();depthTarget.dispose();depthMaterial.dispose();await renderer.dispose();
    }
  },{manifestPath:values.manifest,meshPath:values.mesh,proof,views,boxLabel:values['box-label'],meshLabel:values['mesh-label']});
  if(errors.length)throw Error('Browser errors: '+errors.join('\n'));
  const report={method:'production-native-box-map-diagnostics-v1',sourceManifestSHA256:sha(manifestBytes),
    sourceMeshPayloadSHA256:sha(meshBytes),scriptSHA256:sha(await readFile(new URL(import.meta.url))),
    sourceMeshInspectorHTMLSHA256:values['mesh-html']?sha(await readFile(resolve(values['mesh-html']))):null,
    cameraReferenceSHA256:sha(await readFile(resolve(values.camera))),
    rendererSource:'worker/runtime.mjs restoreBox; three-impostor@0.4.1',
    manifestScaffold:'Unused proxy/tightBounds/face validator fields supplied in memory; embedded render subset and all texture bytes unchanged.',
    originalInspectorHTMLModified:false,browserVersion:browser.version(),...result,files:written};
  await writeFile(join(out,'diagnostic-report.json'),JSON.stringify(report,null,2)+'\n');
  console.log(JSON.stringify({out,files:written.length,backend:result.backend,adapter:result.adapter,triangles:result.triangles}));
}finally{await browser?.close();await server.close();}
