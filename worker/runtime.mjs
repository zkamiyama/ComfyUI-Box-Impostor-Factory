import * as THREE from 'three/webgpu';
import { diffuseColor, mrt, positionWorld, vec3, vec4 } from 'three/tsl';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { ImpostorBaker, ImpostorNodeMaterial } from 'three-impostor';
import {
  FACES, MAP_TYPES, MATERIAL_PROPERTIES, compactPixels, decodeRaw, dimensions, encodeRaw,
  parseGLB, safePath, sha256, validateDescriptor, wrapGLB,
} from './codec.mjs';

const PIN = '4d525ca3aecc5ed2009d6bd9b8c6d6c289248768';
const REQUIRED = ['colorMap', 'normalMap', 'intervalMap', 'rangeMap'];
const CORE_EXTENSIONS = new Set([
  'KHR_lights_punctual','KHR_materials_anisotropy','KHR_materials_clearcoat',
  'KHR_materials_dispersion','KHR_materials_emissive_strength','KHR_materials_ior',
  'KHR_materials_iridescence','KHR_materials_sheen','KHR_materials_specular',
  'KHR_materials_transmission','KHR_materials_unlit','KHR_materials_volume',
  'KHR_mesh_quantization','KHR_texture_transform','EXT_mesh_gpu_instancing','EXT_texture_webp',
]);

function error(code, message, cause) {
  const e = new Error(code + ': ' + message, {cause});
  e.code = code;
  return e;
}
function arrayBuffer(data) {
  if (data instanceof ArrayBuffer) return data;
  if (data instanceof Uint8Array) return data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
  throw error('INVALID_GLB','Expected GLB bytes');
}
function bounds(b) { return {min:b.min.toArray(),max:b.max.toArray()}; }
function validBox(b) {
  return [b?.min,b?.max].every(a => Array.isArray(a) && a.length === 3 && a.every(Number.isFinite)) &&
    b.min.every((v,i) => v <= b.max[i]);
}
function identity() {return new THREE.Matrix4().toArray();}

export function disposeProxy(object) {
  const geometries=new Set(), materials=new Set(), textures=new Set(), images=new Set();
  object.traverse(node => {
    if(node.geometry) geometries.add(node.geometry);
    if(node.material) for(const mat of [].concat(node.material)) materials.add(mat);
  });
  for(const mat of materials) {
    for(const value of Object.values(mat)) if(value?.isTexture) textures.add(value);
    mat.dispose();
  }
  for(const t of textures){ if(t.image?.close) images.add(t.image);t.dispose(); }
  for(const im of images) im.close();
  for(const g of geometries) g.dispose();
}

export async function loadProxy(data) {
  let gltf;
  try {
    const bytes=arrayBuffer(data);
    const {json}=parseGLB(bytes);
    for(const ext of json.extensionsRequired ?? []) {
      if(!CORE_EXTENSIONS.has(ext)) throw error('UNSUPPORTED_GLB','Required GLB extension '+ext+' needs a decoder or material adapter');
    }
    const manager=new THREE.LoadingManager();
    manager.setURLModifier(url=>{
      if(url.startsWith('blob:') || url.startsWith('data:')) return url;
      throw error('EXTERNAL_RESOURCE','GLB resources must be embedded');
    });
    gltf=await new GLTFLoader(manager).parseAsync(bytes,'');
    let count=0;
    gltf.scene.traverse(node=>{
      if(node.isSkinnedMesh || node.isPoints || node.isLine)
        throw error('UNSUPPORTED_GLB','Static triangle meshes are required for the surface bake');
      if(node.isMesh) count++;
    });
    if(!count) throw error('INVALID_GLB','Active GLB scene contains no triangle meshes');
    gltf.scene.updateMatrixWorld(true);
    return gltf.scene;
  } catch(cause) {
    if(gltf?.scene) disposeProxy(gltf.scene);
    if(cause.code) throw cause;
    throw error('INVALID_GLB',cause.message,cause);
  }
}

function measure(proxy) {
  proxy.updateMatrixWorld(true);
  const tight=new THREE.Box3().setFromObject(proxy,true);
  if(!validBox(bounds(tight))) throw error('INVALID_BOUNDS','Missing or non-finite mesh bounds');
  const size=tight.getSize(new THREE.Vector3());
  if(!(Math.max(...size.toArray())>0)) throw error('INVALID_BOUNDS','Degenerate mesh extent');
  return tight;
}

export async function canonicalizeGLB(data,{rotationDeg=[0,0,0]}={}) {
  if(!Array.isArray(rotationDeg)||rotationDeg.length!==3||!rotationDeg.every(Number.isFinite))
    throw error('INVALID_TRANSFORM','Expected three finite XYZ Euler angles in degrees');
  const parsed=parseGLB(arrayBuffer(data));
  const proxy=await loadProxy(data);
  try {
    const rot=new THREE.Matrix4().makeRotationFromEuler(new THREE.Euler(...rotationDeg.map(THREE.MathUtils.degToRad)));
    const group=new THREE.Group();
    group.add(proxy);
    group.applyMatrix4(rot);
    const before=measure(group);
    const c=before.getCenter(new THREE.Vector3());
    const s=1/Math.max(...before.getSize(new THREE.Vector3()).toArray());
    const m=new THREE.Matrix4().makeScale(s,s,s).multiply(new THREE.Matrix4().makeTranslation(-c.x,-c.y,-c.z)).multiply(rot);
    group.matrix.copy(m);group.matrixAutoUpdate=false;
    const tightBounds=bounds(measure(group));
    return {bytes:wrapGLB(parsed,m.toArray()),transform:m.toArray(),inverseTransform:m.clone().invert().toArray(),tightBounds};
  } finally {disposeProxy(proxy);}
}

function backendName(r) {
  if(r.backend?.isWebGPUBackend) return 'webgpu';
  if(r.backend?.isWebGLBackend) return 'webgl2';
  throw error('GPU_INIT','Unsupported renderer backend');
}

function verifyResolution(resolution) {
  if(!Number.isSafeInteger(resolution)||resolution<1||resolution>8192)
    throw error('INVALID_RESOLUTION','Face resolution must be an integer between 1 and 8192');
}

async function useRenderer(options, run) {
  const requested = options.backend === 'webgl' ? 'webgl2' : (options.backend ?? 'auto');
  if(!['webgpu','webgl2','auto'].includes(requested)) throw error('GPU_INIT','Unknown backend '+requested);
  const owned=!options.renderer;
  let renderer=options.renderer;
  try {
    try {
      renderer ??= new THREE.WebGPURenderer({forceWebGL: requested==='webgl2',alpha:true,antialias:false});
      await renderer.init();
    } catch(cause) {throw error('GPU_INIT',cause.message,cause);}
    const backend=backendName(renderer);
    if(requested!=='auto' && backend!==requested) throw error('GPU_INIT','Requested '+requested+' but selected '+backend);
    return await run(renderer,backend);
  } finally {
    if(owned && renderer) await renderer.dispose();
  }
}

function makeCamera(box,face) {
  const size=box.getSize(new THREE.Vector3()).toArray(), c=box.getCenter(new THREE.Vector3());
  const margin=Math.max(...size)*.1, half=size[face.axis]/2;
  const w=size[face.width]/2, h=size[face.height]/2;
  const camera=new THREE.OrthographicCamera(-w,w,h,-h,margin/2,margin+2*half+margin/2);
  camera.position.copy(c).addScaledVector(new THREE.Vector3(...face.side),half+margin);
  camera.up.set(...face.up);camera.lookAt(c);camera.updateMatrixWorld();camera.updateProjectionMatrix();
  return camera;
}

async function pixels(renderer, rt, index, dtype, backend) {
  try {
    const raw=await renderer.readRenderTargetPixelsAsync(rt,0,0,rt.width,rt.height,index);
    return compactPixels(raw,rt.width,rt.height,dtype,backend);
  } catch(cause) {throw error('GPU_READBACK',cause.message,cause);}
}

async function guidePNG(data, resolution) {
  const canvas=document.createElement('canvas');
  canvas.width=resolution;canvas.height=resolution;
  const context=canvas.getContext('2d');
  if(!context) throw error('GUIDE_PNG','Canvas 2D context unavailable');
  const image=context.createImageData(resolution,resolution);
  let hits=0;
  for(let i=0;i<data.length;i+=4) {
    const a=Math.max(0,Math.min(1,THREE.DataUtils.fromHalfFloat(data[i+3])));
    if(a>0) hits++;
    for(let k=0;k<3;k++) {
      const linear=a>0 ? Math.max(0,THREE.DataUtils.fromHalfFloat(data[i+k])/a) : 0;
      const srgb=linear<=.0031308 ? 12.92*linear : 1.055*linear**(1/2.4)-.055;
      image.data[i+k]=255*Math.min(1,srgb);
    }
    image.data[i+3]=255*a;
  }
  context.putImageData(image,0,0);
  const blob=await new Promise((resolve,reject)=>canvas.toBlob(b=>b?resolve(b):reject(error('GUIDE_PNG','PNG encoder failed')),'image/png'));
  return {bytes:new Uint8Array(await blob.arrayBuffer()),hits};
}

async function captureGuides(proxy,box,resolution,renderer,backend) {
  const files=new Map(),faces={};
  const scene=new THREE.Scene();
  // A shallow parented clone keeps canonical transforms below the root, as in the baker.
  const source=proxy.clone();
  source.position.set(0,0,0);source.quaternion.identity();source.scale.set(1,1,1);source.updateMatrixWorld(true);
  scene.add(source);
  const previous={target:renderer.getRenderTarget(),mrt:renderer.getMRT(),auto:renderer.autoClear,alpha:renderer.getClearAlpha(),
    color:renderer.getClearColor(new THREE.Color()).clone()};
  const rt=new THREE.RenderTarget(resolution,resolution,{type:THREE.HalfFloatType,count:2});
  rt.textures[0].name='output';rt.textures[1].name='surfaceDepth';
  let visible=0;
  try {
    renderer.autoClear=true;renderer.setClearColor(0,0);
    const p=positionWorld.sub(vec3(...box.min.toArray())).div(vec3(...box.getSize(new THREE.Vector3()).toArray()));
    for(const [i,f] of FACES.entries()) {
      const camera=makeCamera(box,f);
      const depth=f.side[f.axis]>0?p.element(f.axis).oneMinus():p.element(f.axis);
      renderer.setMRT(mrt({output:vec4(diffuseColor.rgb,1),surfaceDepth:vec4(vec3(depth),1)}));
      renderer.setRenderTarget(rt);renderer.render(scene,camera);
      const color=await pixels(renderer,rt,0,'float16',backend);
      const d=await pixels(renderer,rt,1,'float16',backend);
      const z=new Uint16Array(resolution*resolution);
      // Depth is normalized distance from the viewing side; NaN marks background.
      // Only the primary attachment clears its alpha to zero on both renderers.
      for(let k=0;k<z.length;k++) z[k]=THREE.DataUtils.fromHalfFloat(color[4*k+3])>.5?d[4*k]:0x7e00;
      const png=await guidePNG(color,resolution);visible+=png.hits;
      const guide='guides/'+f.slug+'.png', depthPath='guides/'+f.slug+'.depth16f.bin';
      files.set(guide,png.bytes);files.set(depthPath,encodeRaw(z));
      faces[f.id]={guide,depth:depthPath,depthEncoding:'linear-axis-0-near-1-far;nan-background',
        imageRowOrder:'top-to-bottom',imageColorSpace:'sRGB',foregroundPixels:png.hits,
        up:f.up,side:f.side,tile:[i%3,Math.floor(i/3)],viewMatrix:camera.matrixWorldInverse.toArray(),
        projectionMatrix:camera.projectionMatrix.toArray(),frustum:[camera.left,camera.right,camera.bottom,camera.top,camera.near,camera.far]};
    }
    if(!visible) throw error('GPU_EMPTY','No foreground pixels in any of the six captures');
    return {faces,files};
  } finally {
    renderer.setMRT(previous.mrt);renderer.setRenderTarget(previous.target);renderer.autoClear=previous.auto;
    renderer.setClearColor(previous.color,previous.alpha);rt.dispose();
  }
}

async function inputMetadata(data,options) {
  const canonical=options.canonicalMeta ?? options.canonical;
  if(!canonical) return canonicalizeGLB(data,{rotationDeg:options.rotationDeg});
  const transform=canonical.transform ?? identity(), inverseTransform=canonical.inverseTransform ?? identity();
  for(const m of [transform,inverseTransform])
    if(!Array.isArray(m)||m.length!==16||!m.every(Number.isFinite)) throw error('INVALID_TRANSFORM','Invalid canonical matrix');
  return {bytes:new Uint8Array(arrayBuffer(data)),transform,inverseTransform,tightBounds:canonical.tightBounds};
}

// This entry point is independent of ImpostorBaker: guides can be computed before any atlas exists.
export async function makeSixGuidesGLB(data, options={}) {
  const resolution=options.resolution??256;
  verifyResolution(resolution);
  const canonical=await inputMetadata(data,options), proxy=await loadProxy(canonical.bytes);
  try {
    const box=measure(proxy);
    box.expandByScalar(Math.max(...box.getSize(new THREE.Vector3()).toArray())*.02);
    return await useRenderer(options,(r,b)=>captureGuides(proxy,box,resolution,r,b));
  } finally {disposeProxy(proxy);}
}

async function describeTexture(name,texture,resolution,renderer,backend) {
  const [width,height]=dimensions(name,resolution);
  const dtype=MAP_TYPES[name], fp=dtype==='float16';
  const t=texture.renderTarget;
  if(!t || t.width!==width || t.height!==height ||
    texture.type!==(fp?THREE.HalfFloatType:THREE.UnsignedByteType) || texture.format!==THREE.RGBAFormat)
    throw error('BAKER_CONTRACT','Unexpected upstream format or size on '+name);
  const filter=name==='rangeMap'?'nearest':'linear';
  const sampler=filter==='nearest'?THREE.NearestFilter:THREE.LinearFilter;
  if(texture.minFilter!==sampler||texture.magFilter!==sampler||texture.generateMipmaps ||
    texture.wrapS!==THREE.ClampToEdgeWrapping||texture.wrapT!==THREE.ClampToEdgeWrapping)
    throw error('BAKER_CONTRACT','Unexpected texture filtering on '+name);
  const raw=encodeRaw(await pixels(renderer,t,0,dtype,backend));
  const path='maps/'+name.replace(/Map$/,'')+(fp?'.rgba16f.bin':'.rgba8.bin');
  return {raw,descriptor:{path,width,height,channels:4,dtype,endian:'little',rowOrder:'increasingShaderV',
    rowBytes:width*4*(fp?2:1),byteLength:raw.byteLength,sha256:await sha256(raw),minFilter:filter,magFilter:filter,
    wrap:'clamp-to-edge',mipmaps:false,colorSpace:name==='colorMap'||name==='emissiveMap'?'linear-srgb':'data',
    alpha:name==='colorMap'?'premultiplied-coverage':name==='intervalMap'?'gap-end':name==='rangeMap'?'constant-one':'coverage'}};
}

export function createBoxMaterial(data) {
  const material=new ImpostorNodeMaterial(data);
  // A ray can cross sqrt(3)*R texels diagonally. Occupied steps advance 1.5 texels
  // and empty steps cross one of at most 3*R/8 cell boundaries. Leave room for
  // the first and last cells. The shader still exits at the first hit or box exit.
  material.steps=Math.max(material.steps,2*data.resolution);
  return material;
}

export async function bakeGLB(data,options={}) {
  const resolution=options.resolution??256, requested=options.maps??[];
  verifyResolution(resolution);
  if(!Array.isArray(requested)||new Set(requested).size!==requested.length||requested.some(n=>!MATERIAL_PROPERTIES.includes(n)))
    throw error('INVALID_MAPS','Unsupported or duplicate material map');
  const canonical=await inputMetadata(data,options);
  const proxy=await loadProxy(canonical.bytes);
  try {
    const tightBounds=bounds(measure(proxy)), proxyHash=await sha256(canonical.bytes);
    const asset_id=options.asset_id??(await sha256(new TextEncoder().encode([proxyHash,resolution,...requested].join('/')))).slice(0,24);
    return await useRenderer(options,async(renderer,backend)=>{
      let baked,material,keep=false;
      try {
        try {baked=await new ImpostorBaker(renderer).bake(proxy,{layout:'box',resolution,maps:requested});}
        catch(cause){ throw error('GPU_BAKE',cause.message,cause); }
        const files=new Map([['geometry/proxy.glb',canonical.bytes]]);
        const textures={};
        for(const name of Object.keys(MAP_TYPES)) {
          if(!baked[name]) continue;
          const {raw,descriptor}=await describeTexture(name,baked[name],resolution,renderer,backend);
          textures[name]=descriptor;files.set(descriptor.path,raw);
        }
        const alpha=decodeRaw(files.get(textures.colorMap.path),'float16');
        if(!alpha.some((v,i)=> i%4===3 && THREE.DataUtils.fromHalfFloat(v)>.001))
          throw error('GPU_EMPTY','Upstream bake returned zero coverage in all six faces');
        let faces;
        if(options.guides!==false) {
          const guides=await captureGuides(proxy,baked.box,resolution,renderer,backend);
          faces=guides.faces;
          for(const [p,v] of guides.files) files.set(p,v);
        } else {
          faces=Object.fromEntries(FACES.map((f,i)=>{
            const c=makeCamera(baked.box,f);
            return [f.id,{side:f.side,up:f.up,tile:[i%3,Math.floor(i/3)],
              viewMatrix:c.matrixWorldInverse.toArray(),projectionMatrix:c.projectionMatrix.toArray()}];
          }));
        }
        const manifest={schemaVersion:1,asset_id,mode:'geometry_baseline',layout:'box',resolution,
          proxy:'geometry/proxy.glb',proxySha256:proxyHash,tightBounds,bakeBox:bounds(baked.box),
          transform:canonical.transform,inverseTransform:canonical.inverseTransform,
          faceOrder:FACES.map(f=>f.id),faces,maps:[...baked.maps],textures,baseColor:{...textures.colorMap},
          backend,versions:{three:THREE.REVISION,baker:'0.4.1',bakerCommit:PIN},coordinateSystem:'right-handed-y-up'};
        material=createBoxMaterial(baked);
        const mesh=new THREE.Mesh(baked.geometry,material);
        if(options.beforeDispose) await options.beforeDispose({mesh,renderer,manifest,files});
        if(options.renderer){
          keep=true;
          let released=false;
          return {manifest,files,live:{mesh,dispose(){
            if(!released){material.dispose();baked.dispose();released=true;}
          }}};
        }
        return {manifest,files};
      } finally {if(!keep){material?.dispose();baked?.dispose();}}
    });
  } finally {disposeProxy(proxy);}
}

function validateManifest(m) {
  if(m?.schemaVersion!==1 || m.layout!=='box') throw error('INVALID_MANIFEST','Unsupported schema or layout');
  verifyResolution(m.resolution);
  if(!validBox(m.bakeBox) || m.bakeBox.min.some((v,i)=>v===m.bakeBox.max[i]) ||
    !validBox(m.tightBounds) || !safePath(m.proxy))
    throw error('INVALID_MANIFEST','Invalid bounds or proxy path');
  if(!Array.isArray(m.faceOrder) || m.faceOrder.length!==6 || FACES.some((f,i)=>m.faceOrder[i]!==f.id) ||
     !m.faces || Object.keys(m.faces).length!==6 || FACES.some(f=>!Object.hasOwn(m.faces,f.id)))
    throw error('INVALID_MANIFEST','Invalid six-view layout');
  if(!Array.isArray(m.maps)||new Set(m.maps).size!==m.maps.length||m.maps.some(n=>!MATERIAL_PROPERTIES.includes(n)))
    throw error('INVALID_MANIFEST','Unknown or duplicate material properties');
  const expected=[...REQUIRED];
  if(m.maps.some(n=>n!=='emissive')) expected.push('materialMap');
  if(m.maps.includes('emissive')) expected.push('emissiveMap');
  if(!m.textures||Object.keys(m.textures).length!==expected.length) throw error('INVALID_MANIFEST','Inconsistent texture list');
  const paths=new Set();
  for(const name of expected) {
    const desc=m.textures[name];
    validateDescriptor(desc,name,m.resolution);
    if(paths.has(desc.path)) throw error('INVALID_MANIFEST','Duplicate texture path '+desc.path);
    paths.add(desc.path);
  }
}

export async function restoreBox(manifest,files,renderer,{sampling='standard'}={}) {
  if (!['standard','reduce_bleeding'].includes(sampling)) throw error('INVALID_SAMPLING','Unknown Box render sampling mode');
  validateManifest(manifest);
  const box=new THREE.Box3(new THREE.Vector3(...manifest.bakeBox.min),new THREE.Vector3(...manifest.bakeBox.max));
  const size=box.getSize(new THREE.Vector3()),center=box.getCenter(new THREE.Vector3());
  const geometry=new THREE.BoxGeometry(size.x,size.y,size.z).translate(center.x,center.y,center.z);
  const data={box,geometry,layout:'box',resolution:manifest.resolution,maps:[...manifest.maps]};
  const textures=[];
  let material, disposed=false;
  const dispose=()=>{
    if(disposed) return;
    material?.dispose();geometry.dispose();for(const t of textures) t.dispose();
    disposed=true;
  };
  try {
    for(const [name,d] of Object.entries(manifest.textures)) {
      const bytes=files.get(d.path);
      if(!(bytes instanceof Uint8Array)||bytes.byteLength!==d.byteLength) throw error('INVALID_TEXTURE','Missing or truncated '+name);
      if(await sha256(bytes)!==d.sha256) throw error('TEXTURE_HASH','Checksum mismatch for '+name);
      const t=new THREE.DataTexture(decodeRaw(bytes,d.dtype),d.width,d.height,THREE.RGBAFormat,
        d.dtype==='float16'?THREE.HalfFloatType:THREE.UnsignedByteType);
      // Presentation-only override. Never rewrite the baked interval map or its manifest.
      const filter=name==='intervalMap' && sampling==='reduce_bleeding' ? 'nearest' : d.minFilter;
      t.minFilter=t.magFilter=filter==='nearest'?THREE.NearestFilter:THREE.LinearFilter;
      t.flipY=false;t.wrapS=t.wrapT=THREE.ClampToEdgeWrapping;t.generateMipmaps=false;
      // The stored bytes already carry linear RGB, so use the baker's untagged texture sampling.
      t.colorSpace=THREE.NoColorSpace;t.needsUpdate=true;
      if(renderer?.backend?.isWebGLBackend){
        // The upstream TSL node flips V in the shader for GL render targets. Keep that
        // same UV arithmetic: a CPU-only row change shifts depth at thin ray boundaries.
        const row=d.width*4, original=t.image.data;
        const flipped=new original.constructor(original.length);
        for(let y=0;y<d.height;y++){
          flipped.set(original.subarray(y*row,(y+1)*row),(d.height-1-y)*row);
        }
        t.image.data=flipped;
        const rt=new THREE.RenderTarget(d.width,d.height,{type:t.type,depthBuffer:false,
          minFilter:t.minFilter,magFilter:t.magFilter,generateMipmaps:false});
        textures.push(rt);
        try {
          renderer.initRenderTarget(rt);
          renderer.copyTextureToTexture(t,rt.texture);
        } finally {t.dispose();}
        data[name]=rt.texture;
      } else {
        textures.push(t);
        data[name]=t;
      }
    }
    material=createBoxMaterial(data);
    return {mesh:new THREE.Mesh(geometry,material),dispose};
  } catch(cause){dispose();throw cause;}
}
