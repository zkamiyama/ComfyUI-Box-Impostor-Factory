import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three/webgpu';
import { FACES, compactPixels, decodeRaw, dimensions, encodeRaw, parseGLB, safePath, wrapGLB } from '../worker/codec.mjs';
import { bakeGLB, canonicalizeGLB, disposeProxy, loadProxy, restoreBox } from '../worker/runtime.mjs';
import { makeFixture } from './make_fixture.mjs';

test('WebGPU padded half-float rows and WebGL bottom-up rows pack to identical bit patterns', () => {
  const w=3, h=3, row=w*4, stride=128;
  const want=Uint16Array.from({length:row*h},(_,i)=>0x3000+i);
  const gpu=new Uint16Array((h-1)*stride+row).fill(0xffff);
  const gl=new Uint16Array(row*h);
  for(let y=0;y<h;y++){
    const line=want.subarray(y*row,(y+1)*row);
    gpu.set(line,y*stride);gl.set(line,(h-1-y)*row);
  }
  assert.deepEqual(compactPixels(gpu,w,h,'float16','webgpu'),want);
  assert.deepEqual(compactPixels(gl,w,h,'float16','webgl2'),want);
  assert.throws(()=>compactPixels(gpu.slice(1),w,h,'float16','webgpu'),/stride/);
  assert.deepEqual(decodeRaw(encodeRaw(want),'float16'),want);
  assert.equal(encodeRaw(want)[1],0x30);
  assert.deepEqual(dimensions('rangeMap',41),[18,6]);
});

test('GLB external resources and nonportable map paths are rejected',()=>{
  const glb=parseGLB(makeFixture());
  glb.json.images=[{uri:'https://example.com/texture.jpg'}];
  assert.throws(()=>parseGLB(wrapGLB(glb,new THREE.Matrix4().toArray())),/embedded/);
  for(const path of ['../a','a/../b','/a','a//b','C:\\tex','a%2fb','']) assert.equal(safePath(path),false);
});

test('canonical GLB carries the transform below its root; transforms and binary chunk survive rotation',async()=>{
  const original=makeFixture();
  const c=await canonicalizeGLB(original,{rotationDeg:[0,90,0]});
  assert.deepEqual(parseGLB(c.bytes).tail,Uint8Array.from(parseGLB(original).tail));
  const p=await loadProxy(c.bytes);
  try{
    const b=new THREE.Box3().setFromObject(p,true);
    assert(b.getCenter(new THREE.Vector3()).length()<1e-7);
    assert(Math.abs(Math.max(...b.getSize(new THREE.Vector3()).toArray())-1)<1e-7);
    const product=new THREE.Matrix4().fromArray(c.transform).multiply(new THREE.Matrix4().fromArray(c.inverseTransform));
    assert(product.toArray().every((x,i)=>Math.abs(x-(i%5===0?1:0))<1e-7));
  }finally{disposeProxy(p);}
});

// Only GPU execution is substituted. The baker, camera placement, target formats,
// serialization, manifest checks and reload material/geometry all run normally.
test('native baker camera setup and odd-sized readback survive serialization and reconstruction',async()=>{
  const cameras=[];
  const renderer={
    backend:{isWebGPUBackend:true},autoClear:true,async init(){},
    setRenderTarget(t){this.target=t;},getRenderTarget(){return null;},
    setClearColor(){},getClearAlpha(){return 0;},getClearColor(c){return c.set(0);},
    setMRT(){},getMRT(){return null;},
    render(scene,c){cameras.push({pos:c.position.toArray(),up:c.up.toArray(),l:c.left,r:c.right});},
    async readRenderTargetPixelsAsync(t,x,y,w,h,index){
      const T=t.textures[index].type===THREE.HalfFloatType?Uint16Array:Uint8Array;
      const row=w*4,stride=Math.ceil(row*T.BYTES_PER_ELEMENT/256)*256/T.BYTES_PER_ELEMENT;
      const out=new T((h-1)*stride+row).fill(0);
      for(let j=0;j<h;j++)out.fill(T===Uint16Array?0x3c00:255,j*stride,j*stride+row);
      return out;
    },
  };
  const result=await bakeGLB(makeFixture(),{resolution:41,maps:['roughness'],renderer,guides:false});
  const {manifest,files}=result;
  try {
    assert.deepEqual(cameras.slice(0,6).map(c=>c.up),FACES.map(f=>f.up));
    assert.deepEqual([manifest.textures.rangeMap.width,manifest.textures.rangeMap.height],[18,6]);
    assert.equal(manifest.textures.materialMap.dtype,'uint8');
    assert.equal(manifest.textures.intervalMap.minFilter,'linear');
    assert.equal(manifest.textures.rangeMap.minFilter,'nearest');
    assert.equal(manifest.baseColor.sha256,manifest.textures.colorMap.sha256);
    const restored=await restoreBox(manifest,files);
    try{
      const info=restored.mesh.material.impostor;
      assert.deepEqual(info.box.min.toArray(),result.live.mesh.material.impostor.box.min.toArray());
      assert.equal(info.materialMap.type,THREE.UnsignedByteType);
      assert.equal(info.rangeMap.minFilter,THREE.NearestFilter);
      assert.equal(info.colorMap.image.data[0],0x3c00);
    } finally {restored.dispose();}
    const damaged=Uint8Array.from(files.get(manifest.textures.intervalMap.path));damaged[0]^=1;
    const corrupted=new Map(files).set(manifest.textures.intervalMap.path,damaged);
    await assert.rejects(restoreBox(manifest,corrupted),/Checksum mismatch/);
    const incomplete=structuredClone(manifest);delete incomplete.textures.rangeMap;
    await assert.rejects(restoreBox(incomplete,files),/texture list/);
  } finally {result.live.dispose();}
});
