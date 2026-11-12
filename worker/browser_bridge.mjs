import {
  Scene, RenderTarget, DepthTexture, OrthographicCamera, HemisphereLight, DirectionalLight,
  MeshBasicNodeMaterial, QuadMesh, UnsignedIntType, NoColorSpace, NoToneMapping, Vector3
} from 'three/webgpu';
import { texture, screenUV, vec4, clamp, floor, min, mod } from 'three/tsl';
import { bakeGLB, restoreBox, canonicalizeGLB } from './runtime.mjs';
import { fetchChecked, makeRenderer, backendName, readAsset } from '../web/asset_io.mjs';

import { directions } from './capture_views.mjs';

function base64(bytes) {
  let binary = '';
  for (let i = 0; i < bytes.length; i += 16384)
    binary += String.fromCharCode(...bytes.subarray(i,i+16384));
  return btoa(binary);
}

export async function snapshot(renderer, mesh, bounds) {
  const size = 64; // RGBA8 readback is exactly one 256-byte WebGPU row.
  const scene = new Scene();
  scene.add(mesh, new HemisphereLight(0xf3f5ff, 0xc2bdb5, 1.7));
  const light = new DirectionalLight(0xffffff, 2);
  light.position.set(2, 4, 3);
  scene.add(light);
  const center = new Vector3().fromArray(bounds.min).lerp(new Vector3().fromArray(bounds.max),0.5);
  const radius = new Vector3().fromArray(bounds.max).sub(new Vector3().fromArray(bounds.min)).length()/2;
  if (!(radius > 0) || !Number.isFinite(radius)) throw new Error('Invalid capture bounds');
  const span = radius * 1.08;
  const camera = new OrthographicCamera(-span,span,span,-span, radius*0.5,radius*5);
  const target = new RenderTarget(size,size);
  const depthRead = new RenderTarget(size,size,{depthBuffer:false});
  target.texture.colorSpace = NoColorSpace;
  target.depthTexture = new DepthTexture(size,size,UnsignedIntType);
  // Sample the depth *written by the upstream material*; no substitute geometry pass.
  const material = new MeshBasicNodeMaterial();
  material.toneMapped = false;
  const val = min(floor(clamp(texture(target.depthTexture,screenUV).r,0,1).mul(16777215).add(0.5)),16777215);
  material.fragmentNode = vec4(floor(val.div(65536)), mod(floor(val.div(256)),256), mod(val,256),255).div(255);
  const quad = new QuadMesh(material);
  const frames = [];
  renderer.setSize(size,size,false);
  renderer.setClearColor(0,0);
  renderer.toneMapping = NoToneMapping;
  try {
    for (const [id,x,y,z] of directions) {
      camera.position.set(x,y,z).normalize().multiplyScalar(radius*3).add(center);
      camera.up.set(0,1,0);
      if (!x && !z) camera.up.set(0,0,y > 0 ? -1 : 1);
      camera.lookAt(center);
      camera.updateMatrixWorld();
      renderer.setRenderTarget(target);
      renderer.render(scene,camera);
      const color = await renderer.readRenderTargetPixelsAsync(target,0,0,size,size);
      renderer.setRenderTarget(depthRead);
      quad.render(renderer);
      const depth = await renderer.readRenderTargetPixelsAsync(depthRead,0,0,size,size);
      frames.push({id, width:size, height:size, color:base64(color), depth:base64(depth)});
    }
  } finally {
    renderer.setRenderTarget(null);
    scene.remove(mesh);
    target.dispose();
    depthRead.dispose();
    material.dispose();
  }
  return frames;
}

export async function bake({resolution=256,maps=[],backend='auto',asset_id,verify=false,guides=true,canonical=false,canonicalMeta}={}) {
  const renderer = await makeRenderer(backend);
  let result;
  try {
    const glb = await (await fetchChecked('/job/input.glb')).arrayBuffer();
    result = await bakeGLB(glb,{resolution,maps,backend,asset_id,renderer,guides,canonical:canonicalMeta ?? canonical});
    for (const [path,bytes] of result.files) {
      const res = await fetch('/job/files/'+path.split('/').map(encodeURIComponent).join('/'),
        {method:'PUT',body:bytes});
      if (!res.ok) throw new Error('Cannot stage ' + path + ' (HTTP ' + res.status + ')');
    }
    return {manifest:result.manifest,backend:backendName(renderer),snapshot:verify ?
      await snapshot(renderer,result.live.mesh,result.manifest.bakeBox) : undefined};
  } finally {
    result?.live?.dispose();
    renderer.dispose();
  }
}

export async function reload({backend='auto',verify=true,manifestUrl='/asset/manifest.json'}={}) {
  const renderer = await makeRenderer(backend);
  let restored;
  try {
    const {manifest,files} = await readAsset(manifestUrl);
    restored = await restoreBox(manifest,files,renderer);
    return {backend:backendName(renderer), snapshot:verify ? await snapshot(renderer,restored.mesh,manifest.bakeBox) : undefined};
  } finally {
    restored?.dispose();
    renderer.dispose();
  }
}
export async function canonicalize({rotationDeg=[0,0,0]}={}) {
  const glb = await (await fetchChecked('/job/input.glb')).arrayBuffer();
  const {bytes,transform,inverseTransform,tightBounds} = await canonicalizeGLB(glb,{rotationDeg});
  const res = await fetch('/job/files/canonical.glb',{method:'PUT',body:bytes});
  if (!res.ok) throw new Error('Cannot stage canonical GLB (HTTP ' + res.status + ')');
  return {transform,inverseTransform,tightBounds};
}
globalThis.biBridge = {bake,reload,canonicalize};
