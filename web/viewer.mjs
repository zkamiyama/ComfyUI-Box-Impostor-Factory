import {
  Vector3, HemisphereLight, DirectionalLight, PerspectiveCamera, OrthographicCamera, Scene, DataUtils
} from 'three/webgpu';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { restoreBox, loadProxy, disposeProxy } from '../worker/runtime.mjs';
import { assetURL, readAsset, makeRenderer, backendName } from './asset_io.mjs';

const el = id => document.getElementById(id);
const directions = {
  diagonal:[1,0.66,1], '+X':[1,0,0], '-X':[-1,0,0], '+Y':[0,1,0],
  '-Y':[0,-1,0], '+Z':[0,0,1], '-Z':[0,0,-1]
};

function appendOption(select,value,label) {
  select.add(new Option(label,value));
}

function texturePreview(name, descriptor, bytes) {
  const {width,height,dtype} = descriptor;
  const samples = new DataView(bytes.buffer,bytes.byteOffset,bytes.byteLength);
  const byteWidth = dtype === 'uint8' ? 1 : 2;
  const read = byteWidth === 1 ? i => samples.getUint8(i)/255 : i => DataUtils.fromHalfFloat(samples.getUint16(i*2,true));
  const canvas = document.createElement('canvas');
  canvas.width = width; canvas.height = height;
  const ctx = canvas.getContext('2d');
  const image = ctx.createImageData(width,height);
  const linearToSRGB = x => x <= .0031308 ? x*12.92 : 1.055*Math.pow(x,1/2.4)-.055;
  const isColor = ['colorMap','emissiveMap','baseColor'].includes(name);
  // The first stored row corresponds to shader v=0. The preview is shown with v increasing upwards.
  for (let y=0; y<height; y++) for (let x=0; x<width; x++) {
    const src = 4*((height-1-y)*width+x), dst = 4*(y*width+x);
    const alpha = read(src+3);
    for (let i=0; i<3; i++) {
      const c = read(src+i);
      image.data[dst+i] = 255*(isColor ? linearToSRGB(alpha > .0001 ? c/alpha : 0) : c);
    }
    image.data[dst+3] = 255 * (isColor || name === 'normalMap' ? alpha : 1);
  }
  ctx.putImageData(image,0,0);
  return canvas.toDataURL();
}

function litScene() {
  const scene = new Scene();
  const sun = new DirectionalLight(0xffffff,2.5);
  scene.add(new HemisphereLight(0xf0f5ff,0xd8ccbc,1.8),sun);
  return {scene,sun};
}

async function run() {
  const params = new URLSearchParams(location.search);
  const manifestURL = params.get('manifest');
  if (!manifestURL) throw new Error('manifest パラメーターにアセットのURLを指定してください');
  const {manifest,files,url} = await readAsset(manifestURL, {proxy:true});
  const [renderer,active,proxy] = await Promise.all([
    makeRenderer(params.get('backend') || 'auto',{canvas:el('canvas'),antialias:true}),
    restoreBox(manifest,files), loadProxy(files.get(manifest.proxy).slice().buffer)
  ]);
  const hasBase = manifest.baseColor?.sha256 && manifest.baseColor.sha256 !== manifest.textures.colorMap.sha256;
  const baseline = hasBase ? await restoreBox({...manifest,textures:{
    ...manifest.textures, colorMap:manifest.baseColor}},files) : null;
  el('appearanceLabel').hidden = !hasBase;
  const box = litScene(), mesh = litScene();
  box.scene.add(active.mesh);
  if (baseline) { box.scene.add(baseline.mesh); baseline.mesh.visible = false; }
  mesh.scene.add(proxy);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1,2));
  renderer.setClearColor(0x000000,0);
  const center = new Vector3().fromArray(manifest.bakeBox.min).lerp(new Vector3().fromArray(manifest.bakeBox.max),0.5);
  const radius = new Vector3().fromArray(manifest.bakeBox.max).sub(new Vector3().fromArray(manifest.bakeBox.min)).length()/2;
  const stage = el('stage'), help=el('help');
  let camera, controls, width=0, height=0;

  function placeCamera(reset=false, newPose=false) {
    const orthographic = el('projection').value === 'orthographic';
    const target = !reset && controls ? controls.target.clone() : center.clone();
    const oldDirection = camera?.position.clone().sub(target).normalize();
    const oldUp = camera?.up.clone();
    const fov = 36, tangent=Math.tan(fov*Math.PI/360);
    const visibleAspect = Math.max(.05, (el('mode').value === 'both' ? Math.floor(stage.clientWidth/2) : stage.clientWidth)/Math.max(stage.clientHeight,1));
    const distance = !reset && camera ? camera.position.distanceTo(target) :
      radius*1.1 / Math.sin(Math.atan(tangent * Math.min(visibleAspect,1)));
    const direction = !reset && !newPose && oldDirection ? oldDirection : new Vector3(...directions[el('view').value]).normalize();
    controls?.dispose();
    camera = orthographic ? new OrthographicCamera(-1,1,1,-1,0.01*radius,100*radius) :
      new PerspectiveCamera(fov,1,0.01*radius,100*radius);
    if (!reset && !newPose && oldUp) camera.up.copy(oldUp);
    else if (Math.abs(direction.y)>.99999) camera.up.set(0,0,direction.y>0?-1:1);
    camera.position.copy(target).addScaledVector(direction,distance);
    camera.lookAt(target);
    controls = new OrbitControls(camera,el('canvas'));
    controls.target.copy(target);
    controls.minDistance = radius*.12; controls.maxDistance = radius*35;
    controls.update();
    // Keep apparent magnification when switching projection.
    camera.userData.halfHeight = distance*tangent;
  }

  function setProjection(w,h) {
    const aspect = w/h;
    if (camera.isOrthographicCamera) {
      const span = camera.userData.halfHeight;
      camera.left = -span*aspect; camera.right = span*aspect; camera.top = span; camera.bottom = -span;
    } else camera.aspect=aspect;
    camera.updateProjectionMatrix();
  }

  function renderPane(scene,left,w) {
    setProjection(w,height);
    renderer.setViewport(left,0,w,height);
    renderer.setScissor(left,0,w,height);
    renderer.render(scene,camera);
  }

  function render() {
    const w = stage.clientWidth, h = stage.clientHeight;
    if (!w || !h) return;
    if (width !== w || height !== h) {
      width = w; height = h;
      renderer.setSize(width,height,false);
    }
    if (el('mode').value === 'atlas') return;
    controls.update();
    if (el('mode').value === 'both') {
      const half = Math.floor(width/2);
      renderer.setScissorTest(true);
      renderPane(mesh.scene,0,half);
      renderPane(box.scene,half,width-half);
    } else {
      renderer.setScissorTest(false);
      renderPane(el('mode').value === 'proxy' ? mesh.scene : box.scene,0,width);
    }
  }

  const previews = new Map();
  for (const [name] of Object.entries(manifest.textures)) appendOption(el('map'),name,name);
  if (hasBase) appendOption(el('map'),'baseColor','colorMap（ベイク時）');
  for (const face of manifest.faceOrder) {
    if (manifest.faces?.[face]?.guide) appendOption(el('map'),'guide:'+face,face+' ガイド');
    if (manifest.faces?.[face]?.edit?.view) appendOption(el('map'),'view:'+face,face+' 入力画像');
  }

  function selectMap() {
    const name=el('map').value;
    const [kind,face] = name.split(':');
    if (face) {
      const item=manifest.faces[face];
      el('atlas').src=assetURL(url,kind==='guide' ? item.guide : item.edit.view).href;
    } else {
      const descriptor = name==='baseColor' ? manifest.baseColor : manifest.textures[name];
      if (!previews.has(name)) previews.set(name,texturePreview(name,descriptor,files.get(descriptor.path)));
      el('atlas').src=previews.get(name);
    }
    el('atlas').alt=el('map').selectedOptions[0].text;
  }

  function modeChanged() {
    const mode = el('mode').value, atlas = mode==='atlas';
    el('labels').style.display = mode==='both' ? 'flex' : 'none';
    el('mapLabel').hidden = !atlas; el('atlas').hidden = !atlas;
    el('appearanceLabel').hidden = !hasBase || mode==='proxy' || atlas;
    help.hidden = atlas;
    if (atlas) selectMap();
    else if (mode==='both') placeCamera(true);
  }

  function rotateLight() {
    const a = Number(el('light').value)*Math.PI/180;
    for (const {sun} of [mesh,box]) sun.position.set(Math.sin(a)*3,3,Math.cos(a)*3);
  }
  el('appearance').onchange = () => {
    active.mesh.visible = el('appearance').value==='active';
    if (baseline) baseline.mesh.visible = !active.mesh.visible;
  };
  el('mode').onchange=modeChanged;
  el('fit').onclick=() => placeCamera(true);
  el('view').onchange=() => placeCamera(true,true);
  el('projection').onchange=() => placeCamera(false,false);
  el('map').onchange=selectMap;
  el('background').onchange = () => {
    const value=el('background').value;
    stage.classList.toggle('checker',value==='checker');
    stage.style.backgroundColor = value==='checker' ? '' : value;
  };
  el('light').oninput=rotateLight;
  rotateLight(); placeCamera(true); modeChanged();
  const faceReports=Object.values(manifest.faces||{}).map(f=>f.edit?.status);
  const fallback=faceReports.filter(s=>s==='fallback').length;
  el('status').textContent=[backendName(renderer),manifest.mode,'R='+manifest.resolution,fallback ? '元の色へ戻した面 '+fallback : ''].filter(Boolean).join(' · ');
  renderer.setAnimationLoop(render);
  window.addEventListener('pagehide',() => {
    renderer.setAnimationLoop(null); controls.dispose();
    active.dispose(); baseline?.dispose();
    disposeProxy(proxy);
    renderer.dispose();
  },{once:true});
}

run().catch(error => {
  console.error(error);
  el('status').textContent='表示エラー  '+error.message;
  el('status').classList.add('error');
  document.querySelector('.controls').hidden=true;
  el('help').hidden=true;
});
