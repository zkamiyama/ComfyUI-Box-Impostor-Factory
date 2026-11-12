import {
  Vector3, HemisphereLight, DirectionalLight, PerspectiveCamera, OrthographicCamera, Scene, DataUtils,
  RenderTarget, DepthTexture, UnsignedIntType, HalfFloatType, MeshBasicNodeMaterial, QuadMesh
} from 'three/webgpu';
import {
  diffuseColor, normalWorld, mrt, texture, uniform, uv, vec4, vec3, clamp,
  perspectiveDepthToViewZ, orthographicDepthToViewZ, sRGBTransferEOTF
} from 'three/tsl';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { restoreBox, loadProxy, disposeProxy } from '../worker/runtime.mjs';
import { assetURL, fetchChecked, readAsset, makeRenderer, backendName } from './asset_io.mjs';

const el = id => document.getElementById(id);
const directions = {
  diagonal:[1,0.66,1], '+X':[1,0,0], '-X':[-1,0,0], '+Y':[0,1,0],
  '-Y':[0,-1,0], '+Z':[0,0,1], '-Z':[0,0,-1]
};

function appendOption(select,value,label) {
  select.add(new Option(label,value));
}

function showFaceDecisions(manifest, url) {
  if (!manifest.faceOrder.some(id => manifest.faces[id].edit)) return;
  const panel=el('faceQA');
  const reasons={'empty-foreground':'前景が空', 'transform-limit':'移動量または拡大率が許容値外',
    'silhouette-iou':'輪郭の一致率が基準未満', 'generation-canvas-aspect':'生成画像の縦横比が不一致',
    'silhouette-passed':'輪郭の基準を満たした'};
  const sources={qwen_enhanced:'Qwen',h3_enhanced:'H3',manual_faces:'手動画像'};
  for (const id of manifest.faceOrder) {
    const edit=manifest.faces[id].edit;
    const row=panel.querySelector('tbody').insertRow();
    const color=edit?.status==='accepted' ? (sources[edit.source] || edit.source) : 'ベイク時の色';
    const reason=edit ? (reasons[edit.reason] || edit.reason) : '画像の指定なし';
    for (const text of [id,color,reason]) row.insertCell().textContent=text;
  }
  if (manifest.qa) {
    const report=panel.querySelector('a');
    report.href=assetURL(url,manifest.qa).href;
    report.target='_blank'; report.rel='noopener'; report.hidden=false;
  }
  panel.hidden=false;
}

const geometryChannels={
  intervalMap:['R 正側の表面','G 負側の表面（1 − 座標）','B 空隙の始点','A 空隙の終点'],
  rangeMap:['R 占有範囲の下限','G 占有範囲の上限']
};

function texturePreview(name, descriptor, bytes, channel=null) {
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
      const c = read(src+(channel ?? i));
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

// The first pass runs the original proxy or impostor material, including alpha discard
// and ray-marched fragment depth. The second pass only maps its values to display colors.
function diagnosticPane(radius) {
  const target=new RenderTarget(1,1,{type:HalfFloatType});
  target.texture.name='output';
  target.depthTexture=new DepthTexture(1,1,UnsignedIntType);
  const albedo=mrt({output:diffuseColor});
  const normals=mrt({output:vec4(normalWorld.mul(.5).add(.5),diffuseColor.a)});
  const color=texture(target.texture,uv());
  const depth=texture(target.depthTexture,uv()).r;
  const near=uniform(0), far=uniform(1), centerDepth=uniform(0);
  const perspective=uniform(false,'bool'), viewCenter=new Vector3();
  const viewZ=perspective.select(perspectiveDepthToViewZ(depth,near,far),orthographicDepthToViewZ(depth,near,far));
  const brightness=clamp(centerDepth.add(viewZ).div(2*radius).add(.5),0,1);
  const outputs={
    albedo:color,
    normal:vec4(sRGBTransferEOTF(color.rgb),color.a),
    depth:vec4(sRGBTransferEOTF(vec3(brightness)),color.a)
  };
  const material=new MeshBasicNodeMaterial();
  material.depthWrite=false; material.depthTest=false; material.toneMapped=false;
  const quad=new QuadMesh(material);
  return {
    draw(renderer,scene,camera,mode,w,h,center){
      const ratio=renderer.getPixelRatio();
      target.setSize(Math.max(1,Math.floor(w*ratio)),Math.max(1,Math.floor(h*ratio)));
      const split=renderer.getScissorTest();
      renderer.setScissorTest(false);
      renderer.setRenderTarget(target);
      renderer.setMRT(mode==='normal'?normals:albedo);
      try { renderer.render(scene,camera); }
      finally { renderer.setMRT(null); renderer.setRenderTarget(null); renderer.setScissorTest(split); }
      near.value=camera.near; far.value=camera.far;
      perspective.value=camera.isPerspectiveCamera===true;
      centerDepth.value=-camera.worldToLocal(viewCenter.copy(center)).z;
      if (material.fragmentNode!==outputs[mode]) {
        material.fragmentNode=outputs[mode]; material.needsUpdate=true;
      }
      quad.render(renderer);
    },
    dispose(){target.dispose();material.dispose();}
  };
}

async function guideDepthPreview(url, face) {
  const [bytes,guideBlob]=await Promise.all([
    fetchChecked(assetURL(url,face.depth)).then(r=>r.arrayBuffer()),
    fetchChecked(assetURL(url,face.guide)).then(r=>r.blob())
  ]);
  const guide=await createImageBitmap(guideBlob);
  const canvas=document.createElement('canvas');
  canvas.width=guide.width; canvas.height=guide.height;
  const ctx=canvas.getContext('2d');
  ctx.drawImage(guide,0,0); guide.close();
  if(bytes.byteLength!==canvas.width*canvas.height*2) throw Error('深度ガイドのサイズが一致しません');
  const values=new DataView(bytes), image=ctx.getImageData(0,0,canvas.width,canvas.height);
  for(let i=0;i<image.data.length;i+=4){
    const z=DataUtils.fromHalfFloat(values.getUint16(i/2,true));
    // Early exports store zero at empty pixels. The guide's alpha identifies them.
    if(!Number.isFinite(z)) image.data[i+3]=0;
    image.data[i]=image.data[i+1]=image.data[i+2]=255*(1-z);
  }
  ctx.putImageData(image,0,0);
  return canvas.toDataURL();
}

async function run() {
  const params = new URLSearchParams(location.search);
  const manifestURL = params.get('manifest');
  if (!manifestURL) throw new Error('manifest パラメーターにアセットのURLを指定してください');
  const {manifest,files,url} = await readAsset(manifestURL, {proxy:true});
  showFaceDecisions(manifest,url);
  const selectedSampling=params.get('sampling') || 'standard';
  if(!['standard','reduce_bleeding'].includes(selectedSampling)) throw new Error('不明な補間設定です');
  el('sampling').value=selectedSampling;
  let currentSampling=selectedSampling;
  const renderer = await makeRenderer(params.get('backend') || 'auto',{canvas:el('canvas'),antialias:true});
  const [firstActive,proxy] = await Promise.all([
    restoreBox(manifest,files,renderer,{sampling:currentSampling}), loadProxy(files.get(manifest.proxy).slice().buffer)
  ]);
  let active=firstActive;
  const hasBase = manifest.baseColor?.sha256 && manifest.baseColor.sha256 !== manifest.textures.colorMap.sha256;
  const baselineManifest=hasBase ? {...manifest,textures:{
    ...manifest.textures, colorMap:manifest.baseColor}} : null;
  let baseline = hasBase ? await restoreBox(baselineManifest,files,renderer,{sampling:currentSampling}) : null;
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
  const diagnostics=[];
  function releaseDiagnostics(){for(const pane of diagnostics) pane.dispose();diagnostics.length=0;}

  function placeCamera(reset=false, newPose=false) {
    const orthographic = el('projection').value === 'orthographic';
    const target = !reset && controls ? controls.target.clone() : center.clone();
    const oldDirection = camera?.position.clone().sub(target).normalize();
    const oldUp = camera?.up.clone();
    const fov = 36, tangent=Math.tan(fov*Math.PI/360);
    const paneWidth=el('mode').value === 'both' ? Math.floor(stage.clientWidth/2) : stage.clientWidth;
    const visibleAspect = Math.max(1,paneWidth)/Math.max(stage.clientHeight,1);
    const fitDistance=radius*1.1 / Math.sin(Math.atan(tangent * Math.min(visibleAspect,1)));
    const distance = !reset && camera ? (camera.isOrthographicCamera ?
      camera.userData.halfHeight/(camera.zoom*tangent) : camera.position.distanceTo(target)) : fitDistance;
    const maxDistance=Math.max(radius*35,fitDistance*1.2,distance);
    const far=Math.max(100*radius,maxDistance+2*radius);
    const direction = !reset && !newPose && oldDirection ? oldDirection : new Vector3(...directions[el('view').value]).normalize();
    controls?.dispose();
    camera = orthographic ? new OrthographicCamera(-1,1,1,-1,0.01*radius,far) :
      new PerspectiveCamera(fov,1,0.01*radius,far);
    if (!reset && !newPose && oldUp) camera.up.copy(oldUp);
    else if (Math.abs(direction.y)>.99999) camera.up.set(0,0,direction.y>0?-1:1);
    camera.position.copy(target).addScaledVector(direction,distance);
    camera.lookAt(target);
    controls = new OrbitControls(camera,el('canvas'));
    controls.target.copy(target);
    controls.minDistance = radius*.12; controls.maxDistance = maxDistance;
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
    const mode=el('shading').value;
    if(mode==='lit') renderer.render(scene,camera);
    else {
      const i=left===0?0:1;
      const pane=diagnostics[i] ??= diagnosticPane(radius);
      pane.draw(renderer,scene,camera,mode,w,height,center);
    }
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
  for (const name of Object.keys(manifest.textures)) {
    appendOption(el('map'),name,name);
    for(const [channel,label] of (geometryChannels[name] ?? []).entries())
      appendOption(el('map'),name+':'+channel,name+' · '+label);
  }
  if (hasBase) appendOption(el('map'),'baseColor','colorMap（ベイク時）');
  for (const face of manifest.faceOrder) {
    if (manifest.faces?.[face]?.guide) appendOption(el('map'),'guide:'+face,face+' ガイド');
    if (manifest.faces?.[face]?.depth && manifest.faces[face].depthEncoding==='linear-axis-0-near-1-far;nan-background')
      appendOption(el('map'),'depth:'+face,face+' ガイド深度（手前が白）');
    if (manifest.faces?.[face]?.edit?.view) appendOption(el('map'),'view:'+face,face+' 入力画像');
  }

  async function selectMap() {
    const name=el('map').value;
    const [kind,face] = name.split(':');
    if (kind==='depth') {
      el('atlas').removeAttribute('src');
      try {
        if(!previews.has(name)) previews.set(name,guideDepthPreview(url,manifest.faces[face]));
        const image=await previews.get(name);
        if(el('map').value!==name) return;
        el('atlas').src=image;
      } catch(err) {
        previews.delete(name);
        if(el('map').value===name) el('atlas').alt='深度ガイドを読み込めません: '+err.message;
        return;
      }
    } else if (face && !geometryChannels[kind]) {
      const item=manifest.faces[face];
      el('atlas').src=assetURL(url,kind==='guide' ? item.guide : item.edit.view).href;
    } else {
      const descriptor = kind==='baseColor' ? manifest.baseColor : manifest.textures[kind];
      if (!previews.has(name)) previews.set(name,texturePreview(kind,descriptor,files.get(descriptor.path),face===undefined?null:Number(face)));
      el('atlas').src=previews.get(name);
    }
    el('atlas').alt=el('map').selectedOptions[0].text;
  }

  function modeChanged() {
    const mode = el('mode').value, atlas = mode==='atlas';
    el('labels').style.display = mode==='both' ? 'flex' : 'none';
    el('mapLabel').hidden = !atlas; el('atlas').hidden = !atlas;
    el('appearanceLabel').hidden = !hasBase || mode==='proxy' || atlas;
    el('samplingLabel').hidden = mode==='proxy' || atlas;
    el('shadingLabel').hidden = atlas;
    el('lightLabel').hidden = atlas || el('shading').value!=='lit';
    help.hidden = atlas;
    if (atlas) selectMap();
    releaseDiagnostics();
  }

  function rotateLight() {
    const a = Number(el('light').value)*Math.PI/180;
    for (const {sun} of [mesh,box]) sun.position.set(Math.sin(a)*3,3,Math.cos(a)*3);
  }
  el('appearance').onchange = () => {
    active.mesh.visible = el('appearance').value==='active';
    if (baseline) baseline.mesh.visible = !active.mesh.visible;
  };
  // Rebuild the display material on demand. WebGL render-target samplers are fixed at
  // restoration time; changing their properties in place is not portable to WebGPU.
  let samplingBusy=false;
  el('sampling').onchange = async () => {
    if (samplingBusy || el('sampling').value===currentSampling) return;
    samplingBusy=true;
    el('sampling').disabled=true;
    renderer.setAnimationLoop(null);
    const requested=el('sampling').value;
    let nextActive, nextBaseline;
    try {
      nextActive=await restoreBox(manifest,files,renderer,{sampling:requested});
      nextBaseline=baselineManifest ? await restoreBox(baselineManifest,files,renderer,{sampling:requested}) : null;
      box.scene.remove(active.mesh);
      if (baseline) box.scene.remove(baseline.mesh);
      active.dispose();
      baseline?.dispose();
      active=nextActive;
      baseline=nextBaseline;
      box.scene.add(active.mesh);
      if (baseline) box.scene.add(baseline.mesh);
      active.mesh.visible=el('appearance').value==='active' || !baseline;
      if (baseline) baseline.mesh.visible=!active.mesh.visible;
      currentSampling=requested;
      const prefix=el('status').textContent.replace(' · 色にじみ低減','');
      el('status').textContent=prefix+(requested==='reduce_bleeding'?' · 色にじみ低減':'');
    } catch (err) {
      nextActive?.dispose();
      nextBaseline?.dispose();
      el('sampling').value=currentSampling;
      console.error('Box sampling update failed:',err);
      el('status').textContent='補間設定の変更に失敗: '+err.message;
    } finally {
      samplingBusy=false;
      el('sampling').disabled=false;
      renderer.setAnimationLoop(render);
    }
  };
  el('mode').onchange=modeChanged;
  el('shading').onchange=modeChanged;
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
  el('status').textContent=[backendName(renderer),manifest.mode,'R='+manifest.resolution,fallback ? '元の色へ戻した面 '+fallback : '',currentSampling==='reduce_bleeding'?'色にじみ低減':''].filter(Boolean).join(' · ');
  let closing=false;
  if (renderer.backend.isWebGPUBackend) renderer.backend.device.lost.then(info => {
    if (closing) return;
    renderer.setAnimationLoop(null);
    const fallbackURL=new URL(location.href);
    fallbackURL.searchParams.set('backend','webgl2');
    if (!params.has('backend') || params.get('backend')==='auto') {
      location.replace(fallbackURL);
      return;
    }
    showError(new Error('WebGPU デバイスを失いました。'+(info.message || info.reason || '')));
    const link=document.createElement('a');
    link.href=fallbackURL.href; link.textContent='WebGL2 で開く'; link.style.color='inherit';
    el('status').append(' ',link);
  });
  renderer.setAnimationLoop(render);
  window.addEventListener('pagehide',() => {
    closing=true;renderer.setAnimationLoop(null); controls.dispose();
    releaseDiagnostics(); active.dispose(); baseline?.dispose();
    disposeProxy(proxy);
    renderer.dispose();
  },{once:true});
}

function showError(error) {
  console.error(error);
  el('status').textContent='表示エラー  '+error.message;
  el('status').classList.add('error');
  document.querySelector('.controls').hidden=true;
  el('help').hidden=true;
}

run().catch(showError);
