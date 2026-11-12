// Compare a static CC0 source and its single-view Pixal3D reconstruction in one fixed world frame.
// node tests/compare_pixal_novel_views.mjs ORIGINAL.glb UNPACKED_PIXAL_ASSET OUTPUT_DIR [cameraX,cameraY,cameraZ]
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {mkdir, readFile, writeFile} from 'node:fs/promises';
import {join, resolve} from 'node:path';
import {chromium} from 'playwright';
import {chromiumOptions} from '../worker/chromium.mjs';
import {startServer} from '../worker/server.mjs';

const [original, asset, output, direction = '1,.66,1'] = process.argv.slice(2);
assert(original && asset && output, 'Expected static source GLB, unpacked Pixal3D Box, output directory');
const referenceDirection = direction.split(',').map(Number);
assert(referenceDirection.length === 3 && referenceDirection.every(Number.isFinite) &&
  Math.hypot(referenceDirection[0],referenceDirection[2]) > 0, 'Expected a nonvertical x,y,z reference camera direction');
const sourceHash = createHash('sha256').update(await readFile(original)).digest('hex');
const server = await startServer({input:resolve(original),assetDir:resolve(asset)});
let browser;
try {
  browser = await chromium.launch(chromiumOptions('webgl2'));
  const page = await browser.newPage();
  await page.goto(server.origin + '/bridge');
  const {sheet, ...report} = await page.evaluate(async referenceDirection => {
    const T = await import('three/webgpu');
    const {loadProxy,disposeProxy,restoreBox} = await import('/worker/runtime.mjs');
    const {makeRenderer,fetchChecked,readAsset,assetURL} = await import('/web/asset_io.mjs');
    const manifestURL = new URL('/asset/manifest.json',location.href);
    const {manifest, files} = await readAsset(manifestURL);
    if (!manifest.reference?.camera || manifest.provenance?.meshGeneration?.backend !== 'native_mesh')
      throw new Error('Expected a generated Pixal3D Box with conditioning camera.');
    const load = async url => loadProxy(await (await fetchChecked(url)).arrayBuffer());
    const [source, generated] = await Promise.all([load('/job/input.glb'),load(assetURL(manifestURL,manifest.proxy))]);

    // Source GLB canonicalization used by the viewer: center its real vertices; longest axis = one unit.
    const bounds = object => new T.Box3().setFromObject(object,true);
    const originalBounds = bounds(source);
    const scale = 1 / Math.max(...originalBounds.getSize(new T.Vector3()).toArray());
    const originalCenter = originalBounds.getCenter(new T.Vector3());
    source.applyMatrix4(new T.Matrix4().makeScale(scale,scale,scale)
      .multiply(new T.Matrix4().makeTranslation(...originalCenter.negate().toArray())));
    const sourceBounds = bounds(source);
    // The original reference was captured in the standard square 36-degree viewer camera,
    // with a 2% bake margin on all sides and the viewer's fit-to-box distance.
    const radius = sourceBounds.getSize(new T.Vector3()).addScalar(.04).length() / 2;
    const distance = radius * 1.1 / Math.sin(T.MathUtils.degToRad(18));
    const front = new T.Vector3(...referenceDirection).normalize();
    const basis = new T.Matrix4().lookAt(front,new T.Vector3(),new T.Vector3(0,1,0));
    const eyeRight = new T.Vector3().setFromMatrixColumn(basis,0);
    const eyeUp = new T.Vector3().setFromMatrixColumn(basis,1);
    const camera = new T.PerspectiveCamera(36,1,.01*radius,100*radius);
    function pose(direction) {
      camera.position.copy(direction).normalize().multiplyScalar(distance);
      camera.up.set(0,1,0);
      if (Math.abs(direction.y) === 1 && direction.x === 0 && direction.z === 0)
        camera.up.set(0,0,direction.y>0?-1:1);
      camera.lookAt(0,0,0);
      camera.updateMatrixWorld();
    }
    pose(front);

    // Pixal3D +Z faces the input camera. The one-view reconstruction has no measured
    // depth origin. Place its axial depth midpoint at that of the source.
    generated.applyMatrix4(new T.Matrix4().fromArray(manifest.inverseTransform));
    const generatedBounds=bounds(generated), depthOrigin=generatedBounds.getCenter(new T.Vector3()).z;
    const sourceInCameraFrame = new T.Group();
    sourceInCameraFrame.add(source);
    sourceInCameraFrame.quaternion.setFromRotationMatrix(basis).invert();
    const sourceDepthCenter=bounds(sourceInCameraFrame).getCenter(new T.Vector3()).z;
    sourceInCameraFrame.remove(source);
    source.updateMatrixWorld(true);
    const model = new T.Group();
    model.quaternion.setFromRotationMatrix(basis);
    model.add(generated);
    generated.position.z -= depthOrigin;
    model.position.copy(front).multiplyScalar(sourceDepthCenter);
    const point=new T.Vector3();
    function projectedBox(root) {
      root.updateMatrixWorld(true);
      const box = new T.Box2();
      root.traverse(mesh => {
        if (!mesh.isMesh) return;
        const positions=mesh.geometry.attributes.position;
        for (let i=0;i<positions.count;i++) {
          point.fromBufferAttribute(positions,i).applyMatrix4(mesh.matrixWorld).project(camera);
          box.expandByPoint(point);
        }
      });
      return box;
    }
    const referenceBounds = projectedBox(source);
    const targetHeight = referenceBounds.getSize(new T.Vector2()).y;
    const targetCenter = referenceBounds.getCenter(new T.Vector2());
    const screenToWorld = distance * Math.tan(T.MathUtils.degToRad(18));
    // Use only the reference-view silhouette height and center after orienting the mesh.
    // The same rotation, translation and *uniform* scale are then used for every camera.
    for (let i=0;i<16;i++) {
      const b=projectedBox(model), center=b.getCenter(new T.Vector2());
      model.scale.multiplyScalar(targetHeight / b.getSize(new T.Vector2()).y);
      model.position.addScaledVector(eyeRight,(targetCenter.x-center.x)*screenToWorld)
        .addScaledVector(eyeUp,(targetCenter.y-center.y)*screenToWorld);
    }
    model.updateMatrixWorld(true);
    const size=320;
    const renderer = await makeRenderer('webgl2');
    renderer.setSize(size,size,false);
    renderer.setClearColor(0,0);
    const restored = await restoreBox(manifest,files,renderer);
    restored.mesh.applyMatrix4(model.matrixWorld.clone().multiply(generated.matrix));
    const target=new T.RenderTarget(size,size);
    target.texture.colorSpace=T.SRGBColorSpace;
    target.depthTexture=new T.DepthTexture(size,size,T.UnsignedIntType);
    const {texture,screenUV,min,floor,clamp,mod,vec4}=await import('three/tsl');
    const depthTarget=new T.RenderTarget(size,size,{depthBuffer:false});
    const depthMaterial=new T.MeshBasicNodeMaterial();
    depthMaterial.toneMapped=false;
    const encoded=min(floor(clamp(texture(target.depthTexture,screenUV).r,0,1).mul(16777215).add(.5)),16777215);
    depthMaterial.fragmentNode=vec4(floor(encoded.div(65536)),mod(floor(encoded.div(256)),256),mod(encoded,256),255).div(255);
    const quad=new T.QuadMesh(depthMaterial);
    const scene=new T.Scene();
    scene.add(new T.HemisphereLight(0xf3f5ff,0xc2bdb5,1.7));
    const light=new T.DirectionalLight(0xffffff,2);
    light.position.set(2,4,3);
    scene.add(light);
    async function render(mesh) {
      scene.add(mesh);
      renderer.setRenderTarget(target);
      renderer.render(scene,camera);
      const raw=await renderer.readRenderTargetPixelsAsync(target,0,0,size,size);
      renderer.setRenderTarget(depthTarget);
      quad.render(renderer);
      const z=await renderer.readRenderTargetPixelsAsync(depthTarget,0,0,size,size);
      scene.remove(mesh);
      const topDown=new Uint8ClampedArray(raw.length);
      const depth=new Float32Array(size*size);
      for (let y=0;y<size;y++) {
        topDown.set(raw.subarray((size-1-y)*size*4,(size-y)*size*4),y*size*4);
        for (let x=0;x<size;x++) {
          const p=4*((size-1-y)*size+x);
          const normalized=(65536*z[p]+256*z[p+1]+z[p+2])/16777215;
          depth[y*size+x]=camera.near*camera.far/(camera.far-normalized*(camera.far-camera.near));
        }
      }
      return {rgba:topDown,depth};
    }
    const poses=[['Reference',front],['Right (+X)',new T.Vector3(1,0,0)],['Left (-X)',new T.Vector3(-1,0,0)],
      ['Top (+Y)',new T.Vector3(0,1,0)],['Bottom (-Y)',new T.Vector3(0,-1,0)],
      ['Front (+Z)',new T.Vector3(0,0,1)],['Back (-Z)',new T.Vector3(0,0,-1)],['Opposite',front.clone().negate()]];
    const labels=['CC0 mesh','Pixal3D mesh','Box','Box vs CC0'];
    const tile=140, left=115, header=34;
    const sheet=document.createElement('canvas');
    sheet.width=left+tile*poses.length;sheet.height=header+labels.length*tile+71;
    const ctx=sheet.getContext('2d');
    ctx.fillStyle='#232d37';ctx.fillRect(0,0,sheet.width,sheet.height);
    ctx.fillStyle='white';ctx.font='15px sans-serif';
    labels.forEach((l,i)=>ctx.fillText(l,8,header+(i+.5)*tile));
    const footer = header+labels.length*tile;
    ctx.fillText('Mesh IoU',8,footer+17);ctx.fillText('Box IoU',8,footer+36);
    const buffer=document.createElement('canvas');buffer.width=buffer.height=size;
    const off=buffer.getContext('2d'), values=[];
    function compare(a,b,mask=null) {
      const depthError=[];
      let both=0,missing=0,extra=0,absoluteRGB=0;
      for (let p=0;p<a.rgba.length;p+=4) {
        const seenA=a.rgba[p+3]>=128,seenB=b.rgba[p+3]>=128;
        if(seenA&&seenB) {
          both++;
          depthError.push(Math.abs(a.depth[p/4]-b.depth[p/4]));
          for(let c=0;c<3;c++) absoluteRGB+=Math.abs(a.rgba[p+c]-b.rgba[p+c]);
        }
        else if(seenA) missing++; else if(seenB) extra++;
        if(mask) {
          const color=seenA&&seenB?[224,229,235]:seenA?[241,150,50]:seenB?[55,191,218]:[67,73,79];
          mask.set([...color,255],p);
        }
      }
      if (!both) throw new Error('No shared foreground');
      depthError.sort((a,b)=>a-b);
      return {iou:both/(both+missing+extra),overlapPixels:both,sourceOnlyPixels:missing,generatedOnlyPixels:extra,
        meanAbsoluteRGB8OnOverlap:absoluteRGB/(3*both),surfaceDepthMAE:depthError.reduce((sum,x)=>sum+x,0)/both,
        surfaceDepthP95:depthError[Math.ceil(.95*both)-1]};
    }
    try {
      for (const [i,[label,direction]] of poses.entries()) {
        pose(direction);
        const a=await render(source), b=await render(model), c=await render(restored.mesh);
        const mask=new Uint8ClampedArray(a.rgba.length);
        const mesh=compare(a,b), boxVsSource=compare(a,c,mask);
        values.push({label,direction:direction.toArray(),...mesh,boxVsSource,boxVsProxy:compare(b,c)});
        const x=left+i*tile;
        ctx.textAlign='center';ctx.fillStyle='white';ctx.fillText(label,x+tile/2,22);
        for (const [row,data] of [a.rgba,b.rgba,c.rgba,mask].entries()) {
          const y=header+row*tile;
          off.putImageData(new ImageData(data,size,size),0,0);
          ctx.fillStyle='#697079';ctx.fillRect(x,y,tile-2,tile-2);
          ctx.drawImage(buffer,x,y,tile-2,tile-2);
        }
        ctx.fillStyle='white';
        ctx.fillText(mesh.iou.toFixed(3),x+tile/2,footer+17);
        ctx.fillText(boxVsSource.iou.toFixed(3),x+tile/2,footer+36);
      }
      ctx.textAlign='left';ctx.font='13px sans-serif';
      ctx.fillStyle='#e0e5eb';ctx.fillText('Both',left,footer+58);
      ctx.fillStyle='#f19632';ctx.fillText('CC0 only',left+58,footer+58);
      ctx.fillStyle='#37bfda';ctx.fillText('Box only',left+151,footer+58);
      return {camera:{fovDegrees:36,sourceFitDistance:distance,size,projection:'perspective',referenceDirection:front.toArray()},
        registration:{referenceFrame:'CC0 canonical world: longest dimension = 1',method:'Align Pixal3D +Z with the known reference viewing axis and +Y with camera up. Match silhouette height and image bounding-box center in that camera with one uniform scale and two image-plane translations. Set 3D depth midpoint equal to the source midpoint along the viewing axis.',
          assumption:'The relative depth midpoint is unknown from the reference. No individual view is registered independently.',
          matrixColumnMajor:model.matrixWorld.clone().multiply(generated.matrix).toArray(),inputGeometry:'stored canonical generated glTF',
          uniformScaleAfterUndoingCanonicalization:model.scale.x,sourceDepthMidpoint:sourceDepthCenter},
        generatedCanonicalGlbSha256:manifest.proxySha256,views:values,
        depthUnits:'Absolute view-space depth error on shared foreground; one unit equals the source mesh longest axis.',
        meanNovelBoxIoU:values.slice(1).reduce((a,v)=>a+v.boxVsSource.iou,0)/7,
        meanNovelIoU:values.slice(1).reduce((a,v)=>a+v.iou,0)/7,
        sheet:sheet.toDataURL('image/webp',.91).split(',')[1]};
    } finally {
      renderer.setRenderTarget(null);target.dispose();depthTarget.dispose();depthMaterial.dispose();renderer.dispose();
      restored.dispose();disposeProxy(source);disposeProxy(generated);
    }
  }, referenceDirection);
  report.sourceGlbSha256=sourceHash;
  report.renderer={chromiumVersion:browser.version(),backend:'webgl2',softwareGpuRequested:process.env.BI_SOFTWARE_GPU==='1'};
  await mkdir(output,{recursive:true});
  await writeFile(join(output,'pixal3d-novel-views.webp'),Buffer.from(sheet,'base64'));
  await writeFile(join(output,'pixal3d-novel-views.json'),JSON.stringify(report,null,2)+'\n');
  console.log(JSON.stringify(report.views.map(v=>[v.label,v.iou]),null,2));
} finally { await browser?.close(); await server.close(); }
