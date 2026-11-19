// Exercise the actual Factory viewer with saved Box data. This is visual UI
// evidence, not a substitute for the orthographic float-depth comparison.
// node scripts/capture_factory_rgb_perspective.mjs ASSET_DIR OUT
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {execFileSync} from 'node:child_process';
import {access,mkdir,readFile,writeFile} from 'node:fs/promises';
import {resolve,join} from 'node:path';
import {chromium} from 'playwright';
import {chromiumOptions,presentedForeground} from '../worker/chromium.mjs';
import {startServer} from '../worker/server.mjs';
import {RGB_PARALLAX_PRESET,HYBRID_NORMAL_CONFIDENCE} from '../worker/parallax_material.mjs';

const [assetArg,outArg]=process.argv.slice(2);
assert(assetArg&&outArg,'Expected ASSET_DIR OUT');
const assetDir=resolve(assetArg),out=resolve(outArg),size=768;
try{await access(join(out,'factory-perspective.json'));throw Error('Preserve the existing capture; choose another output directory');}
catch(error){if(error.code!=='ENOENT')throw error;}
await mkdir(out,{recursive:true});
const hash=b=>createHash('sha256').update(b).digest('hex');
const manifestBytes=await readFile(join(assetDir,'manifest.json'));
const manifest=JSON.parse(manifestBytes),sourceFilesSHA256={};
for(const path of ['worker/runtime.mjs','worker/parallax_material.mjs','web/viewer.html','web/viewer.mjs','web/asset_io.mjs','scripts/capture_factory_rgb_perspective.mjs']) {
  sourceFilesSHA256[path]=hash(await readFile(new URL('../'+path,import.meta.url)));
}
const report={schemaVersion:1,method:'factory-viewer-perspective-ui-smoke-v1',status:'running',
  codeRevision:execFileSync('git',['rev-parse','HEAD'],{cwd:new URL('../',import.meta.url),encoding:'utf8'}).trim(),
  assetDir,assetId:manifest.asset_id,manifestSHA256:hash(manifestBytes),sourceFilesSHA256,
  preset:RGB_PARALLAX_PRESET,normalConfidence:HYBRID_NORMAL_CONFIDENCE,
  projection:'perspective',canvasSize:[size,size],deviceScaleFactor:1,
  cameraEvidence:'Factory viewer camera is private. Projection and initial +Z view are selected through its UI; orbit positions are recorded as pointer gestures, not asserted as exact camera matrices.',
  qualitativeOnly:true,originalInspectorModified:false,frames:[],errors:[]};
const save=()=>writeFile(join(out,'factory-perspective.json'),JSON.stringify(report,null,2)+'\n');
await save();
const server=await startServer({assetDir});let browser;
try {
  browser=await chromium.launch({...chromiumOptions('webgl2'),chromiumSandbox:true,
    args:[...chromiumOptions('webgl2').args,'--disable-dev-shm-usage']});
  report.browserVersion=browser.version();
  const page=await browser.newPage({viewport:{width:820,height:1040},deviceScaleFactor:1});
  page.on('pageerror',e=>report.errors.push(e.message));
  page.on('console',m=>{if(m.type()==='error')report.errors.push(m.text());});
  await page.goto(server.origin+'/web/viewer.html?manifest=/asset/manifest.json&backend=webgl2&shading=rgb');
  await page.waitForFunction(()=>/webgl2|表示エラー/.test(document.querySelector('#status').textContent),null,{timeout:120000});
  assert.equal(await page.locator('#status.error').count(),0,await page.locator('#status').textContent());
  await page.selectOption('#mode','box');
  await page.selectOption('#projection','perspective');
  await page.selectOption('#view','+Z');
  await page.evaluate(size=>{
    const stage=document.querySelector('#stage');
    Object.assign(stage.style,{width:size+'px',height:size+'px',minHeight:'0',flex:'0 0 auto',border:'0',background:'transparent'});
    stage.classList.remove('checker');
    document.querySelector('main').style.flex='0 0 auto';
    document.documentElement.style.background='transparent';document.body.style.background='transparent';
  },size);
  await page.locator('#fit').click();
  await page.waitForFunction(size=>{const c=document.querySelector('#canvas');return c.width===size&&c.height===size;},size,{timeout:30000});
  await page.waitForFunction(presentedForeground,1,{timeout:30000});
  const settle=()=>page.evaluate(()=>new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r))));
  const canvas=page.locator('#canvas');
  async function shot(name,gesture){
    await settle();
    const bytes=await canvas.screenshot({type:'png',omitBackground:true}),path=join(out,name+'.png');
    await writeFile(path,bytes);
    const ui=await page.evaluate(()=>Object.fromEntries(['shading','projection','view','sampling','mode'].map(id=>[id,document.getElementById(id).value])));
    report.frames.push({name,path,sha256:hash(bytes),ui,gesture});await save();
  }
  await shot('front-rgb',{operation:'Select +Z and fit through Factory controls'});
  await page.selectOption('#shading','lit');await shot('front-lit',{operation:'Change shading only; same camera'});
  await page.selectOption('#shading','rgb');
  const bounds=await canvas.boundingBox();assert(bounds&&bounds.width===size&&bounds.height===size);
  async function orbitOffset(degrees){
    await page.selectOption('#view','+Z');await settle();
    const x=bounds.x+bounds.width*.5,y=bounds.y+bounds.height*.5,dx=-degrees/360*bounds.height;
    await page.mouse.move(x,y);await page.mouse.down();await page.mouse.move(x+dx,y,{steps:12});await page.mouse.up();
    return {operation:'Horizontal OrbitControls drag from reset +Z',deltaPixels:[dx,0],canvasCssHeight:bounds.height,
      nominalYawDegrees:degrees,actualCameraMatrixAvailable:false};
  }
  await shot('orbit-right-rgb',await orbitOffset(30));
  await shot('orbit-left-rgb',await orbitOffset(-30));
  await page.selectOption('#view','+Z');await settle();
  // Record real presented canvas frames during pointer-driven orbit. No
  // interpolation, edited viewer, image enhancement or regenerated RGB.
  await page.evaluate(()=>{
    const canvas=document.querySelector('#canvas'),stream=canvas.captureStream(20),chunks=[];
    const mimeType=MediaRecorder.isTypeSupported('video/webm;codecs=vp9')?'video/webm;codecs=vp9':'video/webm';
    const recorder=new MediaRecorder(stream,{mimeType,videoBitsPerSecond:3000000});
    recorder.ondataavailable=e=>{if(e.data.size)chunks.push(e.data);};
    globalThis.__BI_PERSPECTIVE_CAPTURE__={recorder,stream,chunks,mimeType};recorder.start();
  });
  const sequence=[0,6,12,18,24,30,24,18,12,6,0,-6,-12,-18,-24,-30,-24,-18,-12,-6,0];
  const cx=bounds.x+bounds.width*.5,cy=bounds.y+bounds.height*.5;
  await page.mouse.move(cx,cy);await page.mouse.down();
  for(const nominalYaw of sequence){await page.mouse.move(cx-nominalYaw/360*bounds.height,cy,{steps:2});await page.waitForTimeout(100);}
  await page.mouse.up();await page.waitForTimeout(300);
  const clip=await page.evaluate(async()=>{
    const state=globalThis.__BI_PERSPECTIVE_CAPTURE__;
    await new Promise(resolve=>{state.recorder.onstop=resolve;state.recorder.stop();});
    for(const track of state.stream.getTracks())track.stop();
    const bytes=new Uint8Array(await new Blob(state.chunks,{type:state.mimeType}).arrayBuffer());
    let text='';for(let i=0;i<bytes.length;i+=16384)text+=String.fromCharCode(...bytes.subarray(i,i+16384));
    delete globalThis.__BI_PERSPECTIVE_CAPTURE__;return{base64:btoa(text),mimeType:state.mimeType};
  });
  const video=Buffer.from(clip.base64,'base64');assert(video.length>1000,'Empty perspective video');
  const videoPath=join(out,'factory-rgb-perspective-orbit.webm');await writeFile(videoPath,video);
  report.video={path:videoPath,sha256:hash(video),bytes:video.length,mimeType:clip.mimeType,captureFramesPerSecond:20,
    gestureSequenceNominalDegrees:sequence,holdMillisecondsPerStep:100};
  for(const [path,digest] of Object.entries(sourceFilesSHA256))assert.equal(hash(await readFile(new URL('../'+path,import.meta.url))),digest,'Source changed: '+path);
  for(const descriptor of Object.values(manifest.textures))assert.equal(hash(await readFile(join(assetDir,descriptor.path))),descriptor.sha256,'Source map changed');
  assert.equal(report.errors.length,0,report.errors.join('\n'));
  report.sourceMapsUnchanged=true;report.status='complete';await save();
  console.log(JSON.stringify({report:join(out,'factory-perspective.json'),frames:report.frames.length,video:report.video,status:report.status}));
} catch(error) {report.status='failed';report.error=String(error.stack??error);await save();throw error;}
finally {await browser?.close();await server.close();}
