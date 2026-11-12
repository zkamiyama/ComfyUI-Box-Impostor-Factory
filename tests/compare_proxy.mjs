// Measure the saved Box against the canonical GLB under matching cameras, material lighting and depth.
// Usage: node tests/compare_proxy.mjs ASSET_DIRECTORY [REPORT.json] [webgl2|webgpu]
import { mkdir, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { chromium } from 'playwright';
import { startServer } from '../worker/server.mjs';
import { contactSheet } from '../worker/evidence.mjs';

const depthAt=(bytes,i)=>(65536*bytes[i]+256*bytes[i+1]+bytes[i+2])/16777215;
const round=x=> +x.toFixed(7);

export function fidelity(proxy,box){
  if(proxy.id!==box.id || proxy.width!==box.width || proxy.height!==box.height) throw new Error('Unmatched views');
  const pixels=proxy.width*proxy.height;
  const [p,b,pd,bd]=[proxy.color,box.color,proxy.depth,box.depth].map(s=>Buffer.from(s,'base64'));
  if([p,b,pd,bd].some(a=>a.length!==4*pixels)) throw new Error('Invalid capture');
  let pCount=0,bCount=0,both=0,rgb=0;
  const dz=[];
  for(let i=0;i<4*pixels;i+=4){
    const seenP=p[i+3]>=128, seenB=b[i+3]>=128;
    if(seenP)pCount++;
    if(seenB)bCount++;
    if(!seenP||!seenB)continue;
    both++;
    for(let j=0;j<3;j++)rgb+=Math.abs(p[i+j]-b[i+j]);
    dz.push(Math.abs(depthAt(pd,i)-depthAt(bd,i)));
  }
  const union=pCount+bCount-both;
  if(!pCount) throw new Error('Empty proxy reference');
  dz.sort((a,b)=>a-b);
  return {id:proxy.id,pixels,proxyPixels:pCount,boxPixels:bCount,overlapPixels:both,silhouetteIoU:round(both/union),
    overlapColorMAE8:both?round(rgb/(3*both)):null,overlapDepthMAE:both?round(dz.reduce((a,b)=>a+b,0)/both):null,
    overlapDepthP95:both?round(dz[Math.ceil(both*.95)-1]):null};
}

export async function compareProxy(asset,reportPath,backend='webgl2'){
  const server=await startServer({assetDir:resolve(asset)});
  let browser;
  try{
    const software=process.env.BI_SOFTWARE_GPU==='1';
    const launch={headless:true,args:software?['--use-angle=swiftshader','--enable-unsafe-swiftshader','--enable-unsafe-webgpu']:[]};
    if(process.env.BI_CHROMIUM) launch.executablePath=process.env.BI_CHROMIUM;
    browser=await chromium.launch(launch);
    const page=await browser.newPage();await page.goto(server.origin+'/bridge');
    const result=await page.evaluate(async backend=>{
      const {snapshot}=await import('/worker/browser_bridge.mjs');
      const {makeRenderer,readAsset,backendName}=await import('/web/asset_io.mjs');
      const {loadProxy,restoreBox,disposeProxy}=await import('/worker/runtime.mjs');
      const renderer=await makeRenderer(backend);
      let mesh,restored;
      try{
        const {manifest,files}=await readAsset('/asset/manifest.json',{proxy:true});
        mesh=await loadProxy(files.get(manifest.proxy));
        restored=await restoreBox(manifest,files,renderer);
        return {backend:backendName(renderer),resolution:manifest.resolution,proxySha256:manifest.proxySha256,
          textures:Object.fromEntries(Object.entries(manifest.textures).map(([k,v])=>[k,v.sha256])),
          reference:await snapshot(renderer,mesh,manifest.bakeBox),actual:await snapshot(renderer,restored.mesh,manifest.bakeBox)};
      } finally{if(mesh)disposeProxy(mesh);restored?.dispose();await renderer.dispose();}
    },backend);
    const {reference,actual,...common}=result;
    const views=reference.map((p,i)=>fidelity(p,actual[i]));
    const stats=vs=>({minimumIoU:Math.min(...vs.map(v=>v.silhouetteIoU)),meanIoU:round(vs.reduce((a,v)=>a+v.silhouetteIoU,0)/vs.length)});
    const report={...common,chromiumVersion:browser.version(),softwareGpuRequested:software,source:'canonical proxy GLB',candidate:'restored Box',
      renderSize:[reference[0].width,reference[0].height],projection:'orthographic',steps:128,
      depthUnits:'normalized orthographic camera depth, overlapping foreground pixels',colorUnits:'absolute RGB channel difference in RGBA8 readback, overlapping foreground pixels',
      axes:stats(views.slice(0,6)),obliques:stats(views.slice(6)),views};
    if(reportPath){
      const folder=dirname(reportPath);await mkdir(folder,{recursive:true});
      await writeFile(reportPath,JSON.stringify(report,null,2)+'\n');
      await writeFile(reportPath.replace(/\.json$/i,'')+'-views.png',contactSheet(reference,actual,result.backend));
    }
    return report;
  } finally {if(browser)await browser.close();await server.close();}
}
if(process.argv[1] && import.meta.url===pathToFileURL(resolve(process.argv[1])).href){
  if(!process.argv[2])throw new Error('An asset directory is required');
  const report=await compareProxy(process.argv[2],process.argv[3],process.argv[4]);
  console.log(JSON.stringify(report,null,2));
}
