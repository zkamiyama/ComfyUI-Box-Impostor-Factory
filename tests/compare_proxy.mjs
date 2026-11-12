// Measure the saved Box against the canonical GLB under matching cameras, material lighting and depth.
// Usage: node tests/compare_proxy.mjs ASSET_DIRECTORY [REPORT.json] [webgl2|webgpu] [--compare-interval] [--size 256]
import { mkdir, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { parseArgs } from 'node:util';
import { chromium } from 'playwright';
import { chromiumOptions } from '../worker/chromium.mjs';
import { startServer } from '../worker/server.mjs';
import { contactSheet } from '../worker/evidence.mjs';

const depthAt=(bytes,i)=>(65536*bytes[i]+256*bytes[i+1]+bytes[i+2])/16777215;
const round=x=> +x.toFixed(7);

// A hole is transparent space that cannot reach an image edge in four-connected pixels.
function backgroundRegions(bytes,width,height) {
  const regions=Uint8Array.from({length:width*height},(_,i)=>bytes[4*i+3]>=128 ? 0 : 1);
  const queue=[];
  function visit(i) {if(regions[i]===1){regions[i]=2;queue.push(i);}}
  for(let x=0;x<width;x++){visit(x);visit((height-1)*width+x);}
  for(let y=0;y<height;y++){visit(y*width);visit(y*width+width-1);}
  for(let head=0;head<queue.length;head++){
    const i=queue[head], x=i%width, y=Math.floor(i/width);
    if(x>0)visit(i-1);if(x+1<width)visit(i+1);if(y>0)visit(i-width);if(y+1<height)visit(i+width);
  }
  return regions;
}

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
  const referenceRegions=backgroundRegions(p,proxy.width,proxy.height);
  const candidateRegions=backgroundRegions(b,box.width,box.height);
  let holePixels=0, filledHolePixels=0, leakedHolePixels=0;
  referenceRegions.forEach((region,i)=>{
    if(region!==1)return;
    holePixels++;
    if(candidateRegions[i]===0)filledHolePixels++;
    if(candidateRegions[i]===2)leakedHolePixels++;
  });
  const union=pCount+bCount-both;
  if(!pCount) throw new Error('Empty proxy reference');
  dz.sort((a,b)=>a-b);
  return {id:proxy.id,pixels,proxyPixels:pCount,boxPixels:bCount,overlapPixels:both,holePixels,filledHolePixels,leakedHolePixels,silhouetteIoU:round(both/union),
    overlapColorMAE8:both?round(rgb/(3*both)):null,overlapDepthMAE:both?round(dz.reduce((a,b)=>a+b,0)/both):null,
    overlapDepthP95:both?round(dz[Math.ceil(both*.95)-1]):null};
}

export async function compareProxy(asset,reportPath,backend='webgl2',compareInterval=false,size=64){
  if (![64,128,256].includes(size)) throw new Error('Capture size must be 64, 128 or 256');
  const server=await startServer({assetDir:resolve(asset)});
  let browser;
  try{
    const software=process.env.BI_SOFTWARE_GPU==='1';
    browser=await chromium.launch(chromiumOptions(backend));
    const page=await browser.newPage();await page.goto(server.origin+'/bridge');
    const result=await page.evaluate(async ({backend,compareInterval,size})=>{
      const {snapshot}=await import('/worker/browser_bridge.mjs');
      const {makeRenderer,readAsset,backendName}=await import('/web/asset_io.mjs');
      const {loadProxy,restoreBox,disposeProxy}=await import('/worker/runtime.mjs');
      const renderer=await makeRenderer(backend);
      let mesh,restored,variantTexture,variantMaterial;
      try{
        const {manifest,files}=await readAsset('/asset/manifest.json',{proxy:true});
        mesh=await loadProxy(files.get(manifest.proxy));
        restored=await restoreBox(manifest,files,renderer);
        const reference=await snapshot(renderer,mesh,manifest.bakeBox,size);
        const actual=await snapshot(renderer,restored.mesh,manifest.bakeBox,size);
        let nearest;
        if(compareInterval){
          const {NearestFilter,RenderTarget,Mesh}=await import('three/webgpu');
          const {ImpostorNodeMaterial}=await import('three-impostor');
          const data=restored.mesh.material.impostor, original=data.intervalMap;
          let sampled;
          if(original.isRenderTargetTexture){
            // Preserve the render-target UV path and copy the same RGBA16F texels.
            const {width,height}=original.image;
            variantTexture=new RenderTarget(width,height,{type:original.type,depthBuffer:false,
              minFilter:NearestFilter,magFilter:NearestFilter,generateMipmaps:false});
            renderer.initRenderTarget(variantTexture);
            renderer.copyTextureToTexture(original,variantTexture.texture);
            sampled=variantTexture.texture;
          } else {
            variantTexture=original.clone();
            variantTexture.minFilter=variantTexture.magFilter=NearestFilter;
            variantTexture.needsUpdate=true;
            sampled=variantTexture;
          }
          variantMaterial=new ImpostorNodeMaterial({...data,intervalMap:sampled});
          const variantMesh=new Mesh(restored.mesh.geometry,variantMaterial);
          nearest=await snapshot(renderer,variantMesh,manifest.bakeBox,size);
        }
        return {backend:backendName(renderer),resolution:manifest.resolution,proxySha256:manifest.proxySha256,
          textures:Object.fromEntries(Object.entries(manifest.textures).map(([k,v])=>[k,v.sha256])),reference,actual,nearest};
      } finally {variantMaterial?.dispose();variantTexture?.dispose();if(mesh)disposeProxy(mesh);restored?.dispose();await renderer.dispose();}
    },{backend,compareInterval,size});
    const {reference,actual,nearest,...common}=result;
    const views=reference.map((p,i)=>fidelity(p,actual[i]));
    const stats=vs=>({minimumIoU:Math.min(...vs.map(v=>v.silhouetteIoU)),meanIoU:round(vs.reduce((a,v)=>a+v.silhouetteIoU,0)/vs.length)});
    const report={...common,chromiumVersion:browser.version(),softwareGpuRequested:software,source:'canonical proxy GLB',candidate:'restored Box',
      renderSize:[reference[0].width,reference[0].height],projection:'orthographic',steps:128,
      depthUnits:'normalized orthographic camera depth, overlapping foreground pixels',colorUnits:'absolute RGB channel difference in RGBA8 readback, overlapping foreground pixels',
      axes:stats(views.slice(0,6)),obliques:stats(views.slice(6)),views};
    const summarize=values=>({
      meanViewColorMAE8:values.every(v=>v.overlapPixels>0) ? round(values.reduce((s,v)=>s+v.overlapColorMAE8,0)/values.length) : null,
      maximumViewDepthP95:values.every(v=>v.overlapPixels>0) ? Math.max(...values.map(v=>v.overlapDepthP95)) : null,
      minimumViewIoU:Math.min(...values.map(v=>v.silhouetteIoU)),
      referenceHolePixels:values.reduce((s,v)=>s+v.holePixels,0),
      filledHolePixels:values.reduce((s,v)=>s+v.filledHolePixels,0),
      leakedHolePixels:values.reduce((s,v)=>s+v.leakedHolePixels,0),
    });
    report.summary=summarize(views);
    if(compareInterval){
      const nearestViews=reference.map((p,i)=>fidelity(p,nearest[i]));
      report.experiment={change:'In-memory nearest min/mag filter on intervalMap only; all stored bytes and upstream shader identical.',
        holeDefinition:'Proxy background enclosed using four-connected pixels; filled is Box foreground, leaked is Box background connected to image border.',
        native:report.summary,intervalNearest:summarize(nearestViews),
        nearestViews:nearestViews.map(({id,silhouetteIoU,overlapColorMAE8,overlapDepthP95,filledHolePixels,leakedHolePixels})=>
          ({id,silhouetteIoU,overlapColorMAE8,overlapDepthP95,filledHolePixels,leakedHolePixels}))};
    }
    if(reportPath){
      const folder=dirname(reportPath);await mkdir(folder,{recursive:true});
      await writeFile(reportPath,JSON.stringify(report,null,2)+'\n');
      const stem=reportPath.replace(/\.json$/i,'');
      if(compareInterval) await writeFile(stem+'-native-nearest.png',contactSheet(actual,nearest,result.backend));
      else await writeFile(stem+'-views.png',contactSheet(reference,actual,result.backend));
    }
    return report;
  } finally {if(browser)await browser.close();await server.close();}
}
if(process.argv[1] && import.meta.url===pathToFileURL(resolve(process.argv[1])).href){
  const {positionals,values}=parseArgs({allowPositionals:true,options:{
    'compare-interval':{type:'boolean',default:false},size:{type:'string',default:'64'}
  }});
  if(!positionals.length || positionals.length>3)throw new Error('Expected an asset directory, optional report path and backend');
  const [asset,path,backend]=positionals;
  const report=await compareProxy(asset,path,backend,values['compare-interval'],Number(values.size));
  console.log(JSON.stringify(report,null,2));
}
