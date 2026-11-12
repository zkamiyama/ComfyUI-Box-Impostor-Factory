#!/usr/bin/env node
import { createHash, randomUUID } from 'node:crypto';
import { copyFile, mkdir, mkdtemp, readFile, rename, rm, stat, writeFile } from 'node:fs/promises';
import { dirname, resolve, join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { parseArgs } from 'node:util';
import { startServer, relativeFile } from './server.mjs';
import { writeEvidence } from './evidence.mjs';
import { validateDescriptor } from './codec.mjs';

import { chromiumOptions, presentedForeground } from './chromium.mjs';

const usage = `node worker/cli.mjs bake --input model.glb --output asset-dir [--resolution 256] [--backend auto|webgpu|webgl2] [--maps roughness,metalness,ao,emissive] [--verify]
node worker/cli.mjs canonicalize --input model.glb --output new-dir
node worker/cli.mjs preview --asset asset-dir [--port 8080]
node worker/cli.mjs thumbnail --asset asset-dir --output hero.png
node worker/cli.mjs inspect --asset asset-dir`;
const allowedMaps = ['roughness','metalness','ao','emissive'];
export async function inspectAsset(dir) {
  const manifest = JSON.parse(await readFile(join(dir,'manifest.json'),'utf8'));
  if (manifest.schemaVersion !== 1 || manifest.layout !== 'box' ||
      !Number.isInteger(manifest.resolution) || manifest.resolution <= 0) throw new Error('Unsupported Box manifest');
  for (const name of ['colorMap','normalMap','intervalMap','rangeMap']) {
    if (!manifest.textures?.[name]) throw new Error('Missing ' + name);
  }
  for (const [name,desc] of [...Object.entries(manifest.textures), ...(manifest.baseColor ? [['colorMap',manifest.baseColor]] : [])]) {
    validateDescriptor(desc,name,manifest.resolution);
    const bytes = await readFile(join(dir,relativeFile(desc.path)));
    if (bytes.length !== desc.byteLength || hash(bytes) !== desc.sha256) throw new Error('Invalid texture: '+desc.path);
  }
  const proxy=await readFile(join(dir,relativeFile(manifest.proxy)));
  if(hash(proxy)!==manifest.proxySha256) throw new Error('Invalid proxy checksum');
  return manifest;
}
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
async function exists(path) {
  try {await stat(path);return true;} catch(e){if(e.code==='ENOENT')return false;throw e;}
}
async function browserCall(server, method, options) {
  const { chromium } = await import('playwright');
  const browser = await chromium.launch(chromiumOptions(options.backend));
  const diagnostics = [];
  try {
    const page = await browser.newPage();
    page.on('pageerror',e=>diagnostics.push(String(e)));
    page.on('console',m=>{if(m.type()==='error')diagnostics.push(m.text().slice(0,1600));});
    if(method==='thumbnail') {
      await page.setViewportSize({width:680,height:580});
      await page.goto(server.origin+'/web/viewer.html?manifest=/asset/manifest.json&backend='+options.backend);
      await page.waitForFunction(()=>/webgpu|webgl2|表示エラー/.test(document.querySelector('#status').textContent));
      if(await page.locator('#status.error').count()) throw new Error(await page.locator('#status').innerText());
      await page.waitForFunction(presentedForeground,1,{timeout:20000});
      return await page.locator('#canvas').screenshot({type:'png'});
    }
    await page.goto(server.origin+'/bridge');
    const result = await page.evaluate(async ({method,options})=>{
      const bridge = await import('/worker/browser_bridge.mjs');
      return bridge[method](options);
    },{method,options});
    if (diagnostics.some(m=>/shader error|invalid shader|validation error|framebuffer is incomplete/i.test(m))) {
      throw new Error(diagnostics.slice(-4).join('\n'));
    }
    return result;
  } catch(e) { throw new Error([method + ': ' + e.message, ...diagnostics.slice(-4)].join('\n')); }
  finally {await browser.close();}
}

async function publish(command, opt) {
  if (!opt.input || !opt.output) throw new Error('--input and --output are required');
  const target = resolve(opt.output), input = resolve(opt.input);
  const inputBytes = await readFile(input);
  if (inputBytes.length<20 || inputBytes.toString('ascii',0,4)!=='glTF' || inputBytes.readUInt32LE(4)!==2 ||
      inputBytes.readUInt32LE(8)!==inputBytes.length) throw new Error('Expected a complete GLB 2.0 file');
  const resolution = Number(opt.resolution);
  if (!Number.isInteger(resolution) || resolution < 8 || resolution > 2048) throw new Error('resolution must be an integer from 8 to 2048');
  if (!['auto','webgpu','webgl2'].includes(opt.backend)) throw new Error('Unknown backend');
  const maps = opt.maps ? opt.maps.split(',') : [];
  if (maps.some(m=>!allowedMaps.includes(m)) || new Set(maps).size!==maps.length) throw new Error('Invalid maps');
  const canonicalMeta = opt['canonical-meta'] ? JSON.parse(await readFile(opt['canonical-meta'],'utf8')) : undefined;
  if(canonicalMeta?.glbSha256 && canonicalMeta.glbSha256!==hash(inputBytes)) throw new Error('Canonical metadata does not match GLB');
  const rotationDeg=['x','y','z'].map(axis=>Number(opt['rotate-'+axis] ?? 0));
  if(!rotationDeg.every(Number.isFinite)) throw new Error('Rotation must use finite degrees');
  if(command==='bake' && rotationDeg.some(x=>x!==0)) throw new Error('Set rotation on the canonicalize command');
  const implementation=createHash('sha256');
  for(const file of ['./runtime.mjs','./codec.mjs','./browser_bridge.mjs','./compare_frames.mjs','./capture_views.mjs','../web/asset_io.mjs','../package-lock.json']) {
    implementation.update(await readFile(new URL(file,import.meta.url)));
  }
  const config = {source:hash(inputBytes), resolution, backend:opt.backend, maps, canonicalMeta, rotationDeg,
    verify:opt.verify, implementation:implementation.digest('hex')};
  const jobKey = hash(JSON.stringify(config));
  const id = opt['asset-id'] ?? jobKey;
  if (!/^[a-zA-Z0-9_-]{1,100}$/.test(id)) throw new Error('Invalid asset id');
  async function reuse() {
    if (command==='bake') {
      const m=await inspectAsset(target);
      if(m.provenance?.jobKey===jobKey && m.asset_id===id) {
        if(opt.verify && opt.evidence) {
          await mkdir(opt.evidence,{recursive:true});
          for(const file of ['roundtrip.json','live-and-reloaded.png'])
            await copyFile(join(target,'qa',file),join(opt.evidence,file));
        }
        return {asset:target,cached:true};
      }
    } else {
      const m=JSON.parse(await readFile(join(target,'canonical.json'),'utf8'));
      if(m.jobKey===jobKey && m.sourceSha256===config.source && hash(await readFile(join(target,'canonical.glb')))===m.glbSha256) return {asset:target,cached:true};
    }
    throw new Error('Output exists for different inputs: '+target);
  }
  if (await exists(target)) return reuse();
  await mkdir(dirname(target),{recursive:true});
  const tmp = await mkdtemp(join(dirname(target),'.box-job-'));
  let server;
  try {
    server=await startServer({input,staging:tmp,assetDir:tmp});
    if (command==='canonicalize') {
      const meta=await browserCall(server,'canonicalize',{rotationDeg});
      Object.assign(meta,{rotationDeg,jobKey,sourceSha256:config.source,glbSha256:hash(await readFile(join(tmp,'canonical.glb')))});
      await writeFile(join(tmp,'canonical.json'),JSON.stringify(meta,null,2)+'\n');
    } else {
      const opts={resolution,maps,backend:opt.backend,asset_id:id,verify:opt.verify,canonicalMeta};
      const live=await browserCall(server,'bake',opts);
      const manifest=live.manifest;
      manifest.provenance={...manifest.provenance,jobKey,sourceSha256:config.source};
      await writeFile(join(tmp,'manifest.json'),JSON.stringify(manifest,null,2)+'\n');
      await inspectAsset(tmp);
      if (opt.verify) {
        const again=await browserCall(server,'reload',{backend:live.backend ?? opt.backend,verify:true});
        const {compareFrames}=await import('./compare_frames.mjs');
        const report=compareFrames(live.snapshot,again.snapshot);
        if(opt.evidence) await writeEvidence(resolve(opt.evidence),live.snapshot,again.snapshot,report,again.backend);
        await writeEvidence(join(tmp,'qa'),live.snapshot,again.snapshot,report,again.backend);
        if (!report.pass) throw new Error('Roundtrip mismatch: '+JSON.stringify(report.views.filter(v=>!v.pass)).slice(0,1800));
      }
    }
    try { await rename(tmp,target); } catch(e) {
      if(!['EEXIST','ENOTEMPTY'].includes(e.code))throw e;
      return await reuse();
    }
    return {asset:target,cached:false};
  } finally { if(server)await server.close();await rm(tmp,{recursive:true,force:true}); }
}
export async function main(argv) {
  const [command,...args]=argv;
  if(!command || ['--help','-h','help'].includes(command)){console.log(usage);return;}
  const opt=parseArgs({args,options:{
    input:{type:'string'},output:{type:'string'},asset:{type:'string'},evidence:{type:'string'},resolution:{type:'string',default:'256'},
    backend:{type:'string',default:'auto'},maps:{type:'string',default:''},verify:{type:'boolean',default:false},
    'rotate-x':{type:'string'},'rotate-y':{type:'string'},'rotate-z':{type:'string'},
    'asset-id':{type:'string'},'canonical-meta':{type:'string'},port:{type:'string',default:'8080'},
  }}).values;
  if(command==='thumbnail'){
    if(!opt.asset || !opt.output) throw new Error('thumbnail requires --asset and --output');
    const dir=resolve(opt.asset), dest=resolve(opt.output);
    await inspectAsset(dir);
    const server=await startServer({assetDir:dir});
    const temp=dest+'.'+randomUUID()+'.tmp';
    try {
      const png=await browserCall(server,'thumbnail',{backend:opt.backend});
      await mkdir(dirname(dest),{recursive:true}); await writeFile(temp,png); await rename(temp,dest);
      console.log(JSON.stringify({preview:dest}));
    } finally {await server.close(); await rm(temp,{force:true});}
    return;
  }
  if(command==='preview' || command==='inspect'){
    if(!opt.asset)throw new Error('--asset is required');
    const dir=resolve(opt.asset);
    const manifest=await inspectAsset(dir);
    if(command==='inspect'){console.log(JSON.stringify({asset_id:manifest.asset_id,maps:Object.keys(manifest.textures)}));return;}
    const server=await startServer({assetDir:dir,port:Number(opt.port)});
    console.log(server.origin+'/web/viewer.html?manifest=/asset/manifest.json');return;
  }
  if(!['bake','canonicalize'].includes(command))throw new Error(usage);
  console.log(JSON.stringify(await publish(command,opt)));
}
if(process.argv[1] && import.meta.url===pathToFileURL(resolve(process.argv[1])).href) {
  main(process.argv.slice(2)).catch(e=>{console.error(e.message);process.exitCode=1;});
}
