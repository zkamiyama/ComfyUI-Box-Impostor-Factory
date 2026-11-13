import assert from 'node:assert/strict';
import { mkdir, readFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { chromium } from 'playwright';
import { startServer } from '../worker/server.mjs';
import { chromiumOptions, presentedForeground } from '../worker/chromium.mjs';
const output=resolve(process.argv[2] ?? '.test-output/chat-preview.png');
await mkdir(dirname(output),{recursive:true});
const server=await startServer();
const browser=await chromium.launch(chromiumOptions('webgl2'));
const errors=[],requests=[];
let monitor;
try {
 const page=await browser.newPage({viewport:{width:1100,height:740},deviceScaleFactor:1});
 if(process.env.BI_CHAT_CDN_TEST==='1'){
   await page.route('**/web/chat_viewer.html',async route=>{
     const res=await route.fetch();
     const body=(await res.text()).replace('src="./chat_loader.mjs"',
       'src="https://cdn.jsdelivr.net/gh/zkamiyama/ComfyUI-Box-Impostor-Factory@ff8d2cc/web/chat_loader.mjs"');
     await route.fulfill({response:res,body});
   });
 }
 page.on('pageerror',e=>{errors.push('PAGE '+e.message);console.error('PAGEERROR',e.message);});
 page.on('console',m=>{if(m.type()==='error'){errors.push('CONSOLE '+m.text().slice(0,800));console.error('CONSOLE',m.text().slice(0,800));}});
 page.on('requestfailed',req=>{errors.push('FAILED '+req.url()+' '+req.failure());console.error('REQUESTFAILED',req.url(),req.failure());});
 page.on('response',r=>{if(r.status()>=400)console.log('HTTP',r.status(),r.url());});
 if(process.env.BI_CHAT_OPAQUE_TEST==='1' || process.argv.includes('--opaque')){
   const source=await readFile(new URL('../web/chat_viewer.html',import.meta.url),'utf8');
   const revision=process.env.BI_CHAT_REMOTE_REV || 'ef0c2a1';
   const html=source.replace('src="./chat_loader.mjs"',
     `src="https://cdn.jsdelivr.net/gh/zkamiyama/ComfyUI-Box-Impostor-Factory@${revision}/web/chat_loader.mjs"`);
   await page.setContent(html,{waitUntil:'domcontentloaded'});
 } else await page.goto(server.origin+'/web/chat_viewer.html',{waitUntil:'domcontentloaded'});
 console.log('PAGE_STARTED',await page.title(),page.url());
 monitor=setInterval(async()=>{
  try {
   const state=await page.locator('#status').innerText();
   console.log('LOADING_STATUS',state);
  }catch(e){console.log('MONITOR_ERR',e.message);}
 },7000);
 await page.waitForFunction(()=>{
   const text=document.querySelector('#status')?.textContent || '';
   return text.includes('webgl2') || text.includes('3D表示に失敗') || text.includes('表示エラー');
 },null,{timeout:100000});
 const status=await page.locator('#status').innerText();
 if(!status.includes('webgl2')) throw Error('Unusable Box viewer: '+status+'; '+errors.join(' / '));
 assert.equal(await page.locator('#chat-overlay').count(),0);
 await page.waitForFunction(presentedForeground,1,{timeout:40000});
 await page.locator('#canvas').screenshot({path:output.replace(/\.png$/,'.initial.png')});
 await page.locator('#mode').selectOption('both');
 await page.waitForFunction(presentedForeground,2,{timeout:40000});
 await page.locator('#mode').selectOption('atlas');
 await page.locator('#map').selectOption('guide:+X');
 await page.waitForFunction(()=>{const im=document.querySelector('#atlas');return im.complete && im.naturalWidth>0;});
 await page.locator('#map').selectOption('depth:+X');
 await page.waitForFunction(()=>{const im=document.querySelector('#atlas');return im.complete && im.naturalWidth>0;});
 await page.locator('#mode').selectOption('box');
 await page.locator('#appearance').selectOption('base');
 await page.locator('#appearance').selectOption('active');
 await page.locator('#view').selectOption('-X');
 await page.locator('#projection').selectOption('orthographic');
 await page.locator('#canvas').screenshot({path:output});
 const final=await page.evaluate(()=>({view:document.querySelector('#view').value,
  projection:document.querySelector('#projection').value}));
 console.log(JSON.stringify({ok:true,status,output,final,errors},null,2));
} finally {
 if(monitor) clearInterval(monitor);
 await browser.close();
 await server.close();
}
