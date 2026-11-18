import {mkdir,writeFile} from 'node:fs/promises';
import {resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
const repo=fileURLToPath(new URL('../../../',import.meta.url));
const {chromium}=await import(repo+'/node_modules/playwright/index.mjs');
const {startServer}=await import(repo+'/worker/server.mjs');
const {chromiumOptions,presentedForeground}=await import(repo+'/worker/chromium.mjs');
const [asset,destination,view='+Z']=process.argv.slice(2);
await mkdir(destination,{recursive:true});
const server=await startServer({assetDir:resolve(asset)});
const browser=await chromium.launch(chromiumOptions('webgl2'));
try {
 const page=await browser.newPage({viewport:{width:1024,height:1024},deviceScaleFactor:1});
 const errors=[];
 page.on('pageerror',e=>errors.push(e.message));
 page.on('console',m=>{if(m.type()==='error')errors.push(m.text().slice(0,600));});
 await page.goto(server.origin+'/web/viewer.html?manifest=/asset/manifest.json&backend=webgl2');
 await page.waitForFunction(()=>/webgpu|webgl2|表示エラー/.test(document.querySelector('#status').textContent),null,{timeout:120000});
 if(await page.locator('#status.error').count())throw Error(await page.locator('#status').innerText());
 await page.selectOption('#mode','box');
 await page.selectOption('#view',view);
 await page.selectOption('#projection','orthographic');
 await page.evaluate(()=>{
  const stage=document.querySelector('#stage');
  Object.assign(stage.style,{width:'1024px',height:'1024px',minHeight:'0',flex:'0 0 auto',border:'0',background:'transparent'});
  stage.classList.remove('checker');
  document.querySelector('main').style.flex='0 0 auto';
  document.documentElement.style.background='transparent';document.body.style.background='transparent';
 });
 await page.locator('#fit').click();
 await page.waitForFunction(()=>document.querySelector('#canvas').width===1024,null,{timeout:20000});
 for(const mode of ['albedo','lit']){
  await page.selectOption('#shading',mode);
  await page.waitForFunction(presentedForeground,1,{timeout:60000});
  await page.evaluate(()=>new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r))));
  const path=destination+'/standard-'+mode+'.png';
  await page.locator('#canvas').screenshot({path,type:'png',omitBackground:true});
  console.log(path);
 }
 await writeFile(destination+'/render-evidence.json',JSON.stringify({asset:resolve(asset),view,width:1024,height:1024,variants:[{variant:'standard',errors,status:await page.locator('#status').innerText()}]},null,2));
} finally {await browser.close();await server.close();}
