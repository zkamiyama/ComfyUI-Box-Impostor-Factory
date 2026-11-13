import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { chromium } from 'playwright';

const [file, screenshot] = process.argv.slice(2);
if (!file) throw new Error('Expected an HTML path');
const browser = await chromium.launch({
  headless: true, executablePath: process.env.BI_CHROMIUM,
  args:['--use-angle=swiftshader', '--enable-unsafe-swiftshader','--disable-gpu-sandbox']
});
try {
  const page=await browser.newPage({viewport:{width:1280,height:760},deviceScaleFactor:1});
  const errors=[];
  page.on('pageerror',e=>errors.push('page '+e.message));
  page.on('console',m=>{if(m.type()==='error') errors.push('console '+m.text())});
  const html=await readFile(file,'utf8');
  await page.setContent(html,{waitUntil:'load'});
  await page.waitForFunction(()=>/^(WebGL2|Canvas 2D)$/.test(document.querySelector('#render-state')?.textContent||''),null,{timeout:60000});
  const state=await page.locator('#render-state').innerText();
  const stats=await page.locator('.stat-number').allInnerTexts();
  console.log('RESULT',JSON.stringify({state,stats,errors,gl2:await page.evaluate(()=>!!document.createElement('canvas').getContext('webgl2'))}));
  await page.locator('#tab-materials').click();
  console.log('MATERIAL_CARDS',await page.locator('.material-card').count());
  await page.locator('.material-card').first().click();
  await page.locator('#tab-textures').click();
  console.log('TEXTURES',await page.locator('.texture-option').count());
  await page.locator('#channel').selectOption('a');
  await page.locator('#uv-overlay').check();
  await page.locator('#tab-details').click();
  await page.locator('[data-view="front"]').click();
  await page.locator('#wireframe').click();
  await page.waitForTimeout(150);
  if(screenshot)await page.screenshot({path:screenshot});
  assert.equal(state,'WebGL2','hardware-compatible browser should render in WebGL2, not fallback');
  assert.equal(errors.length,0,JSON.stringify(errors));
}finally{await browser.close();}
