import assert from 'node:assert/strict';
import test from 'node:test';
import { resolve } from 'node:path';
import { chromium } from 'playwright';
import { startServer } from '../worker/server.mjs';

test('saved Box preview switches proxy, Box and synchronized split view, rotates and shows captured maps', {
  skip: process.env.BI_TEST_ASSET ? false : 'set BI_TEST_ASSET to a baked asset directory'
}, async () => {
  const server = await startServer({assetDir:resolve(process.env.BI_TEST_ASSET)});
  let browser;
  try {
    browser = await chromium.launch({headless:true,executablePath:process.env.BI_CHROMIUM || undefined,
      args:process.env.BI_SOFTWARE_GPU === '1' ? ['--use-angle=swiftshader','--enable-unsafe-swiftshader','--enable-unsafe-webgpu',
        '--enable-features=Vulkan','--use-vulkan=swiftshader','--disable-vulkan-surface'] : []});
    const page = await browser.newPage({viewport:{width:800,height:580}});
    const errors = [];
    page.on('pageerror',e => errors.push(e.message));
    page.on('console',m => {
      if (m.type()==='error' && /THREE|shader error|invalid shader|validation error|framebuffer is incomplete/i.test(m.text())) errors.push(m.text());
    });
    const backend = process.env.BI_TEST_BACKEND || 'auto';
    await page.goto(server.origin+'/web/viewer.html?manifest=/asset/manifest.json&backend='+encodeURIComponent(backend));
    await page.waitForFunction(() => /webgpu|webgl2|表示エラー/.test(document.querySelector('#status').textContent));
    assert.equal(await page.locator('#status.error').count(),0,await page.locator('#status').innerText());
    const canvas = page.locator('#canvas');
    await page.selectOption('#mode','proxy');
    await page.selectOption('#mode','both');
    assert.equal(await page.locator('#labels').isVisible(),true);
    // Check the presented canvas itself. Renderer initialization and offscreen readback can
    // both succeed when a headless WebGPU device cannot allocate a swapchain image.
    await page.waitForFunction(() => {
      const sampled=document.createElement('canvas');sampled.width=128;sampled.height=64;
      const ctx=sampled.getContext('2d');ctx.drawImage(document.querySelector('#canvas'),0,0,128,64);
      const pixels=ctx.getImageData(0,0,128,64).data;
      const covered=[0,0];
      for(let y=0;y<64;y++) for(let x=0;x<128;x++) if(pixels[4*(y*128+x)+3]>127) covered[x>>6]++;
      return covered.every(count=>count>20 && count<4096-20);
    });
    await page.selectOption('#view','+Y');
    await page.selectOption('#projection','orthographic');
    const bounds = await canvas.boundingBox();
    const x=bounds.x+bounds.width*.2, y=bounds.y+bounds.height*.4;
    await page.mouse.move(x,y); await page.mouse.down();
    await page.mouse.move(x+30,y+20,{steps:4}); await page.mouse.up();
    await page.selectOption('#mode','box');
    await page.selectOption('#mode','atlas');
    await page.selectOption('#map','intervalMap');
    await page.waitForFunction(() => document.querySelector('#atlas').naturalWidth > 0);
    assert.equal(await page.locator('#atlas').isVisible(),true);
    await page.selectOption('#mode','box');
    await page.getByRole('button',{name:'全体表示'}).click();
    assert.deepEqual(errors,[]);
  } finally { await browser?.close(); await server.close(); }
});
