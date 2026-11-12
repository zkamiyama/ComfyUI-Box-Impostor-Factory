// Compare an original Box and its extracted copy in separate browser processes.
import { chromium } from 'playwright';
import { chromiumOptions } from '../worker/chromium.mjs';
import { compareFrames } from '../worker/compare_frames.mjs';
import { startServer } from '../worker/server.mjs';

const [original, extracted, backend = 'webgl2'] = process.argv.slice(2);
if (!original || !extracted || !['webgl2', 'webgpu'].includes(backend)) {
  throw new Error('Usage: node tests/compare_archive.mjs original-dir extracted-dir [webgl2|webgpu]');
}

async function capture(directory) {
  const server = await startServer({ assetDir: directory });
  let browser;
  try {
    browser = await chromium.launch(chromiumOptions(backend));
    const page = await browser.newPage();
    await page.goto(server.origin + '/bridge');
    return await page.evaluate(async backend => {
      const { reload } = await import('/worker/browser_bridge.mjs');
      return await reload({ backend, verify: true });
    }, backend);
  } finally {
    try {
      await browser?.close();
    } finally {
      await server.close();
    }
  }
}

const before = await capture(original);
const after = await capture(extracted);
if (before.backend !== backend || after.backend !== backend) {
  throw new Error('Requested renderer backend did not execute both captures.');
}
const report = compareFrames(before.snapshot, after.snapshot, { colorTolerance: 0, depthTolerance: 0 });
console.log(JSON.stringify({ backend, ...report }, null, 2));
if (!report.pass) process.exitCode = 1;
