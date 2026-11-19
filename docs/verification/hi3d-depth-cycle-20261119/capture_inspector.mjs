import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { inflateSync } from 'node:zlib';
import { chromium } from 'playwright';

const [htmlArg, outputArg, referenceArg, modesArg = 'lit', viewsArg = 'front,iso,left,right', expectedTrianglesArg] = process.argv.slice(2);
assert(htmlArg && outputArg && referenceArg, 'Usage: node capture_inspector.mjs VIEWER OUTPUT_DIR REFERENCE_MANIFEST [lit,normals] [front,iso,left,right]');
const html = resolve(htmlArg), output = resolve(outputArg);
await mkdir(output, { recursive: true });
function viewerMetadata(bytes) {
  const payload = bytes.toString('utf8').match(/<script id="scene-payload"[^>]*>([\s\S]*?)<\/script>/)[1].trim();
  const packed = Buffer.from(payload, 'base64');
  const decoded = JSON.parse((packed[0] === 0x78 ? inflateSync(packed) : packed).toString('utf8'));
  const payloadTriangles = (decoded.meshes || []).filter(mesh => mesh.primitive !== "points").reduce((sum, mesh) => sum + Buffer.from(mesh.indices || "", "base64").length / 12, 0);
  return { bounds: decoded.bounds, assetId: decoded.assetId, payloadTriangles, original: decoded.original };
}
let htmlBytes = await readFile(html);
const scene = viewerMetadata(htmlBytes);
if (expectedTrianglesArg) assert.equal(scene.payloadTriangles, Number(expectedTrianglesArg), "Viewer payload must contain every expected triangle");
const referenceBytes = await readFile(resolve(referenceArg));
const reference = JSON.parse(referenceBytes.toString('utf8'));
const bounds = reference.bakeBox || reference.bounds;
const center = b => b.min.map((x, i) => (x + b.max[i]) / 2);
const radius = b => Math.hypot(...b.max.map((x, i) => x - b.min[i])) / 2;
const target = center(bounds), currentCenter = center(scene.bounds);
const referenceRadius = radius(bounds), sourceRadius = radius(scene.bounds);
const distance = referenceRadius * 1.05 / Math.sin(20 * Math.PI / 180);
const halfExtent = distance * Math.tan(20 * Math.PI / 180);
const wheelY = -100 * Math.log(referenceRadius / sourceRadius) / Math.log(0.95);
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const executablePath = process.env.BI_CHROMIUM || '/home/dev/.cache/ms-playwright/chromium_headless_shell-1254/chrome-headless-shell-linux64/chrome-headless-shell';
const report = {
  html, htmlSHA256: hash(htmlBytes), referenceManifest: resolve(referenceArg),
  referenceManifestSHA256: hash(referenceBytes), assetId: scene.assetId || null,
  sourceBounds: scene.bounds, referenceBounds: bounds,
  triangleCounts: { expected: expectedTrianglesArg ? Number(expectedTrianglesArg) : null, meshPayload: scene.payloadTriangles, sourceOriginal: scene.original || null },
  camera: { projection: 'orthographic', target, distance, halfExtent,
    fovBeforeOrthographic: 40, fitFormula: 'radius * 1.05 / sin(20deg)',
    sourceRadius, referenceRadius, perspectiveWheelDeltaY: wheelY,
    matching: 'Existing Fit, wheel, projection, pan and view controls only; HTML byte-identical',
    pixels: [1024, 1024], cssPixels: [800, 800], deviceScaleFactor: 1.28001 },
  lighting: { viewerDefaults: true, studio: 'neutral where supported',
    nativeBox: 'Directional intensity 2.1; hemisphere intensity 1.55; ACES exposure 1',
    note: 'Mesh and Box use their respective official native rendering paths. A/B/C are compared within each path.' },
  modes: {}, errors: [],
};
htmlBytes = null;
const browser = await chromium.launch({
  executablePath, headless: true,
  args: ['--disable-gpu', '--disable-features=Vulkan,WebGPU', '--disable-dev-shm-usage', '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'],
});
try {
  const page = await browser.newPage({ viewport: { width: 800, height: 800 }, deviceScaleFactor: 1.28001, acceptDownloads: true });
  page.setDefaultTimeout(45000);
  page.on('pageerror', error => report.errors.push('page: ' + String(error)));
  page.on('console', message => { if (message.type() === 'error') report.errors.push('console: ' + message.text()); });
  await page.goto(pathToFileURL(html).href, { waitUntil: 'domcontentloaded', timeout: 120000 });
  await page.waitForFunction(() => document.getElementById('render-state')?.textContent.startsWith('Three.js /'), null, { timeout: 120000 });
  assert.match(await page.locator('#render-state').innerText(), /WebGL2/);
  report.graphicsRenderer = await page.evaluate(() => {
    const gl = document.getElementById('viewport').getContext('webgl2');
    const extension = gl.getExtension('WEBGL_debug_renderer_info');
    return gl.getParameter(extension.UNMASKED_RENDERER_WEBGL);
  });
  assert.match(report.graphicsRenderer, /SwiftShader/i, 'Capture requires CPU SwiftShader');
  report.hardwareGPUUsed = false;
  report.browserVersion = await browser.version();
  if (await page.locator('#studio-controls').evaluate(el => !el.hidden)) {
    await page.locator('#tab-materials').click();
    await page.locator('#studio-lighting').selectOption('neutral');
  }
  if (await page.locator('#details-toggle').getAttribute('aria-expanded') === 'true')
    await page.locator('#details-toggle').click();
  for (let i = 0; i < 3; i++) {
    const rect = await page.locator('#viewport').boundingBox();
    const viewport = page.viewportSize();
    if (Math.abs(rect.width - 800) < 0.01 && Math.abs(rect.height - 800) < 0.01) break;
    await page.setViewportSize({ width: Math.round(viewport.width + 800 - rect.width), height: Math.round(viewport.height + 800 - rect.height) });
    await page.waitForTimeout(100);
  }
  const rect = await page.locator('#viewport').boundingBox();
  assert(Math.abs(rect.width - 800) < 0.01 && Math.abs(rect.height - 800) < 0.01, 'Canvas CSS must be 800 square for a 1024 backing image');
  if (await page.locator('#projection').getAttribute('aria-pressed') === 'true')
    await page.locator('#projection').click();
  await page.locator('[data-command=fit]').click();
  await page.mouse.move(rect.x + 400, rect.y + 400);
  const inputDpr = await page.evaluate(() => window.devicePixelRatio);
  report.camera.wheelProtocolDeltaY = wheelY * inputDpr;
  if (Math.abs(wheelY) > 1e-10) await page.mouse.wheel(0, report.camera.wheelProtocolDeltaY);
  await page.waitForTimeout(120);
  await page.locator('#projection').click();

  // Official OrbitControls pan is linear in orthographic screen pixels.
  // Front pan corrects world X/Y; right pan then corrects world Z.
  const delta = target.map((x, i) => x - currentCenter[i]);
  const pxPerUnit = 800 / (2 * halfExtent);
  const pans = [
    { view: 'front', dx: -delta[0] * pxPerUnit, dy: delta[1] * pxPerUnit },
    { view: 'right', dx: delta[2] * pxPerUnit, dy: 0 },
  ];
  for (const pan of pans) {
    if (Math.hypot(pan.dx, pan.dy) < 1e-9) continue;
    await page.locator('[data-view="' + pan.view + '"]').evaluate(el => el.click());
    await page.mouse.move(rect.x + 400, rect.y + 400);
    await page.mouse.down({ button: 'right' });
    await page.mouse.move(rect.x + 400 + pan.dx, rect.y + 400 + pan.dy);
    await page.mouse.up({ button: 'right' });
  }
  report.camera.pans = pans;
  report.viewport = page.viewportSize();
  report.canvasRect = rect;
  await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
  report.devicePixelRatio = await page.evaluate(() => window.devicePixelRatio);
  report.canvasPixels = await page.locator('#viewport').evaluate(el => [el.width, el.height]);
  assert.deepEqual(report.canvasPixels, [1024, 1024]);
  report.backend = await page.locator('#render-state').innerText();
  report.stageState = await page.locator('#stage-state').innerText();
  const directions = { front: [0, 0, 1], iso: [1, 1, 1], left: [-1, 0, 0], right: [1, 0, 0] };
  for (const mode of modesArg.split(',')) {
    assert(['lit', 'normals'].includes(mode));
    const normalButton = page.locator('#normals');
    if (mode === 'normals') assert(!await normalButton.isDisabled(), 'Official normal mode unavailable for this model');
    if ((await normalButton.getAttribute('aria-pressed') === 'true') !== (mode === 'normals'))
      await normalButton.click();
    report.modes[mode] = {};
    for (const [name, vector] of Object.entries(directions)) {
      if (!viewsArg.split(',').includes(name)) continue;
      const button = page.locator('[data-view="' + name + '"]');
      if (name === 'iso' || await button.getAttribute('aria-pressed') !== 'true')
        await button.evaluate(el => el.click());
      await page.waitForTimeout(300);
      await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
      const event = page.waitForEvent('download');
      await page.locator('#snapshot').click();
      const download = await event;
      const imagePath = resolve(output, mode + '-' + name + '.png');
      await download.saveAs(imagePath);
      const norm = Math.hypot(...vector);
      report.modes[mode][name] = {
        path: imagePath, sha256: hash(await readFile(imagePath)),
        direction: vector.map(x => x / norm),
        expectedPosition: target.map((x, i) => x + vector[i] / norm * distance),
      };
      console.log('CAPTURE', mode, name, imagePath);
    }
  }
  await page.screenshot({ path: resolve(output, 'inspector-ui.png') });
  assert.equal(hash(await readFile(html)), report.htmlSHA256, 'Viewer changed during capture');
} finally {
  await browser.close();
  await writeFile(resolve(output, 'capture.json'), JSON.stringify(report, null, 2) + '\n');
}
assert.deepEqual(report.errors, []);
console.log('DONE', output);