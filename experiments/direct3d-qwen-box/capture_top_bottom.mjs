import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {mkdir, readFile, writeFile} from 'node:fs/promises';
import {createRequire} from 'node:module';
import {dirname, resolve} from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';
import {parseArgs} from 'node:util';
import {inflateSync} from 'node:zlib';

// Prepare-only helper for official Inspector's existing top/bottom axis UI.
// No HTML, materials, source assets, or the original yaw capture script are changed.
const {values: options} = parseArgs({options: {
  viewer: {type: 'string'}, reference: {type: 'string'}, out: {type: 'string'},
  size: {type: 'string', default: '512'}, mode: {type: 'string', default: 'default'},
  backend: {type: 'string', default: 'swiftshader'}, chromium: {type: 'string'},
  'playwright-root': {type: 'string'}, help: {type: 'boolean'},
}});
if (options.help) {
  console.log('node capture_top_bottom.mjs --viewer OUTPUT_FROM_VIEW_PY.html --reference COMMON_MANIFEST.json --out NEW_CAPTURE_DIR --chromium FULL_CHROMIUM [--playwright-root REPO]');
  console.log('Save full 512x512 transparent PNGs via existing Top UI, then press the active Top again for Bottom.');
  console.log('Default material, orthographic shared framing, SwiftShader. HTML and assets remain unchanged.');
  process.exit(0);
}
assert(options.viewer && options.reference && options.out && options.chromium,
  '--viewer, --reference, --out and explicit --chromium are required');
assert.equal(options.mode, 'default', 'Only the unchanged official default display is supported');
assert.equal(options.backend, 'swiftshader', 'Use the same SwiftShader backend as the yaw comparison');
const width = Number(options.size), frameCount = 2;
assert.equal(width, 512, 'Require the full unscaled 512x512 frame');
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const html = resolve(options.viewer), output = resolve(options.out);
const htmlBytes = await readFile(html), referenceBytes = await readFile(resolve(options.reference));
const reference = JSON.parse(referenceBytes);
const sceneMatch = htmlBytes.toString().match(/<script id="scene-payload"[^>]*>([\s\S]*?)<\/script>/);
assert(sceneMatch, 'Expected actual 3D-Inspector scene-payload');
const packed = Buffer.from(sceneMatch[1].trim(), 'base64');
const scene = JSON.parse((packed[0] === 0x78 ? inflateSync(packed) : packed).toString());
const bounds = reference.bakeBox ?? reference.bounds ?? reference.referenceBounds;
assert(bounds?.min?.length === 3 && bounds?.max?.length === 3, 'Reference must contain full-object bounds');
const center = bounds => bounds.min.map((value, i) => (value + bounds.max[i]) / 2);
const radius = bounds => Math.hypot(...bounds.max.map((value, i) => value - bounds.min[i])) / 2;
const target = center(bounds), sourceTarget = center(scene.bounds);
const halfTan = Math.tan(20 * Math.PI / 180);
const referenceDistance = radius(bounds) * 1.05 / Math.sin(20 * Math.PI / 180);
const halfExtent = referenceDistance * halfTan;
const sourceDistance = radius(scene.bounds) * 1.05 / Math.sin(20 * Math.PI / 180);
const fitHalfExtent = sourceDistance * halfTan;
const report = {
  html, htmlSHA256: hash(htmlBytes), referenceManifest: resolve(options.reference),
  referenceManifestSHA256: hash(referenceBytes),
  scriptSHA256: hash(await readFile(fileURLToPath(import.meta.url))),
  assetId: scene.assetId ?? null, sourceBounds: scene.bounds, referenceBounds: bounds,
  sourceFaceResolution: scene.nativeBox?.manifest?.resolution ?? null,
  display: {mode: options.mode, nativeBoxDisplay: scene.nativeBox?.display ?? null,
    materialOverride: false, toneMappingOverride: false, background: 'native transparent canvas'},
  camera: {projection: 'orthographic', pixels: [width, width], target, halfExtent,
    expectedDistance: referenceDistance, views: ['top', 'bottom'], axisToleranceDegrees: .03,
    framing: 'full reference bounds: radius * 1.05 / sin(20deg)',
    matrixOrder: 'Three.js column-major; camera looks along local -Z, local +Y is up'},
  framesExpected: frameCount, viewsExpected: ['top', 'bottom'],
  method: 'official-inspector-top-bottom-axis-ui-v1',
  backendRequested: options.backend, completed: false, errors: [], frames: [],
  capture: 'Exact view.py HTML; existing UI and mouse; official Save PNG button',
  captureScheduling: 'Unmodified; camera probe delegates every render unchanged',
  sourceCaptureScript: {path: '/mnt/f/_dev/box-impostor-lab/experiments/zipsplat-qwen-20261120/capture_inspector.mjs',
    sha256: 'a34b872abbad6b197b475692a06d3ade15681e52c55fd0b0c901d9202d7a1ab1'},
  axisBehaviorSource: {path: '/home/dev/Projects/box-impostor-lab/third_party/3D-Inspector-skills/skills/3d-inspector/assets/viewer.html',
    sha256: 'd4964315c8af342cc457375fbb8a71cc57d077061a61c6619bf12e67e05273fd',
    inspectedBehavior: 'setupGizmo creates top and bottom buttons; clicking active top chooses bottom',
    runtimeVerification: 'Both controls exist; aria-pressed and actual camera outward direction are checked'},
  noCropOrResize: true,
};
assert.equal(report.sourceFaceResolution, 512, 'Require the native Box512 asset');
await mkdir(dirname(output), {recursive: true});
await mkdir(output); // Preserve prior and partial captures; never overwrite an existing output directory.
const reportPath = resolve(output, 'capture.json');
const require = createRequire(resolve(options['playwright-root'] ?? process.cwd(), 'package.json'));
const {chromium} = require('playwright');
const browserArgs = ['--disable-dev-shm-usage'];
if (options.backend === 'swiftshader') browserArgs.push('--disable-gpu',
  '--disable-features=Vulkan,WebGPU', '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader');
if (options.backend === 'webgl2') browserArgs.push('--disable-features=WebGPU');
const browserEnv = {...process.env};
delete browserEnv.DISPLAY;
delete browserEnv.WAYLAND_DISPLAY;
const browser = await chromium.launch({headless: true, args: browserArgs,
  executablePath: options.chromium, env: browserEnv});
report.chromiumExecutable = resolve(options.chromium);
report.displayEnvironmentRemoved = true;
try {
  const page = await browser.newPage({viewport: {width: Math.max(800, width), height: width},
    deviceScaleFactor: 1, acceptDownloads: true});
  page.setDefaultTimeout(120000);
  page.on('pageerror', error => report.errors.push('page: ' + String(error)));
  page.on('console', message => {if (message.type() === 'error') report.errors.push('console: ' + message.text());});
  await page.addInitScript(() => {
    const probe = globalThis.__inspectorCaptureProbe = {renders: 0};
    let observed;
    Object.defineProperty(globalThis, 'InspectorThree', {
      configurable: true, get() {return observed;}, set(value) {
        observed = new Proxy(value, {get(target, key, receiver) {
          if (key !== 'createThreeViewer') return Reflect.get(target, key, receiver);
          return async function(...args) {
            const api = await Reflect.apply(target.createThreeViewer, target, args);
            probe.api = api;
            const renderer = api.getRenderer(), originalRender = renderer.render;
            renderer.render = function(scene, camera, ...tail) {
              const result = Reflect.apply(originalRender, this, [scene, camera, ...tail]);
              probe.renders++;
              probe.camera = {type: camera.type, position: camera.position.toArray(),
                quaternion: camera.quaternion.toArray(), up: camera.up.toArray(),
                matrixWorld: camera.matrixWorld.toArray(), matrixWorldInverse: camera.matrixWorldInverse.toArray(),
                projectionMatrix: camera.projectionMatrix.toArray(),
                projectionMatrixInverse: camera.projectionMatrixInverse.toArray(),
                near: camera.near, far: camera.far, top: camera.top, bottom: camera.bottom,
                left: camera.left, right: camera.right, zoom: camera.zoom, fov: camera.fov,
                canvasPixels: [renderer.domElement.width, renderer.domElement.height]};
              return result;
            };
            return api;
          };
        }});
      },
    });
  });
  await page.goto(pathToFileURL(html).href, {waitUntil: 'domcontentloaded', timeout: 180000});
  await page.waitForFunction(() => globalThis.__inspectorCaptureProbe?.api &&
    document.getElementById('render-state')?.textContent.startsWith('Three.js /'), null, {timeout: 180000});
  report.backend = await page.locator('#render-state').innerText();
  report.browserVersion = await browser.version();
  report.graphicsRenderer = await page.evaluate(() => {
    const renderer = globalThis.__inspectorCaptureProbe.api.getRenderer();
    if (renderer.backend?.isWebGPUBackend) return 'WebGPU';
    const gl = renderer.domElement.getContext('webgl2');
    const info = gl?.getExtension('WEBGL_debug_renderer_info');
    return info ? gl.getParameter(info.UNMASKED_RENDERER_WEBGL) : 'WebGL2';
  });
  if (await page.locator('#studio-controls').evaluate(element => !element.hidden)) {
    await page.locator('#tab-materials').click();
    await page.locator('#studio-lighting').selectOption('neutral');
    report.display.studio = 'neutral through existing selector';
  } else report.display.studio = 'no studio selector for this asset';
  for (const id of ['wireframe', 'grid', 'normals']) {
    const pressed = await page.locator('#' + id).getAttribute('aria-pressed') === 'true';
    const desired = id === 'normals' && options.mode === 'normals';
    if (pressed !== desired) await page.locator('#' + id).click();
  }
  if (await page.locator('#details-toggle').getAttribute('aria-expanded') === 'true')
    await page.locator('#details-toggle').click();
  for (let attempt = 0; attempt < 4; attempt++) {
    const rect = await page.locator('#viewport').boundingBox();
    if (Math.abs(rect.width - width) < .01 && Math.abs(rect.height - width) < .01) break;
    const old = page.viewportSize();
    await page.setViewportSize({width: Math.round(old.width + width - rect.width),
      height: Math.round(old.height + width - rect.height)});
    await page.evaluate(() => new Promise(done => requestAnimationFrame(done)));
  }
  const rect = await page.locator('#viewport').boundingBox();
  assert(Math.abs(rect.width - width) < .01 && Math.abs(rect.height - width) < .01);
  const cx = rect.x + width / 2, cy = rect.y + width / 2;
  async function settle() {
    await page.evaluate(() => new Promise(done => requestAnimationFrame(() => requestAnimationFrame(done))));
  }
  async function view(name) {
    // Axis buttons toggle to the opposite side if already active; pass through Iso first.
    await page.locator('[data-view="iso"]').evaluate(element => element.click());
    await settle();
    await page.locator(`[data-view="${name}"]`).evaluate(element => element.click());
    await settle();
  }
  if (await page.locator('#projection').getAttribute('aria-pressed') === 'true')
    await page.locator('#projection').click();
  await page.locator('[data-command="fit"]').click();
  await settle();
  await view('front');
  const wheelDelta = -100 * Math.log(halfExtent / fitHalfExtent) / Math.log(.95);
  await page.mouse.move(cx, cy);
  if (Math.abs(wheelDelta) > 1e-8) await page.mouse.wheel(0, wheelDelta);
  await settle();
  await page.locator('#projection').click();
  await settle();
  const delta = target.map((value, i) => value - sourceTarget[i]);
  const pxPerUnit = width / (2 * halfExtent);
  for (const pan of [{view: 'front', dx: -delta[0] * pxPerUnit, dy: delta[1] * pxPerUnit},
    {view: 'right', dx: delta[2] * pxPerUnit, dy: 0}]) {
    if (Math.hypot(pan.dx, pan.dy) < 1e-9) continue;
    await view(pan.view);
    const moves = Math.ceil(Math.max(Math.abs(pan.dx), Math.abs(pan.dy)) / (width * .3));
    for (let step = 0; step < moves; step++) {
      await page.mouse.move(cx, cy); await page.mouse.down({button: 'right'});
      await page.mouse.move(cx + pan.dx / moves, cy + pan.dy / moves);
      await page.mouse.up({button: 'right'}); await settle();
    }
  }
  await view('front');
  async function state() {
    await settle();
    const camera = await page.evaluate(() => globalThis.__inspectorCaptureProbe.camera);
    assert.equal(camera.type, 'OrthographicCamera');
    const direction = [camera.matrixWorld[8], camera.matrixWorld[9], camera.matrixWorld[10]];
    return {camera, yawDegrees: Math.atan2(direction[0], direction[2]) * 180 / Math.PI,
      pitchDegrees: Math.asin(Math.max(-1, Math.min(1, direction[1] / Math.hypot(...direction)))) * 180 / Math.PI};
  }
  report.camera.wheelDelta = wheelDelta;
  const topButton = page.locator('[data-view="top"]');
  const bottomButton = page.locator('[data-view="bottom"]');
  assert.equal(await topButton.count(), 1, 'Expected the official Top axis button');
  assert.equal(await bottomButton.count(), 1, 'Expected the official Bottom axis button');
  report.axisControls = await page.locator('[data-view="top"],[data-view="bottom"]').evaluateAll(elements =>
    elements.map(element => ({view: element.dataset.view, title: element.title,
      ariaLabel: element.getAttribute('aria-label')})));
  async function pngCoverage(bytes) {
    assert(bytes.subarray(0, 8).equals(Buffer.from([137,80,78,71,13,10,26,10])), 'Expected a PNG');
    assert.equal(bytes.toString('ascii', 12, 16), 'IHDR');
    assert.equal(bytes.readUInt32BE(16), width);
    assert.equal(bytes.readUInt32BE(20), width);
    assert.equal(bytes[24], 8, 'Require 8-bit PNG channels');
    assert.equal(bytes[25], 6, 'Require RGBA PNG');
    // Decode the saved file in a detached canvas solely to inspect alpha.
    // This never changes the Inspector canvas, DOM, renderer, or materials.
    const coverage = await page.evaluate(async ({base64, size}) => {
      const bytes = Uint8Array.from(atob(base64), character => character.charCodeAt(0));
      const bitmap = await createImageBitmap(new Blob([bytes], {type: 'image/png'}));
      if (bitmap.width !== size || bitmap.height !== size) throw new Error('Decoded PNG dimensions changed');
      const canvas = new OffscreenCanvas(size, size), context = canvas.getContext('2d');
      context.drawImage(bitmap, 0, 0);
      bitmap.close();
      const pixels = context.getImageData(0, 0, size, size).data;
      let foregroundPixels = 0, edgePixels = 0, x0 = size, y0 = size, x1 = -1, y1 = -1;
      for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
        if (pixels[(y * size + x) * 4 + 3] <= 127) continue;
        foregroundPixels++;
        if (x === 0 || y === 0 || x === size - 1 || y === size - 1) edgePixels++;
        x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y);
      }
      return {foregroundPixels, edgePixels,
        boundsXYXY: foregroundPixels ? [x0, y0, x1, y1] : null, alphaThresholdUint8: 127};
    }, {base64: bytes.toString('base64'), size: width});
    assert(coverage.foregroundPixels > 1000, 'Empty or tiny foreground');
    assert.equal(coverage.edgePixels, 0, 'Foreground touches the frame edge');
    return coverage;
  }
  async function captureAxis(name, sign, index) {
    const current = await state();
    const matrix = current.camera.matrixWorld;
    const raw = [matrix[8], matrix[9], matrix[10]], norm = Math.hypot(...raw);
    const outward = raw.map(value => value / norm);
    const error = Math.acos(Math.max(-1, Math.min(1, sign * outward[1]))) * 180 / Math.PI;
    assert(Number.isFinite(error) && error < .03, 'Actual camera is not on the requested Y axis');
    assert(Math.abs(Math.hypot(...current.camera.position.map((value, i) => value - target[i]))
      - referenceDistance) < 1e-5, 'Camera target/distance changed');
    assert(Math.abs(current.camera.top / current.camera.zoom - halfExtent) < 1e-5, 'Camera extent changed');
    const actualTarget = current.camera.position.map((value, i) => value - outward[i] * referenceDistance);
    assert(Math.max(...actualTarget.map((value, i) => Math.abs(value - target[i]))) < 1e-5,
      'Actual camera target differs from the shared full-object target');
    assert.equal(await topButton.getAttribute('aria-pressed'), String(name === 'top'));
    assert.equal(await bottomButton.getAttribute('aria-pressed'), String(name === 'bottom'));
    const before = await page.evaluate(() => globalThis.__inspectorCaptureProbe.renders);
    const downloadEvent = page.waitForEvent('download');
    await page.locator('#snapshot').click();
    const download = await downloadEvent;
    const path = resolve(output, name + '.png');
    await download.saveAs(path);
    const after = await page.evaluate(() => ({camera: globalThis.__inspectorCaptureProbe.camera,
      renders: globalThis.__inspectorCaptureProbe.renders}));
    assert.deepEqual(after.camera.matrixWorld, current.camera.matrixWorld, 'Camera moved during Save PNG');
    assert.deepEqual(after.camera.projectionMatrix, current.camera.projectionMatrix, 'Projection changed during Save PNG');
    assert.deepEqual(after.camera.canvasPixels, [width, width]);
    assert(after.renders > before, 'Save PNG did not render');
    const bytes = await readFile(path), coverage = await pngCoverage(bytes);
    report.frames.push({index, view: name, path, sha256: hash(bytes),
      requestedOutward: [0, sign, 0], actualOutward: outward, axisErrorDegrees: error,
      actualTarget, actualYawDegrees: current.yawDegrees, actualPitchDegrees: current.pitchDegrees,
      camera: after.camera, coverage, renderedCallsBeforeReadback: after.renders - before});
    await writeFile(reportPath, JSON.stringify(report, null, 2) + '\n');
    console.log('VIEW', name, 'pitch', current.pitchDegrees.toFixed(6),
      'axisError', error.toFixed(8), 'edgePixels', coverage.edgePixels);
  }
  const started = performance.now();
  await view('top');
  await captureAxis('top', 1, 0);
  assert.equal(await topButton.getAttribute('aria-pressed'), 'true');
  // The official setupGizmo handler toggles an active positive axis to its opposite.
  // Execute that existing button handler; do not invoke or replace camera internals.
  await topButton.evaluate(element => element.click());
  await settle();
  await captureAxis('bottom', -1, 1);
  report.axisBehaviorVerified = 'Pressing active Top selected Bottom; actual +Y and -Y cameras confirmed';
  report.captureElapsedSeconds = (performance.now() - started) / 1000;
  report.captureTimingNote = 'Offline browser capture including UI events and downloads; not interactive GPU FPS';
  await page.screenshot({path: resolve(output, 'inspector-ui.png')});
  assert.equal(hash(await readFile(html)), report.htmlSHA256, 'Generated viewer bytes changed');
  assert.equal(hash(await readFile(resolve(options.reference))), report.referenceManifestSHA256,
    'Shared reference manifest changed');
  assert.equal(report.frames.length, frameCount);
  assert.deepEqual(report.errors, []);
  report.completed = true;
} catch (error) {
  report.errors.push(String(error));
  throw error;
} finally {
  await browser.close();
  await writeFile(reportPath, JSON.stringify(report, null, 2) + '\n');
}
console.log('COMPLETE', reportPath);
