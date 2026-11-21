import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {mkdir, readFile, writeFile} from 'node:fs/promises';
import {createRequire} from 'node:module';
import {resolve} from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';
import {parseArgs} from 'node:util';
import {inflateSync} from 'node:zlib';

const {values: options} = parseArgs({options: {
  viewer: {type: 'string'}, reference: {type: 'string'}, out: {type: 'string'},
  frames: {type: 'string', default: '240'}, size: {type: 'string', default: '512'},
  fps: {type: 'string', default: '24'}, mode: {type: 'string', default: 'default'},
  backend: {type: 'string', default: 'swiftshader'}, chromium: {type: 'string'},
  'playwright-root': {type: 'string'}, sparse: {type: 'boolean'}, help: {type: 'boolean'},
}});
if (options.help) {
  console.log('node capture_inspector.mjs --viewer OUTPUT_FROM_VIEW_PY.html --reference manifest.json --out CAPTURE_DIR [--frames 240 --size 512 --mode default --sparse --playwright-root REPO]');
  console.log('The current official native Box viewer has no unlit UI. Mode unlit fails without changing HTML or materials.');
  process.exit(0);
}
assert(options.viewer && options.reference && options.out, '--viewer, --reference and --out are required');
assert(['default', 'normals', 'unlit'].includes(options.mode));
assert(['swiftshader', 'webgl2', 'auto'].includes(options.backend));
const width = Number(options.size), frameCount = options.sparse ? 8 : Number(options.frames);
const fps = Number(options.fps);
assert(Number.isInteger(width) && width >= 128 && width <= 2048);
assert(Number.isInteger(frameCount) && frameCount > 1 && frameCount <= 1440);
assert(Number.isFinite(fps) && fps > 0);
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
const normalizedAngle = angle => ((angle + 180) % 360 + 360) % 360 - 180;
const angleError = (a, b) => normalizedAngle(a - b);
const report = {
  html, htmlSHA256: hash(htmlBytes), referenceManifest: resolve(options.reference),
  referenceManifestSHA256: hash(referenceBytes),
  scriptSHA256: hash(await readFile(fileURLToPath(import.meta.url))),
  assetId: scene.assetId ?? null, sourceBounds: scene.bounds, referenceBounds: bounds,
  sourceFaceResolution: scene.nativeBox?.manifest?.resolution ?? null,
  display: {mode: options.mode, nativeBoxDisplay: scene.nativeBox?.display ?? null,
    materialOverride: false, toneMappingOverride: false, background: 'native transparent canvas'},
  camera: {projection: 'orthographic', pixels: [width, width], target, halfExtent,
    expectedDistance: referenceDistance, startDegrees: 0, sweepDegrees: 360,
    framing: 'full reference bounds: radius * 1.05 / sin(20deg)',
    matrixOrder: 'Three.js column-major; camera looks along local -Z, local +Y is up'},
  framesExpected: frameCount, playbackFPS: fps, durationSeconds: frameCount / fps,
  backendRequested: options.backend, completed: false, errors: [], frames: [],
  capture: 'Exact view.py HTML; existing UI and mouse; official Save PNG button',
  captureScheduling: 'Unmodified; camera probe delegates every render unchanged',
};
await mkdir(output, {recursive: true});
const reportPath = resolve(output, 'capture.json');
const require = createRequire(resolve(options['playwright-root'] ?? process.cwd(), 'package.json'));
const {chromium} = require('playwright');
const browserArgs = ['--disable-dev-shm-usage'];
if (options.backend === 'swiftshader') browserArgs.push('--disable-gpu',
  '--disable-features=Vulkan,WebGPU', '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader');
if (options.backend === 'webgl2') browserArgs.push('--disable-features=WebGPU');
const browser = await chromium.launch({headless: true, args: browserArgs,
  ...(options.chromium || process.env.BI_CHROMIUM
    ? {executablePath: options.chromium || process.env.BI_CHROMIUM} : {})});
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
  if (options.mode === 'unlit')
    throw new Error('Current official 3D-Inspector native Box UI has no unlit control; HTML and renderer were left unchanged. Use --mode default.');
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
  async function drag(dx) {
    assert(Math.abs(dx) < width * .4);
    await page.mouse.move(cx, cy); await page.mouse.down({button: 'left'});
    await page.mouse.move(cx + dx, cy); await page.mouse.up({button: 'left'});
    return state();
  }
  const initial = await state(), calibrated = await drag(10);
  const degreesPerPixel = angleError(calibrated.yawDegrees, initial.yawDegrees) / 10;
  assert(Math.abs(degreesPerPixel) > .05 && Math.abs(degreesPerPixel) < 3);
  await view('front');
  report.camera.degreesPerCSSPixel = degreesPerPixel;
  report.camera.wheelDelta = wheelDelta;
  async function rotateTo(desired) {
    let current = await state();
    for (let attempt = 0; attempt < 8; attempt++) {
      const error = angleError(desired, current.yawDegrees);
      if (Math.abs(error) < .01) break;
      current = await drag(Math.max(-width * .35, Math.min(width * .35, error / degreesPerPixel)));
    }
    assert(Math.abs(angleError(current.yawDegrees, desired)) < .03, 'Yaw did not converge');
    assert(Math.abs(current.pitchDegrees) < .01, 'Pitch drifted');
    assert(Math.abs(Math.hypot(...current.camera.position.map((value, i) => value - target[i]))
      - referenceDistance) < 1e-5, 'Camera target/distance changed');
    assert(Math.abs(current.camera.top / current.camera.zoom - halfExtent) < 1e-5, 'Camera extent changed');
    return current;
  }
  const started = performance.now();
  for (let index = 0; index < frameCount; index++) {
    const requestedYawDegrees = normalizedAngle(360 * index / frameCount);
    const current = await rotateTo(requestedYawDegrees);
    const before = await page.evaluate(() => globalThis.__inspectorCaptureProbe.renders);
    const downloadEvent = page.waitForEvent('download');
    await page.locator('#snapshot').click();
    const download = await downloadEvent;
    const path = resolve(output, `frame-${String(index).padStart(4, '0')}.png`);
    await download.saveAs(path);
    const after = await page.evaluate(() => ({camera: globalThis.__inspectorCaptureProbe.camera,
      renders: globalThis.__inspectorCaptureProbe.renders}));
    assert.deepEqual(after.camera.matrixWorld, current.camera.matrixWorld, 'Camera moved during capture');
    assert.deepEqual(after.camera.canvasPixels, [width, width]);
    assert(after.renders > before, 'Save PNG did not render');
    report.frames.push({index, path, sha256: hash(await readFile(path)), requestedYawDegrees,
      actualYawDegrees: current.yawDegrees, actualPitchDegrees: current.pitchDegrees,
      camera: after.camera, renderedCallsBeforeReadback: after.renders - before});
    await writeFile(reportPath, JSON.stringify(report, null, 2) + '\n');
    if (index % 24 === 0 || index === frameCount - 1)
      console.log('FRAME', index + 1, '/', frameCount, 'yaw', current.yawDegrees.toFixed(6));
  }
  report.captureElapsedSeconds = (performance.now() - started) / 1000;
  report.captureTimingNote = 'Offline browser capture including UI events and downloads; not interactive GPU FPS';
  await page.screenshot({path: resolve(output, 'inspector-ui.png')});
  assert.equal(hash(await readFile(html)), report.htmlSHA256, 'Generated viewer bytes changed');
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
