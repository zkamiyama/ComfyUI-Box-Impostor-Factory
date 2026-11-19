import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { inflateSync } from 'node:zlib';
import { chromium } from 'playwright';

// Capture an unchanged view.py output through the Inspector's real renderer.
// Camera rotation uses its existing OrbitControls; PNGs are actual canvas readbacks.
const configPath = resolve(process.argv[2] ?? '');
assert(process.argv[2], 'Usage: node capture_field_color.mjs CONFIG.json');
const configBytes = await readFile(configPath);
const config = JSON.parse(configBytes);
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const html = resolve(config.html);
const output = resolve(config.output);
const bytes = await readFile(html);
assert.equal(hash(bytes), config.htmlSHA256, 'Expected exact generated Inspector HTML');
const match = bytes.toString('utf8').match(/<script id="scene-payload"[^>]*>([\s\S]*?)<\/script>/);
assert(match, 'Inspector scene-payload is missing');
const packed = Buffer.from(match[1].trim(), 'base64');
const payload = JSON.parse((packed[0] === 0x78 ? inflateSync(packed) : packed).toString());
assert(payload.nativeBox, 'Native Box payload is required');
assert.equal(payload.assetId, config.assetId, 'Unexpected asset');
const width = config.pixels ?? 640;
const anglesDegrees = config.anglesDegrees;
if (anglesDegrees !== undefined) {
  assert(Array.isArray(anglesDegrees) && anglesDegrees.length >= 1 && anglesDegrees.length <= 1440,
    'anglesDegrees must be a nonempty array of at most 1440 angles');
  assert(anglesDegrees.every(Number.isFinite), 'anglesDegrees must contain finite numbers');
}
const frames = anglesDegrees?.length ?? config.frames ?? 240;
const fps = config.fps ?? 24;
assert(Number.isInteger(width) && width >= 128 && width <= 2048);
assert(Number.isInteger(frames) && frames >= (anglesDegrees ? 1 : 2) && frames <= 1440);
assert(Number.isFinite(fps) && fps > 0);
const bounds = config.referenceBounds ?? payload.bounds;
const midpoint = b => b.min.map((v, i) => (v + b.max[i]) / 2);
const radius = b => Math.hypot(...b.max.map((v, i) => v - b.min[i])) / 2;
const sourceRadius = radius(payload.bounds);
const sourceTarget = midpoint(payload.bounds);
const target = config.framing?.target ?? midpoint(bounds);
const fitDistance = sourceRadius * 1.05 / Math.sin(20 * Math.PI / 180);
const fitHalfExtent = fitDistance * Math.tan(20 * Math.PI / 180);
const halfExtent = config.framing?.halfExtent ?? fitHalfExtent;
assert(target.length === 3 && target.every(Number.isFinite));
assert(Number.isFinite(halfExtent) && halfExtent > 0);
const expectedDistance = halfExtent / Math.tan(20 * Math.PI / 180);
const startDegrees = config.startDegrees ?? 0;
const sweepDegrees = config.sweepDegrees ?? 360;
const normalizedAngle = a => ((a + 180) % 360 + 360) % 360 - 180;
const angleError = (a, b) => normalizedAngle(a - b);
const report = {
  config: configPath, configSHA256: hash(configBytes),
  scriptSHA256: hash(await readFile(new URL(import.meta.url))),
  html, htmlSHA256: config.htmlSHA256, assetId: payload.assetId,
  sourceFaceResolution: payload.nativeBox.manifest.resolution,
  display: payload.nativeBox.display, backendRequested: config.backend ?? 'auto',
  camera: {projection: 'orthographic', pixels: [width, width], target,
    halfExtent, expectedDistance, startDegrees, sweepDegrees, anglesDegrees: anglesDegrees ?? null},
  framesExpected: frames, playbackFPS: fps, durationSeconds: frames / fps,
  capture: 'Official Inspector renderer, original HTML, actual canvas PNG readback',
  captureScheduling: config.deferIntermediateRenders
    ? 'Defer intermediate camera/UI draws; render each recorded frame once before readback'
    : 'Unmodified redraw scheduling',
  interpolation: 'No optical flow, no frame interpolation, no generated frames',
  frames: [], errors: [], completed: false,
};
await mkdir(output, {recursive: true});
const reportPath = resolve(output, 'capture.json');
const browserArgs = ['--disable-dev-shm-usage'];
if (config.backend === 'swiftshader') browserArgs.push(
  '--disable-gpu', '--disable-features=Vulkan,WebGPU', '--use-gl=angle',
  '--use-angle=swiftshader', '--enable-unsafe-swiftshader');
if (config.backend === 'webgl2') browserArgs.push('--disable-features=WebGPU');
const launchOptions = {headless: config.headless !== false, args: browserArgs};
if (config.chromiumExecutable || process.env.BI_CHROMIUM)
  launchOptions.executablePath = config.chromiumExecutable || process.env.BI_CHROMIUM;
const browser = await chromium.launch(launchOptions);
try {
  const page = await browser.newPage({viewport: {width, height: width}, deviceScaleFactor: 1});
  page.setDefaultTimeout(config.timeoutMilliseconds ?? 120000);
  page.on('pageerror', e => report.errors.push('page: ' + String(e)));
  page.on('console', m => {if (m.type() === 'error') report.errors.push('console: ' + m.text());});
  await page.addInitScript(({defer}) => {
    let wrapped;
    const probe = globalThis.__turntableProbe = {renders: 0, suppressed: 0, capturePending: false};
    Object.defineProperty(globalThis, 'InspectorThree', {
      configurable: true, get() {return wrapped;}, set(value) {
        wrapped = new Proxy(value, {get(target, key, receiver) {
          if (key !== 'createThreeViewer') return Reflect.get(target, key, receiver);
          return async function (...args) {
            const api = await Reflect.apply(target.createThreeViewer, target, args);
            probe.api = api;
            const renderer = api.getRenderer(), render = renderer.render;
            renderer.render = function (scene, camera, ...tail) {
              let result;
              if (defer && !probe.capturePending) {
                camera.updateMatrixWorld();
                probe.suppressed++;
              } else {
                result = Reflect.apply(render, this, [scene, camera, ...tail]);
                probe.renders++;
              }
              probe.camera = {
                type: camera.type, position: camera.position.toArray(),
                quaternion: camera.quaternion.toArray(), matrixWorld: camera.matrixWorld.toArray(),
                projectionMatrix: camera.projectionMatrix.toArray(), zoom: camera.zoom,
                top: camera.top, bottom: camera.bottom, left: camera.left, right: camera.right,
              };
              return result;
            };
            return api;
          };
        }});
      },
    });
  }, {defer: config.deferIntermediateRenders === true});
  await page.goto(pathToFileURL(html).href, {waitUntil: 'domcontentloaded', timeout: 180000});
  await page.waitForFunction(() => globalThis.__turntableProbe?.api?.isBox &&
    document.getElementById('render-state')?.textContent.startsWith('Three.js /'),
    null, {timeout: 180000});
  report.backend = await page.locator('#render-state').innerText();
  report.stageState = await page.locator('#stage-state').innerText();
  report.browserVersion = await browser.version();
  report.graphicsRenderer = await page.evaluate(() => {
    const renderer = globalThis.__turntableProbe.api.getRenderer();
    if (renderer.backend?.isWebGPUBackend) return 'WebGPU';
    const gl = renderer.domElement.getContext('webgl2');
    const info = gl?.getExtension('WEBGL_debug_renderer_info');
    return info ? gl.getParameter(info.UNMASKED_RENDERER_WEBGL) : 'WebGL2';
  });
  if (await page.locator('#details-toggle').getAttribute('aria-expanded') === 'true')
    await page.locator('#details-toggle').click();
  for (let i = 0; i < 4; i++) {
    const rect = await page.locator('#viewport').boundingBox();
    if (Math.abs(rect.width - width) < 0.01 && Math.abs(rect.height - width) < 0.01) break;
    const v = page.viewportSize();
    await page.setViewportSize({width: Math.round(v.width + width - rect.width),
      height: Math.round(v.height + width - rect.height)});
  }
  const rect = await page.locator('#viewport').boundingBox();
  assert(Math.abs(rect.width - width) < .01 && Math.abs(rect.height - width) < .01);
  const cx = rect.x + width / 2, cy = rect.y + width / 2;
  await page.evaluate(() => {
    const api = globalThis.__turntableProbe.api;
    api.projection(false); api.fit(); api.view('front'); api.draw();
  });
  const wheelDelta = -100 * Math.log(halfExtent / fitHalfExtent) / Math.log(.95);
  await page.mouse.move(cx, cy);
  if (Math.abs(wheelDelta) > 1e-8) await page.mouse.wheel(0, wheelDelta);
  await page.evaluate(() => new Promise(resolve => requestAnimationFrame(resolve)));
  await page.evaluate(() => {const api=globalThis.__turntableProbe.api; api.projection(true); api.draw();});
  const delta = target.map((v, i) => v - sourceTarget[i]);
  const pxPerUnit = width / (2 * halfExtent);
  for (const pan of [
    {view: 'front', dx: -delta[0] * pxPerUnit, dy: delta[1] * pxPerUnit},
    {view: 'right', dx: delta[2] * pxPerUnit, dy: 0},
  ]) {
    await page.evaluate(view => globalThis.__turntableProbe.api.view(view), pan.view);
    const moves = Math.ceil(Math.max(Math.abs(pan.dx), Math.abs(pan.dy)) / (width * .3));
    for (let move = 0; move < moves; move++) {
      await page.mouse.move(cx, cy);
      await page.mouse.down({button: 'right'});
      await page.mouse.move(cx + pan.dx/moves, cy + pan.dy/moves);
      await page.mouse.up({button: 'right'});
    }
  }
  await page.evaluate(() => {const api=globalThis.__turntableProbe.api; api.view('front'); api.draw();});
  async function state() {
    const camera = await page.evaluate(() => {
      const probe = globalThis.__turntableProbe; probe.api.draw(); return probe.camera;
    });
    assert.equal(camera.type, 'OrthographicCamera');
    const d = [camera.matrixWorld[8], camera.matrixWorld[9], camera.matrixWorld[10]];
    return {camera, yawDegrees: Math.atan2(d[0], d[2]) * 180 / Math.PI,
      pitchDegrees: Math.asin(Math.max(-1,Math.min(1,d[1]/Math.hypot(...d)))) * 180 / Math.PI};
  }
  async function drag(dx) {
    assert(Math.abs(dx) < width * .4, 'Drag must stay inside canvas');
    await page.mouse.move(cx, cy);
    await page.mouse.down({button: 'left'});
    await page.mouse.move(cx + dx, cy);
    await page.mouse.up({button: 'left'});
    return state();
  }
  const initial = await state();
  const calibrated = await drag(10);
  const degreesPerPixel = angleError(calibrated.yawDegrees, initial.yawDegrees) / 10;
  assert(Math.abs(degreesPerPixel) > .05 && Math.abs(degreesPerPixel) < 3);
  await page.evaluate(() => {const api=globalThis.__turntableProbe.api; api.view('front'); api.draw();});
  report.camera.degreesPerCSSPixel = degreesPerPixel;
  report.camera.wheelDelta = wheelDelta;
  async function rotateTo(desired) {
    let current = await state();
    for (let attempt = 0; attempt < 8; attempt++) {
      const error = angleError(desired, current.yawDegrees);
      if (Math.abs(error) < .01) break;
      const dx = Math.max(-width * .35, Math.min(width * .35, error / degreesPerPixel));
      current = await drag(dx);
    }
    assert(Math.abs(angleError(current.yawDegrees, desired)) < .03, 'Yaw did not converge');
    assert(Math.abs(current.pitchDegrees) < .01, 'Turntable pitch drifted');
    const distance = Math.hypot(...current.camera.position.map((v, i) => v - target[i]));
    assert(Math.abs(distance - expectedDistance) < 1e-5, 'Camera target/distance changed');
    assert(Math.abs(current.camera.top/current.camera.zoom-halfExtent) < 1e-5, 'Camera extent changed');
    return current;
  }
  const captureStart = performance.now();
  for (let i = 0; i < frames; i++) {
    const desired = normalizedAngle(anglesDegrees?.[i] ?? (startDegrees + sweepDegrees * i / frames));
    const current = await rotateTo(desired);
    const raster = await page.evaluate(async () => {
      const p = globalThis.__turntableProbe, renderer = p.api.getRenderer();
      const count = p.renders;
      p.capturePending = true;
      try {
        p.api.draw();
        if (renderer.backend?.isWebGPUBackend) await renderer.backend.device.queue.onSubmittedWorkDone();
        else renderer.domElement.getContext('webgl2').finish();
        return {png: renderer.domElement.toDataURL('image/png'), camera: p.camera,
          width: renderer.domElement.width, height: renderer.domElement.height,
          renderedCalls: p.renders-count};
      } finally {p.capturePending=false;}
    });
    assert.equal(raster.width, width); assert.equal(raster.height, width);
    assert(raster.renderedCalls >= 1, 'No actual renderer call before readback');
    assert.deepEqual(raster.camera.matrixWorld, current.camera.matrixWorld);
    const png = Buffer.from(raster.png.split(',')[1], 'base64');
    const path = resolve(output, `frame-${String(i).padStart(4, '0')}.png`);
    await writeFile(path, png);
    report.frames.push({index: i, path, sha256: hash(png),
      requestedYawDegrees: desired, actualYawDegrees: current.yawDegrees,
      actualPitchDegrees: current.pitchDegrees, camera: raster.camera,
      renderedCallsBeforeReadback: raster.renderedCalls});
    if (i % 24 === 0 || i === frames - 1) {
      await writeFile(reportPath, JSON.stringify(report, null, 2) + '\n');
      console.log('FRAME', i + 1, '/', frames, 'yaw', current.yawDegrees.toFixed(5));
    }
  }
  report.captureElapsedSeconds = (performance.now() - captureStart) / 1000;
  report.renderScheduling = await page.evaluate(() => {
    const p=globalThis.__turntableProbe; return {rendered:p.renders,suppressed:p.suppressed};
  });
  report.captureTimingNote = 'Offline capture wall time, including readback/automation; not interactive GPU FPS';
  await page.screenshot({path: resolve(output, 'inspector-ui.png')});
  assert.equal(hash(await readFile(html)), config.htmlSHA256, 'HTML was modified');
  assert.deepEqual(report.errors, []);
  report.completed = true;
} finally {
  await browser.close();
  await writeFile(reportPath, JSON.stringify(report, null, 2) + '\n');
}
console.log('COMPLETE', reportPath);
