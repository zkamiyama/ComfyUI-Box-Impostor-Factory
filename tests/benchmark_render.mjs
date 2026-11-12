// node tests/benchmark_render.mjs ASSET_DIR [webgpu|webgl2] > timing.json
import { resolve } from 'node:path';
import { chromium } from 'playwright';
import { chromiumOptions } from '../worker/chromium.mjs';
import { startServer } from '../worker/server.mjs';

const [asset, backend = 'webgpu'] = process.argv.slice(2);
if (!asset || !['webgpu', 'webgl2'].includes(backend)) {
  throw new Error('Expected saved Box asset directory and optional webgpu or webgl2 backend');
}
const server = await startServer({ assetDir: resolve(asset) });
let browser;
try {
  browser = await chromium.launch(chromiumOptions(backend));
  const page = await browser.newPage();
  await page.goto(server.origin + '/bridge');
  const report = await page.evaluate(async backend => {
    const THREE = await import('three/webgpu');
    const { makeRenderer, readAsset } = await import('/web/asset_io.mjs');
    const { loadProxy, restoreBox, disposeProxy } = await import('/worker/runtime.mjs');
    const { manifest, files } = await readAsset('/asset/manifest.json', { proxy:true });
    const renderer = await makeRenderer(backend);
    const gl = backend === 'webgl2' ? renderer.getContext() : null;
    const device = renderer.backend.device;
    const glInfo = gl?.getExtension('WEBGL_debug_renderer_info');
    const adapter = gl
      ? { renderer:glInfo && gl.getParameter(glInfo.UNMASKED_RENDERER_WEBGL) }
      : { vendor:device.adapterInfo.vendor, architecture:device.adapterInfo.architecture,
          description:device.adapterInfo.description, isFallbackAdapter:device.adapterInfo.isFallbackAdapter };
    let mesh, box;
    try {
      mesh = await loadProxy(files.get(manifest.proxy));
      box = await restoreBox(manifest, files, renderer);
      const center = new THREE.Vector3().fromArray(manifest.bakeBox.min)
        .lerp(new THREE.Vector3().fromArray(manifest.bakeBox.max), 0.5);
      const radius = center.distanceTo(new THREE.Vector3().fromArray(manifest.bakeBox.max));
      const span = radius * 1.08;
      const rows = [];
      const batches = 7, frames = 100;
      renderer.setClearColor(0, 0);
      for (const size of [256, 1024]) {
        const target = new THREE.RenderTarget(size, size);
        try {
          renderer.setSize(size, size, false);
          renderer.setRenderTarget(target);
          // Reading a single aligned scanline waits for both the render pass and its copy.
          const fence = () => renderer.readRenderTargetPixelsAsync(target, 0, 0, 64, 1);
          for (const side of [1, 4]) {
            const extent = span * side;
            const camera = new THREE.OrthographicCamera(-extent, extent, extent, -extent, radius * 0.5, radius * 5);
            camera.position.set(1, 0.5, 1).normalize().multiplyScalar(radius * 3).add(center);
            camera.lookAt(center);
            camera.updateMatrixWorld();
            const across = new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 0);
            const up = new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 1);
            for (const [kind, source] of [['mesh', mesh], ['box', box.mesh]]) {
              const scene = new THREE.Scene();
              scene.add(new THREE.HemisphereLight(0xf3f5ff, 0xc2bdb5, 1.7));
              const sun = new THREE.DirectionalLight(0xffffff, 2);
              sun.position.set(2, 4, 3);
              scene.add(sun);
              for (let y = 0; y < side; y++) for (let x = 0; x < side; x++) {
                const object = source.clone();
                object.position.addScaledVector(across, (2*x + 1 - side) * span)
                  .addScaledVector(up, (2*y + 1 - side) * span);
                scene.add(object);
              }
              await renderer.compileAsync(scene, camera);
              for (let i = 0; i < frames; i++) renderer.render(scene, camera);
              await fence();
              const pixels = await renderer.readRenderTargetPixelsAsync(target, 0, 0, size, size);
              let foreground = 0;
              for (let i = 3; i < pixels.length; i += 4) if (pixels[i] >= 128) foreground++;
              if (!foreground || foreground === size * size) throw new Error('Invalid foreground coverage');
              const batchMsPerFrame = [];
              for (let b = 0; b < batches; b++) {
                const start = performance.now();
                for (let i = 0; i < frames; i++) renderer.render(scene, camera);
                await fence();
                batchMsPerFrame.push((performance.now() - start) / frames);
              }
              const medianMsPerFrame = [...batchMsPerFrame].sort((a, b) => a - b)[Math.floor(batches / 2)];
              rows.push({size, instances:side*side, kind, foreground, medianMsPerFrame, batchMsPerFrame});
              scene.clear();
            }
          }
        } finally {
          renderer.setRenderTarget(null);
          target.dispose();
        }
      }
      return {backend, adapter, resolution:manifest.resolution, proxySha256:manifest.proxySha256,
        sampling:'standard', rayStepLimit:box.mesh.material.steps, batches, framesPerBatch:frames,
        metric:'steady-state offscreen wall time per frame including CPU submission and one synchronized 64x1 readback per batch; separate object draws share geometry and material',
        rows};
    } finally {
      if (mesh) disposeProxy(mesh);
      box?.dispose();
      renderer.dispose();
    }
  }, backend);
  console.log(JSON.stringify({browserVersion:browser.version(), ...report}, null, 2));
} finally {
  await browser?.close();
  await server.close();
}
