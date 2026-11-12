// After check_pbr_bake.py --sampler-input DIRECTORY, run this on the GPU used to bake.
import {resolve} from 'node:path';
import {chromium} from 'playwright';
import {chromiumOptions} from '../worker/chromium.mjs';
import {startServer} from '../worker/server.mjs';

if (!process.argv[2]) throw new Error('Expected directory containing points.json and orm.png');
const server = await startServer({assetDir:resolve(process.argv[2])});
let browser;
try {
  browser = await chromium.launch(chromiumOptions('webgl2'));
  const page = await browser.newPage();
  await page.goto(server.origin + '/bridge');
  const report = await page.evaluate(async () => {
    const points = await (await fetch('/asset/points.json')).json();
    const blob = await (await fetch('/asset/orm.png')).blob();
    const bitmap = await createImageBitmap(blob, {colorSpaceConversion:'none', premultiplyAlpha:'none'});
    const gl = document.createElement('canvas').getContext('webgl2');
    if (!gl) throw new Error('WebGL2 unavailable');
    gl.disable(gl.DITHER);
    const debug = gl.getExtension('WEBGL_debug_renderer_info');
    const adapter = debug && gl.getParameter(debug.UNMASKED_RENDERER_WEBGL);
    gl.bindTexture(gl.TEXTURE_2D, gl.createTexture());
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB8, gl.RGB, gl.UNSIGNED_BYTE, bitmap);
    bitmap.close();
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.generateMipmap(gl.TEXTURE_2D);
    const program = gl.createProgram();
    function addShader(type, source) {
      const shader = gl.createShader(type);
      gl.shaderSource(shader, '#version 300 es\n' + source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
      gl.attachShader(program, shader);
    }
    addShader(gl.VERTEX_SHADER, `
      void main() {
        vec2 v = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
        gl_Position = vec4(v * 2. - 1., 0., 1.);
      }`);
    addShader(gl.FRAGMENT_SHADER, `
      precision highp float;
      uniform sampler2D orm;
      uniform vec2 uv, dx, dy;
      uniform float lod;
      uniform bool useGradient;
      out vec4 color;
      void main() {
        color = useGradient ? textureGrad(orm, uv, dx, dy) : textureLod(orm, uv, lod);
      }`);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
    gl.useProgram(program);
    gl.viewport(0, 0, 1, 1);
    const uniform = name => gl.getUniformLocation(program, name);
    function read(useGradient) {
      gl.uniform1i(uniform('useGradient'), Number(useGradient));
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      const rgba = new Uint8Array(4);
      gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, rgba);
      if (gl.getError() !== gl.NO_ERROR || rgba[3] !== 255) throw new Error('Sampler readback failed');
      return [...rgba.slice(0, 3)];
    }
    const probes = points.map(({face, xy, uv, grad, lod, bakedORM}) => {
      gl.uniform2fv(uniform('uv'), uv);
      gl.uniform2fv(uniform('dx'), grad[0]);
      gl.uniform2fv(uniform('dy'), grad[1]);
      gl.uniform1f(uniform('lod'), lod);
      const gradientORM = read(true), explicitLodORM = read(false);
      return {face, xy, bakedORM, gradientORM, explicitLodORM,
        maxError:Math.max(...bakedORM.map((n, i) => Math.abs(n - gradientORM[i])))};
    });
    return {adapter, probes, maxError:Math.max(...probes.map(p => p.maxError))};
  });
  console.log(JSON.stringify(report, null, 2));
  if (report.probes.length !== 14 || report.maxError > 3) throw new Error('Source material sampler mismatch');
} finally {
  await browser?.close();
  await server.close();
}
