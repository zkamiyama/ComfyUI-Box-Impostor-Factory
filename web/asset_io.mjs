import { sha256 } from '../worker/codec.mjs';
import { WebGPURenderer } from 'three/webgpu';

export async function makeRenderer(backend = 'auto', options = {}) {
  if (!['auto','webgpu','webgl','webgl2'].includes(backend)) throw new Error('Unknown graphics backend: ' + backend);
  const renderer = new WebGPURenderer({ antialias:false, alpha:true, ...options,
    forceWebGL:backend === 'webgl2' || backend === 'webgl' });
  await renderer.init();
  if (backend === 'webgpu' && !renderer.backend.isWebGPUBackend) {
    renderer.dispose();
    throw new Error('WebGPU was requested but is unavailable');
  }
  return renderer;
}

export function backendName(renderer) { return renderer.backend.isWebGPUBackend ? 'webgpu' : 'webgl2'; }

export async function fetchChecked(url) {
  const res = await fetch(url, {cache:'no-store', mode:'same-origin'});
  if (!res.ok) throw new Error('HTTP ' + res.status + ' loading ' + new URL(url, location.href).pathname);
  return res;
}

export function assetURL(manifestURL, path) {
  if (typeof path !== 'string' || !path || path.split('/').some(part => !part || part === '.' || part === '..') ||
      /[\\:%?#]/.test(path)) throw new Error('Invalid asset member path');
  return new URL(path, manifestURL);
}

export async function readAsset(manifestLocation, {proxy = false} = {}) {
  const url = new URL(manifestLocation, location.href);
  if (url.origin !== location.origin) throw new Error('The asset manifest must be on this server');
  const manifest = await (await fetchChecked(url)).json();
  const metadata = Object.values(manifest.textures);
  const paths = metadata.map(m => m.path);
  if (manifest.baseColor?.path) paths.push(manifest.baseColor.path);
  if (proxy) paths.push(manifest.proxy);
  const files = new Map(await Promise.all([...new Set(paths)].map(async path => [
    path, new Uint8Array(await (await fetchChecked(assetURL(url, path))).arrayBuffer())
  ])));
  if (proxy && await sha256(files.get(manifest.proxy)) !== manifest.proxySha256)
    throw new Error('Proxy GLB checksum mismatch');
  return {manifest, files, url};
}
