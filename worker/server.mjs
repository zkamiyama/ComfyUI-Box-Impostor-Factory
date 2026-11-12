import { createServer } from 'node:http';
import { createReadStream, createWriteStream } from 'node:fs';
import { stat, realpath, mkdir } from 'node:fs/promises';
import { pipeline } from 'node:stream/promises';
import { basename, dirname, extname, join, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('../', import.meta.url));
const imports = {
  three: '/vendor/three/build/three.webgpu.js',
  'three/webgpu': '/vendor/three/build/three.webgpu.js',
  'three/tsl': '/vendor/three/build/three.tsl.js',
  'three/addons/': '/vendor/three/examples/jsm/',
  'three-impostor': '/vendor/three-impostor/Impostor.js',
};
const mime = {'.js':'text/javascript','.mjs':'text/javascript','.html':'text/html',
  '.json':'application/json','.png':'image/png','.glb':'model/gltf-binary','.css':'text/css'};
const bootstrap = '<!doctype html><meta charset="utf-8"><script type="importmap">' +
  JSON.stringify({imports}) + '</script>';

export function relativeFile(name) {
  if (!name || name.includes('\\') || name.includes('\0') ||
      name.split('/').some(part => !part || part === '.' || part === '..')) {
    throw new Error('Invalid asset path');
  }
  return name;
}
async function inside(base, relative) {
  const dir = await realpath(base);
  const path = await realpath(join(dir, relativeFile(relative)));
  if (!path.startsWith(dir + sep)) throw new Error('File outside asset directory');
  return path;
}
async function sendFile(res, path, alterHTML = false) {
  const info = await stat(path);
  if (!info.isFile()) throw new Error('Expected file');
  res.setHeader('Content-Type', mime[extname(path)] ?? 'application/octet-stream');
  if (alterHTML) {
    const { readFile } = await import('node:fs/promises');
    res.end((await readFile(path, 'utf8')).replaceAll('__BI_BASE__', ''));
    return;
  }
  res.setHeader('Content-Length', info.size);
  await pipeline(createReadStream(path), res);
}

// Input, temporary output, and assets being previewed are explicit capabilities of this instance.
export async function startServer({ input, staging, assetDir, port = 0 } = {}) {
  let origin;
  const server = createServer(async (req,res) => {
    try {
      if (req.headers.host !== new URL(origin).host) {
        res.writeHead(403).end(); return;
      }
      const path = decodeURIComponent((req.url ?? '/').split('?')[0]);
      res.setHeader('Cache-Control','no-store');
      res.setHeader('X-Content-Type-Options','nosniff');
      if (req.method === 'PUT' && staging && path.startsWith('/job/files/')) {
        if (req.headers.origin !== origin) { res.writeHead(403).end(); return; }
        const name = relativeFile(path.slice('/job/files/'.length));
        if (!/^(canonical\.glb|(?:maps|guides|geometry|previews|qa|views|reference)\/[A-Za-z0-9+_./-]+)$/.test(name)) {
          throw new Error('Invalid generated path');
        }
        const dst = join(staging,name);
        await mkdir(dirname(dst),{recursive:true});
        await pipeline(req,createWriteStream(dst,{flags:'wx'}));
        res.writeHead(201).end(); return;
      }
      if (req.method !== 'GET') { res.writeHead(405).end(); return; }
      if (path === '/bridge') { res.setHeader('Content-Type','text/html');res.end(bootstrap);return; }
      if (path === '/') { res.writeHead(302,{Location:'/web/viewer.html?manifest=/asset/manifest.json'}).end();return; }
      if (path === '/job/input.glb' && input) { await sendFile(res,input);return; }
      if (path.startsWith('/asset/') && assetDir) { await sendFile(res,await inside(assetDir,path.slice(7)));return; }
      for (const [prefix,base] of [
        ['/worker/',join(root,'worker')], ['/web/',join(root,'web')],
        ['/vendor/three/build/',join(root,'node_modules/three/build')],
        ['/vendor/three/examples/jsm/',join(root,'node_modules/three/examples/jsm')],
        ['/vendor/three-impostor/',join(root,'node_modules/three-impostor')],
      ]) {
        if (path.startsWith(prefix)) {
          const file = await inside(base,path.slice(prefix.length));
          await sendFile(res,file,basename(file)==='viewer.html'); return;
        }
      }
      res.writeHead(404).end();
    } catch (error) {
      if (!res.headersSent) res.writeHead(error.code === 'ENOENT' ? 404 : 400);
      res.end('Unavailable path');
    }
  });
  await new Promise((ok,fail)=>{server.once('error',fail);server.listen(port,'127.0.0.1',ok);});
  origin = 'http://127.0.0.1:' + server.address().port;
  return {origin, async close() {
    const pending = new Promise((ok,fail)=>server.close(e=>e?fail(e):ok()));
    server.closeAllConnections(); await pending;
  }};
}
