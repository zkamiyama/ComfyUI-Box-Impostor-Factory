import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdir, mkdtemp, rm, symlink, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import test from 'node:test';
import { FACES, MAP_TYPES, dimensions } from '../worker/codec.mjs';
import { inspectAsset } from '../worker/cli.mjs';
import { makeFixture } from './make_fixture.mjs';

const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const r = 8;
const png = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO0lRjcAAAAASUVORK5CYII=',
  'base64'
);

async function fixture(t) {
  const dir = await mkdtemp(join(tmpdir(),'bi-asset-inspection-'));
  t.after(() => rm(dir,{recursive:true,force:true}));
  async function put(path,bytes) {
    const dst = join(dir,path);
    await mkdir(dirname(dst),{recursive:true});
    await writeFile(dst,bytes);
    return sha(bytes);
  }
  const textures = {};
  for (const name of ['colorMap','normalMap','intervalMap','rangeMap']) {
    const [width,height] = dimensions(name,r);
    const dtype = MAP_TYPES[name];
    const byteLength = width*height*4*(dtype === 'uint8' ? 1 : 2);
    const path = 'maps/'+name+'.bin';
    const digest = await put(path,Buffer.alloc(byteLength,0));
    const filter = name === 'rangeMap' ? 'nearest' : 'linear';
    textures[name] = {path,width,height,dtype,byteLength,sha256:digest,channels:4,
      endian:'little',rowOrder:'increasingShaderV',rowBytes:width*4*(dtype==='uint8'?1:2),
      minFilter:filter,magFilter:filter,wrap:'clamp-to-edge',mipmaps:false};
  }
  const proxy = 'geometry/proxy.glb', proxySha256 = await put(proxy,makeFixture());
  const faces = {};
  for (const face of FACES) {
    const guide = 'guides/'+face.slug+'.png';
    const depth = 'guides/'+face.slug+'.depth16f.bin';
    faces[face.id] = {guide,guideSha256:await put(guide,png),
      depth,depthSha256:await put(depth,Buffer.alloc(r*r*2))};
  }
  const manifest = {schemaVersion:1,layout:'box',asset_id:'a'.repeat(64),resolution:r,
    maps:[],textures,baseColor:{...textures.colorMap},proxy,proxySha256,
    faceOrder:FACES.map(f=>f.id),faces};
  const save = () => writeFile(join(dir,'manifest.json'),JSON.stringify(manifest));
  await save();
  return {dir,manifest,save,put};
}

test('standalone inspection validates six guides, half-float depth and texture hashes',async t=>{
  const {dir,manifest} = await fixture(t);
  assert.deepEqual(await inspectAsset(dir),manifest);
  const guide = join(dir,manifest.faces['+X'].guide);
  await writeFile(guide,Buffer.from('tampered'));
  await assert.rejects(inspectAsset(dir),/Invalid guide \+X/);
});

test('missing or truncated depth sidecars are not accepted',async t=>{
  const {dir,manifest,put} = await fixture(t);
  const depth = manifest.faces['-Y'].depth;
  await put(depth,Buffer.alloc(5));
  await assert.rejects(inspectAsset(dir),/Invalid depth -Y/);
  await rm(join(dir,depth));
  await assert.rejects(inspectAsset(dir),/ENOENT/);
});

test('inconsistent maps, face ordering and duplicate texture names are rejected',async t=>{
  const {dir,manifest,save} = await fixture(t);
  manifest.maps = ['roughness'];
  await save();
  await assert.rejects(inspectAsset(dir),/Inconsistent texture list/);
  manifest.maps = [];
  manifest.faceOrder.reverse();
  await save();
  await assert.rejects(inspectAsset(dir),/Invalid six-face layout/);
  manifest.faceOrder.reverse();
  manifest.textures.normalMap.path = manifest.textures.colorMap.path;
  await save();
  await assert.rejects(inspectAsset(dir),/Duplicate texture path/);
});

test('a declared reference camera crop must pass its recorded checksum',async t=>{
  const {dir,manifest,save,put} = await fixture(t);
  const image = 'reference/cropped.png';
  const digest = await put(image,png);
  manifest.reference = {camera:{image,sha256:digest,width:1,height:1}};
  await save();
  assert.deepEqual((await inspectAsset(dir)).reference,manifest.reference);
  await put(image,Buffer.from('corrupted'));
  await assert.rejects(inspectAsset(dir),/Invalid reference camera/);
});

test('legacy guide metadata without checksums is readable, but the guide must exist',async t=>{
  const {dir,manifest,save} = await fixture(t);
  delete manifest.faces['+Z'].guideSha256;
  await save();
  await inspectAsset(dir);
  await rm(join(dir,manifest.faces['+Z'].guide));
  await assert.rejects(inspectAsset(dir),/ENOENT/);
});

test('symbolic links may not reference files outside the selected Box directory',async t=>{
  if (process.platform === 'win32') return t.skip('Windows symlink creation needs developer privileges');
  const {dir,manifest} = await fixture(t);
  const outside = await mkdtemp(join(tmpdir(),'bi-external-file-'));
  t.after(() => rm(outside,{recursive:true,force:true}));
  const file = join(outside,'guide.png');
  await writeFile(file,png);
  const member = join(dir,manifest.faces['+Z'].guide);
  await rm(member);
  await symlink(file,member);
  await assert.rejects(inspectAsset(dir),/escapes its directory/);
});

test('a changed proxy GLB cannot be accepted under the original digest',async t=>{
  const {dir,manifest} = await fixture(t);
  await writeFile(join(dir,manifest.proxy),makeFixture().subarray(0,32));
  await assert.rejects(inspectAsset(dir),/Invalid proxy GLB/);
});
