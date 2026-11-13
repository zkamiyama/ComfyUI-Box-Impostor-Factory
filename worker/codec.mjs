// Packed atlas rows follow the shader's increasing V. PNG rows follow image top to bottom.
export const FACES = Object.freeze([
  { id: '+X', slug: 'px', side: [1, 0, 0], up: [0, 1, 0], axis: 0, width: 2, height: 1 },
  { id: '-X', slug: 'nx', side: [-1, 0, 0], up: [0, 1, 0], axis: 0, width: 2, height: 1 },
  { id: '+Y', slug: 'py', side: [0, 1, 0], up: [0, 0, -1], axis: 1, width: 0, height: 2 },
  { id: '-Y', slug: 'ny', side: [0, -1, 0], up: [0, 0, 1], axis: 1, width: 0, height: 2 },
  { id: '+Z', slug: 'pz', side: [0, 0, 1], up: [0, 1, 0], axis: 2, width: 0, height: 1 },
  { id: '-Z', slug: 'nz', side: [0, 0, -1], up: [0, 1, 0], axis: 2, width: 0, height: 1 },
]);

export const MAP_TYPES = {
  colorMap: 'float16', normalMap: 'float16', intervalMap: 'float16',
  rangeMap: 'float16', materialMap: 'uint8', emissiveMap: 'float16',
};
export const MATERIAL_PROPERTIES = ['roughness', 'metalness', 'ao', 'emissive'];

export function dimensions(name, resolution) {
  if (!Object.hasOwn(MAP_TYPES, name)) throw new Error('Unknown texture ' + name);
  if (name === 'rangeMap') {
    const cells = Math.ceil(resolution / 8);
    return [3 * cells, cells];
  }
  return [3 * resolution, name === 'intervalMap' ? resolution : 2 * resolution];
}

export function compactPixels(raw, width, height, dtype, backend) {
  const Type = dtype === 'float16' ? Uint16Array : Uint8Array;
  if (!(raw instanceof Type)) throw new Error('Unexpected readback array for ' + dtype);
  const row = width * 4, bytes = row * Type.BYTES_PER_ELEMENT;
  if (backend !== 'webgpu' && backend !== 'webgl2') throw new Error('Unknown backend ' + backend);
  const stride = backend === 'webgpu' ? Math.ceil(bytes / 256) * 256 / Type.BYTES_PER_ELEMENT : row;
  if (raw.length !== (height - 1) * stride + row) throw new Error('Unexpected GPU row stride or payload length');
  const packed = new Type(row * height);
  for (let y = 0; y < height; ++y) {
    const src = backend === 'webgl2' ? height - 1 - y : y;
    packed.set(raw.subarray(src * stride, src * stride + row), y * row);
  }
  return packed;
}

const littleEndian = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;

export function encodeRaw(pixels) {
  if (pixels instanceof Uint8Array || littleEndian)
    return new Uint8Array(pixels.buffer, pixels.byteOffset, pixels.byteLength);
  const output = new Uint8Array(pixels.byteLength), v = new DataView(output.buffer);
  pixels.forEach((x, i) => v.setUint16(i * 2, x, true));
  return output;
}

export function decodeRaw(bytes, dtype) {
  if (!(bytes instanceof Uint8Array)) throw new Error('Expected texture bytes');
  if (dtype === 'uint8') return bytes;
  if (dtype !== 'float16' || bytes.byteLength % 2) throw new Error('Invalid half-float payload');
  if (littleEndian && bytes.byteOffset % 2 === 0)
    return new Uint16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 2);
  const out = new Uint16Array(bytes.byteLength / 2);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (let i = 0; i < out.length; i++) out[i] = view.getUint16(i * 2, true);
  return out;
}

// Pure JS fallback for opaque-origin chat frames. SubtleCrypto is unavailable
// in some sandboxed iframes, but portable Box checksums must still be enforced.
// FIPS 180-4 SHA-256, streaming 64-byte blocks to avoid allocating another atlas.
const SHA256_K = Uint32Array.from([
  0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
  0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
  0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
  0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
  0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
  0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
  0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
  0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2,
]);
const SHA256_INITIAL = [0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
                        0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19];
const rotr = (x,n) => (x >>> n) | (x << (32-n));

export function sha256Fallback(input) {
  const bytes=input instanceof Uint8Array?input:new Uint8Array(input);
  const state=Uint32Array.from(SHA256_INITIAL), w=new Uint32Array(64);
  function block(src, offset) {
    for(let i=0;i<16;i++){
      const n=offset+i*4;
      w[i]=((src[n]<<24)|(src[n+1]<<16)|(src[n+2]<<8)|src[n+3])>>>0;
    }
    for(let i=16;i<64;i++){
      const a=w[i-15],b=w[i-2];
      const s0=rotr(a,7)^rotr(a,18)^(a>>>3);
      const s1=rotr(b,17)^rotr(b,19)^(b>>>10);
      w[i]=(w[i-16]+s0+w[i-7]+s1)>>>0;
    }
    let [a,b,c,d,e,f,g,h]=state;
    for(let i=0;i<64;i++){
      const s1=rotr(e,6)^rotr(e,11)^rotr(e,25);
      const ch=(e&f)^(~e&g);
      const t1=(h+s1+ch+SHA256_K[i]+w[i])>>>0;
      const s0=rotr(a,2)^rotr(a,13)^rotr(a,22);
      const maj=(a&b)^(a&c)^(b&c);
      const t2=(s0+maj)>>>0;
      h=g;g=f;f=e;e=(d+t1)>>>0;d=c;c=b;b=a;a=(t1+t2)>>>0;
    }
    state[0]=(state[0]+a)>>>0;state[1]=(state[1]+b)>>>0;
    state[2]=(state[2]+c)>>>0;state[3]=(state[3]+d)>>>0;
    state[4]=(state[4]+e)>>>0;state[5]=(state[5]+f)>>>0;
    state[6]=(state[6]+g)>>>0;state[7]=(state[7]+h)>>>0;
  }
  const whole=bytes.length-(bytes.length%64);
  for(let i=0;i<whole;i+=64) block(bytes,i);
  const remaining=bytes.length-whole;
  const tail=new Uint8Array(remaining<56?64:128);
  tail.set(bytes.subarray(whole));
  tail[remaining]=0x80;
  const view=new DataView(tail.buffer);
  view.setUint32(tail.length-8,Math.floor(bytes.length/0x20000000),false);
  view.setUint32(tail.length-4,(bytes.length*8)>>>0,false);
  for(let i=0;i<tail.length;i+=64) block(tail,i);
  return Array.from(state,x=>x.toString(16).padStart(8,'0')).join('');
}

export async function sha256(bytes) {
  if(globalThis.crypto?.subtle){
    const hash = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes));
    return Array.from(hash, b => b.toString(16).padStart(2, '0')).join('');
  }
  return sha256Fallback(bytes);
}

export function safePath(path) {
  return typeof path === 'string' && /^[A-Za-z0-9_+.\/-]+$/.test(path) &&
    path.split('/').every(part => part && part !== '.' && part !== '..');
}

export function validateDescriptor(d, name, resolution) {
  const [w,h] = dimensions(name, resolution);
  const type = MAP_TYPES[name], row = w * 4 * (type === 'uint8' ? 1 : 2);
  const filter = name === 'rangeMap' ? 'nearest' : 'linear';
  if (!d || !safePath(d.path) || d.width !== w || d.height !== h ||
      d.dtype !== type || d.channels !== 4 || d.endian !== 'little' ||
      d.rowOrder !== 'increasingShaderV' || d.rowBytes !== row ||
      d.byteLength !== row * h || !/^[0-9a-f]{64}$/.test(d.sha256) ||
      d.minFilter !== filter || d.magFilter !== filter ||
      d.wrap !== 'clamp-to-edge' || d.mipmaps !== false)
    throw new Error('Invalid ' + name + ' descriptor');
}

export function parseGLB(bytes) {
  if (!(bytes instanceof Uint8Array)) bytes = new Uint8Array(bytes);
  const v = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (bytes.length < 20 || v.getUint32(0,true) !== 0x46546c67 ||
      v.getUint32(4,true) !== 2 || v.getUint32(8,true) !== bytes.length)
    throw new Error('Expected a binary glTF 2.0 file');
  const n = v.getUint32(12,true);
  if (v.getUint32(16,true) !== 0x4e4f534a || n % 4 || n > bytes.length - 20)
    throw new Error('Invalid GLB JSON chunk');
  const json = JSON.parse(new TextDecoder('utf-8',{fatal:true}).decode(bytes.subarray(20,20+n)));
  for (const item of [...(json.buffers ?? []), ...(json.images ?? [])])
    if (item.uri !== undefined && (typeof item.uri !== 'string' || !item.uri.startsWith('data:')))
      throw new Error('External GLB resources must be embedded');
  let offset = 20+n;
  while (offset < bytes.length) {
    if (offset + 8 > bytes.length) throw new Error('Truncated GLB chunk header');
    const size = v.getUint32(offset,true);
    if (size % 4 || size > bytes.length-offset-8) throw new Error('Truncated GLB chunk');
    offset += 8+size;
  }
  return {json, tail:bytes.subarray(20+n)};
}

export function wrapGLB({json, tail}, matrix) {
  const scene = json.scenes?.[json.scene ?? 0];
  if (!Array.isArray(json.nodes) || !scene?.nodes?.length) throw new Error('GLB has no scene roots');
  const roots = [...scene.nodes];
  scene.nodes = [json.nodes.length];
  json.nodes.push({name:'BI canonical transform',matrix:[...matrix],children:roots});
  const encoded = new TextEncoder().encode(JSON.stringify(json));
  const n = Math.ceil(encoded.length/4)*4;
  const out = new Uint8Array(20+n+tail.length);
  const view = new DataView(out.buffer);
  [0x46546c67,2,out.length,n,0x4e4f534a].forEach((value,i)=>view.setUint32(i*4,value,true));
  out.fill(0x20,20,20+n); out.set(encoded,20); out.set(tail,20+n);
  return out;
}
