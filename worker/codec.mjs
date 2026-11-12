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

export async function sha256(bytes) {
  const hash = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes));
  return Array.from(hash, b => b.toString(16).padStart(2, '0')).join('');
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
