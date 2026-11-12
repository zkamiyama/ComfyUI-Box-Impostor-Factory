// Self-contained UV fixture. A textured block has an adjacent double-sided cutout card.
import { mkdir, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

// 16×16 RGBA, top-left red, top-right blue, bottom-left green, bottom-right yellow.
// Transparent texels are the inner [2,6)×[2,6) square and the [12,16)×[0,4) notch.
const png = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAASElEQVR4nGO8pSfznwEPcNE9jC7EiMxhwqeZGMCCy2QGBga8LqOJC4gFKC6jqguI8jM6oNgFjOp7A/DavPvhJdq6YNSAwWAAAI4CCxycER+uAAAAAElFTkSuQmCC', 'base64');
const axes = [
  [[1, 0, 0], [0, 0, -1], [0, 1, 0]], [[-1, 0, 0], [0, 0, 1], [0, 1, 0]],
  [[0, 1, 0], [1, 0, 0], [0, 0, -1]], [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
  [[0, 0, 1], [1, 0, 0], [0, 1, 0]], [[0, 0, -1], [-1, 0, 0], [0, 1, 0]],
];
const triangles = [[0, 1], [1, 1], [1, 0], [0, 1], [1, 0], [0, 0]];

function quad(normal, position) {
  return triangles.map(([u, v]) => [...position(u, v), ...normal, u, v]);
}

export function makeTexturedFixture() {
  const center = [-0.5, 0, 0], span = [1, 1.2, 0.5];
  const block = axes.flatMap(([normal, right, up]) => quad(normal, (u, v) =>
    center.map((value, i) => value + span[i] * (normal[i] / 2 + right[i] * (u - 0.5) + up[i] * (0.5 - v)))));
  const card = quad([0, 0, 1], (u, v) => [0.08 + 0.92 * u, 0.7 - 1.2 * v, 0.13]);
  const vertices = [...block, ...card].flat();
  const vertexBytes = Buffer.alloc(vertices.length * 4);
  vertices.forEach((value, index) => vertexBytes.writeFloatLE(value, index * 4));
  const binary = Buffer.alloc(Math.ceil((vertexBytes.length + png.length) / 4) * 4);
  vertexBytes.copy(binary);
  png.copy(binary, vertexBytes.length);
  const accessors = [];
  function primitive(first, count, min, max, material) {
    const index = accessors.length;
    for (const [at, type] of [[0, 'VEC3'], [12, 'VEC3'], [24, 'VEC2']]) {
      accessors.push({ bufferView: 0, byteOffset: first * 32 + at,
        componentType: 5126, count, type, ...(at === 0 ? { min, max } : {}) });
    }
    return { attributes: { POSITION: index, NORMAL: index + 1, TEXCOORD_0: index + 2 }, material };
  }
  const pbr = { baseColorTexture: { index: 0 }, metallicFactor: 0, roughnessFactor: 1 };
  const gltf = {
    asset: { version: '2.0' }, scene: 0, scenes: [{ nodes: [0] }], nodes: [{ mesh: 0 }],
    meshes: [{ primitives: [
      primitive(0, block.length, [-1, -0.6, -0.25], [0, 0.6, 0.25], 0),
      primitive(block.length, card.length, [0.08, -0.5, 0.13], [1, 0.7, 0.13], 1),
    ] }],
    materials: [
      { alphaMode: 'OPAQUE', pbrMetallicRoughness: pbr },
      { alphaMode: 'MASK', alphaCutoff: 0.5, doubleSided: true, pbrMetallicRoughness: pbr },
    ],
    textures: [{ source: 0, sampler: 0 }],
    samplers: [{ magFilter: 9728, minFilter: 9728, wrapS: 33071, wrapT: 33071 }],
    images: [{ bufferView: 1, mimeType: 'image/png' }],
    buffers: [{ byteLength: vertexBytes.length + png.length }],
    bufferViews: [
      { buffer: 0, byteLength: vertexBytes.length, byteStride: 32, target: 34962 },
      { buffer: 0, byteOffset: vertexBytes.length, byteLength: png.length },
    ], accessors,
  };
  const json = Buffer.from(JSON.stringify(gltf));
  const padded = Buffer.alloc(Math.ceil(json.length / 4) * 4, 0x20);
  json.copy(padded);
  const header = Buffer.alloc(20);
  header.write('glTF');
  header.writeUInt32LE(2, 4);
  header.writeUInt32LE(28 + padded.length + binary.length, 8);
  header.writeUInt32LE(padded.length, 12);
  header.write('JSON', 16);
  const chunk = Buffer.alloc(8);
  chunk.writeUInt32LE(binary.length);
  chunk.write('BIN\0', 4);
  return Buffer.concat([header, padded, chunk, binary]);
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  const file = process.argv[2] ?? '.test-output/textured.glb';
  await mkdir(dirname(file), { recursive: true });
  await writeFile(file, makeTexturedFixture());
}
