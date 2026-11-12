// A self-contained, asymmetric GLB with face colors, an off-center tab and top/bottom markings.
import { mkdir, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const faces = [
  [[1, 0, 0], [0, 0, -1], [0, 1, 0], [0.9, 0.08, 0.03]],
  [[-1, 0, 0], [0, 0, 1], [0, 1, 0], [0.05, 0.7, 0.1]],
  [[0, 1, 0], [1, 0, 0], [0, 0, -1], [0.1, 0.16, 0.9]],
  [[0, -1, 0], [1, 0, 0], [0, 0, 1], [0.85, 0.64, 0.06]],
  [[0, 0, 1], [1, 0, 0], [0, 1, 0], [0.07, 0.65, 0.7]],
  [[0, 0, -1], [-1, 0, 0], [0, 1, 0], [0.65, 0.07, 0.65]],
];
const corners = [[-1, -1], [1, -1], [1, 1], [-1, -1], [1, 1], [-1, 1]];

export function makeFixtureGeometry(boxes, quads=[]) {
  const positions = [], normals = [], colors = [];
  function box(center, size, tint) {
    for (const [normal, right, up, faceColor] of faces) {
      for (const [u, v] of corners) {
        positions.push(...center.map((c, axis) =>
          c + size[axis] * (normal[axis] + right[axis] * u + up[axis] * v) / 2));
        normals.push(...normal);
        colors.push(...(tint ?? faceColor));
      }
    }
  }
  for (const [center,size,tint] of boxes) box(center,size,tint);
  for (const [normal,vertices] of quads) {
    const color=faces.find(([n])=>n.every((v,i)=>v===normal[i]))[3];
    for (const i of [0,1,2,0,2,3]) {
      positions.push(...vertices[i]);normals.push(...normal);colors.push(...color);
    }
  }
  const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
  positions.forEach((value, i) => {
    const axis = i % 3;
    min[axis] = Math.min(min[axis], value);
    max[axis] = Math.max(max[axis], value);
  });
  let offset = 0;
  const chunks = [], bufferViews = [], accessors = [];
  for (const values of [positions, normals, colors]) {
    const bytes = Buffer.alloc(values.length * 4);
    values.forEach((value, i) => bytes.writeFloatLE(value, i * 4));
    const index = bufferViews.length;
    bufferViews.push({buffer: 0, byteOffset: offset, byteLength: bytes.length, target: 34962});
    accessors.push({bufferView: index, componentType: 5126, count: values.length / 3, type: 'VEC3'});
    chunks.push(bytes);
    offset += bytes.length;
  }
  Object.assign(accessors[0], {min, max});
  const gltf = {
    asset: {version: '2.0', generator: 'Box Impostor Factory fixture'},
    scene: 0, scenes: [{nodes: [0]}], nodes: [{mesh: 0}],
    meshes: [{primitives: [{attributes: {POSITION: 0, NORMAL: 1, COLOR_0: 2}, material: 0}]}],
    materials: [{pbrMetallicRoughness: {metallicFactor: 0, roughnessFactor: 0.8}}],
    buffers: [{byteLength: offset}], bufferViews, accessors,
  };
  const json = Buffer.from(JSON.stringify(gltf));
  const padded = Buffer.alloc(Math.ceil(json.length / 4) * 4, 0x20);
  json.copy(padded);
  const binary = Buffer.concat(chunks);
  const header = Buffer.alloc(20);
  header.write('glTF');
  header.writeUInt32LE(2, 4);
  header.writeUInt32LE(28 + padded.length + binary.length, 8);
  header.writeUInt32LE(padded.length, 12);
  header.write('JSON', 16);
  const binHeader = Buffer.alloc(8);
  binHeader.writeUInt32LE(binary.length);
  binHeader.write('BIN\0', 4);
  return Buffer.concat([header, padded, binHeader, binary]);
}
export function makeFixture() {
  return makeFixtureGeometry([
    [[0.45, 0.1, -0.22], [1.6, 0.9, 0.68]],
    [[1.35, 0.19, -0.12], [0.48, 0.28, 0.32], [0.92, 0.88, 0.85]],
    [[0.78, 0.57, -0.41], [0.45, 0.05, 0.09], [0.88, 0.9, 0.9]],
    [[0.59, 0.57, -0.30], [0.07, 0.05, 0.24], [0.88, 0.9, 0.9]],
    [[0.04, -0.37, -0.10], [0.21, 0.07, 0.11], [0.05, 0.07, 0.1]],
  ]);
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  const output = process.argv[2] ?? '.test-output/asymmetric.glb';
  await mkdir(dirname(output), {recursive: true});
  await writeFile(output, makeFixture());
  console.log(output);
}
