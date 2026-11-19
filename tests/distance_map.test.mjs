import assert from 'node:assert/strict';
import test from 'node:test';
import {DataUtils} from 'three';
import {buildDistanceMap} from '../worker/distance_map.mjs';

const f32 = Math.fround;
const halfBits = DataUtils.toHalfFloat;

function rangeMap(resolution, valueAt) {
  const cells = Math.ceil(resolution / 8), data = new Uint16Array(12 * cells * cells);
  for (let y = 0; y < cells; y++) {
    for (let axis = 0; axis < 3; axis++) {
      for (let x = 0; x < cells; x++) {
        const [low, high] = valueAt(axis, x, y);
        const i = (y * 3 * cells + axis * cells + x) * 4;
        data[i] = halfBits(low);
        data[i + 1] = halfBits(high);
        data[i + 2] = halfBits(0);
        data[i + 3] = halfBits(1);
      }
    }
  }
  return data;
}

function layout(resolution) {
  const cells = Math.ceil(resolution / 8), columns = Math.ceil(Math.sqrt(cells));
  return {cells, columns, rows: Math.ceil(cells / columns)};
}

function atlasIndex(x, y, z, cells, columns) {
  return (Math.floor(z / columns) * cells + y) * columns * cells + (z % columns) * cells + x;
}

// Deliberately direct upstream translation: Three's half decoding, all 26
// neighbours, a fresh atlas on every pass, and no separable min-filter.
function referenceDistanceMap(rangePixels, resolution) {
  const {cells, columns, rows} = layout(resolution), width = columns * cells, height = rows * cells;
  const sample = (axis, u, v) => {
    const p = (v * 3 * cells + axis * cells + u) * 4;
    return [DataUtils.fromHalfFloat(rangePixels[p]), DataUtils.fromHalfFloat(rangePixels[p + 1])];
  };
  const occupied = (x, y, z) => {
    const cell = [x, y, z], uv = [[z, y], [x, z], [x, y]];
    for (let axis = 0; axis < 3; axis++) {
      const low = f32(f32(cell[axis] * f32(8 / resolution)) - f32(1.5 / resolution + 1 / resolution));
      const high = f32(f32(f32(cell[axis] + 1) * f32(8 / resolution)) + f32(1.5 / resolution + 1 / resolution));
      const range = sample(axis, ...uv[axis]);
      if (!(range[0] <= high && range[1] >= low)) return false;
    }
    return true;
  };
  let current = new Uint8Array(width * height);
  for (let z = 0; z < columns * rows; z++) {
    for (let y = 0; y < cells; y++) for (let x = 0; x < cells; x++) {
      current[atlasIndex(x, y, z, cells, columns)] = z < cells && occupied(x, y, z) ? 0 : 7;
    }
  }
  for (let pass = 0; pass < 6; pass++) {
    const next = new Uint8Array(current.length);
    for (let z = 0; z < columns * rows; z++) {
      for (let y = 0; y < cells; y++) for (let x = 0; x < cells; x++) {
        const index = atlasIndex(x, y, z, cells, columns);
        let nearest = current[index];
        for (let dz = -1; dz <= 1; dz++) for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
          if (dx === 0 && dy === 0 && dz === 0) continue;
          const nx = Math.max(0, Math.min(cells - 1, x + dx));
          const ny = Math.max(0, Math.min(cells - 1, y + dy));
          const nz = Math.max(0, Math.min(cells - 1, z + dz));
          nearest = Math.min(nearest, current[atlasIndex(nx, ny, nz, cells, columns)] + 1);
        }
        next[index] = nearest;
      }
    }
    current = next;
  }
  return {data: current, width, height};
}

test('matches the six-pass 26-neighbour oracle, including non-8 resolution and unused tiles', () => {
  let state = 0x042b0a;
  const random = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
  for (const resolution of [1, 7, 8, 9, 17, 31, 41, 48, 50, 64, 96, 101, 128]) {
    const pixels = rangeMap(resolution, () => {
      if (random() < 0.15) return [1, 0];
      const low = random() * 0.7;
      return [low, low + random() * (1 - low)];
    });
    assert.deepEqual(buildDistanceMap(pixels, resolution), referenceDistanceMap(pixels, resolution), `resolution ${resolution}`);
  }
});

test('single occupied cell yields capped Chebyshev distance throughout the valid volume', () => {
  const resolution = 128, {cells, columns} = layout(resolution), origin = [3, 6, 11];
  const pixels = rangeMap(resolution, axis => {
    const point = (origin[axis] + 0.5) / cells;
    return [point, point];
  });
  const {data} = buildDistanceMap(pixels, resolution);
  for (let z = 0; z < cells; z++) for (let y = 0; y < cells; y++) for (let x = 0; x < cells; x++) {
    const expected = Math.min(7, Math.max(Math.abs(x - origin[0]), Math.abs(y - origin[1]), Math.abs(z - origin[2])));
    assert.equal(data[atlasIndex(x, y, z, cells, columns)], expected, `cell ${x},${y},${z}`);
  }
});

test('all empty cells keep distance seven; full cells keep zero, including R41', () => {
  const empty = buildDistanceMap(rangeMap(64, () => [1, 0]), 64);
  assert.ok(empty.data.every(value => value === 7));
  for (const resolution of [41, 56]) {
    const {cells, columns, rows} = layout(resolution);
    const full = buildDistanceMap(rangeMap(resolution, () => [0, 1]), resolution);
    for (let z = 0; z < columns * rows; z++) for (let y = 0; y < cells; y++) for (let x = 0; x < cells; x++) {
      assert.equal(full.data[atlasIndex(x, y, z, cells, columns)], z < cells ? 0 : 1);
    }
  }
});

test('float32 half-value boundaries reproduce shader arithmetic, not a double precision simplification', () => {
  // R48: float32 low(cell 5) = 0.7812500596, so high=0.78125 must not occupy cell 5.
  let resolution = 48, map = rangeMap(resolution, axis => axis === 0 ? [0, 0.78125] : [0, 1]);
  let result = buildDistanceMap(map, resolution), spec = layout(resolution);
  assert.equal(result.data[atlasIndex(5, 2, 2, spec.cells, spec.columns)], 1);
  assert.deepEqual(result, referenceDistanceMap(map, resolution));
  // R96: float32 high(cell 6) is above 0.609375, which must retain the boundary surface.
  resolution = 96;
  map = rangeMap(resolution, axis => axis === 0 ? [0.609375, 1] : [0, 1]);
  result = buildDistanceMap(map, resolution); spec = layout(resolution);
  assert.equal(result.data[atlasIndex(6, 2, 2, spec.cells, spec.columns)], 0);
  assert.deepEqual(result, referenceDistanceMap(map, resolution));
  // R101: float32 high(cell 5) is just below 0.5.
  resolution = 101;
  map = rangeMap(resolution, axis => axis === 0 ? [0.5, 1] : [0, 1]);
  result = buildDistanceMap(map, resolution); spec = layout(resolution);
  assert.equal(result.data[atlasIndex(5, 2, 2, spec.cells, spec.columns)], 1);
  assert.deepEqual(result, referenceDistanceMap(map, resolution));
});

test('half decoding handles subnormal, signed zero, finite extrema, infinity, and NaN like Three half conversion', () => {
  const resolution = 41, pixels = rangeMap(resolution, () => [0, 1]);
  const bits = [0x0000, 0x8000, 0x0001, 0x03ff, 0x0400, 0x3555, 0x3bff, 0x3c00,
    0x7bff, 0xfbff, 0x7c00, 0xfc00, 0x7e00, 0x8001];
  for (let pixel = 0; pixel < pixels.length / 4; pixel++) {
    pixels[pixel * 4] = bits[pixel % bits.length];
    pixels[pixel * 4 + 1] = bits[(pixel * 3 + 7) % bits.length];
  }
  assert.deepEqual(buildDistanceMap(pixels, resolution), referenceDistanceMap(pixels, resolution));
});

test('keeps input bytes immutable and accepts a Uint16Array subview', () => {
  const pixels = rangeMap(41, (axis, x, y) => [0.05 * axis + 0.03 * y, 0.55 + 0.04 * x]);
  const backing = new Uint16Array(pixels.length + 8).fill(0xbeef);
  backing.set(pixels, 4);
  const before = backing.slice();
  assert.deepEqual(buildDistanceMap(backing.subarray(4, -4), 41), buildDistanceMap(pixels, 41));
  assert.deepEqual(backing, before);
});

test('rejects malformed input instead of silently making a distance cache', () => {
  assert.throws(() => buildDistanceMap(new Uint16Array(0), 0), /positive integer/);
  assert.throws(() => buildDistanceMap(new Uint16Array(0), 41.5), /positive integer/);
  assert.throws(() => buildDistanceMap(new Uint8Array(432), 41), /RGBA16F/);
  assert.throws(() => buildDistanceMap(new Uint16Array(431), 41), /RGBA16F/);
});
