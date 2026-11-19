// CPU equivalent of three-impostor 0.4.2's rangeMap-derived distance cache.
// Both input RGBA half bits and output R8 rows follow increasing shader V.
const CELL = 8;
const REACH = 6;
const f32 = Math.fround;

function halfFloat(bits) {
  const sign = bits & 0x8000 ? -1 : 1;
  const exponent = (bits >>> 10) & 31, fraction = bits & 1023;
  if (exponent === 0) return sign * fraction * 2 ** -24;
  if (exponent === 31) return fraction ? NaN : sign * Infinity;
  return sign * (1024 + fraction) * 2 ** (exponent - 25);
}

// Expanded-cache only outward rounding. The native zero-margin path keeps its
// original arithmetic and bytes. Range extrema are stored as half floats; one
// outward half step covers their round-to-nearest storage error before f32
// interpolation. Invalid range numbers are treated as unknown, hence occupied.
function outwardHalf(bits, direction) {
  const value = halfFloat(bits);
  if (!Number.isFinite(value)) return direction < 0 ? -Infinity : Infinity;
  if (direction < 0) {
    return halfFloat((bits & 0x8000) ? bits + 1 : (bits === 0 ? 0x8001 : bits - 1));
  }
  return halfFloat((bits & 0x8000) ? (bits === 0x8000 ? 1 : bits - 1) : bits + 1);
}

const roundBuffer = new ArrayBuffer(4);
const roundFloat = new Float32Array(roundBuffer);
const roundBits = new Uint32Array(roundBuffer);
function outwardFloat(value, direction) {
  roundFloat[0] = value;
  if (!Number.isFinite(roundFloat[0])) return roundFloat[0];
  const bits = roundBits[0];
  if (direction < 0) roundBits[0] = (bits & 0x80000000) ? bits + 1 : (bits === 0 ? 0x80000001 : bits - 1);
  else roundBits[0] = (bits & 0x80000000) ? (bits === 0x80000000 ? 1 : bits - 1) : bits + 1;
  return roundFloat[0];
}

function minAxis(input, output, cells, stride) {
  const span = cells * stride;
  for (let base = 0; base < input.length; base += span) {
    for (let offset = 0; offset < span; offset += stride) {
      const start = base + offset;
      const before = offset === 0 ? start : start - stride;
      const after = offset + stride === span ? start : start + stride;
      for (let i = 0; i < stride; i++) {
        output[start + i] = Math.min(input[before + i], input[start + i], input[after + i]);
      }
    }
  }
}

/**
 * Build the native 0.4.2 R8 distance atlas from a compact RGBA16F range atlas.
 * rangePixels has 3*ceil(resolution/8) columns and ceil(resolution/8) rows.
 * The returned bytes are suitable for RedFormat/UnsignedByteType, nearest
 * filtering, clamp-to-edge, and no mipmaps. Upload orientation is the caller's.
 * Optional additionalMargins are normalized Box-axis expansion distances.
 * For bounded interval morphing use band / boxSize[axis]. Rebuild whenever
 * resolution, interval/range bytes, box size, or the maximum band changes.
 * Keep the expanded cache separate from the native data's original distanceMap.
 */
export function buildDistanceMap(rangePixels, resolution, { additionalMargins = [0, 0, 0] } = {}) {
  if (!Array.isArray(additionalMargins) || additionalMargins.length !== 3 ||
      additionalMargins.some(v => !Number.isFinite(v) || v < 0)) {
    throw new Error('Expected three finite nonnegative normalized-axis additionalMargins');
  }
  const expanded = additionalMargins.some(v => v > 0);
  if (!Number.isSafeInteger(resolution) || resolution < 1) {
    throw new Error('Expected a positive integer Box resolution');
  }
  const cells = Math.ceil(resolution / CELL), plane = cells * cells;
  if (!(rangePixels instanceof Uint16Array) || rangePixels.length !== 12 * plane) {
    throw new Error('Expected compact RGBA16F rangeMap pixels for the Box resolution');
  }
  const columns = Math.ceil(Math.sqrt(cells)), rows = Math.ceil(cells / columns);
  const width = columns * cells, height = rows * cells;
  const rangeWidth = 3 * cells;
  const rangeLow = new Float32Array(3 * plane), rangeHigh = new Float32Array(3 * plane);
  for (let i = 0; i < rangeLow.length; i++) {
    rangeLow[i] = expanded ? outwardHalf(rangePixels[i * 4], -1) : halfFloat(rangePixels[i * 4]);
    rangeHigh[i] = expanded ? outwardHalf(rangePixels[i * 4 + 1], 1) : halfFloat(rangePixels[i * 4 + 1]);
  }

  // Constants and every arithmetic operation in cellOccupied are shader f32.
  // A double-precision (8*c +/- 2.5)/resolution changes half-float boundary tests.
  const cellSize = f32(CELL / resolution), margin = f32(1.5 / resolution + 1 / resolution);
  const low = Array.from({length: 3}, () => new Float32Array(cells));
  const high = Array.from({length: 3}, () => new Float32Array(cells));
  for (let axis = 0; axis < 3; axis++) {
    // The inherited extra texel covers interval filtering. The tiny added f32
    // guard covers constants/physical-to-normalized conversion for this cache.
    const extra = additionalMargins[axis];
    const numericGuard = 8 * 2 ** -23 * (1 + extra);
    const axisMargin = expanded ? outwardFloat(margin + extra + numericGuard, 1) : margin;
    for (let i = 0; i < cells; i++) {
      const lower = f32(f32(i * cellSize) - axisMargin);
      const upper = f32(f32((i + 1) * cellSize) + axisMargin);
      low[axis][i] = expanded ? outwardFloat(lower, -1) : lower;
      high[axis][i] = expanded ? outwardFloat(upper, 1) : upper;
    }
  }

  const distance = new Uint8Array(plane * cells);
  for (let z = 0; z < cells; z++) {
    for (let y = 0; y < cells; y++) {
      // axisUV is (z,y), (x,z), (x,y), respectively; rangeMap uses nearest filtering.
      const rangeX = y * rangeWidth + z;
      const rangeY = z * rangeWidth + cells, rangeZ = y * rangeWidth + 2 * cells;
      const start = (z * cells + y) * cells;
      for (let x = 0; x < cells; x++) {
        const occupied = rangeLow[rangeX] <= high[0][x] && rangeHigh[rangeX] >= low[0][x] &&
          rangeLow[rangeY + x] <= high[1][y] && rangeHigh[rangeY + x] >= low[1][y] &&
          rangeLow[rangeZ + x] <= high[2][z] && rangeHigh[rangeZ + x] >= low[2][z];
        distance[start + x] = occupied ? 0 : REACH + 1;
      }
    }
  }

  const temporaryX = new Uint8Array(distance.length), temporaryY = new Uint8Array(distance.length);
  const padding = columns * rows > cells ? new Uint8Array(plane).fill(REACH + 1) : null;
  for (let pass = 0; pass < REACH; pass++) {
    // min(old, 1 + min of the 27-cell neighbourhood) is identical to the
    // upstream min(old, neighbour+1) over 26 neighbours. The min is separable.
    minAxis(distance, temporaryX, cells, 1);
    minAxis(temporaryX, temporaryY, cells, cells);
    if (padding) {
      // Unused atlas slices start at 7 and clamp every neighbour's z to the last
      // valid slice, just like upstream. They never affect valid cells.
      const last = (cells - 1) * plane;
      for (let i = 0; i < plane; i++) {
        padding[i] = Math.min(padding[i], temporaryY[last + i] + 1);
      }
    }
    minAxis(temporaryY, temporaryX, cells, plane);
    for (let i = 0; i < distance.length; i++) {
      distance[i] = Math.min(distance[i], temporaryX[i] + 1);
    }
  }

  const data = new Uint8Array(width * height);
  for (let z = 0; z < columns * rows; z++) {
    const tileX = (z % columns) * cells, tileY = Math.floor(z / columns) * cells;
    const source = z < cells ? distance : padding;
    const sourceStart = z < cells ? z * plane : 0;
    for (let y = 0; y < cells; y++) {
      const start = sourceStart + y * cells;
      data.set(source.subarray(start, start + cells), (tileY + y) * width + tileX);
    }
  }
  return {data, width, height};
}
