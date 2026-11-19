// Runtime-only compatibility for legacy schemaVersion 1 Box assets.
// Preserve the original interval/color/normal/range bytes and derive the 0.4.2
// skip field once, before a restore path uploads or flips any range-map rows.
import {
  DataTexture, RedFormat, UnsignedByteType, NearestFilter,
  ClampToEdgeWrapping, NoColorSpace,
} from 'three/webgpu';
import { buildDistanceMap } from './distance_map.mjs';

export function createDistanceTexture(rangePixels, resolution, options = {}) {
  const { data, width, height } = buildDistanceMap(rangePixels, resolution, options);
  const texture = new DataTexture(data, width, height, RedFormat, UnsignedByteType);
  texture.minFilter = texture.magFilter = NearestFilter;
  texture.wrapS = texture.wrapT = ClampToEdgeWrapping;
  texture.flipY = false;
  texture.generateMipmaps = false;
  texture.colorSpace = NoColorSpace;
  texture.unpackAlignment = 1;
  texture.needsUpdate = true;
  return texture;
}

// A separate expanded cache belongs to this restore's morph options. Never
// overwrite the native cache, and register ownership alongside other textures.
export function attachMorphDistanceCache(impostor, rangePixels,
  { shapeSampling = 'native', morphBandTexels = 8 } = {}, ownedTextures = []) {
  if (shapeSampling !== 'interval-morph') return null;
  if (!Number.isFinite(morphBandTexels) || morphBandTexels <= 0)
    throw new Error('Morph cache band must be positive and finite');
  const size = ['x', 'y', 'z'].map(axis => impostor.box.max[axis] - impostor.box.min[axis]);
  if (size.some(value => !Number.isFinite(value) || value <= 0))
    throw new Error('Morph cache requires a finite positive Box size');
  const band = Math.min(...size) * morphBandTexels / impostor.resolution;
  const texture = createDistanceTexture(rangePixels, impostor.resolution,
    { additionalMargins: size.map(value => band / value) });
  impostor.morphDistanceMap = texture;
  impostor.morphDistanceBandTexels = morphBandTexels;
  ownedTextures.push(texture);
  return texture;
}
