// Runtime-only compatibility for legacy schemaVersion 1 Box assets.
// Preserve the original interval/color/normal/range bytes and derive the 0.4.2
// skip field once, before a restore path uploads or flips any range-map rows.
import {
  DataTexture, RedFormat, UnsignedByteType, NearestFilter,
  ClampToEdgeWrapping, NoColorSpace,
} from 'three/webgpu';
import { buildDistanceMap } from './distance_map.mjs';

export function createDistanceTexture(rangePixels, resolution) {
  const { data, width, height } = buildDistanceMap(rangePixels, resolution);
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
