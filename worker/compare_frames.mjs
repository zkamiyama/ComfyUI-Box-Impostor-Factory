import { directions } from './capture_views.mjs';

// Compare actual render outputs from separate browser sessions (RGBA8, packed 24-bit depth).
function bytes(encoded, length) {
  const values = Uint8Array.from(atob(encoded), char => char.charCodeAt(0));
  if (values.length !== length) throw new Error('Unexpected snapshot byte length');
  return values;
}

function depth24(buffer, i) {
  return (buffer[i] << 16) | (buffer[i+1] << 8) | buffer[i+2];
}

export function compareFrames(before, after, { colorTolerance = 1, depthTolerance = 2, requireAllViews = true } = {}) {
  if (!Array.isArray(before) || !Array.isArray(after) || before.length !== after.length || !before.length)
    throw new Error('Two complete, non-empty snapshot sets are required');
  const ids = new Set();
  const views = before.map((a, index) => {
    const b = after[index];
    if (!b || !a.id || ids.has(a.id) || a.id !== b.id || a.width !== b.width ||
        a.height !== b.height || !Number.isInteger(a.width) || !Number.isInteger(a.height) ||
        a.width < 1 || a.height < 1)
      throw new Error('Snapshot views or dimensions differ');
    ids.add(a.id);
    const size = a.width * a.height * 4;
    const [c0, c1, z0, z1] = [a.color, b.color, a.depth, b.depth].map(value => bytes(value, size));
    let colorMax = 0, depthMax = 0, coverageMismatch = 0, depthCoverageMismatch = 0;
    let hitsBefore = 0, hitsAfter = 0, depthsBefore = 0, depthsAfter = 0;
    let colorOutliers = 0, depthOutliers = 0, colorDepthMismatchBefore = 0, colorDepthMismatchAfter = 0;
    for (let i = 0; i < size; i += 4) {
      const opaque0 = c0[i+3] > 127, opaque1 = c1[i+3] > 127;
      hitsBefore += +opaque0;
      hitsAfter += +opaque1;
      coverageMismatch += +(opaque0 !== opaque1);
      const d0 = depth24(z0, i), d1 = depth24(z1, i);
      const hasDepth0 = d0 < 0xffffff, hasDepth1 = d1 < 0xffffff;
      depthsBefore += +hasDepth0;
      depthsAfter += +hasDepth1;
      depthCoverageMismatch += +(hasDepth0 !== hasDepth1);
      colorDepthMismatchBefore += +(opaque0 !== hasDepth0);
      colorDepthMismatchAfter += +(opaque1 !== hasDepth1);
      const depthError = Math.abs(d1-d0);
      depthMax = Math.max(depthMax, depthError);
      depthOutliers += +(depthError > depthTolerance);
      let error = 0;
      for (let ch = 0; ch < 4; ch++) error = Math.max(error, Math.abs(c0[i+ch]-c1[i+ch]));
      colorMax = Math.max(colorMax, error);
      colorOutliers += +(error > colorTolerance);
    }
    // A cleared target (or broken, uniformly full target) must never pass a round trip.
    const pixelCount = size / 4;
    const coverageValid = [hitsBefore, hitsAfter, depthsBefore, depthsAfter].every(n => n > 0 && n < pixelCount);
    return { id:a.id, width:a.width, height:a.height, hitsBefore, hitsAfter, depthsBefore, depthsAfter,
      coverageMismatch, depthCoverageMismatch, colorDepthMismatchBefore, colorDepthMismatchAfter,
      colorMax, depthMax, colorOutliers, depthOutliers,
      pass: coverageValid && !coverageMismatch && !depthCoverageMismatch && !colorDepthMismatchBefore &&
        !colorDepthMismatchAfter && !colorOutliers && !depthOutliers };
  });
  if (requireAllViews && (ids.size !== directions.length || directions.some(([id]) => !ids.has(id))))
    throw new Error('The six axis and twelve oblique camera views are required');
  return { pass:views.every(v => v.pass), views, colorTolerance, depthTolerance,
    worstColorLSB:Math.max(...views.map(v => v.colorMax)), worstDepthLSB:Math.max(...views.map(v => v.depthMax)) };
}
