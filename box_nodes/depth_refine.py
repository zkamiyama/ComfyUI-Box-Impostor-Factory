"""Geometry-anchored RGB guidance for normalized axial depth on an orthographic face."""
import math
import operator

import numpy as np
from scipy import ndimage


def guided_depth(depth, rgb, coverage, radius=2, color_sigma=.10, max_delta=.012):
    """Bilateral smoothing in the original depth units, restricted to continuous covered interiors."""
    z = np.asarray(depth, dtype=np.float32)
    color = np.asarray(rgb, dtype=np.float32)
    alpha = np.asarray(coverage, dtype=np.float32)
    if z.ndim != 2 or not z.size or color.shape != (*z.shape, 3):
        raise ValueError('Expected a depth image and an aligned H × W × 3 RGB image.')
    if alpha.shape == (*z.shape, 1):
        alpha = alpha[..., 0]
    if alpha.shape != z.shape or not np.isfinite(alpha).all() or np.any((alpha < 0) | (alpha > 1)):
        raise ValueError('Expected aligned foreground coverage in [0,1].')
    radius = operator.index(radius)
    if radius < 0 or not math.isfinite(color_sigma) or color_sigma <= 0 or not 0 <= max_delta <= 1:
        raise ValueError('Expected a nonnegative radius, positive color sigma, and max_delta in [0,1].')
    valid = (alpha > .5) & np.isfinite(z)
    if np.any(valid & ((z < 0) | (z > 1))) or np.isinf(z).any():
        raise ValueError('Finite foreground depth must be in [0,1]. Use NaN for uncaptured rays.')
    result = z.copy()
    if radius == 0 or max_delta == 0 or 2 * radius + 1 > min(z.shape) or not valid.any():
        return result
    diameter = 2 * radius + 1
    # A complete, continuous window cannot bridge a silhouette, a hole or an occluding edge.
    z_min = ndimage.minimum_filter(np.where(valid, z, np.inf), size=diameter, mode='constant', cval=np.inf)
    z_max = ndimage.maximum_filter(np.where(valid, z, -np.inf), size=diameter, mode='constant', cval=-np.inf)
    interior = ndimage.minimum_filter(valid, size=diameter, mode='constant', cval=0)
    interior &= (z_max - z_min <= 3 * max_delta)
    if not interior.any():
        return result
    if not np.isfinite(color[valid]).all() or np.any((color[valid] < 0) | (color[valid] > 1)):
        raise ValueError('Expected finite straight RGB values in [0,1] on covered, sampled pixels.')
    height, width = z.shape
    spatial_sigma = max(radius / 2, 1.)
    # Bounded strip working set for 2K faces. The fixed top/bottom rows and edge columns are skipped.
    for y in range(radius, height - radius, 128):
        stop = min(height - radius, y + 128)
        roi = np.s_[y:stop, radius:width-radius]
        active = interior[roi]
        if not active.any():
            continue
        center, center_rgb = z[roi], color[roi]
        weights = np.ones(center.shape, np.float32)
        offsets = np.zeros_like(weights)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue
                other = np.s_[y+dy:stop+dy, radius+dx:width-radius+dx]
                delta = np.where(active, z[other] - center, 0)
                color_delta = np.where(active[..., None], color[other] - center_rgb, 0)
                color_distance = np.einsum('ijk,ijk->ij', color_delta, color_delta)
                spatial = math.exp(-(dx*dx + dy*dy) / (2 * spatial_sigma**2))
                weight = spatial * np.exp(-color_distance / (2 * color_sigma**2) - .5 * (delta / max_delta)**2)
                weights += weight
                offsets += weight * delta
        updated = np.clip(center + np.clip(offsets / weights, -max_delta, max_delta), 0, 1)
        np.copyto(result[roi], updated, where=active)
    return result
