"""Optional bounded transfer of source luminance detail to estimated albedo."""
import numpy as np
from scipy import ndimage


METHOD = 'masked-log-luminance-highpass-v1'


def restore_detail(albedo, candidate, aspect, strength):
    """Keep estimated color while recovering fine contrast from the accepted view.

    The input image can contain fine illumination as well as reflectance detail,
    so this appearance control is opt-in. Alpha edges are excluded from transfer.
    """
    if strength == 0:
        return albedo
    base = albedo[0]
    base = np.where(base <= .04045, base / 12.92, ((base + .055) / 1.055) ** 2.4)
    luma = np.array([.2126, .7152, .0722])
    source_luma = candidate[..., :3] @ luma
    base_luma = base @ luma
    alpha = candidate[..., 3]
    sigma = 3 * candidate.shape[0] / 1024
    sigmas = (sigma, sigma / aspect)
    weight = (alpha > .5).astype('f4')
    denominator = ndimage.gaussian_filter(weight, sigmas)

    def low_frequency(value):
        return ndimage.gaussian_filter(value * weight, sigmas) / np.maximum(denominator, 1e-6)

    source_log = np.log(np.maximum(source_luma, .02))
    base_log = np.log(np.maximum(base_luma, .02))
    residual = (source_log - low_frequency(source_log)) - (base_log - low_frequency(base_log))
    gain = np.clip(strength * residual, -np.log(1.5), np.log(1.5))
    edge_weight = np.clip((ndimage.distance_transform_edt(alpha > .9) - 2) / 4, 0, 1)
    gain *= edge_weight
    corrected = np.clip(base * np.exp(gain[..., None]), 0, 1)
    srgb = np.where(corrected <= .0031308, corrected * 12.92,
                    1.055 * corrected ** (1 / 2.4) - .055)
    return srgb[None]
