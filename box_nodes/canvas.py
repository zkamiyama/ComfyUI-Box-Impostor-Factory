"""Reversible camera-to-square letterboxing for the two native image-edit graphs."""
import hashlib
import numpy as np
from PIL import Image

from . import store

# Horizontal and vertical spans of the Baker camera for each box axis.
SPAN_AXES = {'X': (2, 1), 'Y': (0, 2), 'Z': (0, 1)}


def letterbox_rect(width, height, size):
    if not (np.isfinite(width) and np.isfinite(height) and width > 0 and height > 0):
        raise ValueError('The guide and reference must have positive, finite dimensions.')
    factor = size / max(width, height)
    w = min(size, max(1, round(width * factor)))
    h = min(size, max(1, round(height * factor)))
    return [(size - w) // 2, (size - h) // 2, w, h]


def as_image(value):
    if hasattr(value, 'detach'):
        value = value.detach().float().cpu().numpy()
    a = np.asarray(value, dtype=np.float32)
    if a.ndim != 4 or a.shape[0] != 1 or a.shape[-1] not in (3, 4):
        raise ValueError('The model canvas expects one RGB or RGBA image.')
    if not np.isfinite(a).all():
        raise ValueError('The input contains non-finite colors.')
    return a[0]


def place(image, rect, size):
    x, y, w, h = rect
    color = image[..., :3]
    if image.shape[-1] == 4:
        alpha = image[..., 3:4]
        color = color * alpha + 0.5 * (1 - alpha)
    out = np.full((1, size, size, 3), 0.5, dtype=np.float32)
    for c in range(3):
        channel = Image.fromarray(np.ascontiguousarray(color[..., c]))
        out[0, y:y+h, x:x+w, c] = np.asarray(channel.resize((w,h), Image.Resampling.BILINEAR))
    return out


def undo_guide_canvas(source, canvas):
    """Return the capture-area view of the original generated image at its actual resolution."""
    if canvas.get('version') != 1 or canvas.get('faceId') not in store.FACES:
        raise ValueError('Unknown face canvas transform.')
    size = canvas['size']
    x, y, w, h = canvas['guideRect']
    if type(size) is not int or size < 1 or any(type(v) is not int for v in (x,y,w,h)) or not (
            x >= 0 and y >= 0 and w >= 1 and h >= 1 and x+w <= size and y+h <= size):
        raise ValueError('Invalid guide rectangle.')
    H, W = source.shape[:2]
    if H != W:
        raise ValueError('The generated image must preserve the square model canvas.')
    x0, y0, x1, y1 = [round(v * W / size) for v in (x, y, x+w, y+h)]
    if x1 <= x0 or y1 <= y0:
        raise ValueError('The generated image has no pixels inside the guide rectangle.')
    return source[y0:y1, x0:x1]


class ModelCanvas:
    CATEGORY = 'Box Impostor'
    FUNCTION = 'prepare'
    RETURN_TYPES = ('IMAGE', 'IMAGE', 'BOX_CANVAS')
    RETURN_NAMES = ('ref_canvas', 'guide_canvas', 'canvas')

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'reference': ('IMAGE',), 'guide': ('IMAGE',), 'guides': ('BOX_GUIDES',),
                             'face_id': (list(store.FACES),),
                             'size': ('INT', {'default': 1024, 'min': 32, 'max': 4096, 'step': 32})}}

    def prepare(self, reference, guide, guides, face_id, size=1024):
        import torch
        if face_id not in store.FACES or type(size) is not int or size % 32 or not 32 <= size <= 4096:
            raise ValueError('Expected a valid face and a model canvas size divisible by 32.')
        manifest = store.validate(store.asset_dir(guides), guides, check_bytes=False)
        bounds = manifest['bakeBox']
        span = np.subtract(bounds['max'], bounds['min'])
        reference = as_image(reference)
        guide = as_image(guide)
        W, H = (span[i] for i in SPAN_AXES[face_id[1]])
        source_rect = letterbox_rect(reference.shape[1], reference.shape[0], size)
        guide_rect = letterbox_rect(W, H, size)
        def content_key(image):
            samples = np.ascontiguousarray(image, dtype='<f4')
            return store.key(samples.shape, hashlib.sha256(samples).hexdigest())
        canvas = {'version': 1, 'size': size, 'faceId': face_id, 'guideRect': guide_rect,
                  'guideInputSha256': content_key(guide), 'referenceInputSha256': content_key(reference),
                  'sourceRect': source_rect, 'physicalWidthHeight': [float(W), float(H)],
                  'imageRowOrder': 'top-to-bottom', 'backgroundSRGB': 0.5}
        return torch.from_numpy(place(reference,source_rect,size)), torch.from_numpy(place(guide,guide_rect,size)), canvas
