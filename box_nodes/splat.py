"""Optional appearance guide from ComfyUI's native Gaussian splat rasterizer."""
import hashlib
import inspect
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import affine_transform
from scipy.spatial.transform import Rotation
from scipy.stats import spearmanr

from . import faces, store


SPACES = ['canonical_box', 'source_glb']


def camera_plan(manifest, face_id, space):
    """Map the rectangular Baker frustum to RenderSplat's isotropic pixels."""
    if space not in SPACES or face_id not in store.FACES:
        raise ValueError('Choose a Baker face and the space in which the splat is registered.')
    face = manifest['faces'][face_id]
    frame = np.eye(4) if space == 'canonical_box' else np.asarray(
        manifest['inverseTransform'], dtype=float).reshape(4, 4, order='F')
    to_world = np.linalg.inv(np.asarray(face['viewMatrix']).reshape(4, 4, order='F'))
    basis = frame[:3, :3] @ to_world[:3, :3]
    scale = np.linalg.norm(basis[:, 0])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Invalid coordinate transform.')
    basis /= scale
    if not np.allclose(basis.T @ basis, np.eye(3), atol=1e-5) or np.linalg.det(basis) < 0:
        raise ValueError('The splat camera needs a proper, uniformly scaled coordinate transform.')
    box = manifest['bakeBox']
    spans = np.subtract(box['max'], box['min'])
    center = frame @ np.append(np.asarray(box['min']) + spans / 2, 1.)
    left, right, bottom, top = face['frustum'][:4]
    physical = np.asarray([right - left, top - bottom]) * scale
    if not np.all(np.isfinite(physical) & (physical > 0)):
        raise ValueError('Invalid orthographic guide frustum.')
    r = manifest['resolution']
    # Two samples of padding keep the camera-to-atlas resampling within the raster.
    pixels = min(r / min(physical), 2046 / max(physical))
    content = physical * pixels
    width, height = np.maximum(64, np.minimum(2048, np.ceil(content).astype(int) + 2))
    distance = max(float(np.linalg.norm(spans)) * scale, 1.)
    # RenderSplat's orthographic scale = min(viewport)/(2*tan(FOV/2)*distance).
    fov = 2 * math.degrees(math.atan(min(width, height) / (2 * pixels * distance)))
    xyz = lambda p: dict(zip('xyz', map(float, p)))
    camera = {
        'cameraType': 'orthographic', 'position': xyz(center[:3] + basis[:, 2] * distance),
        'target': xyz(center[:3]), 'quaternion': dict(zip('xyzw', Rotation.from_matrix(basis).as_quat().tolist())),
        'zoom': 1., 'fov': fov,
    }
    return camera, (int(width), int(height)), content


def atlas_pixels(image, content, r):
    """Sample at Baker texel centers. Native RenderSplat samples at integer pixels."""
    h, w = image.shape[:2]
    step = np.asarray(content)[::-1] / r
    offset = np.array([h, w]) / 2 - np.asarray(content)[::-1] / 2 + step / 2
    def sample(channel):
        return affine_transform(channel, np.diag(step), offset=offset,
                                output_shape=(r, r), order=1, mode='nearest', prefilter=False)
    if image.ndim == 2:
        return sample(image)
    return np.stack([sample(channel) for channel in image.transpose(2, 0, 1)], axis=-1)


def splat_hash(splat):
    if splat.positions.ndim != 3 or splat.positions.shape[0] != 1:
        raise ValueError('Connect a single SPLAT for a single Box proxy.')
    total = splat.positions.shape[1]
    count = int(splat.counts[0].item()) if getattr(splat, 'counts', None) is not None else total
    if not 0 < count <= total:
        raise ValueError('The splat must contain at least one Gaussian and have a valid count.')
    hashes = []
    for name in ('positions', 'scales', 'rotations', 'opacities', 'sh'):
        value = np.ascontiguousarray(faces.array(getattr(splat, name)[0, :count]), dtype='<f4')
        if not np.isfinite(value).all():
            raise ValueError(f'SPLAT {name} contains non-finite values.')
        hashes.append((name, value.shape, hashlib.sha256(memoryview(value)).hexdigest()))
    return store.key('native-splat-v1', hashes)


def depth_agreement(mesh, near_brightness, overlap):
    valid = np.isfinite(mesh) & overlap
    n = int(valid.sum())
    if n < 3:
        return {'samples': n, 'status': 'insufficient-overlap', 'spearman': None}
    a, b = mesh[valid], -near_brightness[valid]
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return {'samples': n, 'status': 'constant-depth', 'spearman': None}
    return {'samples': n, 'status': 'measured', 'spearman': float(spearmanr(a, b).statistic)}


class TripoSplatGuide:
    CATEGORY = 'Box Impostor'
    FUNCTION = 'render'
    RETURN_TYPES = ('IMAGE', 'MASK', 'STRING')
    RETURN_NAMES = ('guide', 'foreground', 'qa_json')
    DESCRIPTION = ('Render an already registered SPLAT as an appearance guide in one mesh camera. '
                   'Use Transform Splat for registration and inspect coverage and relative depth in qa_json.')

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'splat': ('SPLAT',), 'guides': ('BOX_GUIDES',),
                             'face_id': (list(store.FACES),), 'coordinate_space': (SPACES,)}}

    def render(self, splat, guides, face_id, coordinate_space):
        try:
            from comfy_extras.nodes_gaussian_splat import RenderSplat
        except ImportError as exc:
            raise RuntimeError('BI TripoSplatGuide requires ComfyUI with native RenderSplat support.') from exc
        import torch
        directory = store.asset_dir(guides)
        manifest = store.validate(directory, guides, check_bytes=False)
        camera, size, content = camera_plan(manifest, face_id, coordinate_space)
        r = manifest['resolution']
        face = manifest['faces'][face_id]
        guide_path = store.contained(directory, face['guide'])
        depth_path = store.contained(directory, face['depth']) if face.get('depth') else None
        if depth_path and (face.get('depthEncoding') != 'linear-axis-0-near-1-far;nan-background'
                           or face.get('imageRowOrder') != 'top-to-bottom'):
            raise ValueError('Unknown proxy depth encoding.')
        digest = splat_hash(splat)
        renderer_sha = store.file_hash(Path(inspect.getfile(RenderSplat)))
        inputs = {'asset': guides, 'faceId': face_id, 'coordinateSpace': coordinate_space,
                  'splatSha256': digest, 'camera': camera, 'nativeSize': size, 'contentPixels': content.tolist(),
                  'resolution': r, 'rendererSha256': renderer_sha, 'meshGuideSha256': store.file_hash(guide_path),
                  'meshDepthSha256': store.file_hash(depth_path) if depth_path else None}
        stage_key = store.key('splat-guide-v1', inputs)

        def draw(style):
            store.check_interrupted()
            out = RenderSplat.execute(splat=splat, width=size[0], height=size[1], frames=1,
                                      splat_scale=1., sharpen=1., headlight_shading=0., opacity_threshold=0.,
                                      background='#808080', render_style=style, camera_info=camera)
            return out.result

        def write(stage):
            color, mask = draw('color')
            rgb = atlas_pixels(faces.array(color)[0], content, r)
            coverage = atlas_pixels(faces.array(mask)[0], content, r)
            np.save(stage / 'guide.npy', rgb, allow_pickle=False)
            np.save(stage / 'foreground.npy', coverage, allow_pickle=False)
            with Image.open(guide_path) as guide:
                mesh_mask = np.asarray(guide.convert('RGBA'))[..., 3] >= 128
            splat_mask = coverage > .5
            qa = faces.scores(splat_mask, mesh_mask)
            qa.update(meshPixels=int(mesh_mask.sum()), splatPixels=int(splat_mask.sum()))
            if depth_path:
                depth, _ = draw('depth')
                near = atlas_pixels(faces.array(depth)[0, ..., 0], content, r)
                np.save(stage / 'relative-depth.npy', near, allow_pickle=False)
                mesh = np.fromfile(depth_path, dtype='<f2').reshape(r, r)
                qa['relativeDepth'] = depth_agreement(mesh, near, mesh_mask & splat_mask)
            else:
                qa['relativeDepth'] = {'samples': 0, 'status': 'mesh-depth-unavailable', 'spearman': None}
            report = {**inputs, 'stageKey': stage_key, 'qa': qa}
            (stage / 'qa.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')

        stage = faces.cache_dir('splat-guides', stage_key, write)
        image = torch.from_numpy(np.load(stage / 'guide.npy', allow_pickle=False)[None])
        mask = torch.from_numpy(np.load(stage / 'foreground.npy', allow_pickle=False)[None])
        return image, mask, (stage / 'qa.json').read_text(encoding='utf-8')
