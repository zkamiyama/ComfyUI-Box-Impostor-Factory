#!/usr/bin/env python3
"""Display registered native atlas head crops with their physical aspect ratios.

This is an image-source comparison, not a mesh render or a novel-view prediction.
No pixel tracking, face alignment, relighting, or retouching is applied.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from evaluate_parallax_sweep import matrix, project, rgba_png, sha


def compare(sweep_path, output):
    sweep_path, output = Path(sweep_path), Path(output)
    if output.exists():
        raise FileExistsError('Preserve completed source comparison: ' + str(output))
    sweep = json.loads(sweep_path.read_text())
    source = sweep['source']
    manifest_path = Path(source['manifestPath'])
    if sha(manifest_path) != source['manifestSHA256']:
        raise ValueError('Source manifest differs from the capture.')
    manifest = json.loads(manifest_path.read_text())
    desc = manifest['textures']['colorMap']
    atlas_path = manifest_path.parent / desc['path']
    if sha(atlas_path) != desc['sha256']:
        raise ValueError('Native color atlas checksum differs.')
    r = manifest['resolution']
    atlas = np.fromfile(atlas_path, dtype='<f2').reshape(2 * r, 3 * r, 4)
    low, high = np.asarray(sweep['headWorldBounds'], dtype='f8')
    corners = np.array(list(itertools.product(*zip(low, high))))
    images, receipts = [], []
    display_height = 320
    output.mkdir(parents=True)
    for face_id in ('+X', '+Z', '-X'):
        face = manifest['faces'][face_id]
        projection = matrix(face['projectionMatrix'])
        if not np.allclose(projection[3], [0, 0, 0, 1], rtol=0, atol=1e-12):
            raise ValueError('Source face must use its recorded orthographic bake camera.')
        x, y, _, _ = project(face, corners, r, r)
        rect = [int(np.floor(x.min() + .5)), int(np.floor(y.min() + .5)),
                int(np.ceil(x.max() + .5)), int(np.ceil(y.max() + .5))]
        rect = [int(np.clip(value, 0, r)) for value in rect]
        x0, y0, x1, y1 = rect
        if x1 <= x0 or y1 <= y0:
            raise ValueError('Projected head bounds fall outside the native face.')
        tile_x, tile_y = face['tile']
        native = atlas[tile_y * r:(tile_y + 1) * r, tile_x * r:(tile_x + 1) * r][::-1]
        crop = native[y0:y1, x0:x1]
        linear = crop.astype('f4')
        np.divide(linear[..., :3], linear[..., 3:4], out=linear[..., :3], where=linear[..., 3:4] > 0)
        encoded = rgba_png(linear)
        key = {'+X': 'px', '+Z': 'pz', '-X': 'nx'}[face_id]
        raw_display = output / (key + '-native-head.png')
        encoded.save(raw_display)
        world_per_pixel = [2 / abs(projection[0, 0]) / r, 2 / abs(projection[1, 1]) / r]
        physical_size = [(x1 - x0) * world_per_pixel[0], (y1 - y0) * world_per_pixel[1]]
        size = [int(round(display_height * physical_size[0] / physical_size[1])), display_height]
        scaled = encoded.resize(size, Image.Resampling.NEAREST)
        backdrop = Image.new('RGBA', size, (230, 230, 230, 255))
        backdrop.alpha_composite(scaled)
        images.append((face_id, backdrop.convert('RGB')))
        receipts.append({'face': face_id, 'atlasTile': face['tile'],
                         'nativeFaceResolution': r, 'nativeCropXYXY': rect,
                         'nativeCropPixels': [x1 - x0, y1 - y0],
                         'nativePremultipliedFloat16CropSHA256': hashlib.sha256(np.ascontiguousarray(crop).tobytes()).hexdigest(),
                         'worldPerNativePixelXY': world_per_pixel, 'physicalCropSizeXY': physical_size,
                         'displayPixels': size, 'resize': 'nearest; only native non-square physical aspect and common display height',
                         'originalBakeViewMatrix': face['viewMatrix'], 'originalBakeProjectionMatrix': face['projectionMatrix'],
                         'image': str(raw_display), 'imageSHA256': sha(raw_display)})
    gap, top = 16, 66
    board = Image.new('RGB', (sum(image.width for _, image in images) + gap * 4, display_height + top + gap), (245, 245, 245))
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 16)
        small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 13)
    except OSError:
        font = small = ImageFont.load_default()
    draw = ImageDraw.Draw(board)
    draw.text((gap, 8), 'H native atlas sources | same fixed world head bounds | physical aspect restored', fill=(30, 30, 30), font=font)
    draw.text((gap, 29), 'Aligned/reprojected atlas colors; separate from the original generated F PNGs.', fill=(60, 60, 60), font=small)
    x = gap
    for face_id, image in images:
        draw.text((x, 48), face_id + ' source', fill=(30, 30, 30), font=small)
        board.paste(image, (x, top))
        x += image.width + gap
    strip = output / 'native-head-source-strip.webp'
    board.save(strip, lossless=True, method=6)
    helper = Path(__file__)
    evaluator = helper.with_name('evaluate_parallax_sweep.py')
    (output / 'source-comparison-helper.py').write_bytes(helper.read_bytes())
    (output / 'evaluator-source.py').write_bytes(evaluator.read_bytes())
    proof = {'kind': 'fixed-world-head-native-atlas-source-comparison-v1',
             'sweep': str(sweep_path), 'sweepSHA256': sha(sweep_path),
             'manifest': str(manifest_path), 'manifestSHA256': sha(manifest_path),
             'nativeColorAtlas': str(atlas_path), 'nativeColorAtlasSHA256': sha(atlas_path),
             'assetId': manifest['asset_id'], 'headWorldBounds': sweep['headWorldBounds'],
             'sourceMeaning': 'Actual H native atlas after recorded alignment/reprojection of frozen F sources; not the original generated source PNG bytes.',
             'colorEncoding': 'Native premultiplied linear Float16 RGBA -> unpremultiply -> piecewise sRGB -> nearest 8-bit rounding; no tone mapping.',
             'faces': receipts, 'strip': str(strip), 'stripSHA256': sha(strip),
             'helperSHA256': sha(helper), 'evaluatorSourceSHA256': sha(evaluator),
             'interpretationLimit': 'Different visible source expressions/features can explain pressure on view blending; this source comparison alone does not establish the cause of every rendered seam.'}
    (output / 'source-projections.json').write_text(json.dumps(proof, indent=2, allow_nan=False) + '\n')
    return proof


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sweep', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compare(args.sweep, args.output), indent=2))
