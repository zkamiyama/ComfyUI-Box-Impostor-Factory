#!/usr/bin/env python3
"""Freeze one original and eight verified Qwen views with identical black compositing.

Requires Python 3.11+ (hashlib.file_digest), NumPy, and Pillow.
Use the existing ComfyUI Python 3.12 environment for this helper, not ZipSplat's
Python 3.10 environment.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def frozen_json(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError('Existing frozen JSON differs: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def copy_exact(source, target, expected):
    if digest(source) != expected:
        raise ValueError('Input changed before copying: ' + str(source))
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if digest(target) != expected:
            raise ValueError('Frozen file differs: ' + str(target))
    else:
        shutil.copyfile(source, target)
    if digest(target) != expected:
        raise ValueError('Copy hash differs: ' + str(target))


def srgb_to_linear(rgb):
    return np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)


def linear_to_srgb(rgb):
    return np.where(rgb <= .0031308, rgb * 12.92, 1.055 * np.maximum(rgb, 0) ** (1 / 2.4) - .055)


def black_composite(rgb, mask):
    linear = srgb_to_linear(rgb.astype(np.float32) / 255)
    composited = linear * (mask.astype(np.float32) / 255)[..., None]
    return np.uint8(np.clip(np.rint(linear_to_srgb(composited) * 255), 0, 255))


def image_stats(rgb, mask, alpha=None):
    foreground = mask >= 128
    ys, xs = np.nonzero(foreground)
    background = rgb[mask <= 2]
    result = {
        'width': int(rgb.shape[1]), 'height': int(rgb.shape[0]),
        'mask_threshold': 128, 'mask_min': int(mask.min()), 'mask_max': int(mask.max()),
        'foreground_pixels': int(foreground.sum()), 'foreground_fraction': float(foreground.mean()),
        'alpha_mass_pixels': float(mask.astype(np.float64).sum() / 255),
        'bbox_xyxy_exclusive': [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
            if xs.size else None,
        'foreground_centroid_xy': [float(xs.mean()), float(ys.mean())] if xs.size else None,
        'touches_image_edge': bool(foreground[0].any() or foreground[-1].any()
                                  or foreground[:, 0].any() or foreground[:, -1].any()),
        'background_median_rgb': np.median(background, axis=0).tolist() if len(background) else None,
        'background_p95_rgb': np.percentile(background, 95, axis=0).tolist() if len(background) else None,
    }
    if alpha is not None:
        original = alpha >= 128
        union = np.count_nonzero(original | foreground)
        result['source_png_alpha'] = {
            'min': int(alpha.min()), 'max': int(alpha.max()),
            'foreground_fraction': float(original.mean()),
            'iou_with_explicit_mask': float(np.count_nonzero(original & foreground) / union) if union else None,
            'used_for_black_composite': False,
        }
    return result


def verify(qwen, original, original_mask):
    manifest = read(qwen / 'manifest.json')
    recipe = read(qwen / 'prepared.json')
    expected_ids = [f'yaw{angle:03d}' for angle in range(0, 360, 45)]
    if manifest.get('complete') is not True:
        raise ValueError('Qwen view generation is not complete; do not freeze a partial set')
    if manifest['recipe'] != recipe or recipe['view_ids'] != expected_ids:
        raise ValueError('Expected the frozen eight-view recipe in nominal 45-degree order')
    if [item['id'] for item in manifest['views']] != expected_ids:
        raise ValueError('Manifest entries are missing, reordered or duplicated')
    if digest(original) != recipe['original_sha256']:
        raise ValueError('Original reference no longer matches the generation recipe')
    entries = [{'id': 'original', 'image': original, 'mask': original_mask, 'role': 'original',
                'image_sha256': digest(original), 'mask_sha256': digest(original_mask),
                'mask_method': 'approximate binary mask: max(R,G,B) > 4 in the original uint8 RGB',
                'source_camera_known': False}]
    prompt_ids, source_image_names = set(), set()
    for recorded in manifest['views']:
        name = recorded['id']
        directory = qwen / 'views' / name
        result = read(directory / 'result.json')
        prepared = read(directory / 'prepared.json')
        graph = read(directory / 'prompt.json')
        history = read(directory / 'history.json')
        if result != recorded:
            raise ValueError('Result differs from manifest: ' + name)
        if prepared['id'] != name or prepared['graph'] != graph:
            raise ValueError('Prepared graph differs: ' + name)
        if digest(directory / 'prompt.json') != result['prompt_sha256']:
            raise ValueError('Prompt hash differs: ' + name)
        if history['prompt'][1] != result['prompt_id'] or history['prompt'][2] != graph:
            raise ValueError('History belongs to another prompt: ' + name)
        status = history['status']
        if status.get('completed') is not True or status.get('status_str') != 'success':
            raise ValueError('Generation did not complete successfully: ' + name)
        cached = [str(node) for event, data in status['messages'] if event == 'execution_cached'
                  for node in data.get('nodes', [])]
        if '12' in cached:
            raise ValueError('Sampler was cached: ' + name)
        if not any(event == 'execution_success' for event, _ in status['messages']):
            raise ValueError('Missing successful execution event: ' + name)
        encoder = graph['10']['inputs']
        if (encoder['images.image_1'] != ['1', 0] or encoder['images.image_2'] != ['2', 0]
                or encoder['prompt'] != recipe['prompt']
                or len([key for key in encoder if key.startswith('images.')]) != 2):
            raise ValueError('Expected the original/coarse two-reference recipe: ' + name)
        sampler = graph['12']['inputs']
        if (sampler['steps'] != recipe['steps'] or sampler['cfg'] != recipe['cfg']
                or sampler['seed'] != result['seed']
                or graph['19']['inputs']['strength_model'] != recipe['lora_strength']
                or graph['9']['inputs']['dtype'] != recipe['cache_dtype']):
            raise ValueError('Generation settings differ: ' + name)
        prompt_ids.add(result['prompt_id'])
        source_image_names.add(graph['1']['inputs']['image'])
        image, mask = Path(result['image']), Path(result['mask'])
        if result['path'] != result['image'] or result['sha256'] != result['image_sha256']:
            raise ValueError('Image aliases disagree: ' + name)
        if digest(image) != result['image_sha256'] or digest(mask) != result['mask_sha256']:
            raise ValueError('Image or mask hash differs: ' + name)
        if digest(result['guide']) != prepared['guide_sha256']:
            raise ValueError('Guide changed: ' + name)
        if result.get('camera_is_calibration') is not False:
            raise ValueError('Guide conditioning must not claim to be image calibration: ' + name)
        if result['camera_conditioning'] != prepared['camera_conditioning']:
            raise ValueError('Guide camera differs: ' + name)
        entries.append({
            'id': name, 'image': image, 'mask': mask, 'role': 'generated',
            'image_sha256': result['image_sha256'], 'mask_sha256': result['mask_sha256'],
            'mask_method': 'provided BiRefNet mask from the same completed Qwen workflow',
            'seed': result['seed'], 'seconds': result['seconds'], 'prompt_id': result['prompt_id'],
            'prompt': directory / 'prompt.json', 'prompt_sha256': result['prompt_sha256'],
            'result_sha256': digest(directory / 'result.json'),
            'history_sha256': digest(directory / 'history.json'),
            'prepared_sha256': digest(directory / 'prepared.json'),
            'guide': result['guide'], 'guide_sha256': prepared['guide_sha256'],
            'nominal_guide_yaw_degrees': int(name[3:]),
            'camera_conditioning': result['camera_conditioning'], 'camera_is_calibration': False,
        })
    if len(prompt_ids) != 8 or len(source_image_names) != 1:
        raise ValueError('Expected eight distinct jobs sharing one original reference')
    return entries, recipe


def contact_grid(images, entries, tile, title):
    band, header = 36, 36
    sheet = Image.new('RGB', (tile * 3, (tile + band) * 3 + header), '#191d22')
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=15)
    draw.text((10, 10), title, fill='white', font=font)
    for index, (image, entry) in enumerate(zip(images, entries)):
        thumb = image.copy()
        thumb.thumbnail((tile, tile), Image.Resampling.LANCZOS)
        x, y = index % 3 * tile, index // 3 * (tile + band) + header
        label = 'Original reference' if entry['id'] == 'original' else entry['id'] + ' (guide label)'
        draw.text((x + 8, y + 9), label, fill='white', font=font)
        image_x, image_y = x + (tile - thumb.width) // 2, y + band + (tile - thumb.height) // 2
        sheet.paste(thumb, (image_x, image_y))
    return sheet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qwen', type=Path, required=True)
    parser.add_argument('--original', type=Path)
    parser.add_argument('--original-mask', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--background', choices=['black'], default='black')
    parser.add_argument('--tile', type=int, default=256)
    args = parser.parse_args()
    qwen = args.qwen.resolve()
    original = (args.original or Path(read(qwen / 'prepared.json')['original'])).resolve()
    original_mask = (args.original_mask or qwen.parent / 'original-mask.png').resolve()
    entries, recipe = verify(qwen, original, original_mask)
    preprocessing = {
        'background': 'black', 'equation': 'sRGB_encode(sRGB_decode(raw_RGB) * explicit_mask / 255)',
        'output': 'uint8 RGB PNG, original width and height; round-to-nearest',
        'raw_png_alpha': 'recorded for diagnostics, not additionally multiplied',
        'resize': 'none', 'crop': 'none', 'applied_to': 'all nine images including original',
        'comparison': 'one frozen preprocessed set shared by B504 and C1024',
    }
    output = args.out.resolve()
    freeze_identity = {'generation_manifest_sha256': digest(qwen / 'manifest.json'),
                       'generation_recipe_sha256': digest(qwen / 'prepared.json'),
                       'original_sha256': digest(original), 'original_mask_sha256': digest(original_mask),
                       'preprocessing': preprocessing,
                       'ordered_ids': [entry['id'] for entry in entries]}
    frozen_json(output / 'freeze.json', freeze_identity)
    input_views, mask_views, diagnostics = [], [], []
    raw_images, derived_images, mask_images = [], [], []
    original_binary, yaw_zero_binary = None, None
    for entry in entries:
        name = entry['id']
        raw_path, mask_path = output / 'raw' / (name + '.png'), output / 'masks' / (name + '.png')
        derived_path = output / 'inputs_black' / (name + '.png')
        copy_exact(entry['image'], raw_path, entry['image_sha256'])
        copy_exact(entry['mask'], mask_path, entry['mask_sha256'])
        with Image.open(raw_path) as source:
            source_mode = source.mode
            rgb = np.asarray(source.convert('RGB'))
            alpha = np.asarray(source.getchannel('A')) if 'A' in source.getbands() else None
        with Image.open(mask_path) as source:
            mask = np.asarray(source.convert('L'))
        if rgb.shape[:2] != mask.shape:
            raise ValueError('Image and mask dimensions differ: ' + name)
        if entry['role'] == 'generated' and rgb.shape[:2] != (recipe['size'], recipe['size']):
            raise ValueError('Unexpected generated image size: ' + name)
        derived = black_composite(rgb, mask)
        derived_path.parent.mkdir(parents=True, exist_ok=True)
        if derived_path.exists():
            with Image.open(derived_path) as previous:
                if previous.mode != 'RGB' or not np.array_equal(np.asarray(previous), derived):
                    raise ValueError('Frozen black composite differs: ' + name)
        else:
            Image.fromarray(derived, 'RGB').save(derived_path)
        derived_hash = digest(derived_path)
        row = {'id': name, 'path': str(derived_path), 'mask': str(mask_path), 'role': entry['role'],
               'sha256': derived_hash, 'mask_sha256': entry['mask_sha256'],
               'raw_path': str(raw_path), 'raw_sha256': entry['image_sha256']}
        for key in ['seed', 'nominal_guide_yaw_degrees', 'camera_conditioning', 'camera_is_calibration']:
            if key in entry:
                row[key] = entry[key]
        if 'prompt' in entry:
            prompt_path = output / 'prompts' / (name + '.json')
            copy_exact(entry['prompt'], prompt_path, entry['prompt_sha256'])
            row.update(prompt=str(prompt_path), prompt_sha256=entry['prompt_sha256'])
        input_views.append(row)
        mask_views.append({'id': name, 'mask_path': str(mask_path), 'mask_sha256': entry['mask_sha256'],
                           'method': entry['mask_method']})
        diagnostics.append({**{key: str(value) if isinstance(value, Path) else value for key, value in entry.items()},
                            'raw_snapshot': str(raw_path), 'mask_snapshot': str(mask_path),
                            'derived_path': str(derived_path), 'derived_sha256': derived_hash,
                            'source_mode': source_mode, 'stats': image_stats(rgb, mask, alpha)})
        raw_images.append(Image.fromarray(rgb, 'RGB'))
        derived_images.append(Image.fromarray(derived, 'RGB'))
        mask_images.append(Image.fromarray(mask, 'L').convert('RGB'))
        if name == 'original':
            original_binary = mask >= 128
        elif name == 'yaw000':
            yaw_zero_binary = mask >= 128
    metadata = {'count': 9, 'original_first': True, 'preprocessing': preprocessing,
                'guide_camera_policy': 'conditioning only, not calibrated poses; ZipSplat estimates poses',
                'generation_manifest': str(qwen / 'manifest.json'),
                'generation_manifest_sha256': freeze_identity['generation_manifest_sha256']}
    frozen_json(output / 'views.json', {**metadata, 'views': input_views})
    frozen_json(output / 'masks.json', {'views': mask_views})
    raw_grid = contact_grid(raw_images, entries, args.tile, 'RAW RGB - full images, source alpha ignored for this display')
    black_grid = contact_grid(derived_images, entries, args.tile, 'FROZEN ZIP INPUT - linear-light mask over black')
    mask_grid = contact_grid(mask_images, entries, args.tile, 'EXPLICIT MASKS - original approximate; generated BiRefNet')
    combined = Image.new('RGB', (raw_grid.width + black_grid.width + 12, raw_grid.height), '#353a42')
    combined.paste(raw_grid, (0, 0))
    combined.paste(black_grid, (raw_grid.width + 12, 0))
    for filename, image in [('contact.png', combined), ('raw-contact.png', raw_grid),
                            ('black-contact.png', black_grid), ('mask-contact.png', mask_grid)]:
        image.save(output / filename)
    duplicate = {'ids': ['original', 'yaw000'], 'kept': True, 'exclusion_policy': 'no automatic exclusion',
                 'interpretation': 'nearby/overlapping observations; these image metrics do not measure camera separation',
                 'raw_bytes_identical': entries[0]['image_sha256'] == entries[1]['image_sha256'],
                 'derived_bytes_identical': input_views[0]['sha256'] == input_views[1]['sha256']}
    if original_binary.shape == yaw_zero_binary.shape:
        union = np.count_nonzero(original_binary | yaw_zero_binary)
        duplicate['unregistered_mask_iou'] = float(np.count_nonzero(original_binary & yaw_zero_binary) / union) if union else None
        a, b = np.asarray(derived_images[0], dtype=np.float64), np.asarray(derived_images[1], dtype=np.float64)
        duplicate['unregistered_black_rgb_rmse_0_255'] = float(np.sqrt(np.mean((a - b) ** 2)))
    duplicate_groups = {}
    for row in input_views:
        duplicate_groups.setdefault(row['sha256'], []).append(row['id'])
    qa = {**metadata, 'status': 'complete', 'source_recipe': recipe,
          'original_mask_limitation': 'maxRGB > 4 threshold is approximate, not an independent segmentation ground truth',
          'images': diagnostics, 'original_yaw000_overlap': duplicate,
          'exact_derived_duplicate_groups': [ids for ids in duplicate_groups.values() if len(ids) > 1],
          'all_generated_masks_clear_of_image_edges': all(not row['stats']['touches_image_edge'] for row in diagnostics[1:]),
          'quality_exclusions': [], 'changed_raw_source_images': False,
          'note': 'This new nine-view dataset also changes generation settings from the older six-view experiment.',
          'artifacts': {name: {'path': str(output / name), 'sha256': digest(output / name)} for name in
                        ['views.json', 'masks.json', 'contact.png', 'raw-contact.png', 'black-contact.png', 'mask-contact.png']}}
    frozen_json(output / 'qa.json', qa)
    print(json.dumps({'output': str(output), 'views': len(input_views),
                      'views_sha256': digest(output / 'views.json'), 'masks_sha256': digest(output / 'masks.json'),
                      'qa_sha256': digest(output / 'qa.json'), 'overlap': duplicate,
                      'bbox': {row['id']: row['stats']['bbox_xyxy_exclusive'] for row in diagnostics}}, indent=2))


if __name__ == '__main__':
    main()
