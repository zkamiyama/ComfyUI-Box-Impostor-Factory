"""Compare actual Inspector frames after changing only F's +X source RGB."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


HELPER = Path('/mnt/f/_dev/box-impostor-lab/results/seam-fade-20261119/compare_seam_variants.py')
spec = importlib.util.spec_from_file_location('capture_qa_helpers', HELPER)
qa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qa)
HEAD = (378, 115, 654, 271)
GHOST = (435, 155, 486, 231)


def front_fidelity(records, config, output):
    prior = json.loads((qa.OLD_EVIDENCE/'F-J-K-front-fidelity.json').read_text())
    residual = json.loads((qa.OLD_EVIDENCE/'K-front-residual-classification.json').read_text())
    crop = prior['headCrop']
    reference_entry = residual['outputs']['K-front-reference-rgba.png']
    interior_entry = residual['outputs']['K-front-source-interior-mask.png']
    for entry in [reference_entry, interior_entry, prior['mask']]:
        assert qa.sha(entry['path']) == entry['sha256']
    reference_image = Image.open(reference_entry['path']).convert('RGBA')
    reference = np.asarray(reference_image)
    mask = np.asarray(Image.open(prior['mask']['path']).convert('L')) > 0
    interior = np.asarray(Image.open(interior_entry['path']).convert('L')) > 0
    assert int(mask.sum()) == 10936 and int(interior.sum()) == 10953
    face_rect = residual['faceRectangleFullImage']
    face = np.zeros(mask.shape, bool)
    face[face_rect[1]-crop[1]:face_rect[3]-crop[1], face_rect[0]-crop[0]:face_rect[2]-crop[0]] = True
    face &= interior
    results = {}
    displays = [('RGB reference', reference_image, None)]
    for name, record in records.items():
        image = record['images'][0].crop(crop)
        array = np.asarray(image)
        difference = np.abs(array[..., :3].astype(float)-reference[..., :3])
        mae = difference.mean(axis=2)
        item = {'sRGBMAE255FixedPriorMask': float(difference[mask].mean()),
                'sampledPixels': int(mask.sum()), 'sourceInterior': {}}
        for region_name, region in [('head', interior), ('faceRectangle', face)]:
            opaque = region & (array[..., 3] > 250)
            item['sourceInterior'][region_name] = {'pixels': int(region.sum()),
                'transparentAlpha0to5': int((region & (array[..., 3] <= 5)).sum()),
                'partialAlpha6to250': int((region & (array[..., 3] > 5) & (array[..., 3] <= 250)).sum()),
                'opaqueMeanRGBErrorOver16': int((opaque & (mae > 16)).sum())}
        results[name] = item
        displays.append((config['cases'][name]['label'], image, item))
    width, height = reference_image.size
    cell = width*3+32
    canvas = Image.new('RGB', (len(displays)*cell+16, height*3+184), qa.BG)
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 16), 'Front preservation / exact prior reference and masks', font=qa.font(25), fill=qa.TEXT)
    draw.text((20, 55), 'Fixed 10,936-pixel fidelity mask; same current renderer for F, Q1 and Q2.', font=qa.font(16), fill=qa.MUTED)
    for i, (label, image, item) in enumerate(displays):
        x = 16+i*cell
        draw.text((x, 88), label, font=qa.font(18), fill=qa.TEXT)
        canvas.paste(qa.backdrop(image).resize((width*3, height*3), Image.Resampling.NEAREST), (x, 120))
        if item:
            draw.text((x, height*3+134), f"MAE {item['sRGBMAE255FixedPriorMask']:.4f} /255", font=qa.font(18), fill=qa.TEXT)
    path = output/'Q1-Q2-box-front-preservation.png'
    canvas.save(path)
    return path, {'referenceImage': reference_entry, 'fidelityMask': prior['mask'],
                  'residualMask': interior_entry, 'headCrop': crop, 'faceRectangle': face_rect,
                  'metrics': results, 'historicalKReferenceMAE': prior['metrics']['K-view']['sRGBMAE255'],
                  'freshFMinusHistoricalKMAE': results['K-view']['sRGBMAE255FixedPriorMask']-prior['metrics']['K-view']['sRGBMAE255'],
                  'interpretation': 'Same registered +Z source and unchanged historical masks. Within-trial comparisons use fresh F-current with the same renderer as Q1/Q2. Image fidelity, not 3D accuracy.'}


def invariants(records, config):
    base = records['K-view']
    results = {}
    for name, record in records.items():
        for report in record['reports']:
            assert report['assetId'] == config['cases'][name]['assetId']
            assert report['display']['lighting'] == 'unlit'
            assert report['display']['sampling'] == 'native'
            assert report['display']['colorSampling'] == 'view'
        for angle in config['angles']:
            anchor = base['frames'][angle]['actualCamera']
            actual = record['frames'][angle]['actualCamera']
            matrix_differences = {key: float(np.max(np.abs(np.asarray(actual[key])-anchor[key])))
                                  for key in ['matrixWorld', 'projectionMatrix']}
            assert all(value <= 1e-8 for value in matrix_differences.values()), (name, angle, matrix_differences)
            a = np.asarray(base['images'][angle])
            b = np.asarray(record['images'][angle])
            alpha_equal = bool(np.array_equal(a[..., 3], b[..., 3]))
            assert alpha_equal, (name, angle, 'Coverage alpha changed')
            delta = np.abs(a[..., :3].astype(float)-b[..., :3].astype(float))
            foreground = (a[..., 3] > 0) | (b[..., 3] > 0)
            results[f'{name}/{angle}'] = {'actualYawDegrees': record['frames'][angle]['actualYawDegrees'],
                'maxMatrixDifference': matrix_differences, 'alphaByteExactToK': alpha_equal,
                'rgbaByteExactToK': bool(np.array_equal(a, b)),
                'foregroundRGBChangedPixelsToK': int(np.count_nonzero(np.any(delta > 0, axis=2) & foreground)),
                'foregroundRGBMaxByteDifferenceToK': float(delta[foreground].max()),
                'foregroundRGBMeanByteDifferenceToK': float(delta[foreground].mean())}
    return results


def board(records, config, angles, crop, scale, title, footer, path):
    names = list(records)
    cw, ch = (crop[2]-crop[0])*scale, (crop[3]-crop[1])*scale
    gutter = 24
    min_col = 300
    cell_width = max(cw, min_col)+gutter
    canvas = Image.new('RGB', (len(names)*cell_width+gutter, len(angles)*(ch+66)+116), qa.BG)
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 16), title, font=qa.font(25), fill=qa.TEXT)
    draw.text((24, 53), 'Same geometry, depth, alpha, camera and view/unlit shader; only +X source RGB changes.', font=qa.font(17), fill=qa.MUTED)
    for row, angle in enumerate(angles):
        for col, name in enumerate(names):
            x, y = gutter+col*cell_width, 88+row*(ch+66)
            draw.text((x, y), config['cases'][name]['label'], font=qa.font(19), fill=qa.TEXT)
            draw.text((x, y+25), f"Yaw {records[name]['frames'][angle]['actualYawDegrees']:+.6f} deg", font=qa.font(15), fill=qa.MUTED)
            im = qa.backdrop(records[name]['images'][angle].crop(crop)).resize((cw, ch), Image.Resampling.NEAREST)
            canvas.paste(im, (x, y+46))
    draw.text((24, canvas.height-24), footer, font=qa.font(15), fill=qa.MUTED)
    canvas.save(path)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    records = {name: qa.load(name, [Path(p) for p in item['captures']], config['angles'])
               for name, item in config['cases'].items()}
    checks = invariants(records, config)
    args.output.mkdir(parents=True, exist_ok=True)
    outputs = [board(records, config, [45], HEAD, 2, 'Actual Box Impostor / +45 degree head comparison',
                     f'Fixed head crop {list(HEAD)}; 2x nearest-neighbor. Raw 1024px Inspector snapshots.',
                     args.output/'Q1-Q2-box-p45-head-comparison.png'),
               board(records, config, [45], GHOST, 4, 'Actual +45 degree overlap / fixed detail',
                     f'Fixed crop {list(GHOST)}; 4x nearest-neighbor. No image retouching.',
                     args.output/'Q1-Q2-box-p45-ghost-comparison.png'),
               board(records, config, [0, 30, 60], HEAD, 2, 'Actual Box Impostor / neighboring camera views',
                     f'Fixed head crop {list(HEAD)}; 2x nearest-neighbor. Same capture path at all angles.',
                     args.output/'Q1-Q2-box-head-other-angles.png')]
    front_path, front_metrics = front_fidelity(records, config, args.output)
    outputs.append(front_path)
    body = Image.new('RGB', (len(records)*360+24, len(config['angles'])*394+104), qa.BG)
    draw = ImageDraw.Draw(body)
    draw.text((24, 16), 'Same actual captures / full model', font=qa.font(25), fill=qa.TEXT)
    for row, angle in enumerate(config['angles']):
        for col, (name, record) in enumerate(records.items()):
            x, y = 24+col*360, 62+row*394
            draw.text((x, y), f"{config['cases'][name]['label']} / {angle:+g} deg", font=qa.font(16), fill=qa.TEXT)
            im = qa.backdrop(record['images'][angle]).resize((336, 336), Image.Resampling.NEAREST)
            body.paste(im, (x, y+30))
    draw.text((24, body.height-24), 'Raw 1024px frames shown at 336px for layout. Native alpha and actual camera matched.', font=qa.font(15), fill=qa.MUTED)
    body_path = args.output/'Q1-Q2-box-body-comparison.png'
    body.save(body_path)
    outputs.append(body_path)
    proof = {'kind': __doc__, 'config': str(args.config), 'configSHA256': qa.sha(args.config),
             'scriptSHA256': qa.sha(__file__), 'helper': str(HELPER), 'helperSHA256': qa.sha(HELPER),
             'anglesDegrees': config['angles'], 'fixedHeadCrop': HEAD, 'fixedGhostCrop': GHOST,
             'baselineNote': config['baselineNote'],
             'invariantChecks': checks, 'frontFidelity': front_metrics,
             'captureReports': {name: record['reports'] for name, record in records.items()},
             'frames': {name: record['frames'] for name, record in records.items()},
             'outputs': [{'path': str(path), 'sha256': qa.sha(path), 'bytes': path.stat().st_size,
                          'size': list(Image.open(path).size)} for path in outputs],
             'interpretation': 'RGB changes measure experimental intervention and front preservation. They are not quality or 3D accuracy scores; inspect ghosting visually on the same fixed crop.',
             'processing': 'Official Inspector snapshots. Fixed crops, nearest-neighbor display, solid background composition only. No retouching, pose correction, sharpening or generated frames.'}
    path = args.output/'Q1-Q2-box-provenance.json'
    path.write_text(json.dumps(proof, indent=2)+'\n')
    print(json.dumps({'proof': str(path), 'proofSHA256': qa.sha(path), 'outputs': proof['outputs'],
                      'checks': checks, 'frontMetrics': front_metrics['metrics']}))


if __name__ == '__main__':
    main()
