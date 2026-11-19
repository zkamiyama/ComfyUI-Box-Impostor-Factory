"""Review existing +60 captures on a light background without changing their RGBA."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


WORK = Path('/mnt/f/_dev/box-impostor-lab/results/qwen-image-20261119')
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    config = json.loads((WORK/'capture-cases.json').read_text())
    records = {}
    for key, case in config['cases'].items():
        cap = json.loads(Path(case['captures'][0]).read_text())
        frame = next(x for x in cap['yawFrames'] if x['requestedYawDegrees'] == 60)
        path = Path(frame['path'])
        assert sha(path) == frame['sha256']
        image = Image.open(path).convert('RGBA')
        records[key] = {'path': str(path), 'sha256': sha(path), 'image': image,
                        'array': np.asarray(image), 'label': case['label']}
    base = records['K-view']['array']
    for record in records.values():
        assert np.array_equal(base[..., 3], record['array'][..., 3])
    coverage = base[..., 3] > 0
    ys, xs = np.nonzero(coverage)
    crop = (max(0, int(xs.min())-12), max(0, int(ys.min())-12),
            min(1024, int(xs.max())+13), min(1024, int(ys.max())+13))
    opaque = base[..., 3] > 250
    eroded = np.asarray(Image.fromarray(opaque.astype('uint8')*255).filter(ImageFilter.MinFilter(7))) > 0
    edge = opaque & ~eroded
    base_dark = np.max(base[..., :3], axis=2) < 16
    metrics = {}
    for key, record in records.items():
        array = record['array']
        dark = np.max(array[..., :3], axis=2) < 16
        new_dark = edge & dark & ~base_dark
        positions = np.argwhere(new_dark)
        metrics[key] = {'opaqueInnerEdgeBandPixels': int(edge.sum()),
                        'maxRGBBelow16InBand': int((edge & dark).sum()),
                        'newDarkPixelsVsBaselineInBand': int(new_dark.sum()),
                        'newDarkPixelCoordinatesXY': [[int(x), int(y)] for y, x in positions]}
    width, height = crop[2]-crop[0], crop[3]-crop[1]
    canvas = Image.new('RGB', (3*(width+24)+24, height+150), (21, 24, 30))
    draw = ImageDraw.Draw(canvas)
    font = lambda size: ImageFont.truetype(FONT, size)
    draw.text((24, 15), 'Actual +60 degree captures / silhouette edge review', font=font(23), fill=(237, 241, 247))
    draw.text((24, 49), 'Original RGBA at 1:1 on a common light background. Same alpha in all cases.', font=font(15), fill=(177, 192, 211))
    for index, record in enumerate(records.values()):
        x = 24+index*(width+24)
        draw.text((x, 79), record['label'], font=font(15), fill=(237, 241, 247))
        image = record['image'].crop(crop)
        bg = Image.new('RGBA', image.size, (235, 238, 242, 255))
        bg.alpha_composite(image)
        canvas.paste(bg.convert('RGB'), (x, 106))
    draw.text((24, canvas.height-25), f'Fixed crop {list(crop)}. No resizing, sharpening, retouching or new render.', font=font(14), fill=(177, 192, 211))
    figure = WORK/'evidence/Q1-Q2-box-p60-edge-review.png'
    canvas.save(figure)
    proof = {'kind': __doc__, 'scriptSHA256': sha(__file__), 'crop': crop,
             'inputs': {key: {field: value for field, value in record.items() if field not in ['image', 'array']}
                        for key, record in records.items()},
             'alphaByteExactAcrossCases': True, 'metrics': metrics,
             'metricDefinition': 'Opaque alpha >250; 3-pixel inner band = opaque minus 7x7 min-filter erosion. Dark means maximum RGB byte <16. New dark pixels are an inspection locator, not a halo or quality classification; natural black shoes/hair may satisfy it.',
             'output': {'path': str(figure), 'sha256': sha(figure), 'bytes': figure.stat().st_size},
             'processing': 'Original snapshot RGBA, fixed common crop, alpha-composited on RGB(235,238,242). No pixel retouching or geometry change.'}
    path = WORK/'evidence/edge-review-provenance.json'
    path.write_text(json.dumps(proof, indent=2)+'\n')
    print(json.dumps({'proof': str(path), 'proofSHA256': sha(path), 'output': proof['output'],
                      'metrics': metrics, 'crop': crop}))


if __name__ == '__main__':
    main()
