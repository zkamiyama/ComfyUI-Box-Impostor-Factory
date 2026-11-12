"""Exercise the pinned native PLY reader and EWA renderer in a real ComfyUI queue."""
import json
from pathlib import Path
import struct

import numpy as np
from PIL import Image


POINTS = np.array([[.20, .09, .06], [-.15, -.11, -.07]])


def write_splat_fixture(inputs):
    # Native 3DGS storage in this frontend maps to three.js world as (x,-y,-z).
    path = inputs / '3d' / 'two-gaussians.ply'
    path.parent.mkdir(exist_ok=True)
    header = ['ply', 'format binary_little_endian 1.0', 'element vertex 2',
              *['property float ' + k for k in ('x', 'y', 'z', 'scale_0', 'scale_1', 'scale_2')],
              *['property uchar ' + k for k in ('red', 'green', 'blue')], 'end_header']
    with path.open('wb') as file:
        file.write(('\n'.join(header)+'\n').encode('ascii'))
        for p, color in zip(POINTS, ((255, 0, 0), (0, 255, 0))):
            file.write(struct.pack('<6f3B', *(p * [1, -1, -1]), *([np.log(.014)] * 3), *color))
    # The legacy Load3D node accepts PLY and requires three recorded viewport images.
    Image.new('RGBA', (32, 32), (255, 255, 255, 255)).save(inputs / 'splat-viewport.png')


def check_splat(queue, node, inputs, output, fixture_name, manifest):
    r = manifest['resolution']
    png = 'splat-viewport.png'
    prompt = {
        '1': node('BICanonicalizeProxy', filename=fixture_name),
        '2': node('BIMakeSixGuides', proxy=['1', 0], resolution=r, backend='webgl2'),
        '3': node('Load3D', model_file='3d/two-gaussians.ply', width=32, height=32,
                  image=dict(image=png, mask=png, normal=png, recording='', camera_info={})),
        '4': node('File3DToSplat', model_3d=['3', 6]),
    }
    faces = list(manifest['faceOrder'])
    for i, face_id in enumerate(faces):
        guide, save = str(2*i+5), str(2*i+6)
        prompt[guide] = node('BITripoSplatGuide', splat=['4', 0], guides=['2', 6],
                             face_id=face_id, coordinate_space='canonical_box')
        prompt[save] = node('SaveImage', images=[guide, 0], filename_prefix='native-splat/' + str(i))
    outputs = queue(prompt)
    reports = []
    for i, face_id in enumerate(faces):
        path = outputs[str(2*i+6)]['images'][0]
        rgb = np.asarray(Image.open(output/'output'/path['subfolder']/path['filename']).convert('RGB'), dtype=float)
        assert rgb.shape == (r, r, 3), (face_id, rgb.shape)
        face = manifest['faces'][face_id]
        vp = np.asarray(face['projectionMatrix']).reshape(4,4, order='F') @ np.asarray(face['viewMatrix']).reshape(4,4, order='F')
        for j, p in enumerate(POINTS):
            clip = vp @ np.append(p, 1.)
            expected = (clip[:2]/clip[3] * [.5, -.5] + .5) * r - .5
            # Difference from neutral background isolates the red/green point even at soft edges.
            weight = np.maximum(0., rgb[..., j] - rgb[..., 1-j])
            yy, xx = np.indices(weight.shape)
            assert weight.max() > 90, (face_id, j, 'missing colored splat', weight.max())
            centroid = np.array([(xx*weight).sum(), (yy*weight).sum()]) / weight.sum()
            assert np.max(np.abs(centroid-expected)) < 1.25, (face_id, j, centroid, expected)
        reports.append(face_id)
    cache = output / 'output' / 'box_impostor' / '.stages' / 'splat-guides'
    checks = [json.loads(p.read_text()) for p in cache.glob('*/qa.json')]
    checks = [row for row in checks if row['asset'] == manifest['asset_id']]
    assert {row['faceId'] for row in checks} == set(faces)
    assert all(row['qa']['splatPixels'] > 0 and row['qa']['meshPixels'] > 0 for row in checks)
    assert all(row['qa']['relativeDepth']['status'] in ('constant-depth', 'measured', 'insufficient-overlap')
               for row in checks)
    dest = output / 'frontend'
    dest.mkdir(exist_ok=True)
    (dest/'native-splat-proof.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
    print('Native Gaussian splat projections matched asymmetric color landmarks on', reports, flush=True)
