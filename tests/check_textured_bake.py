"""Check known glTF texel colors and alpha behavior in both representations."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


RGB = np.array([(218, 46, 28), (30, 91, 223), (39, 189, 80), (226, 158, 34)])
FACES = ('+X', '-X', '+Y', '-Y', '+Z', '-Z')


def verify(directory, backend):
    manifest = json.loads((directory / 'manifest.json').read_bytes())
    assert manifest['backend'] == backend
    resolution = manifest['resolution']
    # Enough pixels to put a sample strictly inside each 16×16 fixture texel.
    assert resolution >= 128
    descriptor = manifest['textures']['colorMap']
    atlas = np.fromfile(directory / descriptor['path'], dtype='<f2').reshape(
        descriptor['height'], descriptor['width'], 4)
    transform = np.array(manifest['transform']).reshape(4, 4).T
    bounds = manifest['bakeBox']
    low, high = np.array(bounds['min']), np.array(bounds['max'])
    samples = holes = max_srgb_error = 0
    for face_id in FACES:
        face = manifest['faces'][face_id]
        proj = np.array(face['projectionMatrix']).reshape(4, 4).T
        view = np.array(face['viewMatrix']).reshape(4, 4).T
        with Image.open(directory / face['guide']) as image:
            guide = np.asarray(image.convert('RGBA'))
        depths = np.fromfile(directory / face['depth'], dtype='<f2').reshape(resolution, resolution)
        np.testing.assert_array_equal(np.isnan(depths), guide[..., 3] == 0,
                                      err_msg=f'{backend} {face_id}: guide depth background')
        assert np.isfinite(depths[guide[..., 3] != 0]).all(), f'{backend} {face_id}: non-finite foreground depth'
        col, row = face['tile']
        axis = 'XYZ'.index(face_id[1])
        side = np.eye(3)[axis] * (1 if face_id[0] == '+' else -1)
        up = np.array([0., 1., 0.]) if axis != 1 else np.array([0., 0., -side[1]])
        right = np.cross(up, side)

        def sample(point, ix, iy, covered):
            nonlocal samples, holes, max_srgb_error
            position = transform @ [*point, 1.]
            clip = proj @ view @ position
            ndc = clip[:2] / clip[3]
            x, y = np.floor((np.array([ndc[0], -ndc[1]]) + 1) * resolution / 2).astype(int)
            captured = guide[y, x]
            baked = atlas[(row + 1) * resolution - 1 - y, col * resolution + x].astype(float)
            detail = f'{backend} {face_id} texel({ix},{iy}) at guide({x},{y})'
            assert captured[3] == (255 if covered else 0), f'{detail}: guide coverage {captured[3]}'
            assert baked[3] == float(covered), f'{detail}: atlas coverage {baked[3]}'
            assert np.isfinite(depths[y, x]) == covered, f'{detail}: depth coverage {depths[y, x]}'
            if covered:
                rgb = RGB[2 * (iy >= 8) + (ix >= 8)]
                linear = baked[:3]
                srgb = np.where(linear <= .0031308, 12.92 * linear,
                                1.055 * np.maximum(linear, 0) ** (1 / 2.4) - .055) * 255
                for label, got in [('guide', captured[:3]), ('atlas', srgb.round().astype(int))]:
                    error = int(np.max(np.abs(got - rgb)))
                    max_srgb_error = max(max_srgb_error, error)
                    assert error <= 1, f'{detail}: {label} RGB {got}, expected {rgb}'
                expected = (position[axis] - low[axis]) / (high[axis] - low[axis])
                if side[axis] > 0:
                    expected = 1 - expected
                assert abs(float(depths[y, x]) - expected) < .001, f'{detail}: surface depth'
            else:
                holes += 1
            samples += 1

        # All six opaque block faces use UV=(0,0) at the camera's upper left.
        for ix, iy in [(1, 7), (10, 5), (3, 11), (12, 14), (3, 3), (13, 1)]:
            u, v = (ix + .5) / 16, (iy + .5) / 16
            point = [-.5, 0, 0] + np.array([1, 1.2, .5]) * (side / 2 + (u - .5) * right + (.5 - v) * up)
            sample(point, ix, iy, True)

        # From +Z and -Z the double-sided card exposes the same physical texels.
        # The center hole and outer corner notch must both contain background depth.
        if axis == 2:
            for iy in range(16):
                for ix in range(16):
                    u, v = (ix + .5) / 16, (iy + .5) / 16
                    cutout = (2 <= ix < 6 and 2 <= iy < 6) or (ix >= 12 and iy < 4)
                    sample([.08 + .92 * u, .7 - 1.2 * v, .13], ix, iy, not cutout)
    return dict(backend=backend, texelSamples=samples, cutoutSamples=holes, maxSRGBError=max_srgb_error)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('backend', choices=('webgl2', 'webgpu'))
    args = parser.parse_args()
    print(json.dumps(verify(args.directory, args.backend)))
