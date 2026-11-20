#!/usr/bin/env python3
"""Export a properly rotated degree-0/1 Gaussian PLY for unchanged 3D-Inspector."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from zip_render_faces import sha, write_json


def canonical_gaussians(packed, transform):
    packed = np.asarray(packed)
    transform = np.asarray(transform, dtype=float)
    if packed.ndim != 2 or packed.shape[1] not in (14, 23):
        raise ValueError('Canonical export currently supports degree-0 or degree-1 packed Gaussians')
    scale = np.linalg.norm(transform[:3, 0])
    rotation = transform[:3, :3] / scale
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or np.linalg.det(rotation) < .999:
        raise ValueError('Canonical transform must be a proper uniform similarity')
    result = np.asarray(packed, dtype='f4').copy()
    result[:, :3] = packed[:, :3] @ transform[:3, :3].T + transform[:3, 3]
    result[:, 3:6] = packed[:, 3:6] * scale
    q_xyzw = Rotation.from_matrix(rotation).as_quat()
    q = np.r_[q_xyzw[3], q_xyzw[:3]]
    source = packed[:, 6:10].astype(float)
    source /= np.linalg.norm(source, axis=1, keepdims=True)
    destination = np.column_stack((q[0] * source[:, 0] - source[:, 1:] @ q[1:],
                                   q[0] * source[:, 1:] + source[:, 0, None] * q[1:] + np.cross(q[1:], source[:, 1:])))
    destination /= np.linalg.norm(destination, axis=1, keepdims=True)
    result[:, 6:10] = destination
    if packed.shape[1] == 23:
        # gsplat basis l=1 is C1 * [-y, z, -x]. Coefficients transform contragrediently to directions.
        basis_to_xyz = np.array([[0., 0., -1.], [-1., 0., 0.], [0., 1., 0.]])
        sh_rotation = basis_to_xyz.T @ rotation @ basis_to_xyz
        original_sh = packed[:, 11:].reshape(-1, 4, 3)
        transformed_sh = result[:, 11:].reshape(-1, 4, 3)
        transformed_sh[:, 1:] = np.einsum('ij,njc->nic', sh_rotation, original_sh[:, 1:])
    return result


def write_ply(path, packed):
    count = len(packed)
    sh = packed[:, 11:].reshape(count, -1, 3)
    dc = sh[:, 0]
    rest = sh[:, 1:].transpose(0, 2, 1).reshape(count, -1)
    opacity = np.clip(packed[:, 10:11], 1e-6, 1 - 1e-6)
    logits = np.log(opacity) - np.log1p(-opacity)
    values = np.concatenate((packed[:, :3], np.zeros((count, 3), dtype='f4'), dc, rest, logits,
                             np.log(np.maximum(packed[:, 3:6], 1e-12)), packed[:, 6:10]), axis=1)
    names = ['x', 'y', 'z', 'nx', 'ny', 'nz'] + [f'f_dc_{i}' for i in range(3)]
    names += [f'f_rest_{i}' for i in range(rest.shape[1])]
    names += ['opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3']
    elements = np.empty(count, dtype=[(name, '<f4') for name in names])
    for index, name in enumerate(names):
        elements[name] = values[:, index]
    header = 'ply\nformat binary_little_endian 1.0\n'
    header += 'comment Canonical similarity with degree-1 spherical harmonics rotated\n'
    header += f'element vertex {count}\n'
    header += ''.join(f'property float {name}\n' for name in names) + 'end_header\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(header.encode('ascii'))
        elements.tofile(stream)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--transform', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--packed-output', type=Path)
    args = parser.parse_args()
    packed = np.load(args.input, allow_pickle=False)
    transform = json.loads(args.transform.read_text())['canonical_from_scene']
    canonical = canonical_gaussians(packed, transform)
    write_ply(args.output, canonical)
    if args.packed_output:
        with args.packed_output.open('xb') as stream:
            np.save(stream, canonical, allow_pickle=False)
    report = {'inputSha256': sha(args.input), 'transformSha256': sha(args.transform),
              'outputSha256': sha(args.output), 'gaussians': len(canonical),
              'packedWidth': canonical.shape[1], 'canonical_from_scene': transform,
              'meanBounds': [canonical[:, :3].min(0).tolist(), canonical[:, :3].max(0).tolist()],
              'method': 'Uniform similarity for means/covariance and corresponding real SH l=1 rotation',
              'shBasis': 'C0, -C1*y, C1*z, -C1*x',
              'shBasisSource': 'https://github.com/nerfstudio-project/gsplat/blob/main/gsplat/cuda/_torch_impl.py',
              'originalGaussiansPreserved': True}
    write_json(args.output.with_suffix(args.output.suffix + '.json'), report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
