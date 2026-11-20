#!/usr/bin/env python3
"""Build one Box frame from gauged, foreground-filtered reconstruction branches."""
import argparse
import copy
import json
from pathlib import Path
import shutil

import numpy as np

from zip_render_faces import FACES, make_face, sha, write_json


def union_bounds(points, quantile=.001, padding_height_fraction=.02):
    robust = {name: np.quantile(value, [quantile, 1 - quantile], axis=0) for name, value in points.items()}
    low = np.min([value[0] for value in robust.values()], axis=0)
    high = np.max([value[1] for value in robust.values()], axis=0)
    padding = (high[1] - low[1]) * padding_height_fraction
    return low, high, low - padding, high + padding, robust, float(padding)


def build(args):
    source = json.loads(args.template.read_text())
    definitions = json.loads(args.branches.read_text())['branches']
    points, records = {}, {}
    for item in definitions:
        name = item['id']
        if name in points:
            raise ValueError('Branch IDs must be unique')
        files = {}
        for key in ('gaussians', 'transform'):
            path = Path(item[key])
            files[key] = path if path.is_absolute() else args.branches.parent / path
        data = np.load(files['gaussians'], allow_pickle=False)
        transform = np.asarray(json.loads(files['transform'].read_text())['canonical_from_scene'], dtype=float)
        scale = float(np.linalg.norm(transform[:3, 0]))
        rotation = transform[:3, :3] / scale
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or np.linalg.det(rotation) < .999:
            raise ValueError('Branch transform is not a proper uniform similarity: ' + name)
        canonical = data[:, :3] @ transform[:3, :3].T + transform[:3, 3]
        if not len(canonical) or not np.isfinite(canonical).all():
            raise ValueError('Branch needs finite foreground points: ' + name)
        points[name] = canonical
        records[name] = {'gaussians': str(files['gaussians'].resolve()), 'gaussiansSha256': sha(files['gaussians']),
                         'transform': str(files['transform'].resolve()), 'transformSha256': sha(files['transform']),
                         'canonicalUniformScaleUnchanged': scale, 'count': len(canonical),
                         'allMeanBounds': [canonical.min(0).tolist(), canonical.max(0).tolist()]}
    low, high, bake_low, bake_high, robust, padding = union_bounds(points, args.quantile, args.padding_height_fraction)
    old_low, old_high = np.asarray(source['bakeBox']['min']), np.asarray(source['bakeBox']['max'])
    for name, values in points.items():
        old_outside = (values < old_low) | (values > old_high)
        outside = (values < bake_low) | (values > bake_high)
        records[name].update(robustMeanBounds=robust[name].tolist(),
                            meansOutsidePreviousFrame=int(old_outside.any(1).sum()),
                            meansOutsideCommonFrame=int(outside.any(1).sum()),
                            outsideCommonFramePerAxis=outside.sum(0).tolist())
    report = {'method': 'Union of per-branch robust foreground bounds after existing gauge transforms',
              'quantile': args.quantile, 'paddingHeightFraction': args.padding_height_fraction,
              'paddingCanonicalUnits': padding, 'templateSha256': sha(args.template),
              'branchDefinitionsSha256': sha(args.branches), 'branches': records,
              'previousBakeBox': source['bakeBox'],
              'tightBounds': {'min': low.tolist(), 'max': high.tolist()},
              'bakeBox': {'min': bake_low.tolist(), 'max': bake_high.tolist()},
              'canonicalTransformsModified': False,
              'limitations': ['Robust bounds exclude a small number of center outliers.',
                              'Gaussian tails may extend beyond center-derived bounds.',
                              'The retained proxy GLB is metadata only and is not these reconstructions.']}
    args.output.mkdir(parents=True, exist_ok=False)
    frame = {key: copy.deepcopy(source[key]) for key in ('schemaVersion', 'asset_id', 'layout', 'proxy',
                   'proxySha256', 'transform', 'inverseTransform', 'faceOrder', 'resolution')}
    frame.update(tightBounds=report['tightBounds'], bakeBox=report['bakeBox'])
    frame['faces'] = {fid: make_face(frame['bakeBox'], source['faces'][fid]) for fid in FACES.values()}
    frame['provenance'] = {'commonFrame': report, 'proxyIsMetadataOnly': True}
    proxy = (args.template.parent / source['proxy']).resolve()
    target = (args.output / source['proxy']).resolve()
    if not proxy.is_relative_to(args.template.parent.resolve()) or not target.is_relative_to(args.output.resolve()):
        raise ValueError('Proxy path escapes frame directory')
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(proxy, target)
    if sha(target) != source['proxySha256']:
        raise ValueError('Template proxy digest changed')
    write_json(args.output / 'manifest.json', frame)
    write_json(args.output / 'bounds.json', report)
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--branches', type=Path, required=True,
                        help='JSON: branches [{id,gaussians,transform}], paths relative to JSON or absolute')
    parser.add_argument('--template', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--quantile', type=float, default=.001)
    parser.add_argument('--padding-height-fraction', type=float, default=.02)
    build(parser.parse_args())


if __name__ == '__main__':
    main()
