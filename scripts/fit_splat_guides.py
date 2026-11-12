"""Fit one XYZ scale and translation to six cached TripoSplat guide silhouettes."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

FACES = ('+X', '-X', '+Y', '-Y', '+Z', '-Z')


def fit_silhouettes(manifest, splat_masks, mesh_masks):
    """Fit a shared diagonal transform in canonical box space, with the splat already oriented."""
    r = manifest['resolution']
    if any(masks[f].shape != (r, r) for masks in (splat_masks, mesh_masks) for f in FACES):
        raise ValueError('Every silhouette must match the manifest resolution.')
    axes = {}
    for f in FACES:
        face = manifest['faces'][f]
        up, side = np.asarray(face['up']), np.asarray(face['side'])
        # Image rows point down, columns right. Each maps to exactly one signed world axis.
        axes[f] = []
        for vector in (-up, np.cross(up, side)):
            axis = int(np.argmax(abs(vector)))
            if np.count_nonzero(vector) != 1 or abs(vector[axis]) != 1:
                raise ValueError('Expected axis-aligned Baker faces in canonical box space.')
            axes[f].append((axis, int(vector[axis])))

    def view_scores(params, moving, fixed):
        scales, shifts = np.exp(params[:3]), params[3:]
        size = next(iter(fixed.values())).shape[0]
        coords = (np.arange(size) + .5) / size - .5
        values = []
        for f in FACES:
            indices, inside = [], []
            for axis, sign in axes[f]:
                source = (coords - sign * shifts[axis]) / scales[axis]
                index = np.floor((source + .5) * size).astype(int)
                inside.append((index >= 0) & (index < size))
                indices.append(index.clip(0, size - 1))
            warped = moving[f][np.ix_(*indices)] & inside[0][:, None] & inside[1][None, :]
            union = np.count_nonzero(warped | fixed[f])
            values.append(np.count_nonzero(warped & fixed[f]) / union if union else 0.)
        return np.asarray(values)

    def sampled(masks):
        size = min(256, r)
        indices = np.floor((np.arange(size) + .5) * r / size).astype(int)
        return {f: masks[f][np.ix_(indices, indices)] for f in FACES}

    moving, fixed = sampled(splat_masks), sampled(mesh_masks)
    size = fixed['+Z'].shape[0]
    scale_guesses, shift_guesses = [[] for _ in range(3)], [[] for _ in range(3)]
    for f in FACES:
        if not moving[f].any() or not fixed[f].any():
            continue
        # Robust projected bounds seed the same transform for each opposing pair of cameras.
        for dim, (axis, sign) in enumerate(axes[f]):
            bounds = [np.quantile((np.nonzero(mask)[dim] + .5) / size - .5, [.02, .98])
                      for mask in (moving[f], fixed[f])]
            old, new = bounds
            if min(np.ptp(old), np.ptp(new)) <= 1 / size:
                continue
            scale = np.ptp(new) / np.ptp(old)
            scale_guesses[axis].append(scale)
            shift_guesses[axis].append(sign * (new.mean() - scale * old.mean()))
    if not any(scale_guesses):
        raise ValueError('No measurable pair of silhouettes for registration.')
    guess = np.r_[np.log([np.median(s) if s else 1. for s in scale_guesses]),
                  [np.median(t) if t else 0. for t in shift_guesses]]

    def refine(start):
        params = start.copy()
        best = view_scores(params, moving, fixed).mean()
        for step in (.18, .09, .045, .0225, .01125, .005625, .0028125):
            for _ in range(4):
                previous = best
                for k in range(6):
                    origin = params.copy()
                    for sign in (-1, 1):
                        trial = origin.copy()
                        trial[k] += sign * (step if k < 3 else step / 4)
                        score = view_scores(trial, moving, fixed).mean()
                        if score > best:
                            best, params = score, trial
                if previous == best:
                    break
        return params

    identity = np.zeros(6)
    candidates = (identity, refine(identity), refine(guess))
    # Evaluate the final choices at their original resolution, including the unchanged splat.
    params = max(candidates, key=lambda p: view_scores(p, splat_masks, mesh_masks).mean())
    scale = np.exp(params[:3])
    bounds = manifest['bakeBox']
    center = np.mean([bounds['min'], bounds['max']], axis=0)
    extent = np.subtract(bounds['max'], bounds['min'])
    world_shift = center * (1 - scale) + extent * params[3:]
    # Native TransformSplat uses the 3DGS frame (X right, Y down, Z forward).
    native_shift = world_shift * [1, -1, -1]
    transform = dict(zip(('translate_x', 'translate_y', 'translate_z'), native_shift.tolist()))
    transform.update(rotate_x=0., rotate_y=0., rotate_z=0.)
    transform.update(zip(('scale_x', 'scale_y', 'scale_z'), scale.tolist()))
    return {'transformSplat': transform,
            'originalIoU': dict(zip(FACES, view_scores(identity, splat_masks, mesh_masks).tolist())),
            'predictedSilhouetteIoU': dict(zip(FACES, view_scores(params, splat_masks, mesh_masks).tolist()))}


def fit_cached(asset_dir, report_path):
    manifest = json.loads((asset_dir / 'manifest.json').read_bytes())
    reference = json.loads(report_path.read_bytes())
    if reference['asset'] != manifest['asset_id'] or reference['coordinateSpace'] != 'canonical_box':
        raise ValueError('Use a canonical_box projection of this Box asset.')
    keys = ('asset', 'splatSha256', 'coordinateSpace', 'resolution', 'rendererSha256')
    moving, fixed = {}, {}
    for path in (asset_dir.parent / '.stages' / 'splat-guides').glob('*/qa.json'):
        record = json.loads(path.read_bytes())
        if any(record.get(k) != reference[k] for k in keys):
            continue
        face = record['faceId']
        stage = path.parent
        if stage.name != record['stageKey'] or face not in FACES:
            raise ValueError('Invalid cached projection identity.')
        guide = (asset_dir / manifest['faces'][face]['guide']).resolve()
        if not guide.is_relative_to(asset_dir.resolve()):
            raise ValueError('Unexpected guide path outside Box asset.')
        if hashlib.sha256(guide.read_bytes()).hexdigest() != record['meshGuideSha256']:
            continue
        with Image.open(guide) as image:
            fixed[face] = np.asarray(image.convert('RGBA'))[..., 3] >= 128
        coverage = np.load(stage / 'foreground.npy', allow_pickle=False)
        if not np.isfinite(coverage).all():
            raise ValueError('Non-finite splat coverage.')
        moving[face] = coverage > .5
    missing = [f for f in FACES if f not in moving]
    if missing:
        raise ValueError('Render the same splat and Box in workflow 08 for the missing faces: ' + ', '.join(missing))
    return {**{k: reference[k] for k in keys},
            'application': 'Apply this TransformSplat after the transform used to render the six input guides.',
            **fit_silhouettes(manifest, moving, fixed)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('asset_dir', type=Path, help='ComfyUI/output/box_impostor/<asset_id>')
    parser.add_argument('qa_json', type=Path, help='qa.json of any of the six original TripoSplat projections')
    args = parser.parse_args()
    print(json.dumps(fit_cached(args.asset_dir, args.qa_json), indent=2, allow_nan=False))
