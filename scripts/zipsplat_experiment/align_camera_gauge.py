#!/usr/bin/env python3
"""Anchor a refined ZipSplat branch to the initial prediction's first camera.

Uses only NumPy. The output is accepted by zip_render_faces.py --transform.
No Gaussian data, SH coefficients, or scale is modified. The matrix composition is
C_branch = C_initial @ T_initial_camera0 @ inverse(T_branch_camera0).

Example:
  python align_camera_gauge.py --initial-cameras initial/cameras.json \
    --branch-cameras ba1008/cameras.json --canonical canonical.json \
    --output ba1008/canonical.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rotation_angle(rotation):
    cosine = np.clip((np.trace(rotation, axis1=-2, axis2=-1) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(cosine))


def validate_poses(poses, label):
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or not len(poses):
        raise ValueError(f"{label} must contain N x 4 x 4 c2w matrices")
    if not np.isfinite(poses).all():
        raise ValueError(f"Nonfinite camera pose in {label}")
    if not np.allclose(poses[:, 3], [0, 0, 0, 1], atol=1e-7):
        raise ValueError(f"Invalid homogeneous camera matrix in {label}")
    rotations = poses[:, :3, :3]
    if not np.allclose(np.swapaxes(rotations, -2, -1) @ rotations, np.eye(3), atol=1e-4):
        raise ValueError(f"Nonorthogonal camera rotation in {label}")
    if not np.allclose(np.linalg.det(rotations), 1, atol=1e-4):
        raise ValueError(f"Improper camera rotation in {label}")
    return poses


def corresponding_poses(initial, branch):
    views = initial["views"]
    initial_ids = [v["id"] for v in views]
    branch_ids = [v["id"] for v in branch["views"]]
    if len(set(initial_ids)) != len(initial_ids) or len(set(branch_ids)) != len(branch_ids):
        raise ValueError("View IDs must be unique")
    if set(initial_ids) != set(branch_ids):
        raise ValueError("Initial and branch camera files describe different view IDs")
    mapping = {v["id"]: i for i, v in enumerate(branch["views"])}
    order = [mapping[view_id] for view_id in initial_ids]
    for view, index in zip(views, order):
        other = branch["views"][index]
        if view.get("sha256") and other.get("sha256") and view["sha256"] != other["sha256"]:
            raise ValueError(f"Input image changed for {view['id']}")
    a = validate_poses(initial["c2w"], "initial")
    b = validate_poses(branch["c2w"], "branch")
    if len(a) != len(initial_ids) or len(b) != len(branch_ids):
        raise ValueError("Pose and view counts do not match")
    return a, b[order], initial_ids, order


def validate_similarity(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("canonical_from_scene must be a finite 4 x 4 matrix")
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-7):
        raise ValueError("Invalid homogeneous canonical matrix")
    scale = np.linalg.norm(matrix[:3, 0])
    if scale <= 0:
        raise ValueError("Canonical similarity has zero scale")
    rotation = matrix[:3, :3] / scale
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-4):
        raise ValueError("Canonical transform must be a proper uniform similarity")
    return matrix, float(scale)


def alignment(initial, branch, canonical):
    a, b, ids, order = corresponding_poses(initial, branch)
    canonical_matrix, canonical_scale = validate_similarity(canonical["canonical_from_scene"])
    gauge = a[0] @ np.linalg.inv(b[0])
    corrected = gauge[None] @ b
    result = canonical_matrix @ gauge
    center_errors = np.linalg.norm(corrected[:, :3, 3] - a[:, :3, 3], axis=1)
    rotation_errors = rotation_angle(np.swapaxes(a[:, :3, :3], -2, -1) @ corrected[:, :3, :3])
    pair_rows, valid_ratios, valid_errors, initial_baselines = [], [], [], []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            da = float(np.linalg.norm(a[i, :3, 3] - a[j, :3, 3]))
            db = float(np.linalg.norm(b[i, :3, 3] - b[j, :3, 3]))
            initial_baselines.append(da)
            pair_rows.append({"ids": [ids[i], ids[j]], "initial_length": da, "branch_length": db})
    nonzero = [value for value in initial_baselines if value > 1e-12]
    median_baseline = float(np.median(nonzero)) if nonzero else None
    floor = max(1e-9, 1e-5 * (median_baseline or 0))
    for row in pair_rows:
        da, db = row["initial_length"], row["branch_length"]
        if min(da, db) > floor:
            ratio = da / db
            error = (db - da) / da
            row.update(initial_over_branch_ratio=ratio, relative_length_error=error)
            valid_ratios.append(ratio)
            valid_errors.append(error)
        else:
            row.update(initial_over_branch_ratio=None, relative_length_error=None)
    drift_rotation = rotation_angle(a[0, :3, :3].T @ b[0, :3, :3])
    return {
        "canonical_from_scene": result.tolist(),
        "method": "Rigid first-camera gauge alignment to a shared initial prediction",
        "composition": "C_branch = C_initial @ T_initial_camera0 @ inverse(T_branch_camera0)",
        "anchor_view_id": ids[0],
        "branch_indices_in_initial_order": order,
        "initial_from_branch": gauge.tolist(),
        "canonical_uniform_scale_unchanged": canonical_scale,
        "automatic_rescale_applied": False,
        "first_camera_drift": {
            "initial_c2w": a[0].tolist(),
            "branch_c2w": b[0].tolist(),
            "raw_rotation_degrees": float(drift_rotation),
            "raw_center_delta_scene_units": (b[0, :3, 3] - a[0, :3, 3]).tolist(),
            "gauge_translation_norm_scene_units": float(np.linalg.norm(gauge[:3, 3])),
            "anchored_matrix_max_abs_error": float(np.abs(corrected[0] - a[0]).max()),
        },
        "camera_residuals_after_rigid_anchor": {
            "view_ids": ids,
            "center_error_scene_units": center_errors.tolist(),
            "rotation_error_degrees": rotation_errors.tolist(),
            "center_rmse_scene_units": float(np.sqrt(np.mean(center_errors ** 2))),
            "center_rmse_over_initial_median_baseline": float(np.sqrt(np.mean(center_errors ** 2)) / median_baseline) if median_baseline else None,
            "max_rotation_error_degrees": float(rotation_errors.max()),
        },
        "camera_baselines": {
            "pairs": pair_rows,
            "valid_pair_count": len(valid_ratios),
            "degeneracy_floor_scene_units": floor,
            "initial_nonzero_median_length": median_baseline,
            "initial_over_branch_ratio_median": float(np.median(valid_ratios)) if valid_ratios else None,
            "initial_over_branch_ratio_range": [min(valid_ratios), max(valid_ratios)] if valid_ratios else None,
            "relative_length_error_rmse": float(np.sqrt(np.mean(np.square(valid_errors)))) if valid_errors else None,
        },
        "limitations": [
            "The first camera is an orientation/position anchor, not calibrated ground truth.",
            "A uniform scale gauge remains. Baseline ratios are diagnostics only and are not applied automatically.",
            "Residual relative-camera changes can reflect pose corrections, inconsistent generated views, or deformation; they are not all global gauge drift.",
            "Use the same canonical Box bounds for all branches; this transform does not refit each branch to a new bounding box.",
        ],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--initial-cameras", type=Path, required=True)
    p.add_argument("--branch-cameras", type=Path, required=True)
    p.add_argument("--canonical", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    paths = {"initial_cameras": args.initial_cameras, "branch_cameras": args.branch_cameras, "initial_canonical": args.canonical}
    data = {key: json.loads(path.read_text(encoding="utf-8")) for key, path in paths.items()}
    result = alignment(data["initial_cameras"], data["branch_cameras"], data["initial_canonical"])
    result["provenance"] = {key: {"path": str(path.resolve()), "sha256": sha(path)} for key, path in paths.items()}
    result["provenance"]["helper_sha256"] = sha(__file__)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output.exists():
        if args.output.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"Refusing to replace a different alignment: {args.output}")
    else:
        temp = args.output.with_name(args.output.name + ".tmp")
        temp.write_text(text, encoding="utf-8")
        os.replace(temp, args.output)
    print(json.dumps({"output": str(args.output), "first_camera_drift": result["first_camera_drift"],
                      "baseline_ratio_median": result["camera_baselines"]["initial_over_branch_ratio_median"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
