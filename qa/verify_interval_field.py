"""Independent CPU properties for the bounded, angle-dependent interval-field prototype.

This does not certify GPU sampling, framebuffer output, or arbitrary ray-step safety.
It tests analytic surfaces, units, support, topology guards, and angle endpoints.
Intervals are [low, high, gap_low, gap_high] in normalized box coordinates.
"""
from pathlib import Path
import json
import math
import numpy as np

OUT = Path(__file__).resolve().parent
R = 1024


def smoothstep(a, b, x):
    t = np.clip((x-a)/(b-a), 0.0, 1.0)
    return t*t*(3.0-2.0*t)


def field(b, intervals, size, view, epsilon=1.5/R, strength=1.0):
    b = np.asarray(b, dtype=float)
    iv = np.asarray(intervals, dtype=float)
    size = np.asarray(size, dtype=float)
    view = np.asarray(view, dtype=float)
    view = view/np.linalg.norm(view, axis=-1, keepdims=True)
    lo, hi, gl, gh = (iv[..., k] for k in range(4))
    outer = np.maximum(b-hi, lo-b)
    gap = np.minimum(b-gl, gh-b)
    f = (np.maximum(outer, gap)-epsilon)*size
    native = np.max(f, axis=-1)
    gap_guard = np.max((gap-epsilon)*size, axis=-1)
    side = view >= 0
    endpoint = np.where(side, hi, lo)
    front = (np.where(side, b-hi, lo-b)-epsilon)*size
    band = 8.0*np.min(size, axis=-1)/R
    valid = (endpoint > 1e-6) & (endpoint < 1.0-1e-6)
    near = 1.0-smoothstep(band[..., None]/2, band[..., None], np.abs(front))
    same_branch = 1.0-smoothstep(band[..., None]/2, band[..., None], np.abs(f-front))
    active = view*view*valid*near*same_branch
    total = active.sum(axis=-1)
    q = active/np.maximum(total[..., None], 1e-12)
    blend = np.sum(q*np.maximum(f, native[..., None]-band[..., None]), axis=-1)
    overlap = np.clip(2.0*(1.0-np.sum(q*q, axis=-1)), 0.0, 1.0)
    gamma = strength*smoothstep(0.05, 0.25, total)*overlap
    has_gap = gh > gl
    lower_span, upper_span = gl-lo, hi-gh
    gap_segment = np.where(b <= gl, lower_span, np.where(b >= gh, upper_span, np.minimum(lower_span, upper_span)))
    local_span = np.maximum(np.where(has_gap, gap_segment, hi-lo), 0.0)
    thin = smoothstep(4.0/R, 8.0/R, np.min(local_span, axis=-1))
    delta = np.minimum(band, 0.25*np.min(local_span*size, axis=-1))*thin
    delta = np.maximum(delta, 0.0)
    value = np.maximum.reduce([native*(1-gamma)+blend*gamma, native-delta, gap_guard])
    support = np.all(hi >= lo, axis=-1)
    value = np.where(support, value, native)
    return {"value": value, "native": native, "delta": delta, "gap": gap_guard,
            "gamma": gamma, "support": support}


def sphere_intervals(b, size, radii):
    b = np.asarray(b)
    size = np.asarray(size)
    p = (b-0.5)*size
    radius = np.broadcast_to(np.asarray(radii), p.shape)
    transverse = np.sum(p*p, axis=-1, keepdims=True)-p*p
    radicand = radius*radius-transverse
    half = np.sqrt(np.maximum(radicand, 0.0))/size
    lo = np.where(radicand >= 0, 0.5-half, 1.0)
    hi = np.where(radicand >= 0, 0.5+half, 0.0)
    return np.stack([lo, hi, np.ones_like(lo), np.zeros_like(lo)], axis=-1)


def radial_root(view, size, radii, use_morph=True, epsilon=1.5/R):
    view = np.asarray(view)
    low, high = 0.0, 0.42
    for _ in range(55):
        t = (low+high)/2
        b = 0.5+view*t/size
        result = field(b, sphere_intervals(b, size, radii), size, view, epsilon=epsilon)
        value = result["value" if use_morph else "native"]
        if value <= 0: low = t
        else: high = t
    return (low+high)/2


def main():
    rng = np.random.default_rng(851070)
    summary = {"description": "CPU properties of bounded projective interval morph; not a true Euclidean SDF", "checks": {}}
    checks = summary["checks"]
    count = 50000
    b = rng.uniform(0.0, 1.0, (count, 3))
    size = rng.uniform(0.3, 3.0, (count, 3))
    view = rng.normal(size=(count, 3))
    lo = rng.uniform(0.05, 0.35, (count, 3))
    hi = rng.uniform(0.65, 0.95, (count, 3))
    gl = lo+0.35*(hi-lo)
    gh = lo+0.65*(hi-lo)
    has_gap = rng.uniform(size=(count, 3)) > 0.5
    gl = np.where(has_gap, gl, 1.0)
    gh = np.where(has_gap, gh, 0.0)
    iv = np.stack([lo, hi, gl, gh], axis=-1)
    out = field(b, iv, size, view)
    tol = 1e-12
    assert np.all(out["value"] <= out["native"]+tol)
    assert np.all(out["value"] >= out["native"]-out["delta"]-tol)
    assert np.all(out["value"][out["native"] <= 0] <= tol)
    assert np.all(out["value"][out["gap"] > 0] > 0)
    checks["bounded_expansion_native_occupancy_and_gap_guard"] = {"samples": count, "pass": True}

    # Uniform physical rescaling must rescale the field, not its zero surface.
    scale = 17.3
    scaled = field(b, iv, size*scale, view)
    error = float(np.max(np.abs(scaled["value"]-scale*out["value"])))
    assert error < 1e-10
    checks["physical_units_uniform_scale"] = {"scale": scale, "max_error": error, "pass": True}

    # Decode one anisotropic physical interval without relying on its normalized width.
    sz = np.array([2.0, 3.0, 5.0])
    pos = np.array([0.625, 0.5, 0.5])
    cube_iv = np.array([[0.2, 0.6, 1, 0], [0.2, 0.8, 1, 0], [0.2, 0.8, 1, 0]])
    unit_result = field(pos, cube_iv, sz, [1, 0, 0], epsilon=0)
    assert abs(float(unit_result["value"])-0.05) < 1e-12
    checks["noncubic_axis_depth_unit"] = {"expected_physical_residual": 0.05, "actual": float(unit_result["value"]), "pass": True}

    # An invalid interval and a genuine thin segment each retain native behavior.
    bad = iv.copy(); bad[:, 1, 0] = 0.8; bad[:, 1, 1] = 0.2
    invalid = field(b, bad, size, view)
    assert np.array_equal(invalid["value"], invalid["native"])
    thin = iv.copy(); thin[:, 1, :] = [0.5-1.5/R, 0.5+1.5/R, 1, 0]
    thin_result = field(b, thin, size, view)
    assert np.array_equal(thin_result["value"], thin_result["native"])
    checks["invalid_support_and_sub4texel_thin_parts"] = {"samples_each": count, "pass": True}

    # Cardinal endpoints, full yaw loop, and pole angles have no angular seam.
    cardinal_errors = []
    for v in np.vstack([np.eye(3), -np.eye(3)]):
        z = field(b, iv, size, v)
        cardinal_errors.append(float(np.max(np.abs(z["value"]-z["native"]))))
    assert max(cardinal_errors) < 1e-12
    loop_a = field(b, iv, size, [0, 0, 1])["value"]
    loop_b = field(b, iv, size, [math.sin(2*math.pi), 0, math.cos(2*math.pi)])["value"]
    loop_error = float(np.max(np.abs(loop_a-loop_b)))
    assert loop_error < 1e-12
    checks["six_cardinal_endpoints_and_yaw_loop"] = {"max_cardinal_error": max(cardinal_errors), "loop_error": loop_error, "pass": True}

    # A plane's exactly consistent depth measurements have exactly its zero set (epsilon=0).
    n = np.array([0.55, 0.4, 0.732]); n /= np.linalg.norm(n)
    plane_size = np.array([1.3, 1.0, 1.7])
    plane_points = rng.normal(size=(2000, 3))*0.025
    plane_points -= (plane_points@n)[:, None]*n
    plane_b = plane_points/plane_size+0.5
    plane_hi = plane_b-(plane_points@n)[:, None]/n/plane_size
    plane_iv = np.stack([np.zeros_like(plane_hi), plane_hi, np.ones_like(plane_hi), np.zeros_like(plane_hi)], axis=-1)
    plane_v = np.abs(rng.normal(size=(2000, 3)))
    plane = field(plane_b, plane_iv, plane_size, plane_v, epsilon=0)
    plane_error = float(np.max(np.abs(plane["value"])))
    assert plane_error < 1e-12
    checks["consistent_plane_zero_surface"] = {"samples": 2000, "max_error": plane_error, "epsilon": 0, "pass": True}

    # A sphere tests curved, consistent views and tangential intervals.
    sphere_size = np.array([1.3, 1.0, 1.7])
    directions = rng.normal(size=(5000, 3)); directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    sphere_b = 0.5+0.3*directions/sphere_size
    sphere_iv = sphere_intervals(sphere_b, sphere_size, [0.3, 0.3, 0.3])
    sphere = field(sphere_b, sphere_iv, sphere_size, directions, epsilon=0)
    sphere_error = float(np.max(np.abs(sphere["value"])))
    assert sphere_error < 1e-10
    checks["consistent_sphere_zero_surface"] = {"samples": 5000, "max_error": sphere_error, "epsilon": 0, "pass": True}

    # Deliberately inconsistent projected radii must yield bounded continuous shape movement.
    angles = np.linspace(0, 360, 721)
    rows = []
    for degrees in angles:
        yaw = math.radians(degrees)
        v = np.array([math.sin(yaw), 0.0, math.cos(yaw)])
        native = radial_root(v, sphere_size, [0.298, 0.3, 0.302], False)
        morphed = radial_root(v, sphere_size, [0.298, 0.3, 0.302], True)
        rows.append([degrees, native, morphed, morphed-native])
    rows = np.array(rows)
    assert abs(rows[0, 2]-rows[-1, 2]) < 1e-10
    assert np.max(rows[:, 3]) > 1e-6
    # A 0.5 degree sampled continuity check is empirical, not a universal proof.
    max_adjacent_change = float(np.max(np.abs(np.diff(rows[:, 2]))))
    assert max_adjacent_change < 0.001
    np.savetxt(OUT/'angle_fixture.csv', rows, delimiter=',', header='yaw_deg,native_radius,morphed_radius,delta_radius', comments='')
    checks["inconsistent_depth_angle_morph"] = {"samples": len(rows), "max_outward_shift": float(np.max(rows[:, 3])), "min_shift": float(np.min(rows[:, 3])), "max_adjacent_radius_change_at_half_degree": max_adjacent_change, "loop_error": float(abs(rows[0, 2]-rows[-1, 2])), "pass": True}
    summary["limitations"] = [
        "An implicit projective field, not an exact Euclidean SDF; no sphere-tracing step guarantee is claimed.",
        "This bounded relaxation expands from native occupancy; it does not linearly transport corresponding facial landmarks between input views.",
        "Native epsilon is per normalized axis. With epsilon enabled, even ideal consistent views can change their small numerical inflation under blending.",
        "Hard gap/invalid support preserves known empty regions; unknown or incorrect input geometry remains unresolved.",
        "A fixed sampling step can still miss sufficiently thin or steep features without a separate bound; these tests do not certify arbitrary rays.",
        "RGB correspondences and visible face doubling require image validation in the renderer; zero-set properties cannot certify appearance."
    ]
    (OUT/'verification.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
