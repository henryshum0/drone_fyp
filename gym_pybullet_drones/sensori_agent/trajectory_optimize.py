from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from gym_pybullet_drones.sensori_agent.trajectory import Node, Segment, Trajectory


def rebuild_trj(trajectory: Trajectory, segment_time):
    return trajectory.build_new(segment_time)


def _poly_derivative(coeffs: np.ndarray, order: int = 1) -> np.ndarray:
    out = np.asarray(coeffs, dtype=float)
    for _ in range(order):
        if out.shape[0] <= 1:
            return np.array([0.0], dtype=float)
        out = np.array([i * out[i] for i in range(1, out.shape[0])], dtype=float)
    return out


def _poly_eval(coeffs: np.ndarray, t: float) -> float:
    return float(np.dot(coeffs, np.power(t, np.arange(coeffs.shape[0], dtype=float))))


def _poly_roots_in_interval(coeffs: np.ndarray, t0: float, t1: float, tol: float = 1e-9) -> list[float]:
    coeffs = np.asarray(coeffs, dtype=float)
    if coeffs.size <= 1:
        return []

    nz = np.where(np.abs(coeffs) > tol)[0]
    if nz.size == 0:
        return []
    coeffs = coeffs[: nz[-1] + 1]
    if coeffs.size <= 1:
        return []

    roots = np.roots(coeffs[::-1])
    valid = []
    for r in roots:
        if abs(np.imag(r)) <= 1e-7:
            rr = float(np.real(r))
            if t0 - 1e-8 <= rr <= t1 + 1e-8:
                valid.append(min(max(rr, t0), t1))
    return valid


def _segment_peak_velocity(seg: Segment) -> float:
    vx = _poly_derivative(seg.coeffs_x, order=1)
    vy = _poly_derivative(seg.coeffs_y, order=1)
    vz = _poly_derivative(seg.coeffs_z, order=1)

    v2 = np.convolve(vx, vx) + np.convolve(vy, vy) + np.convolve(vz, vz)
    dv2 = _poly_derivative(v2, order=1)

    candidates = [0.0, float(seg._duration)]
    candidates.extend(_poly_roots_in_interval(dv2, 0.0, float(seg._duration)))

    max_v2 = 0.0
    for t in candidates:
        max_v2 = max(max_v2, _poly_eval(v2, t))
    return float(np.sqrt(max(max_v2, 0.0)))


def _segment_min_velocity(seg: Segment) -> float:
    vx = _poly_derivative(seg.coeffs_x, order=1)
    vy = _poly_derivative(seg.coeffs_y, order=1)
    vz = _poly_derivative(seg.coeffs_z, order=1)

    v2 = np.convolve(vx, vx) + np.convolve(vy, vy) + np.convolve(vz, vz)
    dv2 = _poly_derivative(v2, order=1)

    # Use interior critical points only, so endpoint-fixed zero velocity does not
    # trivially violate the minimum speed constraint.
    candidates = _poly_roots_in_interval(dv2, 0.0, float(seg._duration))
    eps_t = 1e-6
    candidates = [t for t in candidates if eps_t < t < float(seg._duration) - eps_t]

    if len(candidates) == 0:
        # No interior stationary points -> approximate interior minimum near ends.
        candidates = [eps_t, max(float(seg._duration) - eps_t, eps_t)]

    min_v2 = np.inf
    for t in candidates:
        min_v2 = min(min_v2, _poly_eval(v2, t))

    if not np.isfinite(min_v2):
        return 0.0
    return float(np.sqrt(max(min_v2, 0.0)))


def _segment_peak_normalized_thrust(seg: Segment, thrust_offset: np.ndarray) -> float:
    ax = _poly_derivative(seg.coeffs_x, order=2)
    ay = _poly_derivative(seg.coeffs_y, order=2)
    az = _poly_derivative(seg.coeffs_z, order=2)

    tx = ax.copy()
    ty = ay.copy()
    tz = az.copy()
    tx[0] += float(thrust_offset[0])
    ty[0] += float(thrust_offset[1])
    tz[0] += float(thrust_offset[2])

    t2 = np.convolve(tx, tx) + np.convolve(ty, ty) + np.convolve(tz, tz)
    dt2 = _poly_derivative(t2, order=1)

    candidates = [0.0, float(seg._duration)]
    candidates.extend(_poly_roots_in_interval(dt2, 0.0, float(seg._duration)))

    max_t2 = 0.0
    for t in candidates:
        max_t2 = max(max_t2, _poly_eval(t2, t))
    return float(np.sqrt(max(max_t2, 0.0)))


def _trajectory_peak_metrics(trj: Trajectory, thrust_offset: np.ndarray) -> tuple[float, float]:
    peak_v = 0.0
    peak_thrust = 0.0
    for seg in trj._segments:
        peak_v = max(peak_v, _segment_peak_velocity(seg))
        peak_thrust = max(peak_thrust, _segment_peak_normalized_thrust(seg, thrust_offset))
    return peak_v, peak_thrust


def _trajectory_min_velocity_metric(trj: Trajectory) -> float:
    min_v = np.inf
    for seg in trj._segments:
        min_v = min(min_v, _segment_min_velocity(seg))
    if not np.isfinite(min_v):
        return 0.0
    return float(min_v)

def optimize_trj_time(
    trajectory: Trajectory,
    time_penalty=None,
    min_duration: float = 0.1,
    preserve_total_time: bool = True,
    min_velocity: float | None = 5,
    max_velocity: float | None = None,
    max_normalized_thrust: float | None = None,
    thrust_offset: np.ndarray | None = None,
    constraint_weight: float = 1e6,
    report_peaks: bool = False,
    maxiter: int = 200,
    ftol: float = 1e-4,
    disp: bool = False,
    cache_round_decimals: int = 9,
):
    """Optimize segment durations while re-solving the trajectory at each step.

    The previous version only evaluated the Hessian on fixed coefficients,
    which tends to push every segment to the lower bound. This version rebuilds
    the trajectory for each candidate duration vector so the smoothness cost is
    consistent with the new timing.
    """
    time_initial = np.array([seg._duration for seg in trajectory._segments], dtype=float)
    n_segments = len(time_initial)

    if n_segments == 0:
        raise ValueError("Trajectory has no segments")
    if min_duration <= 0:
        raise ValueError("min_duration must be > 0")
    if cache_round_decimals < 0:
        raise ValueError("cache_round_decimals must be >= 0")

    if constraint_weight < 0:
        raise ValueError("constraint_weight must be >= 0")
    if thrust_offset is None:
        thrust_offset = np.array([0.0, 0.0, 9.81], dtype=float)
    else:
        thrust_offset = np.asarray(thrust_offset, dtype=float)
        if thrust_offset.shape != (3,):
            raise ValueError(f"thrust_offset must have shape (3,), got {thrust_offset.shape}")

    if time_penalty is None:
        time_penalty = np.full(n_segments, 1e-2, dtype=float)
    else:
        time_penalty = np.asarray(time_penalty, dtype=float)
        if time_penalty.shape != (n_segments,):
            raise ValueError(f"time_penalty must have shape ({n_segments},), got {time_penalty.shape}")

    total_time = float(np.sum(time_initial))
    bounds = [(float(min_duration), None) for _ in range(n_segments)]
    constraints = []
    if preserve_total_time:
        constraints.append({"type": "eq", "fun": lambda t: float(np.sum(t) - total_time)})

    # Backward-compatible argument kept for API stability. Hard constraints are
    # enforced explicitly below, so this value is intentionally unused.
    _ = constraint_weight

    eval_cache = {}
    need_peak_velocity = (max_velocity is not None) or bool(report_peaks)
    need_peak_thrust = (max_normalized_thrust is not None) or bool(report_peaks)
    need_min_velocity = False

    def _evaluate_candidate(t):
        t = np.asarray(t, dtype=float)
        if np.any(~np.isfinite(t)) or np.any(t < min_duration):
            return None

        key = tuple(np.round(t, cache_round_decimals).tolist())
        if key in eval_cache:
            return eval_cache[key]

        try:
            trj = rebuild_trj(trajectory, t)
        except Exception:
            eval_cache[key] = None
            return None

        smoothness_cost = 0.0
        for seg in trj._segments:
            smoothness_cost += float(seg.coeffs_x.T @ seg.H_x @ seg.coeffs_x)
            smoothness_cost += float(seg.coeffs_y.T @ seg.H_y @ seg.coeffs_y)
            smoothness_cost += float(seg.coeffs_z.T @ seg.H_z @ seg.coeffs_z)
            smoothness_cost += float(seg.coeffs_psi.T @ seg.H_psi @ seg.coeffs_psi)

        out = {
            "smoothness_cost": float(smoothness_cost),
        }

        if need_min_velocity:
            out["v_min"] = float(_trajectory_min_velocity_metric(trj))

        if need_peak_velocity and need_peak_thrust:
            v_peak, thrust_peak = _trajectory_peak_metrics(trj, thrust_offset)
            out["v_peak"] = float(v_peak)
            out["thrust_peak"] = float(thrust_peak)
        elif need_peak_velocity:
            out["v_peak"] = float(max(_segment_peak_velocity(seg) for seg in trj._segments))
        elif need_peak_thrust:
            out["thrust_peak"] = float(max(_segment_peak_normalized_thrust(seg, thrust_offset) for seg in trj._segments))

        eval_cache[key] = out
        return out

    def _ineq_metric(t, metric_key: str, lower: float | None = None, upper: float | None = None) -> float:
        eval_out = _evaluate_candidate(t)
        if eval_out is None:
            return -1e6
        if lower is not None:
            return float(eval_out[metric_key] - lower)
        if upper is not None:
            return float(upper - eval_out[metric_key])
        raise ValueError("Either lower or upper must be provided")

    if max_velocity is not None:
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda t: _ineq_metric(t, "v_peak", upper=float(max_velocity)),
            }
        )
    if max_normalized_thrust is not None:
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda t: _ineq_metric(t, "thrust_peak", lower=0, upper=float(max_normalized_thrust)),
            }
        )

    def objective(t):
        eval_out = _evaluate_candidate(t)
        if eval_out is None:
            return 1e20

        return eval_out["smoothness_cost"] + float(np.dot(time_penalty, np.asarray(t, dtype=float)))

    method = "SLSQP" if len(constraints) > 0 else "L-BFGS-B"
    min_result = minimize(
        objective,
        np.maximum(time_initial, min_duration),
        method=method,
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": int(maxiter), "disp": bool(disp), "ftol": float(ftol)},
    )

    optimized_time = np.asarray(min_result.x if min_result.success else time_initial, dtype=float)
    optimized_traj = rebuild_trj(trajectory, optimized_time)

    peak_v_final, peak_thrust_final = _trajectory_peak_metrics(optimized_traj, thrust_offset)
    min_v_final = _trajectory_min_velocity_metric(optimized_traj)
    min_result.min_velocity = float(min_v_final)
    min_result.peak_velocity = float(peak_v_final)
    min_result.peak_normalized_thrust = float(peak_thrust_final)

    if report_peaks:
        print(f"min_velocity_analytic: {min_v_final:.6f}")
        print(f"peak_velocity_analytic: {peak_v_final:.6f}")
        print(f"peak_normalized_thrust_analytic: {peak_thrust_final:.6f}")

    return optimized_traj, optimized_time, min_result

if __name__ == "__main__":
    node1 = Node(pos=[0, 0, 0], con_vel=[0, 0, 0], con_acc=[0, 0, 0], psi=0)
    node2 = Node(pos=[10, 0, 0], psi=0,)
    node3 = Node(pos=[12.5, 2, 2.5], psi=0, )
    node4 = Node(pos=[10, 0, 5], psi=0, con_vel=[-10, -4, 0])
    node5 = Node(pos=[8.5, -2, 2.5], psi=0, )
    node6 = Node(pos=[10, 0, 0], con_vel=[0, 0, 0], psi=0, )
    segment = Segment(node1, node2, duration=1)
    segment2 = Segment(node2, node3, duration=1)
    segment3 = Segment(node3, node4, duration=1)
    segment4 = Segment(node4, node5, duration=1)
    segment5 = Segment(node5, node6, duration=1)
    segments = [segment, segment2, segment3, segment4, segment5]
    trajectory = Trajectory(segments)
    optimized_traj, opt_time, result = optimize_trj_time(
        trajectory,
        time_penalty=np.array([100000 for seg in trajectory._segments]),
        preserve_total_time=False,
        max_velocity=20,
        max_normalized_thrust=60,
        report_peaks=True,
    )
    print("optimization_success:", result.success)
    print("optimized_time:", opt_time)
    print(optimized_traj.b_x)
    optimized_traj.visualize(show=True)
