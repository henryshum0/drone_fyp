import importlib

import numpy as np
import pybullet as p

from gym_pybullet_drones.sensori_agent.trajectory.trajectory_optimize import optimize_trj_time


def quat_normalize(q):
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if n < 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    # PyBullet orientation utilities expect normalized quaternions.
    _, q_norm = p.multiplyTransforms(
        [0.0, 0.0, 0.0],
        (q / n).tolist(),
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    )
    return np.asarray(q_norm, dtype=float)


def quat_to_rotmat(quat_xyzw):
    q = quat_normalize(quat_xyzw)
    mat = p.getMatrixFromQuaternion(q.tolist())
    return np.asarray(mat, dtype=float).reshape(3, 3)


def quat_derivative_xyzw(q, w_body):
    x, y, z, w = q
    p_rate, q_rate, r_rate = w_body
    return 0.5 * np.array(
        [
            w * p_rate + y * r_rate - z * q_rate,
            w * q_rate + z * p_rate - x * r_rate,
            w * r_rate + x * q_rate - y * p_rate,
            -x * p_rate - y * q_rate - z * r_rate,
        ],
        dtype=float,
    )


def quat_integrate_body_rate(quat_xyzw, w_body, dt):
    q = quat_normalize(quat_xyzw)
    w = np.asarray(w_body, dtype=float)
    dt = float(dt)
    theta = float(np.linalg.norm(w) * dt)
    if theta < 1e-12:
        return q

    axis = w / (np.linalg.norm(w) + 1e-12)
    half = 0.5 * theta
    s = float(np.sin(half))
    q_delta = [float(axis[0] * s), float(axis[1] * s), float(axis[2] * s), float(np.cos(half))]
    _, q_next = p.multiplyTransforms(
        [0.0, 0.0, 0.0],
        q.tolist(),
        [0.0, 0.0, 0.0],
        q_delta,
    )
    return np.asarray(q_next, dtype=float)


def world_to_body(v_world, quat_xyzw):
    q = quat_normalize(quat_xyzw)
    _, q_inv = p.invertTransform([0.0, 0.0, 0.0], q.tolist())
    v_body, _ = p.multiplyTransforms(
        [0.0, 0.0, 0.0],
        q_inv,
        np.asarray(v_world, dtype=float).tolist(),
        [0.0, 0.0, 0.0, 1.0],
    )
    return np.asarray(v_body, dtype=float)


def quat_to_yaw(quat_xyzw):
    q = quat_normalize(quat_xyzw)
    return float(p.getEulerFromQuaternion(q.tolist())[2])


def quat_derivative_ca(ca, q, w_body):
    x = q[0]
    y = q[1]
    z = q[2]
    w = q[3]
    p_rate = w_body[0]
    q_rate = w_body[1]
    r_rate = w_body[2]
    return 0.5 * ca.vertcat(
        w * p_rate + y * r_rate - z * q_rate,
        w * q_rate + z * p_rate - x * r_rate,
        w * r_rate + x * q_rate - y * p_rate,
        -x * p_rate - y * q_rate - z * r_rate,
    )


def quat_to_rotmat_ca(ca, quat_xyzw):
    x = quat_xyzw[0]
    y = quat_xyzw[1]
    z = quat_xyzw[2]
    w = quat_xyzw[3]
    r11 = 1.0 - 2.0 * (y * y + z * z)
    r12 = 2.0 * (x * y - z * w)
    r13 = 2.0 * (x * z + y * w)
    r21 = 2.0 * (x * y + z * w)
    r22 = 1.0 - 2.0 * (x * x + z * z)
    r23 = 2.0 * (y * z - x * w)
    r31 = 2.0 * (x * z - y * w)
    r32 = 2.0 * (y * z + x * w)
    r33 = 1.0 - 2.0 * (x * x + y * y)
    return ca.vertcat(
        ca.horzcat(r11, r12, r13),
        ca.horzcat(r21, r22, r23),
        ca.horzcat(r31, r32, r33),
    )


def quat_to_yaw_ca(ca, q):
    x = q[0]
    y = q[1]
    z = q[2]
    w = q[3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return ca.atan2(siny_cosp, cosy_cosp)


def build_demo_trajectory(duration_sec=8.0, dt=0.02):
    from gym_pybullet_drones.sensori_agent.acro_templates import (
        SplitSLeftTemplate, BarrelRollLeftTemplate, BarrelRollRightTemplate, PowerloopTemplate, HeartTemplate
    )
    from gym_pybullet_drones.sensori_agent.trajectory.trajectory_generation import build_trajectory_from_template

    template = BarrelRollLeftTemplate()
    traj_obj = build_trajectory_from_template(template, randomized=True)
    traj_obj, optimized_time, _ = optimize_trj_time(
        traj_obj,
        time_penalty=np.array([100 for seg in traj_obj._segments]),
        preserve_total_time=False,
        max_velocity=30,
        min_velocity=0,
        max_normalized_thrust=50,
        report_peaks=True,
    )
    print("optimized demo trajectory time:", optimized_time)
    _ = duration_sec, dt
    return traj_obj


def plot_trajectory_pyplot(reference_traj, actual_positions, actual_quats=None, show=True, save_path=None, axis_stride=10, axis_scale=0.2):
    import matplotlib.pyplot as plt

    ref = np.asarray(reference_traj[:, 0:3], dtype=float)
    ref_quat = np.asarray(reference_traj[:, 3:7], dtype=float)
    act = np.asarray(actual_positions, dtype=float)
    if act.ndim != 2 or act.shape[1] != 3:
        raise ValueError("actual_positions must be shape (N, 3)")
    if actual_quats is None:
        raise ValueError("actual_quats must be provided with shape (N, 4)")
    act_quat = np.asarray(actual_quats, dtype=float)
    if act_quat.ndim != 2 or act_quat.shape[1] != 4:
        raise ValueError("actual_quats must be shape (N, 4)")

    min_len = max(1, min(ref.shape[0], act.shape[0], ref_quat.shape[0], act_quat.shape[0]))
    ref = ref[:min_len]
    ref_quat = ref_quat[:min_len]
    act = act[:min_len]
    act_quat = act_quat[:min_len]
    t = np.arange(min_len)
    axis_idx = np.arange(0, min_len, max(1, int(axis_stride)))

    ref_x = np.zeros((min_len, 3), dtype=float)
    ref_z = np.zeros((min_len, 3), dtype=float)
    act_x = np.zeros((min_len, 3), dtype=float)
    act_z = np.zeros((min_len, 3), dtype=float)
    for i in range(min_len):
        ref_rot = quat_to_rotmat(ref_quat[i])
        act_rot = quat_to_rotmat(act_quat[i])
        ref_x[i], ref_z[i] = ref_rot[:, 0], ref_rot[:, 2]
        act_x[i], act_z[i] = act_rot[:, 0], act_rot[:, 2]

    fig = plt.figure(figsize=(12, 5))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax3d.plot(ref[:, 0], ref[:, 1], ref[:, 2], color="tab:orange", linewidth=2.0, label="reference")
    ax3d.plot(act[:, 0], act[:, 1], act[:, 2], color="tab:blue", linewidth=1.6, label="quadrotor")
    ax3d.quiver(
        ref[axis_idx, 0], ref[axis_idx, 1], ref[axis_idx, 2],
        ref_x[axis_idx, 0], ref_x[axis_idx, 1], ref_x[axis_idx, 2],
        length=float(axis_scale), normalize=True, color="tab:cyan", linewidth=1.0,
    )
    ax3d.quiver(
        ref[axis_idx, 0], ref[axis_idx, 1], ref[axis_idx, 2],
        ref_z[axis_idx, 0], ref_z[axis_idx, 1], ref_z[axis_idx, 2],
        length=float(axis_scale), normalize=True, color="tab:purple", linewidth=1.0,
    )
    ax3d.quiver(
        act[axis_idx, 0], act[axis_idx, 1], act[axis_idx, 2],
        act_x[axis_idx, 0], act_x[axis_idx, 1], act_x[axis_idx, 2],
        length=float(axis_scale), normalize=True, color="tab:green", linewidth=1.0,
    )
    ax3d.quiver(
        act[axis_idx, 0], act[axis_idx, 1], act[axis_idx, 2],
        act_z[axis_idx, 0], act_z[axis_idx, 1], act_z[axis_idx, 2],
        length=float(axis_scale), normalize=True, color="tab:red", linewidth=1.0,
    )
    ax3d.scatter(ref[0, 0], ref[0, 1], ref[0, 2], color="tab:green", s=30, label="start")
    ax3d.set_xlabel("x [m]")
    ax3d.set_ylabel("y [m]")
    ax3d.set_zlabel("z [m]")
    all_pts = np.vstack((ref, act))
    mins = np.min(all_pts, axis=0)
    maxs = np.max(all_pts, axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * np.max(maxs - mins)
    radius = max(radius, 1e-3)
    ax3d.set_xlim(center[0] - radius, center[0] + radius)
    ax3d.set_ylim(center[1] - radius, center[1] + radius)
    ax3d.set_zlim(center[2] - radius, center[2] + radius)
    ax3d.set_box_aspect((1, 1, 1))
    ax3d.set_title("3D Trajectory with X/Z Axes")
    ax3d.legend(loc="upper right")

    ax = fig.add_subplot(1, 2, 2)
    ax.plot(t, ref[:, 0], "--", color="tab:red", linewidth=1.2, label="x_ref")
    ax.plot(t, act[:, 0], color="tab:red", linewidth=1.2, label="x")
    ax.plot(t, ref[:, 1], "--", color="tab:green", linewidth=1.2, label="y_ref")
    ax.plot(t, act[:, 1], color="tab:green", linewidth=1.2, label="y")
    ax.plot(t, ref[:, 2], "--", color="tab:blue", linewidth=1.2, label="z_ref")
    ax.plot(t, act[:, 2], color="tab:blue", linewidth=1.2, label="z")
    ax.set_xlabel("sample index")
    ax.set_ylabel("position [m]")
    ax.set_title("Position Tracking")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)

    plt.tight_layout()
    if save_path is not None and len(str(save_path)) > 0:
        fig.savefig(str(save_path), dpi=160)
        print(f"[DEMO] Saved trajectory plot to {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
