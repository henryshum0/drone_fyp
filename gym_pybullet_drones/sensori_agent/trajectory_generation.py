from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation
from gym_pybullet_drones.sensori_agent.trajectory import Node, Segment, Trajectory
from gym_pybullet_drones.sensori_agent.trajectory_optimize import optimize_trj_time


def _rpy_to_rotmat(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = np.asarray(rpy, dtype=float)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    rot_y = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rot_z = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rot_z @ rot_y @ rot_x


def _waypoint_velocity_from_rpy_speed(rpy: np.ndarray, speed: float) -> np.ndarray:
    rot = _rpy_to_rotmat(rpy)
    local_x = rot[:, 0]
    return float(speed) * local_x


def sample_template(template, randomized: bool = True):
    """Sample a waypoint template and return waypoint positions, orientations, speeds, durations, and accelerations."""
    if hasattr(template, "sample_with_speeds_and_durations"):
        xyzs, rpys, speeds, durations, spawns, max_dist = template.sample_with_speeds_and_durations() if randomized else (
            np.asarray(template.waypoints_xyzs, dtype=float),
            np.asarray(template.waypoints_rpys, dtype=float),
            np.asarray(template.waypoints_speeds, dtype=float),
            np.asarray(template.waypoints_durations, dtype=float),
            template.spawns,
            template.max_dist,
        )
    elif hasattr(template, "sample_with_speeds"):
        xyzs, rpys, speeds, spawns, max_dist = template.sample_with_speeds() if randomized else (
            np.asarray(template.waypoints_xyzs, dtype=float),
            np.asarray(template.waypoints_rpys, dtype=float),
            np.asarray(template.waypoints_speeds, dtype=float),
            template.spawns,
            template.max_dist,
        )
        durations = np.asarray(getattr(template, "waypoints_durations", np.ones(len(xyzs))), dtype=float)
    else:
        xyzs, rpys, spawns, max_dist = template.sample() if randomized else (
            np.asarray(template.waypoints_xyzs, dtype=float),
            np.asarray(template.waypoints_rpys, dtype=float),
            template.spawns,
            template.max_dist,
        )
        speeds = np.asarray(getattr(template, "waypoints_speeds", np.zeros(len(xyzs))), dtype=float)
        durations = np.asarray(getattr(template, "waypoints_durations", np.ones(len(xyzs))), dtype=float)

    xyzs = np.asarray(xyzs, dtype=float)
    rpys = np.asarray(rpys, dtype=float)
    speeds = np.asarray(speeds, dtype=float).reshape(-1)
    durations = np.asarray(durations, dtype=float).reshape(-1)
    accelerations = getattr(template, "waypoints_accelerations", None)
    if xyzs.shape[0] != rpys.shape[0] or xyzs.shape[0] != speeds.shape[0] or xyzs.shape[0] != durations.shape[0]:
        raise ValueError("Template sample must provide matching xyzs, rpys, speeds, and durations lengths")
    
    return xyzs, rpys, speeds, durations, accelerations, spawns, max_dist


def build_trajectory_from_template(
    template,
    randomized: bool = True,
    min_duration: float = 0.1,
    speed_floor: float = 0.1,
) -> Trajectory:
    """Build a trajectory from a sampled waypoint template.

    Velocity at each waypoint is assumed to be aligned with the waypoint's local
    x-axis, scaled by the waypoint speed attribute.
    """
    xyzs, rpys, speeds, durations, accelerations, _, _ = sample_template(template, randomized=randomized)

    if xyzs.shape[0] < 2:
        raise ValueError("A trajectory needs at least two waypoints")

    velocities = np.array([
        _waypoint_velocity_from_rpy_speed(rpy, speed)
        for rpy, speed in zip(rpys, speeds)
    ])

    nodes = []
    for xyz, rpy, vel, acc in zip(xyzs, rpys, velocities, accelerations):
        nodes.append(Node(pos=xyz, psi=float(rpy[2]), con_vel=vel, con_acc=acc))

    segments = []
    for i in range(len(nodes) - 1):
        duration = max(0.5 * (float(durations[i]) + float(durations[i + 1])), float(min_duration))
        segments.append(Segment(nodes[i], nodes[i + 1], duration=duration))

    return Trajectory(segments)


def trajectory_from_template(template, randomized: bool = True, min_duration: float = 0.1):
    """Backward-friendly alias for build_trajectory_from_template()."""
    return build_trajectory_from_template(template, randomized=randomized, min_duration=min_duration)

if __name__ == "__main__":
    from gym_pybullet_drones.gateRL.waypoints.acro_templates import BackRollTemplate, FrontRollTemplate, SplitSLeftTemplate, SplitSRightTemplate, BarrelRollLeftTemplate, BarrelRollRightTemplate
    from gym_pybullet_drones.sensori_agent.trajectory_optimize import optimize_trj_time
    template = BackRollTemplate()
    trajectory = build_trajectory_from_template(template, randomized=True)
    optimized_traj, optimized_time, min_result = optimize_trj_time(
        trajectory,
        time_penalty=np.array([100000 for seg in trajectory._segments]),
        preserve_total_time=False,
        max_velocity=20,
        max_normalized_thrust=60,
        report_peaks=True,
    )
    print("optimized_time:", optimized_time)
    optimized_traj.visualize(show=True)