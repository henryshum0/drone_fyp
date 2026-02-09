import numpy as np
import pybullet as p
from typing import List
from matplotlib import pyplot as plt
from scipy.spatial.transform import Slerp, Rotation as R
    

def interpolate_waypoints(waypoints_xyz, waypoints_rpy, num_points_per_segment=10):
    waypoints_xyz = np.array(waypoints_xyz)
    waypoints_rpy = np.array(waypoints_rpy)

    xyz = catmull_rom_chain(waypoints_xyz, num_points_per_segment)
    waypoints_quat = R.from_euler('xyz', waypoints_rpy).as_quat()
    slerp = Slerp(np.arange(len(waypoints_quat)), R.from_quat(waypoints_quat))
    times = np.linspace(0, len(waypoints_quat) - 1, num=len(xyz))
    interp_rots = slerp(times)
    quats = interp_rots.as_quat(scalar_first=True)
    return xyz, quats

def cubic_hermite_spline(p0, m0, p1, m1, t):
    """Compute the cubic Hermite spline point at parameter t.
    
    Args:
        p0: Start point.
        m0: Start tangent (derivative) at p0.
        p1: End point.
        m1: End tangent (derivative) at p1.
        t: Parameter between 0 and 1.
        
    Returns:
        The interpolated point at parameter t.
    """
    h00 = 2*t**3 - 3*t**2 + 1
    h10 = t**3 - 2*t**2 + t
    h01 = -2*t**3 + 3*t**2
    h11 = t**3 - t**2
    
    return h00*p0 + h10*m0 + h01*p1 + h11*m1

def centripetal_catmull_rom_spline(p0, p1, p2, p3, num_points, alpha=0.5):
    def tj(ti, pi, pj):
        return ((np.linalg.norm(pj - pi))**alpha) + ti
    t0 = 0
    t1 = tj(t0, p0, p1)
    t2 = tj(t1, p1, p2)
    t3 = tj(t2, p2, p3)
    t = np.linspace(t1, t2, num_points).reshape(num_points, 1)
    A1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
    A2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
    A3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
    B1 = (t2 - t) / (t2 - t0) * A1 + (t - t0) / (t2 - t0) * A2
    B2 = (t3 - t) / (t3 - t1) * A2 + (t - t1) / (t3 - t1) * A3
    points = (t2 - t) / (t2 - t1) * B1 + (t - t1) / (t2 - t1) * B2
    return points

def catmull_rom_chain(points, num_points_per_segment, alpha=0.5):
    all_points = []

    # catmull rom will lose the first and last point
    head = points[0] - (points[1] - points[0])
    tail = points[-1] + (points[-1] - points[-2])
    points = np.vstack([head, points, tail])

    for i in range(1, len(points) - 2):
        p0 = points[i - 1]
        p1 = points[i]
        p2 = points[i + 1]
        p3 = points[i + 2]
        segment_points = centripetal_catmull_rom_spline(p0, p1, p2, p3, num_points_per_segment, alpha)
        all_points.append(segment_points)
    all_points = np.vstack(all_points).astype(np.float32)
    mask = (all_points[:-1] != all_points[1:]).any(axis=1)
    mask = np.concatenate(([True], mask))
    return all_points[mask]


if __name__ == "__main__":
    
    waypoints = [
        np.array([1, 0, 0]),
        np.array([0, 0, 0]),
        np.array([-1, 0, 1]),
        np.array([0, 0, 2]),
        np.array([1, 0, 1]),
        np.array([0, 0, 0]),
        np.array([-1, 0, 0])
    ]

    waypoints_rpy = [
        np.array([0, 0, 0]),
        np.array([0, 0, np.pi/2]),
        np.array([0, 0, np.pi]),
        np.array([0, 0, -np.pi/2]),
        np.array([0, 0, 0]),
        np.array([0, 0, np.pi/2]),
        np.array([0, 0, np.pi])
    ]
    points, quats = interpolate_waypoints(waypoints, waypoints_rpy=waypoints_rpy, num_points_per_segment=2)
    rpy = R.from_quat(quats[:,[1,2,3,0]]).as_euler('xyz')
    for pos, ros in zip(points, rpy):
        print("Pos:", pos, "\tRPY:", ros)
    x = points[:,0]
    y = points[:,1]
    z = points[:,2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()