import numpy as np
import pybullet as p
from typing import List
from matplotlib import pyplot as plt
from scipy.spatial.transform import Slerp, Rotation as R
    

def interpolate_waypoints(waypoints_xyz, waypoints_rpy, num_intermediates):
    """
    Compute intermediate waypoints between given waypoints using cubic Hermite splines.
    
    Args:
        waypoints (List[Waypoint]): List of Waypoint objects defining the path.
        num_intermediates (int): Number of intermediate waypoints to generate between each pair of waypoints.
    Returns:
        List[Waypoint]: List of Waypoint objects including intermediates.
    """
    ts = np.linspace(0, 1, num_intermediates+2)
    t = np.array([0., 1.])
    interp_waypoints = []
    for i in range(len(waypoints_xyz)-1):
        p0 = waypoints_xyz[i,0:2]
        p1 = waypoints_xyz[i+1,0:2]
        heading0 = waypoints_rpy[i,2]
        heading1 = waypoints_rpy[i+1,2]
        m0 = np.array([np.cos(heading0), np.sin(heading0)])
        m1 = np.array([np.cos(heading1), np.sin(heading1)])
        spline_points = [cubic_hermite_spline(p0, m0, p1, m1, t) for t in ts]
        spline_points_xy = np.array(spline_points)
        
        # interpolate on the x-z plane
        p0 = waypoints_xyz[i,[0,2]]
        p1 = waypoints_xyz[i+1,[0,2]]
        pitch0 = waypoints_rpy[i,1]
        pitch1 = waypoints_rpy[i+1,1]
        m0 = np.array([np.cos(pitch0), np.sin(pitch0)])
        m1 = np.array([np.cos(pitch1), np.sin(pitch1)])
        spline_points = [cubic_hermite_spline(p0, m0, p1, m1, t) for t in ts]
        spline_points_xz = np.array(spline_points)
        
        # interpolate orientation
        
        q0 = R.from_euler('xyz', waypoints_rpy[i,:]).as_quat()
        q1 = R.from_euler('xyz', waypoints_rpy[i+1,:]).as_quat()
        slerp = Slerp(t, np.array([q0, q1]))
        interp_quats_xyzw = slerp(ts).as_quat()
        interp_quats_wxyz = np.zeros_like(interp_quats_xyzw)
        interp_quats_wxyz[:,0] = interp_quats_xyzw[:,3]
        interp_quats_wxyz[:,1:4] = interp_quats_xyzw[:,0:3]
        
        interp_xyz = np.zeros_like(spline_points_xy)
        interp_xyz[:,:2] = spline_points_xy[:,:]
        interp_xyz[:,2] = spline_points_xz[:,1]
    return interp_xyz, interp_quats_wxyz

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

if __name__ == "__main__":
    waypoints = [
        np.array([0, 0]),
        np.array([1, 1]),
        np.array([0, 2]),
        np.array([-1, 1])
    ]
    d_waypoints = [
        np.array([2, 0]),
        np.array([0, 2]),
        np.array([-2, 0]),
        np.array([0, -2])
    ]
    for i in range(len(waypoints)-1):
        p0 = waypoints[i]
        p1 = waypoints[i+1]
        m0 = d_waypoints[i]
        m1 = d_waypoints[i+1]
        ts = np.linspace(0, 1, 10)
        spline_points = [cubic_hermite_spline(p0, m0, p1, m1, t) for t in ts]
        spline_points = np.array(spline_points)
        plt.plot(spline_points[:,0], spline_points[:,1], 'b-')
    waypoints = np.array(waypoints)
    plt.plot(waypoints[:,0], waypoints[:,1], 'ro')
    plt.show()
    
    