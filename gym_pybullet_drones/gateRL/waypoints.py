import numpy as np
from gym_pybullet_drones.gateRL.interpolate import interpolate_waypoints
from transforms3d.euler import euler2quat, quat2euler

waypoints1 = {
    "pos":np.array([
        np.array([2, 0, 3]),
        np.array([1, 0, 3]),
        np.array([0, 1, 3]),
        np.array([-1, 0, 3]),
        np.array([0, -1, 3]),
        np.array([1, 0, 3]),
    ]),
    "rpy": np.array([
        np.array([0, 0, np.pi]),
        np.array([0, 0, np.pi/4 * 3]),
        np.array([0, 0, np.pi]),
        np.array([0, 0, -np.pi/2]),
        np.array([0, 0, 0]),
        np.array([0, 0, np.pi/2]),
    ]),
    "spawn":[
        {
            "pos": np.array([2.1, 0, 3]),
            "vel": np.array([0, 0, 0]),
            "acc": np.array([0, 0, 0]),
            "rpy": np.array([0, 0, np.pi]),
            "next_waypoints": [0, 1]
        }
    ]
}

waypoints1["pos"], waypoints1["quats"] = interpolate_waypoints(waypoints1["pos"], waypoints_rpy=waypoints1["rpy"], num_points_per_segment=3)
waypoints1["rpy"] = np.array([quat2euler(quat) for quat in waypoints1["quats"]])

waypoints2 = np.array([
    [
        np.array([2, 0, 3]),
        np.array([1, 0, 3]),
        np.array([0, 1, 3]),
        np.array([-1, 0, 3]),
        np.array([0, -1, 3]),
        np.array([1, 0, 3]),
    ],
    [
        np.array([0, 0, np.pi]),
        np.array([0, 0, np.pi/4 * 3]),
        np.array([0, 0, np.pi]),
        np.array([0, 0, -np.pi/2]),
        np.array([0, 0, 0]),
        np.array([0, 0, np.pi/2]),
    ]
])
