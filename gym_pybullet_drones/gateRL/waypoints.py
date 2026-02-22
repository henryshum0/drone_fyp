import numpy as np


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
