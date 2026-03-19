import numpy as np
from transforms3d.euler import euler2quat

RPY_FRONT_UP = np.array([0, 0, 0])
RPY_FRONT_DOWN = np.array([np.pi, 0, 0])
RPY_FRONT_LEFT = np.array([np.pi/2, 0, 0])
RPY_FRONT_RIGHT = np.array([-np.pi/2, 0, 0])
RPY_BACK_UP = np.array([0, 0, np.pi])
RPY_BACK_DOWN = np.array([np.pi, 0, np.pi])
RPY_BACK_LEFT = np.array([np.pi/2, 0, np.pi])
RPY_BACK_RIGHT = np.array([-np.pi/2, 0, np.pi])
RPY_LEFT_UP = np.array([0, 0, np.pi/2])
RPY_LEFT_DOWN = np.array([np.pi, 0, np.pi/2])
RPY_LEFT_FRONT = np.array([np.pi/2, 0, np.pi/2])
RPY_LEFT_BACK = np.array([-np.pi/2, 0, np.pi/2])
RPY_RIGHT_UP = np.array([0, 0, -np.pi/2])
RPY_RIGHT_DOWN = np.array([np.pi, 0, -np.pi/2])
RPY_RIGHT_FRONT = np.array([-np.pi/2, 0, -np.pi/2])
RPY_RIGHT_BACK = np.array([np.pi/2, 0, -np.pi/2])
RPY_DOWN_FRONT = np.array([0, np.pi/2, 0])
RPY_DOWN_BACK = np.array([np.pi, np.pi/2, 0])
RPY_DOWN_LEFT = np.array([-np.pi/2, np.pi/2, 0])
RPY_DOWN_RIGHT = np.array([np.pi/2, np.pi/2, 0])

class WaypointTemplate():
    def __init__(
    self, 
    waypoints_xyzs,
    waypoints_rpys,
    spawns,
    waypoints_normal_distr,
    rpy_choices,
    waypoints_scale,
    max_dist,
    difficulty,
    repeat=1,
    time_limit_sec=5,
    ):
        self.waypoints_xyzs = waypoints_xyzs
        self.waypoints_rpys = waypoints_rpys
        self.spawns = spawns
        self.max_dist = max_dist
        self.xyzs_normal_distr = waypoints_normal_distr
        self.rpy_choices = rpy_choices
        self.waypoints_scale = waypoints_scale
        self.difficulty = difficulty
        self.repeat = repeat
        self.time_limit_sec = time_limit_sec

    def __call__(self):
        p = self.spawns[0]['pos']
        v = self.spawns[0]['vel']
        a = self.spawns[0]['acc']
        rpy = self.spawns[0]['rpy']
        xyzs = self.waypoints_xyzs
        rpys = self.waypoints_rpys
        quats = np.array([euler2quat(*rpy) for rpy in rpys])
        max_dist = self.max_dist
        return (p, v, a, rpy, xyzs, rpys, quats, max_dist)
    def sample(self):
        xyzs = self._randomized_xyzs()
        rpys = self._randomized_rpys()
        return xyzs, rpys, self.spawns, self.max_dist
    

    def _randomized_xyzs(self):
        waypoints_xyzs = np.asarray(self.waypoints_xyzs, dtype=float)
        if waypoints_xyzs.ndim != 2 or waypoints_xyzs.shape[1] != 3:
            raise ValueError("waypoints_xyzs must have shape (n_waypoints, 3).")

        xyzs_normal_distr = np.asarray(self.xyzs_normal_distr, dtype=float)
        if xyzs_normal_distr.shape != (waypoints_xyzs.shape[0], 3, 2):
            raise ValueError(
                "xyzs_normal_distr must have shape (n_waypoints, 3, 2) with [mean, std] per axis."
            )

        means = xyzs_normal_distr[:, :, 0]
        stds = np.maximum(xyzs_normal_distr[:, :, 1], 0.0)

        waypoints_scale = np.asarray(self.waypoints_scale, dtype=float).reshape(-1)
        if waypoints_scale.shape[0] != 2:
            raise ValueError(
                "waypoints_scale must have shape (2,) as [min_scale, max_scale] to scale the full waypoint set."
            )
        scale_low = min(waypoints_scale[0], waypoints_scale[1])
        scale_high = max(waypoints_scale[0], waypoints_scale[1])
        sampled_scale = np.random.uniform(scale_low, scale_high)

        waypoint_noise = np.random.normal(
            loc=means,
            scale=stds,
            size=waypoints_xyzs.shape,
        )
        randomized_waypoints_xyzs = (waypoints_xyzs + waypoint_noise) * sampled_scale

        return randomized_waypoints_xyzs
    
    def _randomized_rpys(self):
        waypoints_rpys = np.asarray(self.waypoints_rpys, dtype=float, copy=True)
        if waypoints_rpys.ndim != 2 or waypoints_rpys.shape[1] != 3:
            raise ValueError("waypoints_rpys must have shape (n_waypoints, 3).")

        for i in range(len(self.waypoints_rpys)):
            waypoints_rpys[i] = self.rpy_choices[i][np.random.randint(0, len(self.rpy_choices[i]))]
        return waypoints_rpys