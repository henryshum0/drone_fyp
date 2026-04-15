import numpy as np
from transforms3d.euler import euler2quat

RPY_FRONT_UP = np.array([0, 0, 0])
RPY_FRONT_DOWN = np.array([np.pi, 0, 0])
RPY_FRONT_LEFT = np.array([-np.pi/2, 0, 0])
RPY_FRONT_RIGHT = np.array([np.pi/2, 0, 0])
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
    waypoints_speeds=None,
    waypoints_durations=None,
    ):
        self.waypoints_xyzs = waypoints_xyzs
        self.waypoints_rpys = waypoints_rpys
        if waypoints_speeds is None:
            self.waypoints_speeds = np.zeros(len(waypoints_xyzs), dtype=float)
        else:
            self.waypoints_speeds = np.asarray(waypoints_speeds, dtype=float)
        if waypoints_durations is None:
            self.waypoints_durations = np.ones(len(waypoints_xyzs), dtype=float)
        else:
            self.waypoints_durations = np.asarray(waypoints_durations, dtype=float)
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
        return (p, v, a, rpy, xyzs, rpys, quats, max_dist, self.repeat, self.time_limit_sec)
    
    def sample(self):
        xyzs = self._randomized_xyzs()
        rpys = self._randomized_rpys()
        return xyzs, rpys, self.spawns, self.max_dist

    def sample_with_speeds(self):
        xyzs = self._randomized_xyzs()
        rpys = self._randomized_rpys()
        speeds = np.asarray(self.waypoints_speeds, dtype=float).copy()
        if speeds.shape[0] != xyzs.shape[0]:
            raise ValueError("waypoints_speeds must have the same length as waypoints_xyzs")
        return xyzs, rpys, speeds, self.spawns, self.max_dist

    def sample_with_speeds_and_durations(self):
        xyzs = self._randomized_xyzs()
        rpys = self._randomized_rpys()
        speeds = np.asarray(self.waypoints_speeds, dtype=float).copy()
        durations = np.asarray(self.waypoints_durations, dtype=float).copy()
        if speeds.shape[0] != xyzs.shape[0]:
            raise ValueError("waypoints_speeds must have the same length as waypoints_xyzs")
        if durations.shape[0] != xyzs.shape[0]:
            raise ValueError("waypoints_durations must have the same length as waypoints_xyzs")
        return xyzs, rpys, speeds, durations, self.spawns, self.max_dist

    def visualize_waypoints(self,
                            randomized=True,
                            show_orientation=True,
                            axis_length=0.25,
                            show_spawn=True,
                            show=True,
                            ax=None,
                            title=None):
        """Visualize waypoints in 3D for quick template debugging.

        Parameters
        ----------
        randomized : bool, optional
            If True, visualize sampled waypoints/rpys; otherwise plot base template values.
        show_orientation : bool, optional
            If True, draw each waypoint's local +X and +Z axes as direction arrows.
        axis_length : float, optional
            Arrow length for waypoint orientation.
        show_spawn : bool, optional
            If True, show spawn position from the first spawn entry.
        show : bool, optional
            If True, call matplotlib show().
        ax : matplotlib axis, optional
            Existing 3D axis to plot into. If None, a new figure/axis is created.
        title : str | None, optional
            Optional plot title. If None, a default title is used.

        Returns
        -------
        tuple
            (fig, ax) matplotlib figure and axis.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError("matplotlib is required for waypoint visualization") from exc

        if randomized:
            xyzs, rpys, _, _ = self.sample()
        else:
            xyzs = np.asarray(self.waypoints_xyzs, dtype=float)
            rpys = np.asarray(self.waypoints_rpys, dtype=float)

        if ax is None:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig = ax.figure

        ax.plot(xyzs[:, 0], xyzs[:, 1], xyzs[:, 2], '-o', color='tab:blue', linewidth=1.5, markersize=5, label='waypoints')

        for i, p in enumerate(xyzs):
            ax.text(p[0], p[1], p[2], f"wp{i}", fontsize=8)

        if show_orientation:
            for p, rpy in zip(xyzs, rpys):
                roll, pitch, yaw = rpy
                cr, sr = np.cos(roll), np.sin(roll)
                cp, sp = np.cos(pitch), np.sin(pitch)
                cy, sy = np.cos(yaw), np.sin(yaw)

                rot_x = np.array([[1, 0, 0],
                                  [0, cr, -sr],
                                  [0, sr, cr]])
                rot_y = np.array([[cp, 0, sp],
                                  [0, 1, 0],
                                  [-sp, 0, cp]])
                rot_z = np.array([[cy, -sy, 0],
                                  [sy, cy, 0],
                                  [0, 0, 1]])
                rot = rot_z @ rot_y @ rot_x

                # Draw local +X axis (forward direction).
                x_axis = rot[:, 0]
                ax.quiver(p[0], p[1], p[2],
                          x_axis[0], x_axis[1], x_axis[2],
                          length=axis_length, normalize=True, color='tab:orange')

                # Draw local +Z axis (up direction).
                z_axis = rot[:, 2]
                ax.quiver(p[0], p[1], p[2],
                          z_axis[0], z_axis[1], z_axis[2],
                          length=axis_length, normalize=True, color='tab:red')

        if show_spawn and len(self.spawns) > 0:
            spawn_pos = np.asarray(self.spawns[0]["pos"], dtype=float)
            ax.scatter(spawn_pos[0], spawn_pos[1], spawn_pos[2], marker='*', s=120, color='tab:green', label='spawn')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(loc='best')
        if title is None:
            title = f"{self.__class__.__name__} ({'sampled' if randomized else 'base'})"
        ax.set_title(title)

        x_range = np.ptp(xyzs[:, 0]) if xyzs.shape[0] > 0 else 1.0
        y_range = np.ptp(xyzs[:, 1]) if xyzs.shape[0] > 0 else 1.0
        z_range = np.ptp(xyzs[:, 2]) if xyzs.shape[0] > 0 else 1.0
        max_range = max(x_range, y_range, z_range, 1e-6)
        center = xyzs.mean(axis=0) if xyzs.shape[0] > 0 else np.zeros(3)
        half = max_range / 2.0 + axis_length
        ax.set_xlim(center[0] - half, center[0] + half)
        ax.set_ylim(center[1] - half, center[1] + half)
        ax.set_zlim(center[2] - half, center[2] + half)

        if show:
            plt.show()

        return fig, ax
    

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