import numpy as np
import matplotlib.pyplot as plt
from gym_pybullet_drones.gateRL.interpolate import interpolate_waypoints
from transforms3d.euler import euler2quat, quat2euler, euler2mat


class WaypointsSet:
    def __init__(self, waypoints_list):
        self.waypoints_list = waypoints_list
        self.refresh()

    def refresh(self):
        self.idx = np.random.randint(len(self.waypoints_list))
        self.waypoints = self.waypoints_list[self.idx]
        self.waypoint_xyzs = self.waypoints["pos"]
        self.waypoint_rpys = self.waypoints["rpy"]
        self.waypoint_quats = np.array([euler2quat(*rpy) for rpy in self.waypoint_rpys])
        self.predefined_spawns = self.waypoints.get("spawn", [])
        self.max_dist_from_waypoint = self.waypoints["max_dist"]

    def set_waypoints_combo(self, idx):
        if idx < 0 or idx >= len(self.waypoints_list):
            raise ValueError(f"Invalid waypoints index: {idx}")
        self.idx = idx
        self.waypoints = self.waypoints_list[self.idx]
        self.waypoint_xyzs = self.waypoints["pos"]
        self.waypoint_rpys = self.waypoints["rpy"]
        self.waypoint_quats = np.array([euler2quat(*rpy) for rpy in self.waypoint_rpys])
        self.predefined_spawns = self.waypoints.get("spawn", [])
        self.max_dist_from_waypoint = self.waypoints["max_dist"]

    def get_n_waypoints_combo(self):
        return len(self.waypoints_list)
    
    def get_n_waypoints(self):
        return len(self.waypoint_xyzs)
    
    def get_waypoints_idx(self):
        return self.idx
    
    def get_waypoints(self):
        return self.waypoints
    
    def get_waypoints_xyzs (self):
        return self.waypoint_xyzs
    
    def get_waypoints_rpys (self):
        return self.waypoint_rpys 
    
    def get_waypoints_quats (self):
        return self.waypoint_quats
    
    def get_predefined_spawns(self):
        return self.predefined_spawns
    
    def get_max_dist_from_waypoint(self):
        return self.max_dist_from_waypoint
# Original simple square
waypoints1 = {
    "pos":np.array([
        np.array([0, 0, 0]),
        np.array([4, 0, 0]),
        np.array([4, 4, 0]),
        np.array([0, 4, 0]),
    ]),
    "rpy": np.array([
        np.array([0, 0, 0]),
        np.array([0, 0, np.pi/2]),
        np.array([0, 0, np.pi]),
        np.array([0, 0, np.pi * 3/2]),
    ]),
    "spawn":[
        {
            "pos": np.array([-.4, .4, -.4]),
            "vel": np.array([0, 0, 0]),
            "acc": np.array([0, 0, 0]),
            "rpy": np.array([0, 0, 0]),
            "next_waypoints": (0, 1)
        }
    ],
    "max_dist": 6.,
}

# Figure-8 pattern (3D)
waypoints_figure8 = {
    "pos": np.array([
        np.array([0, 0, 0]),
        np.array([2, 2, 1]),
        np.array([4, 0, 2]),
        np.array([2, -2, 1]),
        np.array([0, 0, 0]),
        np.array([-2, 2, 1]),
        np.array([-4, 0, 2]),
        np.array([-2, -2, 1]),
    ]),
    "rpy": np.array([
        np.array([0, 0, np.pi/4]),
        np.array([0, 0.1, 0.]),
        np.array([0, 0.2, np.pi* -1/2]),
        np.array([0, 0.1, np.pi]),
        np.array([0, 0, np.pi * 3/4]),
        np.array([0, 0.1, np.pi]),
        np.array([0, 0.2, np.pi * -1 /2 ]),
        np.array([0, 0.1, 0.]),
    ]),
    "spawn": [
        {
            "pos": np.array([-1.1, 0, 0]),
            "vel": np.array([0, 0, 0]),
            "acc": np.array([0, 0, 0]),
            "rpy": np.array([0, 0, 0]),
            "next_waypoints": [0, 1]
        }
    ],
    "max_dist": 4.,
}



# roll
waypoints_up_roll = {
    "pos": np.array([
        np.array([0, 0, 0]),
        np.array([2, 0, 2]),
        np.array([4, 0, 0]),
    ]),
    "rpy": np.array([
        np.array([0, 0, 0]),
        np.array([0, np.pi, 0]),
        np.array([0, 0, 0]),
    ]),
    "spawn": [
        {
            "pos": np.array([-1.1, 0, 0]),
            "vel": np.array([0, 0, 0]),
            "acc": np.array([0, 0, 0]),
            "rpy": np.array([0, 0, 0]),
            "next_waypoints": [0, 1]
        }
    ],
    "max_dist": 4.,
}

waypoints_down_roll = {
    "pos": np.array([
        np.array([0, 0, 0]),
        np.array([2, 0, -2]),
        np.array([4, 0, 0]),
    ]),
    "rpy": np.array([
        np.array([0, 0, 0]),
        np.array([0, np.pi, 0]),
        np.array([0, 0, 0]),
    ]),
    "spawn": [
        {
            "pos": np.array([-1.1, 0, 0]),
            "vel": np.array([0, 0, 0]),
            "acc": np.array([0, 0, 0]),
            "rpy": np.array([0, 0, 0]),
            "next_waypoints": [0, 1]
        }
    ],
    "max_dist": 4.,
}

waypoints_right_loop = {
    "pos": np.array([
        np.array([0, 0, 0]),
        np.array([2, 0, 0]),
        np.array([3, 0, 0]),
        np.array([4, 0, 0]),
        np.array([6, 0, 0]),
    ]),
    "rpy": np.array([
        np.array([0, 0, 0]),
        np.array([0, 0, np.pi * 1/2]),
        np.array([0, 0, np.pi * 1/2]),
        np.array([0, 0, np.pi * 1/2]),
        np.array([0, 0, 0]),
    ]),
    "spawn": [
        {
            "pos": np.array([-1.1, 0, 0]),
            "vel": np.array([0, 0, 0]),
            "acc": np.array([0, 0, 0]),
            "rpy": np.array([0, 0, 0]),
            "next_waypoints": [0, 1]
        }
    ],
    "max_dist": 3.,
}

waypoints_left_loop = {
    "pos": np.array([
        np.array([0, 0, 0]),
        np.array([2, 0, 0]),
        np.array([3, 0, 0]),
        np.array([4, 0, 0]),
        np.array([6, 0, 0]),
    ]),
    "rpy": np.array([
        np.array([0, 0, 0]),
        np.array([0, 0, -np.pi * 1/2]),
        np.array([0, 0, -np.pi * 1/2]),
        np.array([0, 0, -np.pi * 1/2]),
        np.array([0, 0, 0]),
    ]),
    "spawn": [
        {
            "pos": np.array([-1.1, 0, 0]),
            "vel": np.array([0, 0, 0]),
            "acc": np.array([0, 0, 0]),
            "rpy": np.array([0, 0, 0]),
            "next_waypoints": [0, 1]
        }
    ],
    "max_dist": 3.,
}

waypoints_left = {
    "pos": np.array([
        np.array([0, 0, 0]),
        np.array([2, 1, 0]),
        np.array([4, 0, 0]),
    ]),
    "rpy": np.array([
        np.array([0, 0, 0]),
        np.array([0, 0, 0]),
        np.array([0, 0, 0]),
    ]),
    "spawn": [
        {
            "pos": np.array([-1.1, 0, 0]),
            "vel": np.array([0, 0, 0]),
            "acc": np.array([0, 0, 0]),
            "rpy": np.array([0, 0, 0]),
            "next_waypoints": [0, 1]
        }
    ],
    "max_dist": 3.,
}

waypoints_right = {
    "pos": np.array([
        np.array([0, 0, 0]),
        np.array([2, -1, 0]),
        np.array([4, 0, 0]),
    ]),
    "rpy": np.array([
        np.array([0, 0, 0]),
        np.array([0, 0, 0]),
        np.array([0, 0, 0]),
    ]),
    "spawn": [
        {
            "pos": np.array([-1.1, 0, 0]),
            "vel": np.array([0, 0, 0]),
            "acc": np.array([0, 0, 0]),
            "rpy": np.array([0, 0, 0]),
            "next_waypoints": [0, 1]
        }
    ],
    "max_dist": 3.,
}

waypoints_up = {
    "pos": np.array([
        np.array([0, 0, 0]),
        np.array([2, 0, 1]),
        np.array([4, 0, 0]),
    ]),
    "rpy": np.array([
        np.array([0, 0, 0]),
        np.array([0, 0, 0]),
        np.array([0, 0, 0]),
    ]),
    "spawn": [
        {
            "pos": np.array([-1.1, 0, 0]),
            "vel": np.array([0, 0, 0]),
            "acc": np.array([0, 0, 0]),
            "rpy": np.array([0, 0, 0]),
            "next_waypoints": [0, 1]
        }
    ],
    "max_dist": 3.,
}

waypoints_down = {
    "pos": np.array([
        np.array([0, 0, 0]),
        np.array([2, 0, -1]),
        np.array([4, 0, 0]),
    ]),
    "rpy": np.array([
        np.array([0, 0, 0]),
        np.array([0, 0, 0]),
        np.array([0, 0, 0]),
    ]),
    "spawn": [
        {
            "pos": np.array([-1.1, 0, 0]),
            "vel": np.array([0, 0, 0]),
            "acc": np.array([0, 0, 0]),
            "rpy": np.array([0, 0, 0]),
            "next_waypoints": [0, 1]
        }
    ],
    "max_dist": 3.,
}

waypoints_up_left = {
    "pos": np.array([
        np.array([0, 0, 0]),
        np.array([2, 1, 1]),
        np.array([4, 0, 0]),
    ]),
    "rpy": np.array([
        np.array([0, 0, 0]),
        np.array([0, 0, 0]),
        np.array([0, 0, 0]),
    ]),
    "spawn": [
        {
            "pos": np.array([-1.1, 0, 0]),
            "vel": np.array([0, 0, 0]),
            "acc": np.array([0, 0, 0]),
            "rpy": np.array([0, 0, 0]),
            "next_waypoints": [0, 1]
        }
    ],
    "max_dist": 3.,
}

waypoints_up_right = {
    "pos": np.array([
        np.array([0, 0, 0]),
        np.array([2, -1, 1]),
        np.array([4, 0, 0]),
    ]),
    "rpy": np.array([
        np.array([0, 0, 0]),
        np.array([0, 0, 0]),
        np.array([0, 0, 0]),
    ]),
    "spawn": [
        {
            "pos": np.array([-1.1, 0, 0]),
            "vel": np.array([0, 0, 0]),
            "acc": np.array([0, 0, 0]),
            "rpy": np.array([0, 0, 0]),
            "next_waypoints": [0, 1]
        }
    ],
    "max_dist": 3.,
}

waypoints_down_left = {
    "pos": np.array([
        np.array([0, 0, 0]),
        np.array([2, 1, -1]),
        np.array([4, 0, 0]),
    ]),
    "rpy": np.array([
        np.array([0, 0, 0]),
        np.array([0, 0, 0]),
        np.array([0, 0, 0]),
    ]),
    "spawn": [
        {
            "pos": np.array([-1.1, 0, 0]),
            "vel": np.array([0, 0, 0]),
            "acc": np.array([0, 0, 0]),
            "rpy": np.array([0, 0, 0]),
            "next_waypoints": [0, 1]
        }
    ],
    "max_dist": 3.,
}

waypoints_down_right = {
    "pos": np.array([
        np.array([0, 0, 0]),
        np.array([2, -1, -1]),
        np.array([4, 0, 0]),
    ]),
    "rpy": np.array([
        np.array([0, 0, 0]),
        np.array([0, 0, 0]),
        np.array([0, 0, 0]),
    ]),
    "spawn": [
        {
            "pos": np.array([-1.1, 0, 0]),
            "vel": np.array([0, 0, 0]),
            "acc": np.array([0, 0, 0]),
            "rpy": np.array([0, 0, 0]),
            "next_waypoints": [0, 1]
        }
    ],
    "max_dist": 3.,
}

waypoints_return_up = {
    "pos": np.array([
        np.array([0, 0, 0]),
        np.array([0, 0, 1]),
        np.array([-2, 0, 0]),
    ]),
    "rpy": np.array([
        np.array([0, 0, 0]),
        np.array([0, 0, np.pi]),
        np.array([0, 0, np.pi]),
    ]),
    "spawn": [
        {
            "pos": np.array([-1.1, 0, 0]),
            "vel": np.array([0, 0, 0]),
            "acc": np.array([0, 0, 0]),
            "rpy": np.array([0, 0, 0]),
            "next_waypoints": [0, 1]
        }
    ],
    "max_dist": 3.,
}

waypoints_return_down = {
    "pos": np.array([
        np.array([0, 0, 0]),
        np.array([0, 0, -1]),
        np.array([-2, 0, 0]),
    ]),
    "rpy": np.array([
        np.array([0, 0, 0]),
        np.array([0, 0, np.pi]),
        np.array([0, 0, np.pi]),
    ]),
    "spawn": [
        {
            "pos": np.array([-1.1, 0, 0]),
            "vel": np.array([0, 0, 0]),
            "acc": np.array([0, 0, 0]),
            "rpy": np.array([0, 0, 0]),
            "next_waypoints": [0, 1]
        }
    ],
    "max_dist": 3.,
}

waypoints_return_left = {
    "pos": np.array([
        np.array([0, 0, 0]),
        np.array([0, 1, 0]),
        np.array([-2, 0, 0]),
    ]),
    "rpy": np.array([
        np.array([0, 0, 0]),
        np.array([0, 0, np.pi]),
        np.array([0, 0, np.pi]),
    ]),
    "spawn": [
        {
            "pos": np.array([-1.1, 0, 0]),
            "vel": np.array([0, 0, 0]),
            "acc": np.array([0, 0, 0]),
            "rpy": np.array([0, 0, 0]),
            "next_waypoints": [0, 1]
        }
    ],
    "max_dist": 3.,
}

waypoints_return_right = {
    "pos": np.array([
        np.array([0, 0, 0]),
        np.array([0, -1, 0]),
        np.array([-2, 0, 0]),
    ]),
    "rpy": np.array([
        np.array([0, 0, 0]),
        np.array([0, 0, np.pi]),
        np.array([0, 0, np.pi]),
    ]),
    "spawn": [
        {
            "pos": np.array([-1.1, 0, 0]),
            "vel": np.array([0, 0, 0]),
            "acc": np.array([0, 0, 0]),
            "rpy": np.array([0, 0, 0]),
            "next_waypoints": [0, 1]
        }
    ],
    "max_dist": 3.,
}

waypoints_return_left_up = {
    "pos": np.array([
        np.array([0, 0, 0]),
        np.array([0, 1, 1]),
        np.array([-2, 0, 0]),
    ]),
    "rpy": np.array([
        np.array([0, 0, 0]),
        np.array([0, 0, np.pi]),
        np.array([0, 0, np.pi]),
    ]),
    "spawn": [
        {
            "pos": np.array([-1.1, 0, 0]),
            "vel": np.array([0, 0, 0]),
            "acc": np.array([0, 0, 0]),
            "rpy": np.array([0, 0, 0]),
            "next_waypoints": [0, 1]
        }
    ],
    "max_dist": 3.,
}

waypoints_return_left_down = {
    "pos": np.array([
        np.array([0, 0, 0]),
        np.array([0, 1, -1]),
        np.array([-2, 0, 0]),
    ]),
    "rpy": np.array([
        np.array([0, 0, 0]),
        np.array([0, 0, np.pi]),
        np.array([0, 0, np.pi]),
    ]),
    "spawn": [
        {
            "pos": np.array([-1.1, 0, 0]),
            "vel": np.array([0, 0, 0]),
            "acc": np.array([0, 0, 0]),
            "rpy": np.array([0, 0, 0]),
            "next_waypoints": [0, 1]
        }
    ],
    "max_dist": 3.,
}

waypoints_return_right_up = {
    "pos": np.array([
        np.array([0, 0, 0]),
        np.array([0, -1, 1]),
        np.array([-2, 0, 0]),
    ]),
    "rpy": np.array([
        np.array([0, 0, 0]),
        np.array([0, 0, np.pi]),
        np.array([0, 0, np.pi]),
    ]),
    "spawn": [
        {
            "pos": np.array([-1.1, 0, 0]),
            "vel": np.array([0, 0, 0]),
            "acc": np.array([0, 0, 0]),
            "rpy": np.array([0, 0, 0]),
            "next_waypoints": [0, 1]
        }
    ],
    "max_dist": 3.,
}

waypoints_return_right_down = {
    "pos": np.array([
        np.array([0, 0, 0]),
        np.array([0, -1, -1]),
        np.array([-2, 0, 0]),
    ]),
    "rpy": np.array([
        np.array([0, 0, 0]),
        np.array([0, 0, np.pi]),
        np.array([0, 0, np.pi]),
    ]),
    "spawn": [
        {
            "pos": np.array([-1.1, 0, 0]),
            "vel": np.array([0, 0, 0]),
            "acc": np.array([0, 0, 0]),
            "rpy": np.array([0, 0, 0]),
            "next_waypoints": [0, 1]
        }
    ],
    "max_dist": 3.,
}

waypoints_left_U = {
    "pos": np.array([
        np.array([0, 0, 0]),
        np.array([1, 1, 0]),
        np.array([2, 0, 0]),
    ]),
    "rpy": np.array([
        np.array([0, 0, 0]),
        np.array([0, 0, np.pi * -1/2]),
        np.array([0, 0, 0]),
    ]),
    "spawn": [
        {
            "pos": np.array([-1.1, 0, 0]),
            "vel": np.array([0, 0, 0]),
            "acc": np.array([0, 0, 0]),
            "rpy": np.array([0, 0, 0]),
            "next_waypoints": [0, 1]
        }
    ],
    "max_dist": 3.,
}

waypoints_right_U = {
    "pos": np.array([
        np.array([0, 0, 0]),
        np.array([1, -1, 0]),
        np.array([2, 0, 0]),
    ]),
    "rpy": np.array([
        np.array([0, 0, 0]),
        np.array([0, 0, np.pi * 1/2]),
        np.array([0, 0, 0]),
    ]),
    "spawn": [
        {
            "pos": np.array([-1.1, 0, 0]),
            "vel": np.array([0, 0, 0]),
            "acc": np.array([0, 0, 0]),
            "rpy": np.array([0, 0, 0]),
            "next_waypoints": [0, 1]
        }
    ],
    "max_dist": 3.,
}


all_waypoints = [
    waypoints_up_roll,
    waypoints_down_roll,
    waypoints_right_loop,
    waypoints_left_loop,
    waypoints_left,
    waypoints_right,
    waypoints_up,
    waypoints_down,
    waypoints_up_left,
    waypoints_up_right,
    waypoints_down_left,
    waypoints_down_right,
    waypoints_return_up,
    waypoints_return_down,
    waypoints_return_left,
    waypoints_return_right,
    waypoints_return_left_up,
    waypoints_return_left_down,
    waypoints_return_right_up,
    waypoints_return_right_down,
    waypoints_left_U,
    waypoints_right_U,
]


def _normalize_waypoint_sets(waypoint_sets=None):
    if waypoint_sets is None:
        discovered = {}
        for name, value in globals().items():
            if not isinstance(value, dict):
                continue
            if "pos" not in value or "rpy" not in value:
                continue
            if name.startswith("waypoint"):
                discovered[name] = value
        waypoint_sets = discovered

    if isinstance(waypoint_sets, dict) and "pos" in waypoint_sets and "rpy" in waypoint_sets:
        return {"waypoints": waypoint_sets}

    if isinstance(waypoint_sets, dict):
        return waypoint_sets

    return {f"waypoints_{idx}": wp for idx, wp in enumerate(waypoint_sets)}


def visualize_all_waypoints(waypoint_sets=None, show_heading=True, show_z_axis=True, figsize_scale=5):
    """Visualize one or many waypoint sets in 3D.

    Args:
        waypoint_sets: Optional dict of {name: waypoint_dict} or iterable of waypoint_dict.
                       If None, all waypoint dicts defined in this module are plotted.
        show_heading: If True, draw heading arrows from waypoint yaw.
        show_z_axis: If True, draw each waypoint's local z-axis in world frame.
        figsize_scale: Scale factor for subplot figure size.
    """
    waypoint_sets = _normalize_waypoint_sets(waypoint_sets)
    if len(waypoint_sets) == 0:
        raise ValueError("No waypoint sets found to visualize.")

    n_sets = len(waypoint_sets)
    n_cols = min(2, n_sets)
    n_rows = int(np.ceil(n_sets / n_cols))

    fig = plt.figure(figsize=(figsize_scale * n_cols, figsize_scale * n_rows))

    for i, (name, data) in enumerate(waypoint_sets.items(), start=1):
        pos = np.asarray(data["pos"], dtype=float)
        rpy = np.asarray(data["rpy"], dtype=float)
        spawn = data.get("spawn", [])

        ax = fig.add_subplot(n_rows, n_cols, i, projection="3d")
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], "-o", linewidth=1.5, markersize=4, label="path")
        ax.scatter(pos[0, 0], pos[0, 1], pos[0, 2], c="green", s=45, label="start")
        ax.scatter(pos[-1, 0], pos[-1, 1], pos[-1, 2], c="red", s=45, label="end")

        R = np.array([euler2mat(*angles) for angles in rpy])
        x_dirs = R[:, :, 0]  # Local x-axis in world frame
        z_dirs = R[:, :, 2]  # Local z-axis in world frame

        if show_heading:
            arrow_len = 0.35 
            ax.quiver(pos[:, 0], pos[:, 1], pos[:, 2], x_dirs[:, 0] * arrow_len, x_dirs[:, 1] * arrow_len, x_dirs[:, 2] * arrow_len, color="tab:orange", linewidth=1.2)

        if show_z_axis:
            z_axis_len = 0.35

            ax.quiver(
                pos[:, 0],
                pos[:, 1],
                pos[:, 2],
                z_dirs[:, 0] * z_axis_len,
                z_dirs[:, 1] * z_axis_len,
                z_dirs[:, 2] * z_axis_len,
                color="tab:cyan",
                linewidth=1.2,
            )

        for s_idx, s in enumerate(spawn):
            s_pos = np.asarray(s["pos"], dtype=float)
            ax.scatter(s_pos[0], s_pos[1], s_pos[2], c="purple", s=40, marker="x", label="spawn" if s_idx == 0 else None)

        ax.set_title(name)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        ax.grid(True, alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        if show_heading:
            labels.append("heading")
            handles.append(plt.Line2D([0], [0], color="tab:orange", lw=1.8))
        if show_z_axis:
            labels.append("z-axis")
            handles.append(plt.Line2D([0], [0], color="tab:cyan", lw=1.8))
        ax.legend(handles, labels, loc="best")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    loops = {
        "waypoints_up_roll": waypoints_up_roll,
        "waypoints_down_roll": waypoints_down_roll,
        "waypoints_right_loop": waypoints_right_loop,
        "waypoints_left_loop": waypoints_left_loop,
    }

    updownsleftright = {
        "waypoints_up": waypoints_up,
        "waypoints_down": waypoints_down,
        "waypoints_left": waypoints_left,
        "waypoints_right": waypoints_right,
        "waypoints_up_left": waypoints_up_left,
        "waypoints_up_right": waypoints_up_right,
        "waypoints_down_left": waypoints_down_left,
        "waypoints_down_right": waypoints_down_right,
    }
    leftrightU = {
        "waypoints_left_U": waypoints_left_U,
        "waypoints_right_U": waypoints_right_U,
    }

    visualize_all_waypoints(loops)
    visualize_all_waypoints(updownsleftright)
    visualize_all_waypoints(leftrightU)
