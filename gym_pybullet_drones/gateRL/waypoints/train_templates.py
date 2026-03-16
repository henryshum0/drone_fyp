from .WaypointTemplate import *

class DownUpTemplate(WaypointTemplate):
    def __init__(self,):
        waypoints_xyzs = np.array([
            np.array([0, 0, 0]),
            np.array([0, 0, -1]),
        ])  
        waypoints_rpys = np.array([
            np.array([0, 0, 0]),
            np.array([0, 0, 0]),
        ])
        spawns = [
            {
                "pos": np.array([-1.1, 0, 0]),
                "vel": np.array([0, 0, 0]),
                "acc": np.array([0, 0, 0]),
                "rpy": np.array([0, 0, 0]),
                "next_waypoints": [0, 1]
            }
        ]
        waypoints_normal_distr = np.array([
            [[0, 0], [0, 0], [0, 0]],
            [[0, 0], [0, 1], [0, 0]],
        ])
        waypoints_rpys_choices = [
            [RPY_FRONT_DOWN],
            [RPY_FRONT_UP, RPY_BACK_UP, RPY_LEFT_UP, RPY_RIGHT_UP],
        ]
        waypoints_scale = [1, 2.0]
        max_dist = 5

        super().__init__(
            waypoints_xyzs=waypoints_xyzs,
            waypoints_rpys=waypoints_rpys,
            spawns=spawns,
            waypoints_normal_distr=waypoints_normal_distr,
            rpy_choices=waypoints_rpys_choices,
            waypoints_scale=waypoints_scale,
            max_dist=max_dist,
            difficulty="easy"
        )

class UpDownTemplate(WaypointTemplate):
    def __init__(self,):
        waypoints_xyzs = np.array([
            np.array([0, 0, 0]),
            np.array([0, 0, 1]),
        ])  
        waypoints_rpys = np.array([
            np.array([0, 0, 0]),
            np.array([0, 0, 0]),
        ])
        spawns = [
            {
                "pos": np.array([-1.1, 0, 0]),
                "vel": np.array([0, 0, 0]),
                "acc": np.array([0, 0, 0]),
                "rpy": np.array([0, 0, 0]),
                "next_waypoints": [0, 1]
            }
        ]
        waypoints_normal_distr = np.array([
            [[0, 0], [0, 0], [0, 0]],
            [[0, 0], [0, 1], [0, 0]],
        ])
        waypoints_rpys_choices = [
            [RPY_FRONT_UP],
            [RPY_FRONT_DOWN, RPY_BACK_DOWN, RPY_LEFT_DOWN, RPY_RIGHT_DOWN],
        ]
        waypoints_scale = [1, 2.0]
        max_dist = 5

        super().__init__(
            waypoints_xyzs=waypoints_xyzs,
            waypoints_rpys=waypoints_rpys,
            spawns=spawns,
            waypoints_normal_distr=waypoints_normal_distr,
            rpy_choices=waypoints_rpys_choices,
            waypoints_scale=waypoints_scale,
            max_dist=max_dist,
            difficulty="easy"
        )

class FrontFrontTemplate(WaypointTemplate):
    def __init__(self,):
        waypoints_xyzs = np.array([
            np.array([0, 0, 0]),
            np.array([1, 0, 0]),
        ])  
        waypoints_rpys = np.array([
            np.array([0, 0, 0]),
            np.array([0, 0, 0]),
        ])
        spawns = [
            {
                "pos": np.array([-1.1, 0, 0]),
                "vel": np.array([0, 0, 0]),
                "acc": np.array([0, 0, 0]),
                "rpy": np.array([0, 0, 0]),
                "next_waypoints": [0, 1]
            }
        ]
        waypoints_normal_distr = np.array([
            [[0, 0], [0, 0], [0, 0]],
            [[0, 0], [0, 2], [0, 2]],
        ])
        waypoints_rpys_choices = [
            [RPY_FRONT_UP, RPY_BACK_UP, RPY_LEFT_UP, RPY_RIGHT_UP],
            [RPY_FRONT_UP, RPY_BACK_UP, RPY_LEFT_UP, RPY_RIGHT_UP],
        ]
        waypoints_scale = [.5, 5.0]
        max_dist = 10

        super().__init__(
            waypoints_xyzs=waypoints_xyzs,
            waypoints_rpys=waypoints_rpys,
            spawns=spawns,
            waypoints_normal_distr=waypoints_normal_distr,
            rpy_choices=waypoints_rpys_choices,
            waypoints_scale=waypoints_scale,
            max_dist=max_dist,
            difficulty="easy"
        )

class FrontBackTemplate(WaypointTemplate):
    def __init__(self,):
        waypoints_xyzs = np.array([
            np.array([0, 0, 0]),
            np.array([-1, 0, 0]),
        ])  
        waypoints_rpys = np.array([
            np.array([0, 0, 0]),
            np.array([0, 0, 0]),
        ])
        spawns = [
            {
                "pos": np.array([-1.1, 0, 0]),
                "vel": np.array([0, 0, 0]),
                "acc": np.array([0, 0, 0]),
                "rpy": np.array([0, 0, 0]),
                "next_waypoints": [0, 1]
            }
        ]
        waypoints_normal_distr = np.array([
            [[0, 0], [0, 0], [0, 0]],
            [[0, 0], [0, 2], [0, 2]],
        ])
        waypoints_rpys_choices = [
            [RPY_FRONT_UP, RPY_BACK_UP, RPY_LEFT_UP, RPY_RIGHT_UP],
            [RPY_FRONT_UP, RPY_BACK_UP, RPY_LEFT_UP, RPY_RIGHT_UP],
        ]
        waypoints_scale = [.5, 5.0]
        max_dist = 10

        super().__init__(
            waypoints_xyzs=waypoints_xyzs,
            waypoints_rpys=waypoints_rpys,
            spawns=spawns,
            waypoints_normal_distr=waypoints_normal_distr,
            rpy_choices=waypoints_rpys_choices,
            waypoints_scale=waypoints_scale,
            max_dist=max_dist,
            difficulty="easy"
        )