from .WaypointTemplate import *
import numpy as np

class SideZTemplate(WaypointTemplate):
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
            np.array([[0, 0], [0, 0], [0, 0]]),
            np.array([[0, 0], [0, 1], [0, 1]]),
        ])
        waypoints_rpys_choices = [
            [RPY_FRONT_DOWN, RPY_BACK_DOWN, RPY_LEFT_DOWN, RPY_RIGHT_DOWN],
            [RPY_FRONT_LEFT, RPY_FRONT_RIGHT, RPY_BACK_LEFT, RPY_BACK_RIGHT, RPY_LEFT_FRONT, RPY_LEFT_BACK, RPY_RIGHT_FRONT, RPY_RIGHT_BACK],
        ]
        waypoints_scale = [0.5, 2.0]
        max_dist = 8

        super().__init__(
            waypoints_xyzs=waypoints_xyzs,
            waypoints_rpys=waypoints_rpys,
            spawns=spawns,
            waypoints_normal_distr=waypoints_normal_distr,
            rpy_choices=waypoints_rpys_choices,
            waypoints_scale=waypoints_scale,
            max_dist=max_dist,
            difficulty="hard"
        )

class BackSideZTemplate(WaypointTemplate):
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
            np.array([[0, 0], [0, 0], [0, 0]]),
            np.array([[0, 0], [0, 1], [0, 1]]),
        ])
        waypoints_rpys_choices = [
            [RPY_FRONT_DOWN, RPY_BACK_DOWN, RPY_LEFT_DOWN, RPY_RIGHT_DOWN],
            [RPY_FRONT_LEFT, RPY_FRONT_RIGHT, RPY_BACK_LEFT, RPY_BACK_RIGHT, RPY_LEFT_FRONT, RPY_LEFT_BACK, RPY_RIGHT_FRONT, RPY_RIGHT_BACK],
        ]
        waypoints_scale = [0.5, 2.0]
        max_dist = 8

        super().__init__(
            waypoints_xyzs=waypoints_xyzs,
            waypoints_rpys=waypoints_rpys,
            spawns=spawns,
            waypoints_normal_distr=waypoints_normal_distr,
            rpy_choices=waypoints_rpys_choices,
            waypoints_scale=waypoints_scale,
            max_dist=max_dist,
            difficulty="medium"
        )

class UpRollTemplate(WaypointTemplate):
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
            np.array([[0, 0], [0, 0], [0, 0]]),
            np.array([[0, 1], [0, 1], [0, 0]]),
        ])
        waypoints_rpys_choices = [
            [RPY_FRONT_DOWN, RPY_BACK_DOWN, RPY_LEFT_DOWN, RPY_RIGHT_DOWN],
            [RPY_BACK_DOWN, RPY_LEFT_DOWN, RPY_RIGHT_DOWN, RPY_FRONT_DOWN],
        ]
        waypoints_scale = [0.5, 2.0]
        max_dist = 8

        super().__init__(
            waypoints_xyzs=waypoints_xyzs,
            waypoints_rpys=waypoints_rpys,
            spawns=spawns,
            waypoints_normal_distr=waypoints_normal_distr,
            rpy_choices=waypoints_rpys_choices,
            waypoints_scale=waypoints_scale,
            max_dist=max_dist,
            difficulty="hard"
        )

class DownRollTemplate(WaypointTemplate):
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
            np.array([[0, 0], [0, 0], [0, 0]]),
            np.array([[0, 1], [0, 1], [0, 0]]),
        ])
        waypoints_rpys_choices = [
            [RPY_FRONT_DOWN, RPY_BACK_DOWN, RPY_LEFT_DOWN, RPY_RIGHT_DOWN],
            [RPY_BACK_DOWN, RPY_LEFT_DOWN, RPY_RIGHT_DOWN, RPY_FRONT_DOWN],
        ]
        waypoints_scale = [0.5, 2.0]
        max_dist = 5

        super().__init__(
            waypoints_xyzs=waypoints_xyzs,
            waypoints_rpys=waypoints_rpys,
            spawns=spawns,
            waypoints_normal_distr=waypoints_normal_distr,
            rpy_choices=waypoints_rpys_choices,
            waypoints_scale=waypoints_scale,
            max_dist=max_dist,
            difficulty="hard"
        )