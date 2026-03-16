import numpy as np
from .WaypointTemplate import *

class ZeroTemplate(WaypointTemplate):
    def __init__(self,):
        waypoints_xyzs = np.array([
            np.array([0, 0, 0]),
        ])  
        waypoints_rpys = np.array([
            np.array([0, 0, 0]),
        ])
        spawns = [
            {
                "pos": np.array([-1.1, 0, 0]),
                "vel": np.array([0, 0, 0]),
                "acc": np.array([0, 0, 0]),
                "rpy": np.array([0, 0, 0]),
                "next_waypoints": [0]
            }
        ]
        waypoints_normal_distr = np.array([
            np.array([[0, 0], [0, 0], [0, 0]]),
        ])
        waypoints_rpys_choices = [
            [RPY_FRONT_UP, RPY_FRONT_DOWN],
        ]
        waypoints_scale = [1.0, 1.0]
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
        
class EasyTemplate1(WaypointTemplate):
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
            np.array([[0, 1], [0, 1], [0, 1]]),
        ])
        waypoints_rpys_choices = [
            [RPY_FRONT_UP, RPY_BACK_UP],
            [RPY_FRONT_UP, RPY_LEFT_UP, RPY_RIGHT_UP, RPY_FRONT_LEFT, RPY_FRONT_RIGHT],
        ]
        waypoints_scale = [0.25, 3]
        max_dist = 7

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
    
class EasyTemplate2(WaypointTemplate):
    def __init__(self,):
        waypoints_xyzs = np.array([
            np.array([0, 0, 0]),
            np.array([1, 1, 0]),
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
            np.array([[0, 1], [0, 1], [0, 1]]),
        ])
        waypoints_rpys_choices = [
            [RPY_FRONT_UP, RPY_BACK_UP],
            [RPY_FRONT_UP, RPY_LEFT_UP, RPY_RIGHT_UP, RPY_FRONT_LEFT, RPY_FRONT_RIGHT],
        ]
        waypoints_scale = [0.25, 3]
        max_dist = 7

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

class EasyTemplate3(WaypointTemplate):
    def __init__(self,):
        waypoints_xyzs = np.array([
            np.array([0, 0, 0]),
            np.array([1, -1, 0]),
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
            np.array([[0, 1], [0, 1], [0, 1]]),
        ])
        waypoints_rpys_choices = [
            [RPY_FRONT_UP, RPY_BACK_UP],
            [RPY_FRONT_UP, RPY_LEFT_UP, RPY_RIGHT_UP, RPY_FRONT_LEFT, RPY_FRONT_RIGHT],
        ]
        waypoints_scale = [0.25, 3]
        max_dist = 7

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

class EasyTemplate4(WaypointTemplate):
    def __init__(self,):
        waypoints_xyzs = np.array([
            np.array([0, 0, 0]),
            np.array([-1, 0, 0]),
        ])  
        waypoints_rpys = np.array([
            np.array([0, 0, 0]),
            np.array([0, 0, np.pi]),
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
            np.array([[0, 1], [0, 1], [0, 1]]),
        ])
        waypoints_rpys_choices = [
            [RPY_FRONT_UP, RPY_BACK_UP],
            [RPY_BACK_UP, RPY_RIGHT_UP, RPY_LEFT_UP, RPY_BACK_RIGHT, RPY_BACK_LEFT],
        ]
        waypoints_scale = [0.25, 3]
        max_dist = 7

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

class EasyTemplate5(WaypointTemplate):
    def __init__(self,):
        waypoints_xyzs = np.array([
            np.array([0, 0, 0]),
            np.array([-1, -1, 0]),
        ])  
        waypoints_rpys = np.array([
            np.array([0, 0, 0]),
            np.array([0, 0, np.pi]),
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
            np.array([[0, 1], [0, 1], [0, 1]]),
        ])
        waypoints_rpys_choices = [
            [RPY_FRONT_UP, RPY_BACK_UP],
            [RPY_BACK_UP, RPY_RIGHT_UP, RPY_LEFT_UP, RPY_BACK_RIGHT, RPY_BACK_LEFT],
        ]
        waypoints_scale = [0.25, 3]
        max_dist = 7

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

class EasyTemplate6(WaypointTemplate):
    def __init__(self,):
        waypoints_xyzs = np.array([
            np.array([0, 0, 0]),
            np.array([-1, 1, 0]),
        ])  
        waypoints_rpys = np.array([
            np.array([0, 0, 0]),
            np.array([0, 0, np.pi]),
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
            np.array([[0, 1], [0, 1], [0, 1]]),
        ])
        waypoints_rpys_choices = [
            [RPY_FRONT_UP, RPY_BACK_UP],
            [RPY_BACK_UP, RPY_RIGHT_UP, RPY_LEFT_UP, RPY_BACK_RIGHT, RPY_BACK_LEFT],
        ]
        waypoints_scale = [0.25, 3]
        max_dist = 7

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