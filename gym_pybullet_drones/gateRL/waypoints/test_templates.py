from .WaypointTemplate import *

class TestTemplate1(WaypointTemplate):
    def __init__(self,):
        waypoints_xyzs = np.array([
            np.array([0, 0, 0]),
            np.array([1, 0, 0]),
            np.array([1.5, 0, 0]),
            np.array([2, 0, 0]),
            np.array([2.5, 0, 0]),
            np.array([4, 0, 0]),
        ])  
        waypoints_rpys = np.array([
            np.array([0, 0, 0]),
            np.array([np.pi, 0, np.pi/2]),
            np.array([np.pi, 0, np.pi/2]),
            np.array([np.pi, 0, np.pi/2]),
            np.array([np.pi, 0, np.pi/2]),
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

class TestTemplate2(WaypointTemplate):
    def __init__(self,):
        waypoints_xyzs = np.array([
            np.array([0, 0, 0]),
            np.array([2, 0, 1]),
            np.array([4, 0, 0]),
            np.array([6, 0, 1]),
            np.array([8, 0, 0]),
            np.array([9, 0, 0]),
        ])  
        waypoints_rpys = np.array([
            np.array([0, 0, 0]),
            np.array([0, np.pi, 0]),
            np.array([0, np.pi, 0]),
            np.array([0, np.pi, 0]),
            np.array([0, np.pi, 0]),
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

class TestTemplate3(WaypointTemplate):
    def __init__(self,):
        waypoints_xyzs = np.array([
            np.array([0, 0, 0]),
            np.array([2, 0.5, 0]),
            np.array([4, -0.5, 0]),
            np.array([6, 0.5, 0]),
            np.array([8, -0.5, 0]),
            np.array([9, 0, 0]),
        ])  
        waypoints_rpys = np.array([
            np.array([0, 0, 0]),
            np.array([np.pi/6, 0, 0]),
            np.array([-np.pi/6, 0, 0]),
            np.array([np.pi/6, 0, 0]),
            np.array([-np.pi/6, 0, 0]),
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

class TestTemplate4(WaypointTemplate):
    def __init__(self,):
        waypoints_xyzs = np.array([
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.75, 0.0]),
            np.array([2.0, 0.0, 0.0]),
            np.array([1.0, -0.75, 0.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([-1.0, 0.75, 0.0]),
            np.array([-2.0, 0.0, 0.0]),
            np.array([-1.0, -0.75, 0.0]),
            np.array([0.0, 0.0, 0.0]),
        ])
        waypoints_rpys = np.array([
            np.array([0.0, 0.0, np.arctan2(0.75, 1.0)]),
            np.array([0.0, 0.0, np.arctan2(-0.75, 1.0)]),
            np.array([0.0, 0.0, np.arctan2(-0.75, -1.0)]),
            np.array([0.0, 0.0, np.arctan2(0.75, -1.0)]),
            np.array([0.0, 0.0, np.arctan2(0.75, -1.0)]),
            np.array([0.0, 0.0, np.arctan2(-0.75, -1.0)]),
            np.array([0.0, 0.0, np.arctan2(-0.75, 1.0)]),
            np.array([0.0, 0.0, np.arctan2(0.75, 1.0)]),
            np.array([0.0, 0.0, 0.0]),
        ])
        spawns = [
            {
                "pos": np.array([-1.1, 0.0, 0.0]),
                "vel": np.array([0.0, 0.0, 0.0]),
                "acc": np.array([0.0, 0.0, 0.0]),
                "rpy": np.array([0.0, 0.0, 0.0]),
                "next_waypoints": [0, 1]
            }
        ]
        waypoints_normal_distr = np.array([
            np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
            np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
            np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
            np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
            np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
            np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
            np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
            np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
            np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
        ])
        waypoints_rpys_choices = [
        ]
        waypoints_scale = [1.0, 1.0]
        max_dist = 10

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