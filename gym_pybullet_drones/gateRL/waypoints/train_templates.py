from .WaypointTemplate import *


class OmniTemplate(WaypointTemplate):
    def __init__(self,):
        waypoints_xyzs = np.array([
            np.array([0, 0, 0]),
            np.array([0, 0, 0]),
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
            [[0, 3], [0, 3], [0, 2]],
        ])
        waypoints_rpys_choices = [
            [RPY_FRONT_UP, RPY_FRONT_DOWN, RPY_FRONT_LEFT, RPY_FRONT_RIGHT],
            [
                RPY_FRONT_UP, RPY_FRONT_DOWN, RPY_FRONT_UP, RPY_FRONT_DOWN, RPY_FRONT_LEFT, RPY_FRONT_RIGHT,
                RPY_BACK_UP, RPY_BACK_DOWN, RPY_BACK_UP, RPY_BACK_DOWN, RPY_BACK_LEFT, RPY_BACK_RIGHT,
                RPY_LEFT_UP, RPY_LEFT_DOWN, RPY_LEFT_UP, RPY_LEFT_DOWN, RPY_LEFT_FRONT, RPY_LEFT_BACK,
                RPY_RIGHT_UP, RPY_RIGHT_DOWN, RPY_RIGHT_UP, RPY_RIGHT_DOWN, RPY_RIGHT_FRONT, RPY_RIGHT_BACK,
            ],
        ]
        waypoints_scale = [.6, .6]
        max_dist = 4
        repeat = 3
        time_limit_sec = 6

        self.r = [1., 3]
        self.phi = [0, 2*np.pi]
        self.theta = [0, np.pi]

        super().__init__(
            waypoints_xyzs=waypoints_xyzs,
            waypoints_rpys=waypoints_rpys,
            spawns=spawns,
            waypoints_normal_distr=waypoints_normal_distr,
            rpy_choices=waypoints_rpys_choices,
            waypoints_scale=waypoints_scale,
            max_dist=max_dist,
            difficulty="easy",
            repeat=repeat,
            time_limit_sec=time_limit_sec,
        )

    def _randomized_xyzs(self):
        r = np.random.uniform(self.r[0], self.r[1])
        phi = np.random.uniform(self.phi[0], self.phi[1])
        theta = np.random.uniform(self.theta[0], self.theta[1])
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return np.array([[0, 0, 0], [x, y, z]])
    
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
            [[0, 0.5], [0, 1], [0, 0]],
        ])
        waypoints_rpys_choices = [
            [RPY_FRONT_DOWN],
            [RPY_FRONT_UP, RPY_BACK_UP, RPY_LEFT_UP, RPY_RIGHT_UP],
        ]
        waypoints_scale = [.5, 3.0]
        max_dist = 6
        repeat = 2
        time_limit_sec = 6

        super().__init__(
            waypoints_xyzs=waypoints_xyzs,
            waypoints_rpys=waypoints_rpys,
            spawns=spawns,
            waypoints_normal_distr=waypoints_normal_distr,
            rpy_choices=waypoints_rpys_choices,
            waypoints_scale=waypoints_scale,
            max_dist=max_dist,
            difficulty="easy",
            repeat=repeat,
            time_limit_sec=time_limit_sec,
        )

class DownDownTemplate(WaypointTemplate):
    def __init__(self,):
        waypoints_xyzs = np.array([
            np.array([0, 0, 0]),
            np.array([2, 0, 0]),
            np.array([3, 0, 0]),
        ])  
        waypoints_rpys = np.array([
            np.array([0, 0, 0]),
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
            [[0, 0], [0, 1], [0, 0.5]],
            [[0, 0], [0, 0], [0, 0]],
        ])
        waypoints_rpys_choices = [
            [RPY_BACK_DOWN],
            [RPY_FRONT_DOWN, RPY_BACK_DOWN, RPY_LEFT_DOWN, RPY_RIGHT_DOWN],
            [RPY_BACK_DOWN],
        ]
        waypoints_scale = [.5, 3]
        max_dist = 6
        repeat = 3
        time_limit_sec = 6
        super().__init__(
            waypoints_xyzs=waypoints_xyzs,
            waypoints_rpys=waypoints_rpys,
            spawns=spawns,
            waypoints_normal_distr=waypoints_normal_distr,
            rpy_choices=waypoints_rpys_choices,
            waypoints_scale=waypoints_scale,
            max_dist=max_dist,
            difficulty="easy",
            repeat=repeat,
            time_limit_sec=time_limit_sec,
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
            [[0, 1], [0, 1], [0, 0]],
        ])
        waypoints_rpys_choices = [
            [RPY_FRONT_UP],
            [RPY_BACK_DOWN, RPY_LEFT_DOWN, RPY_RIGHT_DOWN],
        ]
        waypoints_scale = [.75, 3]
        max_dist = 6
        repeat = 3
        time_limit_sec = 6
        super().__init__(
            waypoints_xyzs=waypoints_xyzs,
            waypoints_rpys=waypoints_rpys,
            spawns=spawns,
            waypoints_normal_distr=waypoints_normal_distr,
            rpy_choices=waypoints_rpys_choices,
            waypoints_scale=waypoints_scale,
            max_dist=max_dist,
            repeat=repeat,
            time_limit_sec=time_limit_sec,
            difficulty="easy"
        )

class FrontFrontTemplate(WaypointTemplate):
    def __init__(self,):
        waypoints_xyzs = np.array([
            np.array([0, 0, 0]),
            np.array([.75, 0, 0]),
            np.array([1.5, 0, 0]),
        ])  
        waypoints_rpys = np.array([
            np.array([0, 0, 0]),
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
            [[0, 0], [0, 1], [0, 1]],
            [[0, 0], [0, 1], [0, 1]],
        ])
        waypoints_rpys_choices = [
            [RPY_FRONT_UP, RPY_BACK_UP, RPY_LEFT_UP, RPY_RIGHT_UP],
            [RPY_FRONT_UP, RPY_BACK_UP, RPY_LEFT_UP, RPY_RIGHT_UP],
            [RPY_FRONT_UP, RPY_BACK_UP, RPY_LEFT_UP, RPY_RIGHT_UP],
        ]
        waypoints_scale = [1, 2]
        max_dist = 5
        repeat = 3
        time_limit_sec = 6

        super().__init__(
            waypoints_xyzs=waypoints_xyzs,
            waypoints_rpys=waypoints_rpys,
            spawns=spawns,
            waypoints_normal_distr=waypoints_normal_distr,
            rpy_choices=waypoints_rpys_choices,
            waypoints_scale=waypoints_scale,
            max_dist=max_dist,
            repeat=repeat,
            time_limit_sec=time_limit_sec,
            difficulty="easy"
        )

class FrontBackTemplate(WaypointTemplate):
    def __init__(self,):
        waypoints_xyzs = np.array([
            np.array([0, 0, 0]),
            np.array([-0.75, 0, 0]),
            np.array([-1.5, 0, 0]),
        ])  
        waypoints_rpys = np.array([
            np.array([0, 0, 0]),
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
            [[0, 0], [0, 1], [0, 1]],
            [[0, 0], [0, 1], [0, 1]],
        ])
        waypoints_rpys_choices = [
            [RPY_FRONT_UP, RPY_BACK_UP, RPY_LEFT_UP, RPY_RIGHT_UP],
            [RPY_FRONT_UP, RPY_BACK_UP, RPY_LEFT_UP, RPY_RIGHT_UP],
            [RPY_FRONT_UP, RPY_BACK_UP, RPY_LEFT_UP, RPY_RIGHT_UP],
        ]
        waypoints_scale = [1, 2]
        max_dist = 5
        repeat = 3
        time_limit_sec = 6
        super().__init__(
            waypoints_xyzs=waypoints_xyzs,
            waypoints_rpys=waypoints_rpys,
            spawns=spawns,
            waypoints_normal_distr=waypoints_normal_distr,
            rpy_choices=waypoints_rpys_choices,
            waypoints_scale=waypoints_scale,
            max_dist=max_dist,
            repeat=repeat,
            time_limit_sec=time_limit_sec,
            difficulty="easy"
        )

class SideSideTemplate1(WaypointTemplate):
    def __init__(self,):
        waypoints_xyzs = np.array([
            np.array([0, 0, 0]),
            np.array([2, 0, 0]),
            np.array([4, 0, 0]),
        ])  
        waypoints_rpys = np.array([
            np.array([0, 0, 0]),
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
            [RPY_FRONT_RIGHT, RPY_FRONT_LEFT],
            [RPY_FRONT_RIGHT, RPY_FRONT_LEFT, RPY_BACK_RIGHT, RPY_BACK_LEFT, RPY_LEFT_FRONT, RPY_LEFT_BACK, RPY_RIGHT_FRONT, RPY_RIGHT_BACK],
            [RPY_FRONT_UP],
        ]
        waypoints_scale = [1, 1]
        max_dist = 5
        repeat = 3
        time_limit_sec = 6
        super().__init__(
            waypoints_xyzs=waypoints_xyzs,
            waypoints_rpys=waypoints_rpys,
            spawns=spawns,
            waypoints_normal_distr=waypoints_normal_distr,
            rpy_choices=waypoints_rpys_choices,
            waypoints_scale=waypoints_scale,
            max_dist=max_dist,
            repeat=repeat,
            time_limit_sec=time_limit_sec,
            difficulty="easy",
        )

class SideSideTemplate2(WaypointTemplate):
    def __init__(self,):
        waypoints_xyzs = np.array([
            np.array([0, 0, 0]),
            np.array([-2, 0, 0]),
            np.array([-4, 0, 0]),
        ])  
        waypoints_rpys = np.array([
            np.array([0, 0, 0]),
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
            [RPY_FRONT_RIGHT, RPY_FRONT_LEFT],
            [RPY_FRONT_RIGHT, RPY_FRONT_LEFT, RPY_BACK_RIGHT, RPY_BACK_LEFT, RPY_LEFT_FRONT, RPY_LEFT_BACK, RPY_RIGHT_FRONT, RPY_RIGHT_BACK],
            [RPY_BACK_UP],
        ]
        waypoints_scale = [1, 1]
        max_dist = 5
        repeat = 3
        time_limit_sec = 6
        super().__init__(
            waypoints_xyzs=waypoints_xyzs,
            waypoints_rpys=waypoints_rpys,
            spawns=spawns,
            waypoints_normal_distr=waypoints_normal_distr,
            rpy_choices=waypoints_rpys_choices,
            waypoints_scale=waypoints_scale,
            max_dist=max_dist,
            difficulty="easy",
            repeat=repeat,
            time_limit_sec=time_limit_sec,
        )