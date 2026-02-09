import numpy as np

class TrackSettings():
    def __init__(self, waypoints_xyz, waypoints_rpy, easy_settings, medium_settings, hard_settings):
        self.waypoints_xyz = waypoints_xyz
        self.waypoints_rpy = waypoints_rpy
        self.easy_settings = easy_settings
        self.medium_settings = medium_settings
        self.hard_settings = hard_settings
        
        
class Track1(TrackSettings):
    def __init__(self):
        super().__init__(
            waypoints_xyz = [
                np.array([2, 0, 3]),
                np.array([1, 0, 3]),
                np.array([0, 1, 3]),
                np.array([-1, 0, 3]),
                np.array([0, -1, 3]),
                np.array([1, 0, 3]),
            ],
            waypoints_rpy = [
                np.array([0, 0, np.pi]),
                np.array([0, 0, np.pi/4 * 3]),
                np.array([0, 0, np.pi]),
                np.array([0, 0, -np.pi/2]),
                np.array([0, 0, 0]),
                np.array([0, 0, np.pi/2]),
            ],
            easy_settings = {
                "num_waypoints_per_segment": 10,
                "max_dist_from_next_waypoint": 0.5,
            },
            medium_settings = {
                "num_waypoints_per_segment": 5,
                "max_dist_from_next_waypoint": 1,
            },
            hard_settings = {
                "num_waypoints_per_segment": 2,
                "max_dist_from_next_waypoint": 2.,
            },
        )
