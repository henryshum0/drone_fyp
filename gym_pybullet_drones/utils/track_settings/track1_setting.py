import numpy as np
from track_settings.track_settings import TrackSettings
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
                np.array([0, 0, np.pi/2]),
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

	




