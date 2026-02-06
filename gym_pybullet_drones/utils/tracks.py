import numpy as np
from waypoints import interpolate_waypoints
from transforms3d.euler import euler2quat, quat2euler
class  Track():
    def __init__(self, track_settings):
        self.waypoints_xyz = track_settings.waypoints_xyz
        self.waypoints_rpy = track_settings.waypoints_rpy
        self.easy_settings = track_settings.easy_settings
        self.medium_settings = track_settings.medium_settings
        self.hard_settings = track_settings.hard_settings
        
        
    def reset(self, difficulty:str):
        self._set_track(difficulty)
        self._set_spawn_point()
    
    def step(self):
        pass
    
    def get_next_waypoints(self):
        pass
    
    def _set_track(self, difficulty:str):
        if difficulty == "easy":
            num_points_per_segment = self.easy_settings["num_waypoints_per_segment"]
        elif difficulty == "medium":
            num_points_per_segment = self.medium_settings["num_waypoints_per_segment"]
        elif difficulty == "hard":
            num_points_per_segment = self.hard_settings["num_waypoints_per_segment"]
        else:
            raise ValueError("Invalid difficulty level. Choose from 'easy', 'medium', or 'hard'.")
        self.waypoints, self.quats = interpolate_waypoints(self.waypoints_xyz, self.waypoints_rpy, num_points_per_segment)
        
    def _set_spawn_point(self,):
        idx = np.random.randint(0, len(self.waypoints))
        self.spawn_point = self.waypoints[idx]
        self.spawn_quat = self.quats[idx]
        self.waypoints = self.waypoints[idx+1:]
        self.quats = self.quats[idx+1:]
        
        
        
if __name__ == "__main__":
    from track_settings.track1_setting import Track1     
    settings = Track1()
    track = Track(settings)
    track.reset("easy")
    print("Spawn point:", track.spawn_point)
    print("Spawn RPY:", track.spawn_quat)
    print("Waypoints shape:", track.waypoints.shape)
    print("Quaternions shape:", track.quats.shape)
    print(track.waypoints[0])
    print(track.quats[0])