import numpy as np
from waypoints import interpolate_waypoints
from transforms3d.euler import euler2quat, quat2euler
class  Track():
    def __init__(self, tracks):
        self.tracks = tracks
        
        
    def reset(self, difficulty:str):
        track_idx = np.random.randint(0, len(self.tracks))
        self._set_track(difficulty, track_idx)
        self._set_spawn_point()
    
    def step(self):
        if self.waypoints.shape[0] > 0:
            self.waypoints = self.waypoints[1:]
            self.quats = self.quats[1:]
    
    def get_next_waypoints(self):
        if self.waypoints.shape[0] == 0:
            return np.array([[np.inf, np.inf, np.inf], [np.inf, np.inf, np.inf]]),\
                    np.array([[np.inf, np.inf, np.inf, np.inf],[np.inf, np.inf, np.inf, np.inf]])
        elif self.waypoints.shape[0] == 1:
            return np.array([self.waypoints[0], [np.inf, np.inf, np.inf]]),\
                    np.array([self.quats[0], [np.inf, np.inf, np.inf, np.inf]])
        else:
            return self.waypoints[:2], self.quats[:2]
            
    
    def _set_track(self, difficulty:str, track_idx:int):
        track = self.tracks[track_idx]
        if difficulty == "easy":
            num_points_per_segment = track.easy_settings["num_waypoints_per_segment"]
        elif difficulty == "medium":
            num_points_per_segment = track.medium_settings["num_waypoints_per_segment"]
        elif difficulty == "hard":
            num_points_per_segment = track.hard_settings["num_waypoints_per_segment"]
        else:
            raise ValueError("Invalid difficulty level. Choose from 'easy', 'medium', or 'hard'.")
        self.waypoints, self.quats = interpolate_waypoints(track.waypoints_xyz, track.waypoints_rpy, num_points_per_segment)
        
    def _set_spawn_point(self,):
        idx = np.random.randint(0, len(self.waypoints))
        self.spawn_point = self.waypoints[idx]
        self.spawn_quat = self.quats[idx]
        self.waypoints = self.waypoints[idx+1:]
        self.quats = self.quats[idx+1:]
        
        
        
if __name__ == "__main__":
    from track_settings.track1_setting import Track1     
    settings = [Track1()]
    track = Track(settings)
    track.reset("easy")
    print("Spawn point:", track.spawn_point)
    print("Spawn RPY:", track.spawn_quat)
    print("Waypoints shape:", track.waypoints.shape)
    print("Quaternions shape:", track.quats.shape)
    print(track.waypoints[0])
    print(track.quats[0])