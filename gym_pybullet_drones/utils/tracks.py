import numpy as np
from gym_pybullet_drones.utils.waypoints import interpolate_waypoints
from transforms3d.euler import euler2quat, quat2euler
class  Track():
    def __init__(self, tracks):
        self.tracks = tracks
        
        
    def reset(self, difficulty:str):
        track_idx = np.random.randint(0, len(self.tracks))
        self._set_track(difficulty, track_idx)
        self._set_spawn_point()
    
    def step(self):
        if self.waypoints_xyz.shape[0] > 0:
            self.waypoints_xyz = self.waypoints_xyz[1:]
            self.waypoint_quats = self.waypoint_quats[1:]
            if self.waypoints_rpy.shape[0] == 0:
                return False
            else:
                return True
        else:
            return False
                
    def get_next_waypoints(self):
        if self.waypoints_xyz.shape[0] == 0:
            return np.array([[np.inf, np.inf, np.inf], [np.inf, np.inf, np.inf]]),\
                    np.array([[np.inf, np.inf, np.inf, np.inf],[np.inf, np.inf, np.inf, np.inf]]),\
                    np.array([[np.inf, np.inf, np.inf], [np.inf, np.inf, np.inf]])
        elif self.waypoints_xyz.shape[0] == 1:
            return np.array([self.waypoints_xyz[0], [np.inf, np.inf, np.inf]]),\
                    np.array([self.waypoint_quats[0], [np.inf, np.inf, np.inf, np.inf]]),\
                    np.array([self.waypoints_rpy[0], [np.inf, np.inf, np.inf]])
        else:
            return self.waypoints_xyz[:2], self.waypoint_quats[:2], self.waypoints_rpy[:2]
        
    def get_spawn_point(self):
        return self.spawn_point, self.spawn_quat, self.spawn_rpy
            
    def get_waypoints_xyz(self):
        return self.waypoints_xyz
    def get_waypoints_quats(self): 
        return self.waypoint_quats
    def get_waypoints_rpy(self):
        return self.waypoints_rpy
    
    def _set_track(self, difficulty:str, track_idx:int):
        track = self.tracks[track_idx]
        if difficulty == "easy":
            self.num_points_per_segment = track.easy_settings["num_waypoints_per_segment"]
        elif difficulty == "medium":
            self.num_points_per_segment = track.medium_settings["num_waypoints_per_segment"]
        elif difficulty == "hard":
            self.num_points_per_segment = track.hard_settings["num_waypoints_per_segment"]
        else:
            raise ValueError("Invalid difficulty level. Choose from 'easy', 'medium', or 'hard'.")
        self.waypoints_xyz, self.waypoint_quats = interpolate_waypoints(track.waypoints_xyz, track.waypoints_rpy, self.num_points_per_segment)
        self.waypoints_rpy = np.array([quat2euler(quat) for quat in self.waypoint_quats])
        self.full_xyz = self.waypoints_xyz
        self.full_quats = self.waypoint_quats
        self.full_rpy = self.waypoints_rpy
        
    def _set_spawn_point(self,):
        idx = np.random.randint(0, len(self.waypoints_xyz) - self.num_points_per_segment)
        self.spawn_point = self.waypoints_xyz[idx]
        self.spawn_quat = self.waypoint_quats[idx]
        self.spawn_rpy = self.waypoints_rpy[idx]
        self.waypoints_xyz = self.waypoints_xyz[idx+1:]
        self.waypoint_quats = self.waypoint_quats[idx+1:]
        self.waypoints_rpy = self.waypoints_rpy[idx+1:]
        
        
        
if __name__ == "__main__":
    from track_settings.track1_setting import Track1     
    settings = [Track1()]
    track = Track(settings)
    track.reset("easy")
    xyz = track.full_xyz
    rpy = track.full_rpy
    for pos, rots in zip(xyz, rpy):
        print("Pos:", pos, "\tRpy:", rots)
    print(track.step())
