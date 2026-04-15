from transforms3d.euler import mat2euler, euler2quat, euler2mat
from copy import deepcopy
import numpy as np
from typing import List

from gym_pybullet_drones.control.CustomCTBRControl import CTBRPIDControl
from gym_pybullet_drones.gateRL.waypoints.WaypointTemplate import *
from gym_pybullet_drones.gateRL.waypoints.easy_templates import *
from gym_pybullet_drones.gateRL.waypoints.hard_templates import *
from gym_pybullet_drones.utils.enums import DroneModel

class EnvState1():
    def __init__(self, 
                 waypoints_templates:List[WaypointTemplate], 
                 train=True,
                 drone_model=DroneModel.CF2X,
                 ctrl_freq=200,
                 K=10,
                 dt=0.01,
                 low=-1,
                 high=1,
                 replay_p=0.3,
                 full_p = 0.2,
                 init_len_sec=10,
                 max_len_sec=10,
                 k_f = None, 
                 k_m = None,
                 T = None,
                 I = None,
                 param_distributions=None,
                 ):
        self.waypoints_templates = waypoints_templates
        self.len_sec = init_len_sec
        self.max_len_sec = max_len_sec
        self.replay = []
        self.replay_size = 100
        self.replay_p = replay_p
        self.full_p = full_p
        self.K = K
        self.dt = dt
        self.low = low
        self.high = high
        self.TRAIN = train
        self.TRAIN_PLANNING = False
        self.controller = CTBRPIDControl(drone_model=drone_model, ctrl_freq=ctrl_freq)

        # actual physical params for the drone
        self.k_f = k_f
        self.k_m = k_m
        self.T = T
        self.I = I
        self.param_distributions = param_distributions

    def set_K(self, K):
        self.K = K
    
    def set_dt(self, dt):
        self.dt = dt
    
    def set_low(self, low):
        self.low = low
    
    def set_high(self, high):
        self.high = high
    
    def save_config(self, config, total_reward, len_sec, initial_next_waypoints=None):
        if initial_next_waypoints is None and len(config) > 10:
            initial_next_waypoints = config[10]
        if len(self.replay) == 0:
            self.replay.append((config, total_reward, len_sec, initial_next_waypoints))
            return
        for idx, entry in enumerate(self.replay):
            if entry[1] < total_reward:
                self.replay.insert(idx, (config, total_reward, len_sec, initial_next_waypoints))
                if len(self.replay) > self.replay_size:
                    self.replay.pop(-1)
                break
        return

    def get_env_config(self):
        if self.TRAIN:
            p = np.random.rand()
            if p < self.replay_p and len(self.replay) > 0:
                config = self.replay[int(np.random.uniform(low=0.2, high=0.5)* len(self.replay))][0]
                self.replay.pop(-1)
            elif p < self.replay_p + self.full_p:
                idx = np.random.randint(len(self.waypoints_templates))
                config = self._get_env_config(self.waypoints_templates[idx], from_start=True)
            else:
                idx = np.random.randint(len(self.waypoints_templates))
                config = self._get_env_config(self.waypoints_templates[idx])
        else:
            idx = np.random.randint(len(self.waypoints_templates))
            config = self._get_env_config(self.waypoints_templates[idx])

        return tuple(config)

    def get_rpm(self, control_timestep, thrust, cur_body_rate, target_body_rate):
        return self.controller.compute_delayed_control(control_timestep, thrust, cur_body_rate, target_body_rate, self.T)

    def _get_env_config(self, template, from_start=False):
        wp_xyzs, wp_rpys, _, max_dist = template.sample()
        wp_quats = np.array([euler2quat(*rpy) for rpy in wp_rpys])
        n_waypoints = len(wp_xyzs)

        spawn_wp_idx = np.random.randint(n_waypoints) if not from_start else 0
        spawn_wp_pos = np.asarray(wp_xyzs[spawn_wp_idx], dtype=float)
        spawn_wp_rpy = np.asarray(wp_rpys[spawn_wp_idx], dtype=float)
        spawn_wp_rot = euler2mat(*spawn_wp_rpy)
        spawn_wp_forward = spawn_wp_rot[:, 0]

        spawn_speed = 0.1
        p = spawn_wp_pos
        v = spawn_speed * spawn_wp_forward
        a = np.zeros(3)
        p, v, a = self._expand_flat(p, v, a)
        rpy = self._rpy_from_pva(p, v, a)

        if spawn_wp_idx < n_waypoints - 1:
            initial_next_waypoints = (spawn_wp_idx, spawn_wp_idx + 1)
        else:
            initial_next_waypoints = (spawn_wp_idx,)

        len_sec = template.time_limit_sec if from_start else self.len_sec
        return (p, v, a, rpy, wp_xyzs, wp_rpys, wp_quats, max_dist, template.repeat, len_sec, initial_next_waypoints)
    
    def _expand_flat(self, p, v, a):
        for _ in range(self.K):
            j = np.random.uniform(low=self.low, high=self.high, size=(3,))
            a = a - j * self.dt
            v = v - a * self.dt - j * self.dt**2 / 2
            p = p - v * self.dt - a * self.dt**2 / 2 - j * self.dt**3 / 6
        return p, v, a

    def _rpy_from_pva(self, p, v, a):
        z = (a - np.array([0, 0, -9.81])) / (np.linalg.norm(a - np.array([0, 0, -9.81])))
        x = (v - np.dot(v, z) * z) / (np.linalg.norm(v - np.dot(v, z) * z))
        y = np.cross(z, x)
        R = np.vstack((x, y, z)).T
        return mat2euler(R)
    

# if __name__ == "__main__":
    
#     waypoints_templates = [OneBackTemplate(), BackSideZTemplate()]
#     print(waypoints_templates[0]())
#     env_state = EnvState(waypoints_templates=waypoints_templates, env_config_size=10, p_easy=1, K=100, dt=0.01, low=-50, high=50)
    # for _ in range(100):
    #     config = env_state.get_random_env_config()
    #     print("p:", config[0])
    #     print("v:", config[1])
    #     print("a:", config[2])
    #     print("rpy:", config[3])
    #     print("wp_xyzs:", config[4])
    #     print("wp_rpys:", config[5])
    #     print("wp_quats:", config[6])
    #     print("max_dist:", config[7])
    #     print("-----")
    #     input()