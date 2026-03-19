from transforms3d.euler import mat2euler, euler2quat
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
                 history_p=0.3,
                 k_f = None, 
                 k_m = None,
                 T = None,
                 I = None,
                 param_distributions=None,
                 ):
        self.waypoints_templates = waypoints_templates
        self.history = []
        self.history_size = 100
        self.history_p = history_p
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
    
    def save_config(self, config, total_reward):
        if len(self.history) == 0:
            self.history.append((config, total_reward))
            return
        for idx, entry in enumerate(self.history):
            if entry[1] < total_reward:
                self.history.insert(idx, (config, total_reward))
                if len(self.history) > self.history_size:
                    self.history.pop(0)
                break
        return

    def get_env_config(self):
        if self.TRAIN:
            if np.random.rand() < self.history_p and len(self.history) > 0:
                config = self.history[-1][0]
                self.history.pop(-1)
            else:
                idx = np.random.randint(len(self.waypoints_templates))
                config = self._get_env_config(self.waypoints_templates[idx])
        else:
            idx = np.random.randint(len(self.waypoints_templates))
            config = self.waypoints_templates[idx]()
        return deepcopy(config)

    def get_rpm(self, control_timestep, thrust, cur_body_rate, target_body_rate):
        return self.controller.compute_delayed_control(control_timestep, thrust, cur_body_rate, target_body_rate, self.T)

    def _get_env_config(self, template):
        wp_xyzs, wp_rpys, _, max_dist = template.sample()
        wp_quats = np.array([euler2quat(*rpy) for rpy in wp_rpys])
        p = np.zeros(3)
        v = np.array([0.1, 0, 0])
        a = np.zeros(3)
        p, v, a = self._expand_flat(p, v, a)
        rpy = self._rpy_from_pva(p, v, a)
        return (p, v, a, rpy, wp_xyzs, wp_rpys, wp_quats, max_dist, template.repeat, template.time_limit_sec)
    
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