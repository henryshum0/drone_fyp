from gym_pybullet_drones.envs.SensorEnv import SensorEnv
from gym_pybullet_drones.control.CustomCTBRControl import CTBRPIDControl
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType, ImageType, EnvStateType
from gym_pybullet_drones.sensors.camera import CameraSensor
from gym_pybullet_drones.sensori_agent.vins_frontend import VinsMonoFrontend, FrontendConfig
from gym_pybullet_drones.utils.constants import VEC_X, VEC_Z

from scipy.spatial.transform import Rotation as R
from copy import copy, deepcopy


import os
from typing import List, Optional, Dict, Any
import numpy as np
import pybullet as p
import gymnasium as gym
from gymnasium import spaces
from collections import deque

class SensoriAgentEnv(SensorEnv):
    def __init__(self, 
                 drone_model=DroneModel.RACE, 
                 physics=Physics.PYB_GND_DRAG_DW, 
                 pyb_freq=200,
                 ctrl_freq=200,
                 network_freq=100,
                 gui=False, 
                 frontend_config: Optional[FrontendConfig]=None,
                ):
        super().__init__(drone_model=drone_model,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         camera_enabled=True,
                         use_egl_renderer=True,
                         camera_fps=30,
                         camera_width=640,
                         camera_height=480,
                         camera_fov_deg=90,
                         )

        self.frontend_config = frontend_config if frontend_config is not None else FrontendConfig()
        self.vins_frontend: Optional[VinsMonoFrontend] = None
        self.frontend_last_output = None
        self._last_frontend_timestamp = 0.0

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._init_frontend()
        self.frontend_last_output = None
        self._last_frontend_timestamp = 0.0
        self._update_frontend()
        if info is None:
            info = {}
        info.update(self._frontend_info_dict())
        return obs, info

    def step(self, action):
        super().step(action)
        obs = self._computeObs()
        reward = self._computeReward()
        terminated = self._computeTerminated()
        truncated = self._computeTruncated()
        info = self._computeInfo()
        return 

    def _init_frontend(self):
        if self.camera is None:
            return
        K = np.array(
            [
                [self.camera.fx, 0.0, self.camera.cx],
                [0.0, self.camera.fy, self.camera.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((5,), dtype=np.float64)
        self.vins_frontend = VinsMonoFrontend(
            K=K,
            dist_coeffs=dist_coeffs,
            config=self.frontend_config,
        )

    def _random_camera_intrinsics(self):
        if self.camera is None:
            return
        fov = float(np.random.choice([60, 90, 120]))
        self.camera.set_intrinsics(
            fx=self.camera.width / (2 * np.tan(fov * np.pi / 360)),
            fy=self.camera.height / (2 * np.tan(fov * np.pi / 360)),
        )

        if self.vins_frontend is not None:
            K = np.array(
                [
                    [self.camera.fx, 0.0, self.camera.cx],
                    [0.0, self.camera.fy, self.camera.cy],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            self.vins_frontend.K = K

    def _update_frontend(self):
        rgb = self.camera.get_rgb()
        timestamp = float(self.step_counter / self.PYB_FREQ)
        self.frontend_last_output = self.vins_frontend.process(rgb, timestamp)
        self._last_frontend_timestamp = timestamp

    def _frontend_info_dict(self) -> Dict[str, Any]:
        if self.frontend_last_output is None:
            return {
                "frontend_ready": False,
                "frontend_timestamp": self._last_frontend_timestamp,
                "frontend_num_features": 0,
                "frontend_num_new": 0,
                "frontend_is_keyframe": False,
            }

        out = self.frontend_last_output
        return {
            "frontend_ready": True,
            "frontend_timestamp": out.timestamp,
            "frontend_num_features": int(len(out.feature_ids)),
            "frontend_num_new": int(out.num_new),
            "frontend_is_keyframe": bool(out.is_keyframe),
            "frontend_feature_ids": out.feature_ids.copy(),
            "frontend_points": out.points.copy(),
            "frontend_points_undistorted": out.undistorted_points.copy(),
            "frontend_track_lengths": out.track_lengths.copy(),
        }

    