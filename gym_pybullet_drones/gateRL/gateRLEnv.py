from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.control.CustomCTBRControl import CTBRPIDControl
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType, ImageType
from gym_pybullet_drones.gateRL.envState import EnvState
from gym_pybullet_drones.gateRL.waypoints.WaypointTemplate import WaypointTemplate

from gym_pybullet_drones.examples.customPID_test import init_rate_tracking_data, append_rate_tracking_data, plot_rate_tracking

from transforms3d.quaternions import rotate_vector, qconjugate, mat2quat, qmult, quat2mat
from transforms3d.euler import euler2quat, quat2euler
from copy import deepcopy


import os
from typing import List
import numpy as np
import pybullet as p
import gymnasium as gym
from gymnasium import spaces
from collections import deque

class GateRLEnv(BaseAviary):
    
    def __init__(self,
                 waypoints:List[WaypointTemplate],
                 drone_model: DroneModel=DroneModel.CF2X,
                 neighbourhood_radius: float=np.inf,
                 physics: Physics=Physics.PYB_GND_DRAG_DW,
                 pyb_freq: int = 500,
                 ctrl_freq: int = 500,
                 network_freq: int = 100,
                 episode_len_sec = 20,
                 train=True,
                 p_easy=0.8,
                 K=10,
                 flat_low = -1,
                 flat_high = 1,
                 reward_weights=None,
                 use_reward_shaping=False,
                 gui=False,
                 record=False,
                 debug=False,
                 debug_pause=False,
                 ):
        """Initialization of a generic single and multi-agent RL environment.

        Attributes `vision_attributes` and `dynamics_attributes` are selected
        based on the choice of `obs` and `act`; `obstacles` is set to True 
        and overridden with landmarks for vision applications; 
        `user_debug_gui` is set to False for performance.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        num_drones : int, optional
            The desired number of drones in the aviary.
        neighbourhood_radius : float, optional
            Radius used to compute the drones' adjacency matrix, in meters.
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        pyb_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, waypoint or velocity with PID control; etc.)

        """
        self.OBS_TYPE = ObservationType.KIN
        self.ACT_TYPE = ActionType.RPM
        self.TRAIN = train

        #DEBUG
        self.DEBUG = debug
        self.DEBUG_PAUSE = debug_pause
        
        #### Create integrated controllers #########################
        self.NETWORK_FREQ = network_freq
        self.NETWORK_TIMESTEP = 1.0 / self.NETWORK_FREQ
        self.network_step_counter = 0
        self.PYB_PER_NETWORK = int(pyb_freq / network_freq)
        self.episode_len_sec = episode_len_sec

        # Drone states

        self.MAX_ROLL_RATE = 4 *np.pi
        self.MAX_PITCH_RATE = 4 * np.pi
        self.MAX_YAW_RATE = 4 * np.pi
        self.MAX_MASS_NORMALIZED_THRUST = 20
        self.ACTION_SCALE = np.array([self.MAX_MASS_NORMALIZED_THRUST,
                                      self.MAX_ROLL_RATE,
                                      self.MAX_PITCH_RATE,
                                      self.MAX_YAW_RATE])
        self.INIT_VELS = np.array([[0, 0, 0]])
        self.INIT_ACCS = np.array([[0, 0, 0]])

        # Env states
        self.CROSSED_WAYPOINT = False
        self.TIMEOUT = False
        self.OUT_OF_BOUND = False
        self.CRASHED = False
        self.ALL_WAYPOINTS_CROSSED = False
        self.next_waypoints = (0, 1)
        self.env_state_manager = EnvState(
            waypoints_templates=waypoints, 
            p_easy=p_easy,
            K=K,
            dt=self.NETWORK_TIMESTEP,
            low=flat_low,
            high=flat_high,
            train=train,
            ctrl_freq=ctrl_freq,
            drone_model=drone_model,
            T=0.075,
        )
        self.max_dist_from_next_wp = None
        self.waypoints_xyz:np.ndarray = None
        self.waypoints_rpy:np.ndarray = None
        self.waypoints_quats:np.ndarray = None
        self.predefined_spawns = None
        
        # reward function parameters
        self.ALLOWED_BOUNDS = .5
        self.A_perror =0.5
        self.B_perror =0.1
        self.A_theta_error = np.pi
        self.B_theta_error = np.pi/2
        self.C = 1.
        self.USE_REWARD_SHAPING = use_reward_shaping
        
        if reward_weights is None:
            self.w_r_aero = 100
            self.w_r_pa = 1
            self.w_r_theta_error = 2

            self.w_r_aero_shaped = 1
            self.w_r_pa_shaped = 1
            self.w_r_theta_error_shaped = 1

            self.w_r_act = -1
            self.w_r_act_change = -.5
            self.w_r_yaw = -1.5

            self.TIME_PENALTY = -0
            self.OUT_OF_BOUND_PENALTY = -1000
        else:
            self.w_r_aero = reward_weights['aero']
            self.w_r_pa = reward_weights['pa']
            self.w_r_theta_error = reward_weights['theta_error']

            self.w_r_aero_shaped = reward_weights['aero_shaped']
            self.w_r_pa_shaped = reward_weights['pa_shaped']
            self.w_r_theta_error_shaped = reward_weights['theta_error_shaped']

            self.w_r_act = reward_weights['act']
            self.w_r_act_change = reward_weights['act_change']
            self.w_r_yaw = reward_weights['yaw']

            self.TIME_PENALTY = reward_weights['time_penalty']
            self.OUT_OF_BOUND_PENALTY = reward_weights['out_of_bound_penalty']

        super().__init__(drone_model=drone_model,
                         num_drones=1,
                         neighbourhood_radius=neighbourhood_radius,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record, 
                         obstacles=True, # visualize waypoints
                         user_debug_gui=gui, # drawing drone axis
                         vision_attributes=False,
                         compute_returns_per_step=False,
                         ground_plane=False,
                         )
        

    def step(self, action):
        self.action_prev = self.action.copy()
        self.action = action.copy()
        # network is at lower frequency than pyb, aggrewaypoint steps
        for _ in range(self.PYB_PER_NETWORK): 
            super().step(action.copy())    
        obs = self._computeObs()
        terminated = self._computeTerminated()
        truncated = self._computeTruncated()
        reward = self._computeReward()
        info = self._computeInfo()

        self._print_debug()

        if self.CROSSED_WAYPOINT:
            self.CROSSED_WAYPOINT = False
        self.network_step_counter += 1

        return obs, reward, terminated, truncated, info

    def set_episode_len(self, new_len):
        """Set the episode length in seconds."""
        self.episode_len_sec = new_len

    def set_K(self, K):
        self.env_state_manager.set_K(K)

    def set_hard_template_percentage(self, hard_pct):
        self.env_state_manager.set_hard_template_percentage(hard_pct)

    def reset(self, seed=None, options=None):
        self._update_env_config()
        return super().reset(seed=seed, options=options)

    def _update_env_config(self):
        config = self.env_state_manager.get_env_config()
        self.INIT_XYZS[0] = config[0]
        self.INIT_RPYS[0] = config[3]
        self.current_waypoint_idx = 0
        self.waypoints_xyz = config[4]
        self.waypoints_rpy = config[5]
        self.waypoints_quats = config[6]
        self.max_dist_from_next_wp = config[7]
        if len(self.waypoints_xyz) > 1:
            self.next_waypoints = (0, 1)
        else:
            self.next_waypoints = (0, -1)

    def _housekeeping(self):
        self.action_prev = np.zeros(4).astype(np.float32)
        self.action = np.zeros(4).astype(np.float32)
        self.prev_obs = np.zeros(25).astype(np.float32)
        self.obs = np.zeros(25).astype(np.float32)
        self.CROSSED_WAYPOINT = False
        self.TIMEOUT = False
        self.OUT_OF_BOUND = False
        self.CRASHED = False
        self.ALL_WAYPOINTS_CROSSED = False
        self.network_step_counter = 0
        super()._housekeeping()
        self._updateAndStoreKinematicInformation()
        if self.waypoints_xyz is not None and self.waypoints_quats is not None:
            self.drone_state = self._getDroneStateVector(0)
            self.drone_state_prev = self._getDroneStateVector(0)
            self.p_a = rotate_vector(self.drone_state[0:3] - self.waypoints_xyz[self.current_waypoint_idx], qconjugate(self.waypoints_quats[self.current_waypoint_idx]))
            self.p_a_prev = self.p_a.copy()
            self.p_error = np.linalg.norm(self.p_a)
            self.p_error_prev = self.p_error.copy()
            self.theta_error = self._compute_theta_error(self.drone_state[3:7][[3, 0, 1, 2]], self.waypoints_quats[self.current_waypoint_idx])
            self.theta_error_prev = self.theta_error
        
    def _addObstacles(self): # visualize waypoints
        if self.USER_DEBUG and self.waypoints_quats is not None and self.waypoints_xyz is not None:
            waypoint_xyz = self.waypoints_xyz
            waypoint_quats = self.waypoints_quats


            def draw_waypoint_with_x_arrow(pos, quat_wxyz, length=0.3, head_len=0.08, head_angle_deg=25.0):
                R = quat2mat(quat_wxyz)
                x_dir = R[:, 0]
                y_dir = R[:, 1]
                z_dir = R[:, 2]

                # Main axes
                tip = pos + length * x_dir
                p.addUserDebugLine(pos, tip, [1, 0, 0], lineWidth=2, physicsClientId=self.CLIENT)  
                z_tip = pos + length * z_dir
                p.addUserDebugLine(pos, z_tip, [0, 0, 1], lineWidth=2, physicsClientId=self.CLIENT)

                # Arrowhead for X: two lines forming a "V" in the plane spanned by x/y
                a = np.deg2rad(head_angle_deg)

                # directions pointing backward relative to +X, rotated toward ±Y
                d1 = (-np.cos(a) * x_dir + np.sin(a) * y_dir)
                d2 = (-np.cos(a) * x_dir - np.sin(a) * y_dir)

                p.addUserDebugLine(tip, tip + head_len * d1, [1, 0, 0], lineWidth=2, physicsClientId=self.CLIENT)
                p.addUserDebugLine(tip, tip + head_len * d2, [1, 0, 0], lineWidth=2, physicsClientId=self.CLIENT)

                # Optional: add a 3rd head line using Z for better 3D readability
                d3 = (-np.cos(a) * x_dir + np.sin(a) * z_dir)
                d4 = (-np.cos(a) * x_dir - np.sin(a) * z_dir)
                p.addUserDebugLine(tip, tip + head_len * d3, [1, 0, 0], lineWidth=2, physicsClientId=self.CLIENT)
                p.addUserDebugLine(tip, tip + head_len * d4, [1, 0, 0], lineWidth=2, physicsClientId=self.CLIENT)
            
            for pos, quat_wxyz in zip(waypoint_xyz, waypoint_quats):
                draw_waypoint_with_x_arrow(pos, quat_wxyz)
        
    
    def _actionSpace(self):
        act_lower_bound = np.array([-1, -1, -1, -1])
        act_upper_bound = np.array([1, 1, 1, 1])
        return spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)
    
    def _preprocessAction(self, action):
        action = (action + 1) / 2 * self.ACTION_SCALE # scale action from [-1,1] to actual values
        state = self._getDroneStateVector(0)
        drone_quat_xyzw = state[3:7].copy()
        drone_quat_wxyz = drone_quat_xyzw[[3, 0, 1, 2]]
        angular_vel_world = state[13:16].copy()
        cur_body_rate = rotate_vector(angular_vel_world, qconjugate(drone_quat_wxyz))

        rpm = self.env_state_manager.get_rpm(
            control_timestep=self.NETWORK_TIMESTEP,
            thrust=action[0],
            target_body_rate=action[1:4],
            cur_body_rate=cur_body_rate,
        )
        rpm = np.reshape(rpm, (1, 4))
        self.rpm = rpm
        return rpm
    
    def _observationSpace(self):
        lo = -np.inf
        hi = np.inf
        obs_env_lower_bound = np.array([lo, lo, lo, -1 , -1, -1, -1, lo, lo, lo, -1, -1, -1, -1])
        obs_env_upper_bound = np.array([hi, hi, hi, 1, 1, 1, 1, hi, hi, hi, 1, 1, 1, 1])
        obs_ego_lower_bound = np.array([-1, -1, -1, -1, lo, lo, lo, -1, -1, -1, -1] )
        obs_ego_upper_bound = np.array([ 1, 1, 1, 1, hi, hi, hi, 1, 1, 1, 1])

        obs_lower = np.hstack([obs_env_lower_bound, obs_ego_lower_bound])
        obs_upper = np.hstack([obs_env_upper_bound, obs_ego_upper_bound])
        return spaces.Box(low=obs_lower, high=obs_upper, dtype=np.float32)

    def _computeObs(self):
        self.prev_obs = self.obs.copy()
        obs = np.zeros(25).astype(np.float32)
        self.drone_state = self._getDroneStateVector(0)

        drone_pos = self.drone_state[0:3].copy()
        drone_quat = self.drone_state[3:7].copy()
        drove_vel = self.drone_state[10:13].copy()
        drone_ori_wxyz = drone_quat[[3, 0, 1, 2]] # xyzw to wxyz
        drone_vel_b = rotate_vector(drove_vel, qconjugate(drone_ori_wxyz)).astype(np.float32)
        drone_last_action = self.action_prev.copy()

        # calculate position of drone in waypoint frame
        waypoint_pos = self.waypoints_xyz[self.current_waypoint_idx]
        waypoint_quat = self.waypoints_quats[self.current_waypoint_idx]
        self.p_a_prev = self.p_a.copy()
        self.p_a = rotate_vector(drone_pos - waypoint_pos, qconjugate(waypoint_quat))  

        # compute if crossed waypoint
        if (self.p_a[0] > 0 and self.p_a_prev[0] <= 0 and np.abs(self.p_a[1]) < self.ALLOWED_BOUNDS and np.abs(self.p_a[2]) < self.ALLOWED_BOUNDS):
            self.CROSSED_WAYPOINT = True

            # store to calculate reward upon crossing waypoint
            self.p_error_reward = np.linalg.norm(self.p_a)
            self.theta_error_reward = self._compute_theta_error(drone_ori_wxyz, waypoint_quat)

            self._update_next_waypoints()
            waypoint_pos = self.waypoints_xyz[self.current_waypoint_idx]
            waypoint_quat = self.waypoints_quats[self.current_waypoint_idx] 
            self.p_a = rotate_vector(drone_pos - waypoint_pos, qconjugate(waypoint_quat))
            self.p_a_prev = self.p_a.copy()

            # when just crossed the waypoint, shaped reward for p_error and theta_error should be 0
            self.p_error_prev = np.linalg.norm(self.p_a)
            self.p_error = self.p_error_prev.copy()
            self.theta_error_prev = self._compute_theta_error(drone_ori_wxyz, waypoint_quat)
            self.theta_error = self.theta_error_prev.copy()
        else:

            # compute p_error
            self.p_error_prev = self.p_error.copy()
            self.p_error = np.linalg.norm(self.p_a)
            # compute z-axis alignment error
            self.theta_error_prev = self.theta_error.copy()
            self.theta_error = self._compute_theta_error(drone_ori_wxyz, waypoint_quat)

        waypoint_1_pos_rel = self.waypoints_xyz[self.next_waypoints[0]] - drone_pos
        waypoint_2_pos_rel = self.waypoints_xyz[self.next_waypoints[1]] - drone_pos
        waypoint_1_quat = self.waypoints_quats[self.next_waypoints[0]]
        waypoint_2_quat = self.waypoints_quats[self.next_waypoints[1]]

        obs [0:3] = rotate_vector(waypoint_1_pos_rel, qconjugate(drone_ori_wxyz))
        obs [3:7] = qmult(qconjugate(drone_ori_wxyz), waypoint_1_quat)
        obs[7:10] = rotate_vector(waypoint_2_pos_rel, qconjugate(drone_ori_wxyz))
        obs[10:14] = qmult(qconjugate(drone_ori_wxyz), waypoint_2_quat)
        
        obs[14:18] = drone_ori_wxyz
        obs[18:21] = drone_vel_b
        obs[21:25] = drone_last_action
        self.acceleration = (self.drone_state[10:13] - self.drone_state_prev[10:13]) / self.NETWORK_TIMESTEP

        # Guard against any non-finite values entering the policy network.
        if not np.isfinite(obs).all():
            obs = np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)
        if not np.isfinite(self.acceleration).all():
            self.acceleration = np.nan_to_num(self.acceleration, nan=0.0, posinf=0.0, neginf=0.0)

        self.obs = obs.astype(np.float32).copy()
        self.drone_state_prev = self.drone_state.copy()

        return obs
    
    def _compute_theta_error(self, drone_quat, waypoint_quat):
        R_drone = quat2mat(drone_quat)
        R_waypoint = quat2mat(waypoint_quat)
        z_waypoint = R_waypoint[:, 2]
        z_drone = R_drone[:, 2]
        cos_theta = np.clip(np.dot(z_waypoint, z_drone), -1.0, 1.0)
        theta_error = np.arccos(cos_theta)
        return theta_error
    
    def _update_next_waypoints(self):
        if self.CROSSED_WAYPOINT:
            if self.next_waypoints[1] < self.waypoints_xyz.shape[0] - 1:
                next_waypoints = (self.next_waypoints[1], self.next_waypoints[1] + 1)
            else:
                next_waypoints = (self.next_waypoints[1], -1)
            self.next_waypoints = next_waypoints
        self.current_waypoint_idx = self.next_waypoints[0]
       
    def _computeTerminated(self):
        waypoint_pos_rel = self.obs[0:3]
        if np.linalg.norm(waypoint_pos_rel) > self.max_dist_from_next_wp:
            self.OUT_OF_BOUND = True
            return True
    
        # Fix: Convert network_step_counter to seconds before comparing
        elapsed_time = self.network_step_counter * self.NETWORK_TIMESTEP
        if elapsed_time >= self.episode_len_sec:
            self.TIMEOUT = True
            return True

        if self.current_waypoint_idx == -1:
            self.ALL_WAYPOINTS_CROSSED = True
            return True
            
        return False
    
    def _computeTruncated(self):
        return False
    
    def _computeReward(self):
        def activation(x, A, B):
            C1 = x > A - B
            C2 = x <= A - B
            output = (A - x) * C1 + (B - 1 + np.power(10, A - x -B)) * C2
            return output

        action = self.action.copy()
        action_prev = self.action_prev.copy()
        
        # compute sparse reward at waypoint crossing
        if self.CROSSED_WAYPOINT:
            r_pa = activation(self.p_error_reward, self.A_perror, self.B_perror) / activation(0, self.A_perror, self.B_perror) # normalize to [0, 1]
            r_theta_error = activation(self.theta_error_reward, self.A_theta_error, self.B_theta_error) / activation(0, self.A_theta_error, self.B_theta_error) # normalize to [0, 1]
            r_aero = r_pa + r_theta_error + self.C
        else:
            r_pa = 0
            r_theta_error = 0
            r_aero = 0

        if self.USE_REWARD_SHAPING:
        # reward shaping
            r_pa_shaped = self.w_r_pa_shaped * (self.p_error_prev - self.p_error)
            r_theta_error_shaped = self.w_r_theta_error_shaped * (self.theta_error_prev - self.theta_error)
            r_aero_shaped = r_pa_shaped + r_theta_error_shaped
        else:
            r_pa_shaped = 0
            r_theta_error_shaped = 0
            r_aero_shaped = 0

        # calculate r_act and r_act_change
        r_act = np.linalg.norm(action[1:], ord=1) / 3
        r_act_change = np.linalg.norm(action - action_prev, ord=2) / np.sqrt(16)

        # calculate r_yaw
        drone_vel = self.drone_state[10:13].copy()
        drone_quat = self.drone_state[3:7].copy()
        drone_quat_wxyz = drone_quat[[3, 0, 1, 2]]
        v_b = rotate_vector(drone_vel, qconjugate(drone_quat_wxyz))
        v_b[2] = 0.0
        speed = np.linalg.norm(v_b)
        if speed < 1e-6:
            r_yaw = 0
        else:
            v_dir = v_b / speed
            cos_yaw = np.dot(v_dir, np.array([1.0, 0.0, 0.0], dtype=np.float32))
            cos_yaw = np.clip(cos_yaw, -1.0, 1.0)
            yaw_diff = np.arccos(cos_yaw)
            r_yaw = yaw_diff/np.pi # normalize to [0, 1]
        
        self.r_pa = r_pa 
        self.r_theta_error = r_theta_error 
        self.r_aero = r_aero
        self.r_pa_shaped = r_pa_shaped
        self.r_theta_error_shaped = r_theta_error_shaped
        self.r_aero_shaped = r_aero_shaped
        self.r_act =  r_act
        self.r_act_change = r_act_change
        self.r_yaw =  r_yaw
        
        self.reward = self.w_r_aero * self.r_aero + self.w_r_act * self.r_act + self.w_r_act_change * self.r_act_change + self.w_r_yaw * self.r_yaw + self.w_r_aero_shaped * self.r_aero_shaped
        if self.OUT_OF_BOUND:
            self.reward = self.reward + self.OUT_OF_BOUND_PENALTY
        self.reward = self.reward + self.TIME_PENALTY

        return self.reward.copy()
        
    def _computeInfo(self):
        if self.CROSSED_WAYPOINT:
            passed_waypoint = True
        else:
            passed_waypoint = False

        info = {
            "passed_waypoint": passed_waypoint,
            "next_waypoints": deepcopy(self.next_waypoints),
            "pos": self._getDroneStateVector(0)[0:3].copy(),
            "vel": self._getDroneStateVector(0)[10:13].copy(),
            "acc": self.acceleration.copy(),
            "curr_waypoint_idx": self.current_waypoint_idx,
            "timeout": self.TIMEOUT,
            "out_of_bound": self.OUT_OF_BOUND,
            "crashed": self.CRASHED,
            "all_waypoints_crossed": self.ALL_WAYPOINTS_CROSSED,
        }
        return info


    def _print_debug(self):
        if not self.DEBUG:
            return
        print("========= Step {} =========".format(self.network_step_counter))
        print("Episode Length:", self.network_step_counter * self.NETWORK_TIMESTEP, f" ({self.episode_len_sec})")
        print(f"obs: {self.obs}")
        print(f"action: {self.action}")
        print(f"velocity: {self._getDroneStateVector(0)[10:13]}")
        print(f"acceleration: {self.acceleration}")
        print(f"terminated: {self._computeTerminated()}")
        print(f"truncated: {self._computeTruncated()}")
        print(f"info: {self._computeInfo()}")
        print(f"p_a: {self.p_a}")
        print(f"p_a_prev: {self.p_a_prev}")
        print(f"p_error: {self.p_error}")
        print(f"p_error_prev: {self.p_error_prev}")
        print(f"theta_error: {self.theta_error}")
        print(f"theta_error_prev: {self.theta_error_prev}")
        print("================================")
        print("reward:", self.reward)
        print("r_pa:", self.r_pa)
        print("r_theta_error:", self.r_theta_error)
        print("r_aero:", self.r_aero, "weighted:", self.w_r_aero * self.r_aero)
        print("r_pa_shaped:", self.r_pa_shaped)
        print("r_theta_error_shaped:", self.r_theta_error_shaped)
        print("r_aero_shaped:", self.r_aero_shaped, "weighted:", self.w_r_aero_shaped * self.r_aero_shaped)
        print("r_act:", self.r_act, "weighted:", self.w_r_act * self.r_act)
        print("r_act_change:", self.r_act_change, "weighted:", self.w_r_act_change * self.r_act_change)
        print("r_yaw:", self.r_yaw, "weighted:", self.w_r_yaw * self.r_yaw)
        print("================================")
        print("spawn:")
        print("xyz:", self.INIT_XYZS[0])
        print("rpy:", self.INIT_RPYS[0])
        print("================================")
        input("Press Enter to continue...") if self.DEBUG_PAUSE else None


if __name__ == "__main__":
    from gym_pybullet_drones.gateRL.waypoints.easy_templates import EasyTemplate1
    env = GateRLEnv(waypoints=[EasyTemplate1()],p_easy=1, gui=True, debug=True, debug_pause=False, K=120, flat_low=-20, flat_high=20)
    obs = env.reset()
    done = False
    n_done=0
    while n_done < 1000:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if done:
            n_done += 1
            obs = env.reset()
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            input()