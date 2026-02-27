from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.control.CustomCTBRControl import CTBRPIDControl
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType, ImageType
from gym_pybullet_drones import asset_directory
from gym_pybullet_drones.gateRL.interpolate import interpolate_waypoints
from gym_pybullet_drones.gateRL.procedualLearning import ProcedualLearning

from transforms3d.quaternions import rotate_vector, qconjugate, mat2quat, qmult, quat2mat
from transforms3d.euler import euler2quat, quat2euler
from copy import deepcopy


import os
import numpy as np
import pybullet as p
import gymnasium as gym
from gymnasium import spaces
from collections import deque

class GateRLEnv(BaseAviary):
    
    def __init__(self,
                 waypoints:dict,
                 procedual_learning: ProcedualLearning = None,
                 drone_model: DroneModel=DroneModel.CF2X,
                 neighbourhood_radius: float=np.inf,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 500,
                 ctrl_freq: int = 500,
                 network_freq: int = 100,
                 episode_len_sec = 20,
                 train=True,
                 p_adap=0.5,
                 p_buff=0.4,
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
        self.procedual_learning = procedual_learning

        #DEBUG
        self.DEBUG = debug
        self.DEBUG_PAUSE = debug_pause
        
        #### Create integrated controllers #########################
        self.ctrl = CTBRPIDControl(drone_model=drone_model,
                                  ctrl_freq=ctrl_freq,
                                  )
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
        self.MAX_DIST_FROM_WAYPOINT = waypoints["max_dist"]
        self.MIN_Z = 1.5
        self.MAX_Z  = 4
        self.MAX_Y = 2
        self.MIN_Y = -2
        self.MAX_X = 3
        self.MIN_X = -2
        self.crossed_waypoint = False
        self.timeout = False
        self.out_of_bound = False
        self.crashed = False
        self.no_waypoints_remain = False
        self.next_waypoints = (0, 1)
        self.waypoints_xyz:np.ndarray = waypoints["pos"]
        self.waypoints_rpy:np.ndarray = waypoints["rpy"]
        self.waypoints_quats:np.ndarray = np.array([euler2quat(*rpy) for rpy in self.waypoints_rpy])
        self.predefined_spawns = waypoints["spawn"]
        self.P_ADAP = p_adap
        self.P_BUFF = p_buff
        assert self.P_ADAP + self.P_BUFF <= 1.0, "Probabilities for adaptive buffer and experience buffer should sum to less than or equal to 1.0"


        # spaces
        self.prev_obs = np.zeros((1, 28)) # obs is (1,24) for compatibility with parent class
        self.obs = np.zeros((1, 28))
        self.action_prev = np.zeros((1,4))
        self.action = np.zeros((1,4))
        
        # reward function parameters
        self.A_perror =0.4
        self.B_perror =0.1
        self.A_theta_error = np.pi / 2
        self.B_theta_error = 0.
        self.A_vel_dir = np.pi
        self.B_vel_dir = np.pi * 3/4
        self.C = 1.
        self.w_0 = 100
        self.w_1 = -0.1
        self.w_2 = -0.2
        self.w_3 = -0.1
        self.w_4 = 0.6
        self.ALLOWED_BOUNDS = .5
        self.PER_STEP_REWARD = 0.5

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

        self.action = action
        # network is at lower frequency than pyb, aggrewaypoint steps
        for _ in range(self.PYB_PER_NETWORK): 
            super().step(action.copy())    
        obs = self._computeObs()
        self._computeCrossedWaypoint()
        reward = self._computeReward()
        terminated = self._computeTerminated()
        truncated = self._computeTruncated()
        info = self._computeInfo()

        self._print_debug()

        if self.crossed_waypoint:
            self._update_next_waypoints()
            self.crossed_waypoint = False
        self.network_step_counter += 1
        self.action_prev = action
        return obs, reward, terminated, truncated, info

    def set_episode_len(self, new_len):
        """Set the episode length in seconds."""
        self.episode_len_sec = new_len

    def _housekeeping(self):
        if self.procedual_learning is not None and self.TRAIN:
            spawn = self.procedual_learning.sample_spawn(self.network_step_counter, training=self.TRAIN)
        else:
            spawn = self.predefined_spawns[0]
        self._set_spawn(spawn)
        self.action_prev = np.zeros((1,4)).astype(np.float32)
        self.action = np.zeros((1,4)).astype(np.float32)
        self.prev_obs = np.zeros((1, 28)).astype(np.float32)
        self.obs = np.zeros((1, 28)).astype(np.float32)
        self.p_a = np.zeros(3).astype(np.float32)
        self.p_a_prev = np.zeros(3).astype(np.float32)
        self.drone_state = np.zeros(20).astype(np.float32)
        self.drone_state_prev = np.zeros(20).astype(np.float32)
        self.crossed_waypoint = False
        self.crossed_waypoint = False
        self.timeout = False
        self.out_of_bound = False
        self.crashed = False
        self.no_waypoints_remain = False
        self.network_step_counter = 0
        self.curr_waypoint_idx = -1
        super()._housekeeping()
        
    def _addObstacles(self): # visualize waypoints
        if self.USER_DEBUG:
            waypoint_xyz = self.waypoints_xyz
            waypoint_quats = self.waypoints_quats


            def draw_waypoint_with_x_arrow(pos, quat_wxyz, length=0.3, head_len=0.08, head_angle_deg=25.0):
                R = quat2mat(quat_wxyz)
                x_dir = R[:, 0]
                y_dir = R[:, 1]
                z_dir = R[:, 2]

                # Main axes
                tip = pos + length * x_dir
                p.addUserDebugLine(pos, tip, [1, 0, 0], lineWidth=2, physicsClientId=self.CLIENT)                 # X (main)

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
        act_lower_bound = np.array([[-1, -1, -1, -1] for i in range(self.NUM_DRONES)])
        act_upper_bound = np.array([[1, 1, 1, 1] for i in range(self.NUM_DRONES)])
        return spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)
    
    def _preprocessAction(self, action):
        action = (action + 1) / 2 * self.ACTION_SCALE # scale action from [-1,1] to actual values
        cur_body_rate = rotate_vector(self._getDroneStateVector(0)[13:16], self.obs[0, 17:21])
        rpm = self.ctrl.computeControl(
            control_timestep=self.CTRL_TIMESTEP,
            thrust=action[0, 0],
            target_body_rate=action[0, 1:4],
            cur_body_rate=cur_body_rate,
        )
        rpm = np.reshape(rpm, (1, 4))
        self.rpm = rpm
        return rpm
    
    def _observationSpace(self):
        lo = -np.inf
        hi = np.inf
        obs_env_lower_bound = np.array([[lo, lo, lo, -1 , -1, -1, -1, lo, lo, lo, -1, -1, -1, -1] for i in range(self.NUM_DRONES)])
        obs_env_upper_bound = np.array([[hi, hi, hi, 1, 1, 1, 1, hi, hi, hi, 1, 1, 1, 1] for i in range(self.NUM_DRONES)])
        obs_ego_lower_bound = np.array([[lo, lo, 0, -1, -1, -1, -1, lo, lo, lo, -1, -1, -1, -1] for i in range(self.NUM_DRONES)])
        obs_ego_upper_bound = np.array([[hi, hi, hi, 1, 1, 1, 1, hi, hi, hi, 1, 1, 1, 1] for i in range(self.NUM_DRONES)])

        obs_lower = np.hstack([obs_env_lower_bound, obs_ego_lower_bound])
        obs_upper = np.hstack([obs_env_upper_bound, obs_ego_upper_bound])
        return spaces.Box(low=obs_lower, high=obs_upper, dtype=np.float32)

    def _computeObs(self):
        self.prev_obs = self.obs.copy()
        obs = np.zeros((self.NUM_DRONES, 28)).astype(np.float32)
        self.drone_state = self._getDroneStateVector(0)
        drone_pos = self.drone_state[0:3]
        drone_quat = self.drone_state[3:7]
        drone_ori_wxyz= drone_quat[[3, 0, 1, 2]] # xyzw to wxyz
        drone_vel_b = rotate_vector(self.drone_state[10:13], qconjugate(drone_ori_wxyz)).astype(np.float32)
        drone_last_action = self.action_prev[0]

        waypoint_1_pos_rel = self.waypoints_xyz[self.next_waypoints[0]] - drone_pos
        waypoint_2_pos_rel = self.waypoints_xyz[self.next_waypoints[1]] - drone_pos
        waypoint_1_quat = self.waypoints_quats[self.next_waypoints[0]]
        waypoint_2_quat = self.waypoints_quats[self.next_waypoints[1]]

        obs [0, 0:3] = rotate_vector(waypoint_1_pos_rel, qconjugate(drone_ori_wxyz))
        obs [0, 3:7] = qmult(qconjugate(drone_ori_wxyz), waypoint_1_quat)
        obs[0, 7:10] = rotate_vector(waypoint_2_pos_rel, qconjugate(drone_ori_wxyz))
        obs[0, 10:14] = qmult(qconjugate(drone_ori_wxyz), waypoint_2_quat)

            
        obs[0, 14:17] = drone_pos
        obs[0, 17:21] = drone_ori_wxyz
        obs[0, 21:24] = drone_vel_b
        obs[0, 24:28] = drone_last_action
        self.acceleration = (self.drone_state[10:13] - self.drone_state_prev[10:13]) / self.NETWORK_TIMESTEP

        self.obs = obs.astype(np.float32)
        self.drone_state_prev = self.drone_state.copy()

        return obs.copy()
       
    def _computeCrossedWaypoint(self):
        # calculate position of drone in waypoint frame
        if self.curr_waypoint_idx == self.next_waypoints[0]:
            waypoint_pos = self.waypoints_xyz[self.curr_waypoint_idx]
            waypoint_quat = self.waypoints_quats[self.curr_waypoint_idx]
            drone_pos = self.obs[0, 14:17]
            drone_quat = self.obs[0, 17:21]
            drone_pos_prev = self.prev_obs[0, 14:17]
            drone_quat_prev = self.prev_obs[0, 17:21]
            p_a = rotate_vector(drone_pos - waypoint_pos, qconjugate(waypoint_quat))
            p_a_prev = rotate_vector(drone_pos_prev - waypoint_pos, qconjugate(waypoint_quat))
            if self.DEBUG:
                self.p_a = p_a
                self.p_a_prev = p_a_prev
            if (p_a[0] > 0 and p_a_prev[0] <= 0 and np.abs(p_a[1]) < self.ALLOWED_BOUNDS and np.abs(p_a[2]) < self.ALLOWED_BOUNDS):
                self.crossed_waypoint = True
        else:
            self.curr_waypoint_idx = self.next_waypoints[0] # the obs is oudated for p_a_prev computation, wait for next step


    def _computeTerminated(self):
        waypoint_pos_rel = self.obs[0,0:3]
        if np.linalg.norm(waypoint_pos_rel) > self.MAX_DIST_FROM_WAYPOINT and self._get_num_waypoints_remain() > 0:
            self.out_of_bound = True
            return True
    
        # Fix: Convert network_step_counter to seconds before comparing
        elapsed_time = self.network_step_counter * self.NETWORK_TIMESTEP
        if elapsed_time >= self.episode_len_sec:
            self.timeout = True
            return True
    
        # if self.obs[0, 14] < self.MIN_X or self.obs[0, 14] > self.MAX_X or self.obs[0, 15] < self.MIN_Y or self.obs[0, 15] > self.MAX_Y or self.obs[0,16] < self.MIN_Z or self.obs[0,16] > self.MAX_Z:
        #     self.out_of_bound = True
        #     return True
    
        return False
        
    def _computeTruncated(self):

        if self._get_num_waypoints_remain() == 0:
            self.no_waypoints_remain = True
            return True
        return False
    
    def _computeReward(self):
        def activation(x, A, B):
            C1 = x > A - B
            C2 = x <= A - B
            output = (A - x) * C1 + (B - 1 + np.power(10, A - x -B)) * C2
            return output
        waypoint_pos_rel = self.obs[0, 14:17] - self.waypoints_xyz[self.next_waypoints[0]]
        waypoint_ori = self.waypoints_quats[self.next_waypoints[0]]
        action = self.action[0]
        action_prev = self.action_prev[0]
        v_b = self.obs[0, 21:24]
        
        # calculate position erroer
        p_a = rotate_vector(waypoint_pos_rel, qconjugate(waypoint_ori))
        p_error = np.linalg.norm(p_a)
        
        r_te = 0
        r_pa = 0
        if (self.crossed_waypoint):
        
            # calculate z-axis alignment error
            q_drone = self.obs[0, 17:21]
            R_drone = quat2mat(q_drone)
            R_waypoint = quat2mat(waypoint_ori)
            z_waypoint = R_waypoint[:, 2]
            z_drone = R_drone[:, 2]
            cos_theta = np.clip(np.dot(z_waypoint, z_drone), -1.0, 1.0)
            theta_error = np.arccos(cos_theta)

            r_pa_max = activation(0, self.A_perror, self.B_perror)
            r_te_max = activation(0, self.A_theta_error, self.B_theta_error)
            ratio = r_pa_max / (r_te_max + 1e-6) # to balance the two components of the reward
            r_pa = activation(p_error, self.A_perror, self.B_perror) / ratio
            r_te = activation(theta_error, self.A_theta_error, self.B_theta_error)
        
            # calculate r_aero
            r_pa /= (2 *r_te_max + 1e-6)
            r_te /= (2 *r_te_max + 1e-6)


        if p_a[0] > 0: # when drone is in front of the waypoint
            r_pos = -3
        else:
            r_pos = 0
            if np.abs(p_a[0]) < self.ALLOWED_BOUNDS:
                r_pos += self.ALLOWED_BOUNDS / (np.abs(p_a[0]) + self.ALLOWED_BOUNDS)
            else:
                r_pos += self.ALLOWED_BOUNDS / (np.abs(p_a[0]) + np.exp(self.ALLOWED_BOUNDS ))
            if np.abs(p_a[1]) < self.ALLOWED_BOUNDS:
                r_pos += self.ALLOWED_BOUNDS / (np.abs(p_a[1]) + self.ALLOWED_BOUNDS)
            else:
                r_pos += self.ALLOWED_BOUNDS / (np.abs(p_a[1]) + np.exp(self.ALLOWED_BOUNDS ))
            if np.abs(p_a[2]) < self.ALLOWED_BOUNDS:
                r_pos += self.ALLOWED_BOUNDS / (np.abs(p_a[2]) + self.ALLOWED_BOUNDS)
            else:
                r_pos += self.ALLOWED_BOUNDS / (np.abs(p_a[2]) + np.exp(self.ALLOWED_BOUNDS ))

        vel_dir = rotate_vector(v_b, qconjugate(self.obs[0, 17:21]))
        vel_dir = vel_dir / (np.linalg.norm(vel_dir) + 1e-6)
        waypoint_dir = - waypoint_pos_rel / (np.linalg.norm(waypoint_pos_rel) + 1e-6)
        angle_error = np.arccos(np.clip(np.dot(vel_dir, waypoint_dir), -1.0, 1.0))
        r_vel = activation(angle_error, self.A_vel_dir, self.B_vel_dir) / activation(0, self.A_vel_dir, self.B_vel_dir)

        # calculate r_act and r_act_change
        r_act = np.linalg.norm(action[1:], ord=1) / 3
        r_act_change = np.linalg.norm(action - action_prev, ord=2) / 4


        v_b[2] = 0.0
        
        speed = np.linalg.norm(v_b)
        if speed < 1e-6:
            r_yaw = 0
        else:
            v_dir = v_b / speed
            cos_yaw = np.dot(v_dir, np.array([1.0, 0.0, 0.0], dtype=np.float32))
            cos_yaw = np.clip(cos_yaw, -1.0, 1.0)
            r_yaw = np.arccos(cos_yaw) / np.pi

        
        self.r_pa = r_pa 
        self.r_te = r_te 
        self.r_aero = self.r_pa + self.r_te
        self.r_act =  r_act
        self.r_act_change = r_act_change
        self.r_yaw =  r_yaw
        self.r_pos =  r_pos
        self.r_vel = r_vel
        self.r_continuous = self.r_pos + self.r_vel

        self.reward = self.w_0 * self.r_aero + self.w_1 * self.r_act + self.w_2 * self.r_act_change + self.w_3 * self.r_yaw + self.w_4 * self.r_continuous

        return self.reward.copy()
        
    def _computeInfo(self):
        if self.crossed_waypoint:
            passed_waypoint = True
        else:
            passed_waypoint = False

        info = {
            "passed_waypoint": passed_waypoint,
            "next_waypoints": deepcopy(self.next_waypoints), 
            "acc": self.acceleration.copy(),
            "curr_waypoint_idx": self.curr_waypoint_idx,
            "timeout": self.timeout,
            "out_of_bound": self.out_of_bound,
            "crashed": self.crashed,
            "no_waypoints_remain": self.no_waypoints_remain,
        }
        return info

    def _set_spawn(self, spawn):
        p = spawn["pos"]
        v = spawn["vel"]
        a = spawn["acc"]
        rpy = spawn["rpy"]
        next_waypoints = spawn["next_waypoints"]
        self.INIT_XYZS[0] = p
        self.INIT_RPYS[0] = rpy
        self.next_waypoints = next_waypoints
        self.curr_waypoint_idx = next_waypoints[0]

    def _get_num_waypoints_remain(self):
        if self.next_waypoints[0] == -1:
            return 0
        elif self.next_waypoints[1] == -1:
            return 1
        else:
            return self.waypoints_xyz.shape[0] - self.next_waypoints[1]
    
    def _update_next_waypoints(self):
        next_waypoints = list(self.next_waypoints)
        next_waypoints[0] = next_waypoints[1]
        if next_waypoints[1] < self.waypoints_xyz.shape[0] - 1:
            next_waypoints[1] += 1
        else:
            next_waypoints[1] = 0

        self.next_waypoints = tuple(next_waypoints)


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
        print("================================")
        print("reward:", self.reward)
        print("r_aero:", self.r_aero, " (r_pa:", self.r_pa, ", r_te:", self.r_te, ")")
        print("r_continuous:", self.r_continuous, " (r_pos:", self.r_pos, ", r_vel:", self.r_vel, ")")
        print("r_act:", self.r_act)
        print("r_act_change:", self.r_act_change)
        print("r_yaw:", self.r_yaw)
        print("================================")
        print("spawn:")
        print("xyz:", self.INIT_XYZS[0])
        print("rpy:", self.INIT_RPYS[0])
        print("================================")
        input("Press Enter to continue...") if self.DEBUG_PAUSE else None