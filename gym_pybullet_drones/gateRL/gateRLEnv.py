from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.control.CustomCTBRControl import CTBRPIDControl
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType, ImageType
from gym_pybullet_drones import asset_directory
from gym_pybullet_drones.gateRL.interpolate import interpolate_waypoints
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
                 drone_model: DroneModel=DroneModel.CF2X,
                 neighbourhood_radius: float=np.inf,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 500,
                 ctrl_freq: int = 500,
                 network_freq: int = 100,
                 episode_len_sec = 20,
                 gui=False,
                 record=False,
                 debug=False,
                 train=True,
                 p_adap=0.5,
                 p_buff=0.4,
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
        self.DEBUG = debug
        self.TRAIN = train

        #### Create integrated controllers #########################
        self.ctrl = CTBRPIDControl(drone_model=drone_model,
                                  ctrl_freq=ctrl_freq,
                                  )
        self.NETWORK_FREQ = network_freq
        self.NETWORK_TIMESTEP = 1.0 / self.NETWORK_FREQ
        self.network_step_counter = 0
        self.PYB_PER_NETWORK = int(pyb_freq / network_freq)
        self.EPISODE_LEN_SEC = episode_len_sec

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
        self.MAX_DIST_FROM_WAYPOINT = 4.
        self.MIN_Z = 1.5
        self.crossed_waypoint = False
        self.timeout = False
        self.out_of_bound = False
        self.crashed = False
        self.no_waypoints_remain = False
        self.next_waypoints = (0, 1)
        self.waypoints_xyz:np.ndarray = waypoints["pos"]
        self.waypoints_rpy:np.ndarray = waypoints["rpy"]
        self.waypoints_quats:np.ndarray = np.array([euler2quat(*rpy) for rpy in self.waypoints_rpy])
        self.predefined_spawn:np.ndarray = waypoints["spawn"]
        self.experience_buffer = []
        self.adaptive_buffer = []
        self.P_ADAP = p_adap
        self.P_BUFF = p_buff
        assert self.P_ADAP + self.P_BUFF <= 1.0, "Probabilities for adaptive buffer and experience buffer should sum to less than or equal to 1.0"


        # spaces
        self.prev_obs = np.zeros((1, 28)) # obs is (1,24) for compatibility with parent class
        self.obs = np.zeros((1, 28))
        self.action_prev = np.zeros((1,4))
        self.action = np.zeros((1,4))
        
        # reward function parameters
        self.A_perror =2
        self.B_perror =0.
        self.A_theta_error = np.pi / 2
        self.B_theta_error = 0.
        self.C = 20.
        self.w_0 = 0.3
        self.w_1 = 0.15
        self.w_2 = 0.35
        self.w_3 = 0.2
        self.ALLOWED_BOUNDS = .5

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
                         )
        

    def step(self, action):

        self.action = action
        # network is at lower frequency than pyb, aggrewaypoint steps
        for _ in range(self.PYB_PER_NETWORK): 
            super().step(action)    
        obs = self._computeObs()
        self._computeCrossedWaypoint(-self.obs[0, 0:3], -self.prev_obs[0, 0:3], self.waypoints_quats[self.curr_waypoint_idx])
        reward = self._computeReward()
        terminated = self._computeTerminated()
        truncated = self._computeTruncated()
        info = self._computeInfo()

        if self.DEBUG:
            # input()
            self._print_debug()

        if self.crossed_waypoint:
            self._update_next_waypoints()
            self.crossed_waypoint = False
        reward += 0.5
        self.network_step_counter += 1
        self.action_prev = action
        return obs, reward, terminated, truncated, info

    def _housekeeping(self):
        self._set_spawn()
        self.action_prev = np.zeros((1,4)).astype(np.float32)
        self.action = np.zeros((1,4)).astype(np.float32)
        self.prev_obs = np.zeros((1, 28)).astype(np.float32)
        self.obs = np.zeros((1, 28)).astype(np.float32)
        self.p_a = np.zeros(3).astype(np.float32)
        self.p_a_prev = np.zeros(3).astype(np.float32)
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
        drone_state = self._getDroneStateVector(0)
        drone_pos = drone_state[0:3]
        drone_quat = drone_state[3:7]
        drone_ori_wxyz= drone_quat[[3, 0, 1, 2]] # xyzw to wxyz
        drone_vel_b = rotate_vector(drone_state[10:13], drone_ori_wxyz).astype(np.float32)
        drone_last_action = self.action_prev[0]

        waypoint_1_pos_rel = self.waypoints_xyz[self.next_waypoints[0]] - drone_pos
        waypoint_2_pos_rel = self.waypoints_xyz[self.next_waypoints[1]] - drone_pos
        waypoint_1_quat = self.waypoints_quats[self.next_waypoints[0]]
        waypoint_2_quat = self.waypoints_quats[self.next_waypoints[1]]

        if not self.crossed_waypoint:
            obs [0, 0:3] = waypoint_1_pos_rel
            obs [0, 3:7] = waypoint_1_quat
            obs[0, 7:10] = waypoint_2_pos_rel
            obs[0, 10:14] = waypoint_2_quat
        else:
            obs [0, 0:3] = waypoint_2_pos_rel
            obs [0, 3:7] = waypoint_2_quat
            new_waypoint_pos_rel = self.waypoints_xyz[self.next_waypoints[1]] - drone_pos
            new_waypoint_quat = self.waypoints_quats[self.next_waypoints[1]]
            obs[0, 7:10] = new_waypoint_pos_rel
            obs[0, 10:14] = new_waypoint_quat
            
        obs[0, 14:17] = drone_pos
        obs[0, 17:21] = drone_ori_wxyz
        obs[0, 21:24] = drone_vel_b
        obs[0, 24:28] = drone_last_action
        self.obs = obs
        return obs.astype(np.float32)
        #compute whether passed through waypoint
       
    def _computeCrossedWaypoint(self, pos_rel, pos_rel_prev, ori):
        # calculate position of drone in waypoint frame
        if self.curr_waypoint_idx == self.next_waypoints[0]:
            p_a = rotate_vector(pos_rel, qconjugate(ori))
            p_a_prev = rotate_vector(pos_rel_prev, qconjugate(ori))
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

        if self.obs[0,16] < self.MIN_Z and self._get_num_waypoints_remain() > 0: 
            self.crashed = True
            return True
        return False
        
    def _computeTruncated(self):
        if self.network_step_counter / self.NETWORK_FREQ  > self.EPISODE_LEN_SEC:
            self.timeout = True
            return True
        if self._get_num_waypoints_remain() == 0:
            self.no_waypoints_remain = True
            return True
        return False
    
    def _computeReward(self):
        def activation(x, A, B):
            C1 = x > A - B
            C2 = x <= A - B
            output = (A - x) * C1 + (B - 1 + np.exp(A - x -B)) * C2
            return output
        waypoint_pos_rel = self.obs[0, 0:3]
        waypoint_ori = self.obs[0, 3:7]
        action = self.action[0]
        action_prev = self.action_prev[0]
        if (self.crossed_waypoint):
            # calculate position erroer
            p_a = rotate_vector(-waypoint_pos_rel, qconjugate(waypoint_ori))
            p_error = np.linalg.norm(p_a)

            # calculate z-axis alignment error
            q_drone = self.obs[0, 17:21]
            R_drone = quat2mat(q_drone)
            R_waypoint = quat2mat(waypoint_ori)
            z_waypoint = R_waypoint[:, 2]
            z_drone = R_drone[:, 2]
            cos_theta = np.clip(np.dot(z_waypoint, z_drone), -1.0, 1.0)
            theta_error = np.arccos(cos_theta)

            # calculate r_aero
            r_aero = activation(theta_error, self.A_theta_error, self.B_theta_error) + activation(p_error, self.A_perror, self.B_perror) + self.C
        else: 
            r_aero = 0

        # calculate r_act and r_act_change
        r_act = - np.linalg.norm(action[1:], ord=1)
        r_act_change = - np.linalg.norm(action - action_prev, ord=2)

        v_b = self.obs[0, 21:24]
        v_b[2] = 0.0
        
        speed = np.linalg.norm(v_b)
        if speed < 1e-6:
            r_yaw = 0
        else:
            v_dir = v_b / speed
            cos_yaw = np.dot(v_dir, np.array([1.0, 0.0, 0.0], dtype=np.float32))
            cos_yaw = np.clip(cos_yaw, -1.0, 1.0)
            r_yaw = -np.arccos(cos_yaw)
        return self.w_0 * r_aero + self.w_1 * (r_act) + self.w_2 * (r_act_change) + self.w_3 * (r_yaw)
        
    def _computeInfo(self):
        if self.crossed_waypoint:
            passed_waypoint = True
        else:
            passed_waypoint = False

        info = {
            "passed_waypoint": passed_waypoint,
            "next_waypoints": deepcopy(self.next_waypoints), 
            "curr_waypoint_idx": self.curr_waypoint_idx,
            "timeout": self.timeout,
            "out_of_bound": self.out_of_bound,
            "crashed": self.crashed,
            "no_waypoints_remain": self.no_waypoints_remain,
        }
        return info

    def _set_spawn(self):
        # in training, spawn is sampled from adaptive set with p1, sampled from experience buffer with p2 and predefined initial states with p3
        prb = np.random.rand()

        if self.TRAIN and self.network_step_counter > 0: 
            if prb < self.P_ADAP:
                assert len(self.adaptive_buffer) > 0, "Adaptive buffer is empty, cannot sample spawn"
                spawn = np.random.choice(self.adaptive_buffer)
            elif prb < self.P_ADAP + self.P_BUFF:
                assert len(self.experience_buffer) > 0, "Experience buffer is empty, cannot sample spawn"
                spawn = np.random.choice(self.experience_buffer)
            else:
                # predefined initial states
                spawn = np.random.choice(self.predefined_spawn)
        else:
            spawn = np.random.choice(self.predefined_spawn)
        pos = spawn["pos"]
        vel = spawn["vel"]
        acc = spawn["acc"]
        rpy = spawn["rpy"]
        next_waypoints = spawn["next_waypoints"]
        self.INIT_XYZS[0] = pos
        self.INIT_RPYS[0] = rpy
        self.next_waypoints = next_waypoints
        self.curr_waypoint_idx = next_waypoints[0]
    def update_experience_buffer(self, buffer):
        self.experience_buffer = deepcopy(buffer)

    def update_adaptive_buffer(self, buffer):
        self.adaptive_buffer = deepcopy(buffer)


    def _get_num_waypoints_remain(self):
        if self.next_waypoints[0] == -1:
            return 0
        elif self.next_waypoints[1] == -1:
            return 1
        else:
            return self.waypoints_xyz.shape[0] - self.next_waypoints[1]
    
    def _update_next_waypoints(self):
        next_waypoints = list(self.next_waypoints)
        if self.next_waypoints[0] < self.waypoints_xyz.shape[0] - 1:
            next_waypoints[0] += 1
            if self.next_waypoints[1] < self.waypoints_xyz.shape[0] - 1:
                next_waypoints[1] += 1
            else:
                next_waypoints[1] = -1
        else:
            next_waypoints[0] = -1
            next_waypoints[1] = -1
        self.next_waypoints = tuple(next_waypoints)


    def _print_debug(self):
        print(f"obs: {self.obs}")
        print(f"action: {self.action}")
        print(f"reward: {self._computeReward()}")
        print(f"terminated: {self._computeTerminated()}")
        print(f"truncated: {self._computeTruncated()}")
        print(f"info: {self._computeInfo()}")
        print(f"p_a: {self.p_a}")
        print(f"p_a_prev: {self.p_a_prev}")
        print("====")
        print("spawn:")
        print("xyz:", self.INIT_XYZS[0])
        print("rpy:", self.INIT_RPYS[0])
        print("====")
        print("experience_buffer:")
        for exp in self.experience_buffer:
            print(exp)
        print("adaptive_buffer:")
        for exp in self.adaptive_buffer:
            print(exp)
        print("====")        