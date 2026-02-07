from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.control.CustomCTBRControl import CTBRPIDControl
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType, ImageType
from gym_pybullet_drones import asset_directory
from gym_pybullet_drones.utils.waypoints import interpolate_waypoints
from transforms3d.quaternions import rotate_vector, qconjugate, mat2quat, qmult
from gym_pybullet_drones.utils.tracks import Track
from gym_pybullet_drones.utils.track_settings.track_settings import TrackSettings

import os
import numpy as np
import pybullet as p
import gymnasium as gym
from gymnasium import spaces
from collections import deque

class GateRLEnv(BaseAviary): #TODO: spawn point, waypoints, waypoints visualization
    
    def __init__(self,
                 tracks: list[TrackSettings],
                 drone_model: DroneModel=DroneModel.CF2X,
                 neighbourhood_radius: float=np.inf,
                 physics: Physics=Physics.PYB_DRAG,
                 pyb_freq: int = 500,
                 ctrl_freq: int = 500,
                 network_freq: int = 100,
                 episode_len_sec = 20,
                 gui=False,
                 record=False,
                 debug=False,
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
        ####
        self.OBS_TYPE = ObservationType.KIN
        self.ACT_TYPE = ActionType.RPM
        #### Create integrated controllers #########################
        self.ctrl = CTBRPIDControl(drone_model=drone_model,
                                  ctrl_freq=ctrl_freq,
                                  )
        self.NETWORK_FREQ = network_freq
        self.NETWORK_TIMESTEP = 1.0 / self.NETWORK_FREQ
        self.network_step_counter = 0
        self.PYB_PER_NETWORK = int(pyb_freq / network_freq)
        self.EPISODE_LEN_SEC = episode_len_sec
        
        self.DIFFICULTY = "easy"
        self.TRACK = Track(tracks)

        self.MAX_ROLL_RATE = 2 *np.pi
        self.MAX_PITCH_RATE = 2 * np.pi
        self.MAX_YAW_RATE = 2 * np.pi
        self.MAX_MASS_NORMALIZED_THRUST = 15
        self.ACTION_SCALE = np.array([self.MAX_MASS_NORMALIZED_THRUST,
                                      self.MAX_ROLL_RATE,
                                      self.MAX_PITCH_RATE,
                                      self.MAX_YAW_RATE])
        
        self.drone_init_ranges = {
            "x": [-1.0, 1.0],
            "y": [-1.0, 1.0],
            "z": [1.0, 2.0],
            "roll": [-0.1, 0.1],
            "pitch": [-0.1, 0.1],
            "yaw": [-np.pi, np.pi],
        }

        self.prev_obs = np.zeros((1, 28)) # obs is (1,24) for compatibility with parent class
        self.obs = np.zeros((1, 28))
        self.infered_action_prev = np.zeros((1,4))
        self.infered_action = np.zeros((1,4))
        
        self.target_waypoints = [0, 1]
        
        self.DEBUG = debug

        self.MAX_DIST_FROM_WAYPOINT = 5.
        self.crossed_waypoint = False
        # reward function parameters
        self.A_perror =2
        self.B_perror =0.
        self.A_theta_error = np.pi / 2
        self.B_theta_error = 0.
        self.C = 20.
        self.w_0 = 0.94
        self.w_1 = 0.002
        self.w_2 = 0.002
        self.w_3 = 0.002
        self.boundary = 3

        super().__init__(drone_model=drone_model,
                         num_drones=1,
                         neighbourhood_radius=neighbourhood_radius,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record, 
                         obstacles=True, # visualize waypoints
                         user_debug_gui=True, # drawing drone axis
                         vision_attributes=False,
                         compute_returns_per_step=False,
                         )
        

    def step(self, action):
        self.prev_obs = self.obs
        self.infered_action_prev = self.infered_action
        self.infered_action = action
        self.infered_action[0,0] = (self.infered_action[0,0] + 1.2) / 2.2

        # network is at lower frequency than pyb, aggrewaypoint steps
        for _ in range(self.PYB_PER_NETWORK): 
            super().step(action)
        # self.step_counter -= (self.PYB_PER_NETWORK - 1)

        obs = self._computeObs()
        self.obs = obs
        self._computeCrossedWaypoint()
        reward = self._computeReward()
        terminated = self._computeTerminated()
        truncated = self._computeTruncated()
        info = self._computeInfo()

        if self.DEBUG:
            print("\n Step:", self.network_step_counter,
                  "\n Infered action:", self.infered_action,
                  "\n Position:", self.obs[0,14:17],
                  "\n Waypoint 1 pos rel:", self.obs[0,0:3],
                  "\n Waypoint 2 pos rel:", self.obs[0,7:10],
                  "\n Reward:", reward,
                  "\n Terminated:", terminated,
                  "\n Truncated:", truncated,
                  "\n Info:", info,
                  )
        
        # while computing obs, also set the flag for passing through waypoint
        if self.crossed_waypoint:
            self.crossed_waypoint = False
            if not self.TRACK.step(): # step() return true if there are more waypoints left
                reward += 100
                terminated = True
                truncated = False

        self.network_step_counter += 1
        return obs, reward, terminated, truncated, info

    def _housekeeping(self):
        self.TRACK.reset(self.DIFFICULTY)
        self.INIT_XYZS[0], _, self.INIT_RPYS[0] = self.TRACK.get_spawn_point()
        self.infered_action_prev = np.zeros((1,4)).astype(np.float32)
        self.infered_action = np.zeros((1,4)).astype(np.float32)
        self.prev_obs = np.zeros((1, 28)).astype(np.float32)
        self.obs = np.zeros((1, 28)).astype(np.float32)
        self.crossed_waypoint = False
        super()._housekeeping()
        
    def _addObstacles(self): # visualize waypoints
        waypoint_xyz = self.TRACK.get_waypoints_xyz()
        waypoint_quats = self.TRACK.get_waypoints_quats()
        waypoint_quats = waypoint_quats[:, [3,0,1,2]] # convert wxyz to xyzw for pybullet
        
        def draw_waypoint(pos, quat, length=0.3):
            rot_matrix = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
            x_axis = rot_matrix[:, 0] * length
            # y_axis = rot_matrix[:, 1] * length
            # z_axis = rot_matrix[:, 2] * length
            p.addUserDebugLine(pos, pos + x_axis, [1, 0, 0],)
            # p.addUserDebugLine(pos, pos + y_axis, [0, 1, 0],)
            # p.addUserDebugLine(pos, pos + z_axis, [0, 0, 1],)
        
        for pos, quat in zip(waypoint_xyz, waypoint_quats):
            draw_waypoint(pos, quat)
        
    
    def _actionSpace(self):
        act_lower_bound = np.array([[-1, -1, -1, -1] for i in range(self.NUM_DRONES)])
        act_upper_bound = np.array([[1, 1, 1, 1] for i in range(self.NUM_DRONES)])
        return spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)
    
    def _preprocessAction(self, action):

        action = self.infered_action * self.ACTION_SCALE
        cur_body_rate = rotate_vector(self._getDroneStateVector(0)[13:16], self.obs[0, 17:21])
        rpm = self.ctrl.computeControl(
            control_timestep=self.CTRL_TIMESTEP,
            thrust=action[0, 0],
            target_body_rate=action[0, 1:4],
            cur_body_rate=cur_body_rate,
        )
        rpm = np.reshape(rpm, (1, 4))
        if self.DEBUG:
            print(" Preprocess action:",
                  "\n  Infered action (normalized):", self.infered_action,
                  "\n  Infered action (scaled):", action,
                  "\n  RPM:", rpm,
                  )
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

    def _computeObs(self): #TODO: waypoints xyz and rpy need change
        obs = np.zeros((self.NUM_DRONES, 28)).astype(np.float32)
        waypoints_xyz, waypoints_quats, _ = self.TRACK.get_next_waypoints()
        drone_state = self._getDroneStateVector(0)
        drone_pos = drone_state[0:3]
        drone_ori_wxyz = np.zeros(4)
        drone_quat = drone_state[3:7]
        drone_ori_wxyz[0] = drone_quat[3]
        drone_ori_wxyz[1:4] = drone_quat[0:3]
        drone_vel_b = rotate_vector(drone_state[10:13], drone_ori_wxyz).astype(np.float32)
        drone_last_action = self.infered_action[0]
        waypoint_1_pos_rel = waypoints_xyz[0] - drone_pos
        waypoint_2_pos_rel = waypoints_xyz[1] - drone_pos
        obs [0, 0:3] = waypoint_1_pos_rel
        obs [0, 3:7] = waypoints_quats[0]
        obs[0, 7:10] = waypoint_2_pos_rel
        obs[0, 10:14] = waypoints_quats[1]
        obs[0, 14:17] = drone_pos
        obs[0, 17:21] = drone_ori_wxyz
        obs[0, 21:24] = drone_vel_b
        obs[0, 24:28] = drone_last_action
        return obs.astype(np.float32)
        #compute whether passed through waypoint
       
    def _computeCrossedWaypoint(self):
        waypoint1_pos_rel = self.obs[0, 0:3]
        waypoint1_ori = self.obs[0, 3:7]
        waypoint1_pos_rel_prev = self.prev_obs[0, 0:3]
        waypoint1_ori_prev = self.prev_obs[0, 3:7]
        
        p_a = rotate_vector(-waypoint1_pos_rel, qconjugate(waypoint1_ori))
        p_a_prev = rotate_vector(-waypoint1_pos_rel_prev, qconjugate(waypoint1_ori_prev))
        if (p_a[0] > 0 and p_a_prev[0] <= 0 and np.abs(p_a[1]) < self.boundary and np.abs(p_a[2]) < self.boundary):
            self.crossed_waypoint = True

    def _computeTruncated(self):
        waypoint_pos_rel = self.obs[0,0:3]
        if np.linalg.norm(waypoint_pos_rel) > self.MAX_DIST_FROM_WAYPOINT:
            return True

        if self.obs[0,16] < 0.2:
            return True
        return False
        
    def _computeTerminated(self):
        if self.network_step_counter / self.NETWORK_FREQ  > self.EPISODE_LEN_SEC:
            # print("Episode timed out")
            return True
        return False
    
    def _activation(self, x, A, B):
        output = (A - x) if x > A - B else (B - 1 + np.exp(A - x -B))
        return output
    
    def _computeReward(self):
        waypoint_pos_rel = self.obs[0, 0:3]
        waypoint_ori = self.obs[0, 3:7]
        p_drone = self.obs[0, 14:17]
        if (self.crossed_waypoint):
            # calculate position erroer
            p_a = rotate_vector(-waypoint_pos_rel, qconjugate(waypoint_ori))
            p_error = np.linalg.norm(p_a)

            # calculate z-axis alignment error
            q_drone = self.obs[0, 17:21]
            R_drone = np.array(p.getMatrixFromQuaternion(q_drone)).reshape(3, 3)
            R_waypoint = np.array(p.getMatrixFromQuaternion(waypoint_ori)).reshape(3, 3)
            z_waypoint = R_waypoint[:, 2]
            z_drone = R_drone[:, 2]
            cos_theta = np.clip(np.dot(z_waypoint, z_drone), -1.0, 1.0)
            theta_error = np.arccos(cos_theta)

            # calculate r_aero
            r_aero = self._activation(theta_error, self.A_theta_error, self.B_theta_error) + self._activation(p_error, self.A_perror, self.B_perror) + self.C
        else: 
            r_aero = 0

        # calculate r_act and r_act_change
        r_act = - np.sum(np.abs(self.infered_action[0,1:]))
        r_act_change = - np.linalg.norm(self.infered_action[0] - self.infered_action_prev[0], ord=2)

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
            return {"passed_waypoint": True}
        else:
            return {"passed_waypoint": False}
        
        
if __name__ == "__main__":
    from gym_pybullet_drones.utils.track_settings import track1_setting
    env = GateRLEnv(tracks=[track1_setting.Track1()], gui=True, debug=True)
    obs, info = env.reset(seed=42, options={})
    for i in range(1000):
        input()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print("Episode ended")
            break
    env.close() 