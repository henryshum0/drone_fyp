from dataclasses import dataclass

from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.sensori_agent.MPC.simple_mpc_controller import SimpleQuadrotorMPC
from gym_pybullet_drones.utils.enums import DroneModel
from gym_pybullet_drones.control.CustomCTBRControl import CTBRPIDControl
from gym_pybullet_drones.sensori_agent.trajectory.trajectory import Trajectory
from gym_pybullet_drones.sensors.imu import IMU
from gym_pybullet_drones.sensors.camera import CameraSensor

import numpy as np

@dataclass
class Rollout():
	def __init__(self,):
		self.len = 0
		self.gt_x = []
		self.gt_q = []
		self.gt_v = []
		self.gt_w = []
		self.ref_x = []
		self.ref_q = []
		self.ref_v = []
		self.ref_w = []
		self.mpc_act = []
		self.network_act = []
		self.feature_track = []
		self.imu_acc = []
		self.imu_gyro = []

	def add_data(self, data:dict):
		self.gt_x.append(data["gt_x"])
		self.gt_q.append(data["gt_q"])
		self.gt_v.append(data["gt_v"])
		self.gt_w.append(data["gt_w"])
		self.ref_x.append(data["ref_x"])
		self.ref_q.append(data["ref_q"])
		self.ref_v.append(data["ref_v"])
		self.ref_w.append(data["ref_w"])
		self.mpc_act.append(data["mpc_act"])
		self.network_act.append(data["network_act"])
		self.feature_track.append(data["feature_track"])
		self.imu_acc.append(data["imu_acc"])
		self.imu_gyro.append(data["imu_gyro"])
		self.len += 1

class TrainEnv(BaseAviary):

	def __init__(
		self, 
		gui: bool = False,
	):
		super().__init__(
			drone_model=DroneModel.RACE,
			num_drones=1,
			pyb_freq=500,
			ctrl_freq=500,
			use_egl_renderer=True,
			user_debug_gui=gui,
			gui=gui,
			compute_returns_per_step=False,
		)
		
		self.rate_controller = CTBRPIDControl(
			drone_model=self.DRONE_MODEL,
			ctrl_freq=self.CTRL_FREQ,
		)
		self.mpc_controller = SimpleQuadrotorMPC(
			max_thrust=60,
			max_velocity=30,
			max_roll_pitch_rate=5 * np.pi,
			max_yaw_rate=3 * np.pi,
			gravity=self.G,
		)
		self.network = None

		self.mpc_freq = 50
		self.feature_tracker_freq = 15
		self.imu_freq = 100
		self.horizon_dt = 0.05
		self.horizon = 10
		self.trj_freq = 50

		self.motor_delay = 0.075
		
		self.motor_t_range = (0.06, 0.08)
		self.factor_thrust_to_weight_range = (0.8, 1.2)
		self.factor_inertial_range = (0.8, 1.2)
		
		self.n_tf_points = 40

	def reset(self, trajectory: Trajectory):
		self.trajectory = trajectory
		obs, info = super().reset()
		self.mpc_controller.reset(
			trajectory_obj=trajectory,
			trajectory_sample_freq=float(self.trj_freq),
		)
		self.motor_delay = np.random.uniform(*self.motor_t_range)
		self.factor_thrust_to_weight = np.random.uniform(*self.factor_thrust_to_weight_range)
		self.factor_inertial = np.random.uniform(*self.factor_inertial_range)
		return obs, info

	def step(self, action):
		pass

	def _get_action(self):
		pass

