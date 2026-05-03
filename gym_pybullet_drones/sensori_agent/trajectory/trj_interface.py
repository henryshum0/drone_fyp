from gym_pybullet_drones.sensori_agent.trajectory.trajectory_generation import Trajectory, build_trajectory_from_template
from gym_pybullet_drones.sensori_agent.trajectory.trajectory_optimize import optimize_trj_time
import numpy as np
from scipy.spatial.transform import Slerp
class TrajectoryInterface:
	@staticmethod
	def _slerp(quat_a: np.ndarray, quat_b: np.ndarray, alpha: float) -> np.ndarray:
		quat_a = np.asarray(quat_a, dtype=float)
		quat_b = np.asarray(quat_b, dtype=float)
		quat_a /= np.linalg.norm(quat_a)
		quat_b /= np.linalg.norm(quat_b)

		dot = float(np.dot(quat_a, quat_b))
		if dot < 0.0:
			quat_b = -quat_b
			dot = -dot

		dot = float(np.clip(dot, -1.0, 1.0))
		if dot > 0.9995:
			quat = quat_a + alpha * (quat_b - quat_a)
			return quat / np.linalg.norm(quat)

		theta_0 = np.arccos(dot)
		sin_theta_0 = np.sin(theta_0)
		if abs(sin_theta_0) < 1e-12:
			return quat_a

		theta = theta_0 * alpha
		s0 = np.sin(theta_0 - theta) / sin_theta_0
		s1 = np.sin(theta) / sin_theta_0
		return s0 * quat_a + s1 * quat_b

	def __init__(self, trajectory_obj: Trajectory, sample_freq: float):
		self.reset(trajectory_obj, sample_freq)

	def reset(self, trajectory_obj: Trajectory, sample_freq: float):
		self.trajectory_obj = trajectory_obj
		self.trajectory_obj, optimized_time, _ = optimize_trj_time(
			self.trajectory_obj,
			time_penalty=np.array([100 for seg in self.trajectory_obj._segments]),
			preserve_total_time=False,
			max_velocity=30,
			min_velocity=0,
			max_normalized_thrust=50,
			report_peaks=True,
		)
		print("optimized trajectory time:", optimized_time)

		self.sample_freq = sample_freq
		self.REF_DT = 1.0 / self.sample_freq
		self.ref_states = self.trajectory_obj.sample_full_state(sampling_rate=self.sample_freq)
		self.duration = float(self.ref_states["t"][-1])

	def get_discrete_state_mpc(self, t: float) -> np.ndarray:
		x = np.zeros(13, dtype=float)
		t = float(np.clip(t, 0.0, self.duration))

		pos_samples = self.ref_states["pos"]
		quat_samples = self.ref_states["quat"]
		vel_samples = self.ref_states["vel"]
		body_rate_samples = self.ref_states["body_rate"]
		last_idx = len(pos_samples) - 1

		if last_idx <= 0:
			x[0:3] = pos_samples[0, 0:3]
			x[3:7] = quat_samples[0, 0:4]
			x[7:10] = vel_samples[0, 0:3]
			x[10:13] = body_rate_samples[0, 0:3]
			return x

		sample_pos = t / self.REF_DT
		idx = int(np.floor(sample_pos))
		if idx >= last_idx:
			idx = last_idx
			x[0:3] = pos_samples[idx, 0:3]
			x[3:7] = quat_samples[idx, 0:4]
			x[7:10] = vel_samples[idx, 0:3]
			x[10:13] = body_rate_samples[idx, 0:3]
			return x

		alpha = sample_pos - idx
		next_idx = idx + 1
		x[0:3] = (1.0 - alpha) * pos_samples[idx, 0:3] + alpha * pos_samples[next_idx, 0:3]
		x[3:7] = self._slerp(quat_samples[idx, 0:4], quat_samples[next_idx, 0:4], alpha)
		x[7:10] = (1.0 - alpha) * vel_samples[idx, 0:3] + alpha * vel_samples[next_idx, 0:3]
		x[10:13] = (1.0 - alpha) * body_rate_samples[idx, 0:3] + alpha * body_rate_samples[next_idx, 0:3]
		return x

if __name__ == "__main__":
	from gym_pybullet_drones.sensori_agent.acro_templates import\
		PowerloopTemplate, SplitSLeftTemplate, SplitSRightTemplate, BarrelRollLeftTemplate, BarrelRollRightTemplate,\
		HeartTemplate
	
	template = HeartTemplate()
	trj_interface = TrajectoryInterface(build_trajectory_from_template(template, randomized=True), sample_freq=200.0)
	for t in np.arange(0, 0.05, 0.01):
		print("t:", t, "state:", trj_interface.get_discrete_state_mpc(t))