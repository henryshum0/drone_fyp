from gym_pybullet_drones.sensors.sensor import Sensor
import numpy as np


class IMU(Sensor):
	"""Simple IMU model with Gaussian noise and random-walk bias.

	The IMU reports body-frame linear acceleration and angular velocity.
	"""

	def __init__(
		self,
		freq,
		pyb_freq,
		client_id,
		accel_noise_std_range=(1e-3, 5e-3),
        gyro_noise_std_range=(1e-4, 5e-4),
        accel_bias_std_range=(1e-4, 5e-4),
        gyro_bias_std_range=(1e-5, 5e-5),
		gravity=9.81,
		rng=None,
	):
		
		super().__init__(
			freq = freq,
			pyb_freq = pyb_freq,
			client_id = client_id,
		)
		self.DT = 1.0 / self.freq
		self.accel_noise_std_range = self._as_std_range(accel_noise_std_range)
		self.gyro_noise_std_range = self._as_std_range(gyro_noise_std_range)
		self.accel_noise_std = 0.0
		self.gyro_noise_std = 0.0

		self.accel_bias_std_range = self._as_std_range(accel_bias_std_range)
		self.gyro_bias_std_range = self._as_std_range(gyro_bias_std_range)
		self.accel_bias_std = 0.0
		self.gyro_bias_std = 0.0
		self.gravity = float(gravity)

		self._rng = rng if rng is not None else np.random.default_rng()
		self._prev_vel_world = np.zeros(3, dtype=float)
		self._has_prev_vel = False

		self.accel_actual = np.zeros(3, dtype=float)
		self.gyro_actual = np.zeros(3, dtype=float)
		self.accel_noisy = np.zeros(3, dtype=float)
		self.gyro_noisy = np.zeros(3, dtype=float)
		self.accel_bias = np.zeros(3, dtype=float)
		self.gyro_bias = np.zeros(3, dtype=float)

		self.prev_step_counter = -1
		self._sample_model_sds()


	def set_std_sampling_ranges(
		self,
		accel_noise_std_range=None,
		gyro_noise_std_range=None,
		accel_bias_std_range=None,
		gyro_bias_std_range=None,
	):
		"""Set optional [min, max] ranges for per-reset SD resampling."""
		if accel_noise_std_range is not None:
			self.accel_noise_std_range = self._as_std_range(accel_noise_std_range)
		if gyro_noise_std_range is not None:
			self.gyro_noise_std_range = self._as_std_range(gyro_noise_std_range)
		if accel_bias_std_range is not None:
			self.accel_bias_std_range = self._as_std_range(accel_bias_std_range)
		if gyro_bias_std_range is not None:
			self.gyro_bias_std_range = self._as_std_range(gyro_bias_std_range)

	def _read_actual(self, true_linear_acc_body, true_angular_velocity_body, gravity_body=None):
		"""Return ideal IMU measurements with gravity always injected.

		Parameters
		----------
		true_linear_acc_body : array-like, shape (3,)
			True body-frame linear acceleration in m/s^2.
		true_angular_velocity_body : array-like, shape (3,)
			True body-frame angular velocity in rad/s.
		Returns
		-------
		accel_actual : np.ndarray, shape (3,)
			Ideal accelerometer measurement in m/s^2.
		gyro_actual : np.ndarray, shape (3,)
			Ideal gyroscope measurement in rad/s.
		"""
		true_acc = self._as_vec3(true_linear_acc_body)
		true_omega = self._as_vec3(true_angular_velocity_body)

		if gravity_body is None:
			gravity_body = np.array([0.0, 0.0, self.gravity], dtype=float)
		else:
			gravity_body = self._as_vec3(gravity_body)

		# Gravity injection is mandatory for accelerometer actual readings.
		accel_actual = true_acc + gravity_body
		gyro_actual = true_omega
		return accel_actual, gyro_actual

	def _read_noisy(self, true_linear_acc_body, true_angular_velocity_body, gravity_body=None):
		"""Return noisy/bias-corrupted IMU measurements.

		Gravity injection is always applied before noise and bias.

		Parameters
		----------
		true_linear_acc_body : array-like, shape (3,)
			True body-frame linear acceleration in m/s^2.
		true_angular_velocity_body : array-like, shape (3,)
			True body-frame angular velocity in rad/s.

		Returns
		-------
		accel_meas : np.ndarray, shape (3,)
			Simulated accelerometer measurement in m/s^2.
		gyro_meas : np.ndarray, shape (3,)
			Simulated gyroscope measurement in rad/s.
		"""
		accel_actual, gyro_actual = self._read_actual(
			true_linear_acc_body=true_linear_acc_body,
			true_angular_velocity_body=true_angular_velocity_body,
			gravity_body=gravity_body,
		)
		self._update_random_walk_bias(self.DT)

		factor = 1/np.sqrt(self.DT)
		accel_noise = self._rng.normal(0.0, self.accel_noise_std * factor, size=3)
		gyro_noise = self._rng.normal(0.0, self.gyro_noise_std * factor, size=3)

		accel_meas = accel_actual + self.accel_bias + accel_noise
		gyro_meas = gyro_actual + self.gyro_bias + gyro_noise
		return accel_meas, gyro_meas

	def reset(self, seed=None, resample_model_sds=True):
		"""Reset dynamic IMU states and optionally resample model SDs."""
		if seed is not None:
			self._rng = np.random.default_rng(seed)

		self._prev_vel_world.fill(0.0)
		self._has_prev_vel = False
		self.prev_step_counter = -1

		self.accel_actual.fill(0.0)
		self.gyro_actual.fill(0.0)
		self.accel_noisy.fill(0.0)
		self.gyro_noisy.fill(0.0)
		self.accel_bias.fill(0.0)
		self.gyro_bias.fill(0.0)

		if resample_model_sds:
			self._sample_model_sds()

	def update_from_kinematics(self, vel_world, ang_vel_world, quat_xyzw, step_counter):
		"""Update IMU from world-frame kinematics.

		Parameters
		----------
		vel_world : array-like, shape (3,)
			Current world-frame linear velocity in m/s.
		ang_vel_world : array-like, shape (3,)
			Current world-frame angular velocity in rad/s.
		quat_xyzw : array-like, shape (4,)
			Body orientation quaternion in PyBullet format [x, y, z, w].
		step_counter : int
			Current simulation step counter.

		Returns
		-------
		tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
			(accel_actual, gyro_actual, accel_noisy, gyro_noisy)
		"""
		if step_counter <= self.prev_step_counter and self.prev_step_counter >= 0:
			self.reset(resample_model_sds=True)
		if (not super().should_update(step_counter)):
			return
		vel_world = self._as_vec3(vel_world)
		ang_vel_world = self._as_vec3(ang_vel_world)
		dt = self.DT
		if dt <= 0.0:
			raise ValueError(f"dt must be positive, got {dt}")

		if self._has_prev_vel:
			steps_elapsed = step_counter - self.prev_step_counter
			if steps_elapsed <= 0:
				steps_elapsed = 1
			acc_world = (vel_world - self._prev_vel_world) / (dt * steps_elapsed)
		else:
			acc_world = np.zeros(3, dtype=float)

		self._prev_vel_world = np.array(vel_world, copy=True)
		self._has_prev_vel = True

		rot_mat = self._quat_xyzw_to_rotmat(quat_xyzw)
		acc_body = rot_mat.T @ acc_world
		gyro_body = rot_mat.T @ ang_vel_world
		gravity_world = np.array([0.0, 0.0, self.gravity], dtype=float)
		gravity_body = rot_mat.T @ gravity_world

		self.accel_actual, self.gyro_actual = self._read_actual(
			acc_body,
			gyro_body,
			gravity_body=gravity_body,
		)
		self.accel_noisy, self.gyro_noisy = self._read_noisy(
			acc_body,
			gyro_body,
			gravity_body=gravity_body,
		)

		self.prev_step_counter = step_counter
		self.timestamp = step_counter

	def get_timestamp(self):
		"""Return timestamp of the latest IMU update in seconds."""
		return self.timestamp

	def get_noisy(self):
		"""Return the latest noisy IMU measurements."""
		return self.accel_noisy.copy(), self.gyro_noisy.copy()
	
	def get_actual(self):
		"""Return the latest ideal IMU measurements."""
		return self.accel_actual.copy(), self.gyro_actual.copy()

	@staticmethod
	def _quat_xyzw_to_rotmat(quat_xyzw):
		q = np.asarray(quat_xyzw, dtype=float).reshape(-1)
		if q.size != 4:
			raise ValueError(f"Expected quaternion [x, y, z, w], got shape {np.asarray(quat_xyzw).shape}")

		x, y, z, w = q
		n = x * x + y * y + z * z + w * w
		if n < 1e-12:
			return np.eye(3)

		s = 2.0 / n
		xx, yy, zz = x * x * s, y * y * s, z * z * s
		xy, xz, yz = x * y * s, x * z * s, y * z * s
		wx, wy, wz = w * x * s, w * y * s, w * z * s

		return np.array([
			[1.0 - (yy + zz), xy - wz, xz + wy],
			[xy + wz, 1.0 - (xx + zz), yz - wx],
			[xz - wy, yz + wx, 1.0 - (xx + yy)],
		], dtype=float)

	@staticmethod
	def _as_vec3(value, default=None):
		if value is None:
			if default is None:
				raise ValueError("Expected a 3D vector, got None")
			return np.full(3, float(default), dtype=float)

		arr = np.asarray(value, dtype=float).reshape(-1)
		if arr.size != 3:
			raise ValueError(f"Expected a 3D vector, got shape {np.asarray(value).shape}")
		return arr

	@staticmethod
	def _as_std_range(value):
		if value is None:
			return None
		arr = np.asarray(value, dtype=float).reshape(-1)
		if arr.size != 2:
			raise ValueError(f"Expected std range [min, max], got shape {np.asarray(value).shape}")
		lo, hi = float(arr[0]), float(arr[1])
		if lo < 0.0 or hi < 0.0:
			raise ValueError(f"Standard deviation range values must be >= 0, got [{lo}, {hi}]")
		if lo > hi:
			raise ValueError(f"Expected std range [min, max], got [{lo}, {hi}]")
		return (lo, hi)

	def _sample_model_sds(self):
		self.accel_noise_std = self._sample_std(self.accel_noise_std_range)
		self.gyro_noise_std = self._sample_std(self.gyro_noise_std_range)
		self.accel_bias_std = self._sample_std(self.accel_bias_std_range)
		self.gyro_bias_std = self._sample_std(self.gyro_bias_std_range)

	def _sample_std(self, std_range):
		lo, hi = std_range
		return float(self._rng.uniform(lo, hi))

	def _update_random_walk_bias(self, dt):
		dt = float(dt)
		if dt <= 0.0:
			return
		sqrt_dt = np.sqrt(dt)
		if self.accel_bias_std > 0.0:
			self.accel_bias += self._rng.normal(0.0, self.accel_bias_std * sqrt_dt, size=3)
		if self.gyro_bias_std > 0.0:
			self.gyro_bias += self._rng.normal(0.0, self.gyro_bias_std * sqrt_dt, size=3)
