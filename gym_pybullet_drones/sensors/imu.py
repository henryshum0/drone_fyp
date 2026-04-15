from gym_pybullet_drones.sensors.sensor import Sensor
import numpy as np


class IMU(Sensor):
	"""Simple IMU model with additive bias and Gaussian noise.

	The IMU reports body-frame linear acceleration and angular velocity.
	"""

	def __init__(
		self,
		freq,
		pyb_freq,
		client_id,
		accel_noise_std=0.02,
		gyro_noise_std=0.005,
		accel_bias=None,
		gyro_bias=None,
		gravity=9.81,
		rng=None,
	):
		
		super().__init__(
			freq = freq,
			pyb_freq = pyb_freq,
			client_id = client_id,
		)
		self.DT = 1.0 / self.freq
		self.accel_noise_std = float(accel_noise_std)
		self.gyro_noise_std = float(gyro_noise_std)
		self.gravity = float(gravity)

		self.accel_bias = self._as_vec3(accel_bias, default=0.0)
		self.gyro_bias = self._as_vec3(gyro_bias, default=0.0)

		self._rng = rng if rng is not None else np.random.default_rng()
		self._prev_vel_world = np.zeros(3, dtype=float)
		self._has_prev_vel = False

		self.accel_actual = np.zeros(3, dtype=float)
		self.gyro_actual = np.zeros(3, dtype=float)
		self.accel_noisy = np.zeros(3, dtype=float)
		self.gyro_noisy = np.zeros(3, dtype=float)

		self.prev_step_counter = -1


	def set_noise_std(self, accel_noise_std=None, gyro_noise_std=None):
		"""Set noise standard deviations."""
		if accel_noise_std is not None:
			self.accel_noise_std = float(accel_noise_std)
		if gyro_noise_std is not None:
			self.gyro_noise_std = float(gyro_noise_std)

	def set_bias(self, accel_bias=None, gyro_bias=None):
		"""Set fixed IMU biases (3D vectors)."""
		if accel_bias is not None:
			self.accel_bias = self._as_vec3(accel_bias)
		if gyro_bias is not None:
			self.gyro_bias = self._as_vec3(gyro_bias)

	def read_actual(self, true_linear_acc_body, true_angular_velocity_body, gravity_body=None):
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

	def read_noisy(self, true_linear_acc_body, true_angular_velocity_body, gravity_body=None):
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
		accel_actual, gyro_actual = self.read_actual(
			true_linear_acc_body=true_linear_acc_body,
			true_angular_velocity_body=true_angular_velocity_body,
			gravity_body=gravity_body,
		)

		accel_noise = self._rng.normal(0.0, self.accel_noise_std, size=3)
		gyro_noise = self._rng.normal(0.0, self.gyro_noise_std, size=3)

		accel_meas = accel_actual + self.accel_bias + accel_noise
		gyro_meas = gyro_actual + self.gyro_bias + gyro_noise
		return accel_meas, gyro_meas

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
		if (not super().should_update(step_counter)):
			return
		vel_world = self._as_vec3(vel_world)
		ang_vel_world = self._as_vec3(ang_vel_world)
		dt = self.DT
		if dt <= 0.0:
			raise ValueError(f"dt must be positive, got {dt}")

		if self._has_prev_vel:
			acc_world = (vel_world - self._prev_vel_world) / (dt * (step_counter - self.prev_step_counter))
		else:
			acc_world = np.zeros(3, dtype=float)

		self._prev_vel_world = np.array(vel_world, copy=True)
		self._has_prev_vel = True

		rot_mat = self._quat_xyzw_to_rotmat(quat_xyzw)
		acc_body = rot_mat.T @ acc_world
		gyro_body = rot_mat.T @ ang_vel_world
		gravity_world = np.array([0.0, 0.0, self.gravity], dtype=float)
		gravity_body = rot_mat.T @ gravity_world

		self.accel_actual, self.gyro_actual = self.read_actual(
			acc_body,
			gyro_body,
			gravity_body=gravity_body,
		)
		self.accel_noisy, self.gyro_noisy = self.read_noisy(
			acc_body,
			gyro_body,
			gravity_body=gravity_body,
		)

		self.prev_step_counter = step_counter

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