import numpy as np
import pybullet as p
from gym_pybullet_drones.utils.constants import VEC_X, VEC_Z
from gym_pybullet_drones.sensors.sensor import Sensor

class CameraSensor(Sensor):
	"""PyBullet camera sensor that stores the latest rendered images.

	The camera projection is built from pinhole intrinsics (fx, fy, cx, cy).
	Call `update(position, orientation_xyzw)` every simulation step to refresh
	the internally stored RGB/depth frames.
	"""

	def __init__(
		self,
		width,
		height,
		fov=100,
		near=0.01,
		far=1000.0,
		client_id=0,
		pyb_freq=None,
		fps=None,
	):
		self.width = int(width)
		self.height = int(height)
		self.fov = float(fov)
		self.near = float(near)
		self.far = float(far)
		self.fps = None if fps is None else int(fps)

		if self.width <= 0 or self.height <= 0:
			raise ValueError("width and height must be positive")
		if self.near <= 0.0 or self.far <= self.near:
			raise ValueError("near must be > 0 and far must be > near")
		if self.fps is None or self.fps <= 0:
			raise ValueError("fps must be positive integer")

		super().__init__(
			freq = self.fps,
			pyb_freq = pyb_freq,
			client_id = client_id,
		)

		self._projection_matrix = self._build_projection_matrix()
		self._view_matrix = np.eye(4, dtype=float).reshape(-1).tolist()
		
		self.rgb = np.zeros((self.height, self.width, 3), dtype=np.uint8)
		# self.depth = np.zeros((self.height, self.width), dtype=np.float32)


	def update(self, position, orientation_xyzw, step_counter):
		"""Capture and store a new frame from the provided camera pose.

		Parameters
		----------
		position : array-like, shape (3,)
			Camera world position.
		orientation_xyzw : array-like, shape (4,)
			Camera orientation quaternion in PyBullet format [x, y, z, w].

		Notes
		-----
		This method updates the internal frame buffers only. Retrieve frames with
		`get_rgb()`, `get_depth()`, or `get_frames()`.
		"""
		self.new_frame_captured = False
		if not super().should_update(step_counter):
			return

		cam_pos = self._as_vec3(position)
		quat = np.asarray(orientation_xyzw, dtype=float).reshape(-1)
		if quat.size != 4:
			raise ValueError(
				f"Expected quaternion [x, y, z, w], got shape {np.asarray(orientation_xyzw).shape}"
			)

		rot = np.array(p.getMatrixFromQuaternion(quat), dtype=float).reshape(3, 3)

		# Camera forward axis is +X and up axis is +Z in local camera frame.
		forward = rot @ VEC_X
		up = rot @ VEC_Z
		target = cam_pos + forward

		self._view_matrix = p.computeViewMatrix(
			cameraEyePosition=cam_pos.tolist(),
			cameraTargetPosition=target.tolist(),
			cameraUpVector=up.tolist(),
		)

		_, _, rgba, _, _ = p.getCameraImage(
			width=self.width,
			height=self.height,
			viewMatrix=self._view_matrix,
			projectionMatrix=self._projection_matrix,
			renderer=p.ER_BULLET_HARDWARE_OPENGL,
			flags=p.ER_NO_SEGMENTATION_MASK,
			physicsClientId=self.client_id,
		)

		rgba = np.asarray(rgba, dtype=np.uint8).reshape(self.height, self.width, 4)
		self.rgb = rgba[:, :, :3]
		# self.depth = np.asarray(depth, dtype=np.float32).reshape(self.height, self.width)

		self.new_frame_captured = True
		self.timestamp = step_counter
		
	def get_timestamp(self):
		"""Return timestamp of the latest captured frame in seconds."""
		return self.timestamp
	
	def get_rgb(self):
		"""Return latest RGB frame as uint8 array (H, W, 3)."""
		return self.rgb

	def get_depth(self):
		"""Return latest depth buffer as float32 array (H, W)."""
		return self.depth

	# def get_frames(self):
	# 	"""Return latest (rgb, depth) tuple."""
	# 	return self.rgb, self.depth

	def _build_projection_matrix(self):
		proj = p.computeProjectionMatrixFOV(
			fov=self.fov,
			aspect=float(self.width) / float(self.height),
			nearVal=self.near,
			farVal=self.far,
		)

		return proj

	@staticmethod
	def _as_vec3(value):
		arr = np.asarray(value, dtype=float).reshape(-1)
		if arr.size != 3:
			raise ValueError(f"Expected a 3D vector, got shape {np.asarray(value).shape}")
		return arr
