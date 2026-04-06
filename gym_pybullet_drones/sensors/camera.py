import numpy as np
import pybullet as p


class CameraSensor:
	"""PyBullet camera sensor that stores the latest rendered images.

	The camera projection is built from pinhole intrinsics (fx, fy, cx, cy).
	Call `update(position, orientation_xyzw)` every simulation step to refresh
	the internally stored RGB/depth/segmentation frames.
	"""

	def __init__(
		self,
		width,
		height,
		fx,
		fy,
		cx,
		cy,
		near=0.01,
		far=1000.0,
		client_id=0,
		renderer=None,
		control_freq=None,
		fps=None,
	):
		self.width = int(width)
		self.height = int(height)
		self.fx = float(fx)
		self.fy = float(fy)
		self.cx = float(cx)
		self.cy = float(cy)
		self.near = float(near)
		self.far = float(far)
		self.client_id = int(client_id)
		self.renderer = p.ER_TINY_RENDERER if renderer is None else renderer
		self.control_freq = None if control_freq is None else float(control_freq)
		self.fps = None if fps is None else float(fps)

		if self.width <= 0 or self.height <= 0:
			raise ValueError("width and height must be positive")
		if self.fx <= 0.0 or self.fy <= 0.0:
			raise ValueError("fx and fy must be positive")
		if self.near <= 0.0 or self.far <= self.near:
			raise ValueError("near must be > 0 and far must be > near")
		if self.fps is not None and self.fps <= 0.0:
			raise ValueError("fps must be positive")
		if self.control_freq is not None and self.control_freq <= 0.0:
			raise ValueError("control_freq must be positive")
		if (self.fps is None) != (self.control_freq is None):
			raise ValueError("control_freq and fps must be provided together")

		self._projection_matrix = self._build_projection_matrix()
		self._view_matrix = np.eye(4, dtype=float).reshape(-1).tolist()
		self._capture_interval_steps = 1
		self._update_counter = 0
		self.new_frame_captured = False
		if self.fps is not None and self.control_freq is not None:
			self._capture_interval_steps = max(1, int(round(self.control_freq / self.fps)))

		self.rgb = np.zeros((self.height, self.width, 3), dtype=np.uint8)
		self.depth = np.zeros((self.height, self.width), dtype=np.float32)
		self.segmentation = np.zeros((self.height, self.width), dtype=np.int32)

	def update(self, position, orientation_xyzw):
		"""Capture and store a new frame from the provided camera pose.

		Parameters
		----------
		position : array-like, shape (3,)
			Camera world position.
		orientation_xyzw : array-like, shape (4,)
			Camera orientation quaternion in PyBullet format [x, y, z, w].
		"""
		self._update_counter += 1
		if ((self._update_counter - 1) % self._capture_interval_steps) != 0:
			self.new_frame_captured = False
			return self.rgb, self.depth, self.segmentation

		cam_pos = self._as_vec3(position)
		quat = np.asarray(orientation_xyzw, dtype=float).reshape(-1)
		if quat.size != 4:
			raise ValueError(
				f"Expected quaternion [x, y, z, w], got shape {np.asarray(orientation_xyzw).shape}"
			)

		rot = np.array(p.getMatrixFromQuaternion(quat), dtype=float).reshape(3, 3)

		# Camera forward axis is +X and up axis is +Z in local camera frame.
		forward = rot @ np.array([.2, 0.0, 0.0], dtype=float)
		up = rot @ np.array([0.0, 0.0, .2], dtype=float)
		target = cam_pos + forward

		self._view_matrix = p.computeViewMatrix(
			cameraEyePosition=cam_pos.tolist(),
			cameraTargetPosition=target.tolist(),
			cameraUpVector=up.tolist(),
		)

		_, _, rgba, depth, seg = p.getCameraImage(
			width=self.width,
			height=self.height,
			viewMatrix=self._view_matrix,
			projectionMatrix=self._projection_matrix,
			renderer=self.renderer,
			physicsClientId=self.client_id,
		)

		rgba = np.asarray(rgba, dtype=np.uint8).reshape(self.height, self.width, 4)
		self.rgb = rgba[:, :, :3]
		self.depth = np.asarray(depth, dtype=np.float32).reshape(self.height, self.width)
		self.segmentation = np.asarray(seg, dtype=np.int32).reshape(self.height, self.width)
		self.new_frame_captured = True

		return self.rgb, self.depth, self.segmentation

	def get_rgb(self):
		"""Return latest RGB frame as uint8 array (H, W, 3)."""
		return self.rgb

	def get_depth(self):
		"""Return latest depth buffer as float32 array (H, W)."""
		return self.depth

	def get_segmentation(self):
		"""Return latest segmentation mask as int32 array (H, W)."""
		return self.segmentation

	def get_frames(self):
		"""Return latest (rgb, depth, segmentation) tuple."""
		return self.rgb, self.depth, self.segmentation

	def _build_projection_matrix(self):
		w = float(self.width)
		h = float(self.height)
		n = self.near
		f = self.far

		proj = np.array(
			[
				[2.0 * self.fx / w, 0.0, (w - 2.0 * self.cx) / w, 0.0],
				[0.0, 2.0 * self.fy / h, (2.0 * self.cy - h) / h, 0.0],
				[0.0, 0.0, (n + f) / (n - f), (2.0 * n * f) / (n - f)],
				[0.0, 0.0, -1.0, 0.0],
			],
			dtype=float,
		)

		# PyBullet expects column-major flattening for projection matrix input.
		return proj.T.reshape(-1).tolist()

	@staticmethod
	def _as_vec3(value):
		arr = np.asarray(value, dtype=float).reshape(-1)
		if arr.size != 3:
			raise ValueError(f"Expected a 3D vector, got shape {np.asarray(value).shape}")
		return arr
