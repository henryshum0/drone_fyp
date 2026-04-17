import numpy as np
import pybullet as p
from gymnasium import spaces
import importlib

from gym_pybullet_drones.control.CustomCTBRControl import CTBRPIDControl
from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.sensori_agent.trajectory_optimize import optimize_trj_time


class MPCControlEnv(BaseAviary):
	"""Single-drone trajectory-tracking environment with internal MPC.

	The environment is not intended for RL. `step()` ignores external actions,
	solves an MPC problem at 50Hz, and applies the first control in a receding
	horizon loop.
	"""

	def __init__(
		self,
		drone_model: DroneModel = DroneModel.RACE,
		physics: Physics = Physics.PYB,
		pyb_freq: int = 500,
		ctrl_freq: int = 500,
		mpc_freq: int = 50,
		gui: bool = False,
		record: bool = False,
		obstacles: bool = False,
		output_folder: str = "results",
		episode_len_sec: float = 10.0,
		horizon: int = 50,
		max_thrust: float = 60.0,
		max_vel: float = 20.0,
		max_roll_pitch_rate: float = 5.0 * np.pi,
		max_yaw_rate: float = 2.0 * np.pi,
		qx_weights=None,
		r_weights=None,
		rd_weights=None,
		mpc_backend: str = "casadi",
		ipopt_max_iter: int = 10,
	):
		if ctrl_freq != 50:
			raise ValueError("MPCControlEnv requires ctrl_freq=50 for 50Hz trajectory sampling")
		if pyb_freq % ctrl_freq != 0:
			raise ValueError("pyb_freq must be divisible by ctrl_freq")
		self.MPC_FREQ = mpc_freq
        
		self.EPISODE_LEN_SEC = float(episode_len_sec)
		self.HORIZON = int(horizon)
		self.MPC_DT = 1.0 / float(ctrl_freq)
		self.HORIZON_TIME_SEC = self.HORIZON * self.MPC_DT
		self.MASS_NORMALIZED_THRUST_BOUNDS = (0.0, float(max_thrust))
		self.BODY_RATE_BOUNDS = np.array(
			[
				[-float(max_roll_pitch_rate), float(max_roll_pitch_rate)],
				[-float(max_roll_pitch_rate), float(max_roll_pitch_rate)],
				[-float(max_yaw_rate), float(max_yaw_rate)],
			],
			dtype=float,
		)
		self.VEL_BOUNDS = np.array(
			[
				[-float(max_vel), float(max_vel)],
				[-float(max_vel), float(max_vel)],
				[-float(max_vel), float(max_vel)],
			],
			dtype=float,
		)

		self.STATE_DIM = 13  # [p(3), q_xyzw(4), v(3), w_body(3)]
		self.OUTPUT_DIM = 4  # [collective_thrust, p_rate, q_rate, r_rate]
		self.CTRL_DIM = 4  # [collective_thrust, p_cmd, q_cmd, r_cmd]
		self.OBS_DIM = self.STATE_DIM + self.STATE_DIM + self.CTRL_DIM + 1

		self.Qx = np.diag(
			np.array(
				qx_weights
				if qx_weights is not None
				else [
					2.0,
					2.0,
					2.0,
					2.0,
					2.0,
					2.0,
					2.0,
					0.,
					0.,
					0.,
					0.15,
					0.15,
					0.15,
				],
				dtype=float,
			)
		)
		self.R = np.diag(np.array(r_weights if r_weights is not None else [0.0, 0.0, 0.0, 0.], dtype=float))
		self.Rd = np.diag(np.array(rd_weights if rd_weights is not None else [0.001, 0.01, 0.01, 0.01], dtype=float))
		self.MPC_BACKEND = str(mpc_backend).lower()
		self.IPOPT_MAX_ITER = int(ipopt_max_iter)

		self.reference_trajectory = np.zeros((1, self.STATE_DIM), dtype=float)
		self.reference_dt = self.MPC_DT
		self.ref_step = 0
		self._last_mpc_cost = 0.0
		self._last_status = "not_solved"
		self._last_iterations = 0
		self._last_constraint_violation = 0.0
		self._last_success = False

		self._u_prev = np.array([9.8, 0.0, 0.0, 0.0], dtype=float)
		self._u_warm = np.tile(self._u_prev, (self.HORIZON, 1))
		self._last_rpm = np.zeros(4, dtype=float)
		self._casadi_solver = None
		self._acados_solver = None
		self._cached_lb = None
		self._cached_ub = None

		super().__init__(
			drone_model=drone_model,
			num_drones=1,
			neighbourhood_radius=np.inf,
			initial_rpys=None,
			physics=physics,
			pyb_freq=pyb_freq,
			ctrl_freq=ctrl_freq,
			gui=gui,
			record=record,
			obstacles=obstacles,
			user_debug_gui=gui,
			vision_attributes=False,
			output_folder=output_folder,
			compute_returns_per_step=False,
			ground_plane=True,
		)

		self.ctbr_controller = CTBRPIDControl(drone_model=drone_model, ctrl_freq=ctrl_freq, g=self.G)
		self._init_mpc_backend()

	def _actionSpace(self):
		# Internal-MPC env ignores external action. Keep a dummy box for Gym compatibility.
		return spaces.Box(low=np.array([0.0], dtype=np.float32), high=np.array([0.0], dtype=np.float32), dtype=np.float32)

	def _observationSpace(self):
		low = -np.inf * np.ones(self.OBS_DIM, dtype=np.float32)
		high = np.inf * np.ones(self.OBS_DIM, dtype=np.float32)
		return spaces.Box(low=low, high=high, dtype=np.float32)

	def reset(self, seed=None, options=None):
		self._load_reference_from_options(options)
		self.ref_step = 0
		self._u_prev = np.array([0, 0.0, 0.0, 0.0], dtype=float)
		self._u_warm = np.tile(self._u_prev, (self.HORIZON, 1))
		self._last_mpc_cost = 0.0
		self._last_status = "reset"
		self._last_iterations = 0
		self._last_constraint_violation = 0.0
		self._last_success = False
		self._last_rpm = np.zeros(4, dtype=float)

		self.INIT_XYZS[0] = self.reference_trajectory[0, 0:3].copy()
		super().reset(seed=seed, options=options)
		return self._computeObs(), self._computeInfo()

	def step(self, action=None):
		_ = action
		for _ in range(int(self.CTRL_FREQ // self.MPC_FREQ)):
			super().step(np.zeros(1, dtype=np.float32))
		obs = self._computeObs()
		reward = self._computeReward()
		terminated = self._computeTerminated()
		truncated = self._computeTruncated()
		info = self._computeInfo()
		self.ref_step += 1
		return obs, reward, terminated, truncated, info

	def _preprocessAction(self, action):
		_ = action
		x0 = self._get_current_state()
		u_cmd, mpc_info = self._solve_mpc(x0)

		thrust = float(np.clip(u_cmd[0], self.MASS_NORMALIZED_THRUST_BOUNDS[0], self.MASS_NORMALIZED_THRUST_BOUNDS[1]))
		target_body_rate = np.clip(
			u_cmd[1:4],
			self.BODY_RATE_BOUNDS[:, 0],
			self.BODY_RATE_BOUNDS[:, 1],
		)

		cur_body_rate = x0[10:13]
		rpm = self.ctbr_controller.computeControl(
			control_timestep=self.CTRL_TIMESTEP,
			thrust=thrust,
			cur_body_rate=cur_body_rate,
			target_body_rate=target_body_rate,
		)
		self._last_rpm = rpm.copy()

		self._u_prev = np.array([thrust, target_body_rate[0], target_body_rate[1], target_body_rate[2]], dtype=float)
		self._last_mpc_cost = float(mpc_info["cost"])
		self._last_status = str(mpc_info["status"])
		self._last_iterations = int(mpc_info["nit"])
		self._last_constraint_violation = float(mpc_info["constraint_violation"])
		self._last_success = bool(mpc_info["success"])

		return rpm.reshape(1, 4)

	def _computeObs(self):
		x = self._get_current_state()
		x_ref = self._get_ref_state(self.ref_step)
		obs = np.hstack([x, x_ref, self._u_prev, np.array([self._last_mpc_cost], dtype=float)])
		return obs.astype(np.float32)

	def _computeReward(self):
		# Non-RL environment: reward is intentionally non-driving.
		return 0.0

	def _computeTerminated(self):
		return self._check_ground_collision()

	def _computeTruncated(self):
		timeout = (self.ref_step * self.MPC_DT) >= self.EPISODE_LEN_SEC
		ref_finished = self.ref_step >= (len(self.reference_trajectory) - 1)
		return bool(timeout or ref_finished)

	def _computeInfo(self):
		x = self._get_current_state()
		x_ref = self._get_ref_state(self.ref_step)
		y = self._u_prev.copy()
		y_ref = self._get_output_ref(x_ref)
		return {
			"mpc_success": self._last_success,
			"mpc_status": self._last_status,
			"mpc_iterations": self._last_iterations,
			"mpc_cost": self._last_mpc_cost,
			"constraint_violation": self._last_constraint_violation,
			"ref_step": self.ref_step,
			"state_error_norm": float(np.linalg.norm(x - x_ref)),
			"output_error_norm": float(np.linalg.norm(y - y_ref)),
			"applied_u": self._u_prev.copy(),
			"output_ref": y_ref.copy(),
			"applied_rpm": self._last_rpm.copy(),
		}

	def _load_reference_from_options(self, options):
		if options is None or "trajectory" not in options:
			self.reference_trajectory = self._default_hover_reference()
			self.reference_dt = self.MPC_DT
			return

		traj = np.asarray(options["trajectory"], dtype=float)
		if traj.ndim != 2:
			raise ValueError("trajectory must be a 2D numpy array")

		self.reference_dt = float(options.get("trajectory_dt", self.MPC_DT))
		if abs(self.reference_dt - self.MPC_DT) > 1e-9:
			raise ValueError("trajectory_dt must be 0.02 seconds for 50Hz")

		# Supported trajectory layouts:
		# - [p(3), q(4), v(3), w(3)] => 13 columns
		# - [p(3), q(4), v(3)] => 10 columns (w filled with zeros)
		# - [t, p(3), q(4), v(3), w(3)] => 14 columns (t ignored)
		# - [t, p(3), q(4), v(3)] => 11 columns (t ignored, w zeros)
		if traj.shape[1] == 13:
			x_ref = traj
		elif traj.shape[1] == 10:
			x_ref = np.hstack([traj, np.zeros((traj.shape[0], 3), dtype=float)])
		elif traj.shape[1] == 14:
			x_ref = traj[:, 1:14]
		elif traj.shape[1] == 11:
			x_ref = np.hstack([traj[:, 1:11], np.zeros((traj.shape[0], 3), dtype=float)])
		else:
			raise ValueError("Unsupported trajectory width. Expected 10, 11, 13, or 14 columns")

		x_ref[:, 3:7] = np.array([self._quat_normalize(q) for q in x_ref[:, 3:7]])
		x_ref[:, 7:10] = np.clip(x_ref[:, 7:10], self.VEL_BOUNDS[:, 0], self.VEL_BOUNDS[:, 1])
		x_ref[:, 10:13] = np.clip(x_ref[:, 10:13], self.BODY_RATE_BOUNDS[:, 0], self.BODY_RATE_BOUNDS[:, 1])
		self.reference_trajectory = x_ref.astype(float)

	def _default_hover_reference(self):
		n = max(2, int(self.EPISODE_LEN_SEC * 50.0))
		ref = np.zeros((n, self.STATE_DIM), dtype=float)
		ref[:, 2] = 1.0
		ref[:, 3:7] = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
		return ref

	def _get_ref_state(self, k):
		idx = int(np.clip(k, 0, len(self.reference_trajectory) - 1))
		return self.reference_trajectory[idx].copy()

	def _get_horizon_ref_state(self, pred_idx):
		"""Reference state for prediction step pred_idx in the future window.

		pred_idx=0 corresponds to t+dt, and pred_idx=H-1 corresponds to t+T.
		"""
		return self._get_ref_state(self.ref_step + int(pred_idx) + 1)

	def _get_current_state(self):
		state = self._getDroneStateVector(0)
		pos = state[0:3].astype(float)
		quat_xyzw = self._quat_normalize(state[3:7].astype(float))
		vel_world = state[10:13].astype(float)
		ang_vel_world = state[13:16].astype(float)
		ang_vel_body = self._world_to_body(ang_vel_world, quat_xyzw)
		return np.hstack([pos, quat_xyzw, vel_world, ang_vel_body])

	def _get_output_ref(self, x_ref):
		_ = x_ref
		# Output tracking reference for [thrust, p, q, r].
		thrust_ref = float(np.clip(self.G, self.MASS_NORMALIZED_THRUST_BOUNDS[0], self.MASS_NORMALIZED_THRUST_BOUNDS[1]))
		return np.array([thrust_ref, 0.0, 0.0, 0.0], dtype=float)

	def _solve_mpc(self, x0):
		if self.MPC_BACKEND == "acados" and self._acados_solver is not None:
			return self._solve_mpc_acados(x0)
		return self._solve_mpc_casadi(x0)

	def _init_mpc_backend(self):
		if self.MPC_BACKEND not in ("casadi", "acados"):
			raise ValueError("mpc_backend must be either 'casadi' or 'acados'")

		if self.MPC_BACKEND == "acados":
			self._acados_solver = self._try_build_acados_solver()
			if self._acados_solver is None:
				self._last_status = "acados_unavailable_fallback_casadi"

		self._build_casadi_solver()

	def _build_casadi_solver(self):
		ca = self._get_casadi_module()
		nu = self.HORIZON * self.CTRL_DIM
		u_var = ca.SX.sym("u", nu)

		# Parameters pack: [x0(13), x_ref[0:H](13 each), u_prev(4)]
		nx0 = self.STATE_DIM
		nxref = self.HORIZON * self.STATE_DIM
		nup = self.CTRL_DIM
		p_dim = nx0 + nxref + nup
		p_var = ca.SX.sym("p", p_dim)

		x0 = p_var[0:nx0]
		xref_flat = p_var[nx0:nx0 + nxref]
		u_prev = p_var[nx0 + nxref:nx0 + nxref + nup]

		x = x0
		cost = 0
		for i in range(self.HORIZON):
			u_i = u_var[i * self.CTRL_DIM:(i + 1) * self.CTRL_DIM]
			x_ref_i = xref_flat[i * self.STATE_DIM:(i + 1) * self.STATE_DIM]

			x = self._predict_dynamics_ca(x, u_i)

			dx = x - x_ref_i
			if i == 0:
				du = u_i - u_prev
			else:
				u_prev_i = u_var[(i - 1) * self.CTRL_DIM:i * self.CTRL_DIM]
				du = u_i - u_prev_i

			cost += ca.mtimes([dx.T, ca.DM(self.Qx), dx])
			cost += ca.mtimes([u_i.T, ca.DM(self.R), u_i])
			cost += ca.mtimes([du.T, ca.DM(self.Rd), du])

		nlp = {"x": u_var, "p": p_var, "f": cost}
		opts = {
			"ipopt.print_level": 0,
			"print_time": False,
			"ipopt.max_iter": self.IPOPT_MAX_ITER,
			# "ipopt.tol": 1e-4,
			# "ipopt.acceptable_tol": 5e-3,
			# "ipopt.acceptable_iter": 5,
			"ipopt.sb": "yes",
		}
		self._casadi_solver = ca.nlpsol("mpc_solver", "ipopt", nlp, opts)

		lb = []
		ub = []
		for _ in range(self.HORIZON):
			lb.append(self.MASS_NORMALIZED_THRUST_BOUNDS[0])
			ub.append(self.MASS_NORMALIZED_THRUST_BOUNDS[1])
			lb.extend(self.BODY_RATE_BOUNDS[:, 0].tolist())
			ub.extend(self.BODY_RATE_BOUNDS[:, 1].tolist())
		self._cached_lb = np.array(lb, dtype=float)
		self._cached_ub = np.array(ub, dtype=float)

	def _build_casadi_param_vector(self, x0):
		x_ref_list = []
		for i in range(self.HORIZON):
			x_ref_i = self._get_horizon_ref_state(i)
			x_ref_list.append(x_ref_i)

		x_ref_flat = np.concatenate(x_ref_list)
		return np.concatenate([x0, x_ref_flat, self._u_prev])

	def _solve_mpc_casadi(self, x0):
		_ = self._get_casadi_module()
		params = self._build_casadi_param_vector(x0)
		z0 = self._u_warm.reshape(-1)

		try:
			sol = self._casadi_solver(
				x0=z0,
				p=params,
				lbx=self._cached_lb,
				ubx=self._cached_ub,
			)
			u_opt = np.array(sol["x"]).reshape(-1)
			u_seq = u_opt.reshape(self.HORIZON, self.CTRL_DIM)
			u0 = u_seq[0]
			self._u_warm = np.vstack([u_seq[1:], u_seq[-1:]])
			status = self._casadi_solver.stats().get("return_status", "unknown")
			nit = int(self._casadi_solver.stats().get("iter_count", 0))
			success = "success" in str(status).lower() or "solve_succeeded" in str(status).lower()
			cost = float(sol["f"])
		except Exception as exc:
			u0 = self._u_prev.copy()
			self._u_warm = np.tile(u0, (self.HORIZON, 1))
			status = f"casadi_exception:{type(exc).__name__}"
			nit = 0
			success = False
			cost = float(self._rollout_cost(x0, self._u_warm))

		constraint_violation = self._estimate_constraint_violation(x0, self._u_warm)
		return u0, {
			"success": bool(success),
			"status": str(status),
			"nit": int(nit),
			"cost": float(cost),
			"constraint_violation": float(constraint_violation),
		}

	def _try_build_acados_solver(self):
		# Optional hook for acados migration. If acados is unavailable, we keep CasADi path active.
		try:
			importlib.import_module("acados_template")
		except Exception:
			return None
		return None

	@staticmethod
	def _get_casadi_module():
		try:
			return importlib.import_module("casadi")
		except Exception as exc:
			raise RuntimeError("casadi is required for MPC backend. Install with 'pip install casadi'.") from exc

	def _solve_mpc_acados(self, x0):
		# Placeholder path until acados model/export is wired; fallback keeps environment operational.
		_ = x0
		return self._solve_mpc_casadi(x0)

	def _rollout_cost(self, x0, u_seq):
		x = x0.copy()
		cost = 0.0
		u_prev = self._u_prev.copy()

		for i in range(self.HORIZON):
			x_ref = self._get_horizon_ref_state(i)

			u = u_seq[i]
			x = self._predict_dynamics(x, u)

			dx = x - x_ref
			du = u - u_prev

			cost += float(dx.T @ self.Qx @ dx)
			cost += float(u.T @ self.R @ u)
			cost += float(du.T @ self.Rd @ du)

			u_prev = u

		return cost

	def _predict_dynamics(self, x, u):
		pos = x[0:3]
		quat = x[3:7]
		vel = x[7:10]

		thrust = float(np.clip(u[0], self.MASS_NORMALIZED_THRUST_BOUNDS[0], self.MASS_NORMALIZED_THRUST_BOUNDS[1]))
		w_next = np.clip(u[1:4], self.BODY_RATE_BOUNDS[:, 0], self.BODY_RATE_BOUNDS[:, 1])

		q_dot = self._quat_derivative_xyzw(quat, w_next)
		quat_next = self._quat_normalize(quat + self.MPC_DT * q_dot)

		thrust_world = self._quat_to_rotmat(quat_next) @ np.array([0.0, 0.0, thrust], dtype=float)
		acc_world = thrust_world - np.array([0.0, 0.0, self.G], dtype=float)

		vel_next = vel + self.MPC_DT * acc_world
		vel_next = np.clip(vel_next, self.VEL_BOUNDS[:, 0], self.VEL_BOUNDS[:, 1])

		pos_next = pos + self.MPC_DT * vel_next

		return np.hstack([pos_next, quat_next, vel_next, w_next])

	def _state_to_output(self, x):
		pos = x[0:3]
		vel = x[7:10]
		yaw = self._quat_to_yaw(x[3:7])
		return np.hstack([pos, vel, np.array([yaw], dtype=float)])

	def _predict_dynamics_ca(self, x, u):
		ca = self._get_casadi_module()
		pos = x[0:3]
		quat = x[3:7]
		vel = x[7:10]

		thrust = ca.fmin(ca.fmax(u[0], self.MASS_NORMALIZED_THRUST_BOUNDS[0]), self.MASS_NORMALIZED_THRUST_BOUNDS[1])
		w_next = ca.vertcat(
			ca.fmin(ca.fmax(u[1], self.BODY_RATE_BOUNDS[0, 0]), self.BODY_RATE_BOUNDS[0, 1]),
			ca.fmin(ca.fmax(u[2], self.BODY_RATE_BOUNDS[1, 0]), self.BODY_RATE_BOUNDS[1, 1]),
			ca.fmin(ca.fmax(u[3], self.BODY_RATE_BOUNDS[2, 0]), self.BODY_RATE_BOUNDS[2, 1]),
		)

		q_dot = self._quat_derivative_ca(quat, w_next)
		quat_next = quat + self.MPC_DT * q_dot
		quat_next = quat_next / (ca.sqrt(ca.sumsqr(quat_next)) + 1e-9)

		rot = self._quat_to_rotmat_ca(quat_next)
		thrust_world = ca.mtimes(rot, ca.vertcat(0.0, 0.0, thrust))
		acc_world = thrust_world - ca.vertcat(0.0, 0.0, self.G)

		vel_next = vel + self.MPC_DT * acc_world
		vel_next = ca.vertcat(
			ca.fmin(ca.fmax(vel_next[0], self.VEL_BOUNDS[0, 0]), self.VEL_BOUNDS[0, 1]),
			ca.fmin(ca.fmax(vel_next[1], self.VEL_BOUNDS[1, 0]), self.VEL_BOUNDS[1, 1]),
			ca.fmin(ca.fmax(vel_next[2], self.VEL_BOUNDS[2, 0]), self.VEL_BOUNDS[2, 1]),
		)

		pos_next = pos + self.MPC_DT * vel_next
		return ca.vertcat(pos_next, quat_next, vel_next, w_next)

	@classmethod
	def _quat_derivative_ca(cls, q, w_body):
		ca = cls._get_casadi_module()
		x = q[0]
		y = q[1]
		z = q[2]
		w = q[3]
		p_rate = w_body[0]
		q_rate = w_body[1]
		r_rate = w_body[2]
		return 0.5 * ca.vertcat(
			w * p_rate + y * r_rate - z * q_rate,
			w * q_rate + z * p_rate - x * r_rate,
			w * r_rate + x * q_rate - y * p_rate,
			-x * p_rate - y * q_rate - z * r_rate,
		)

	@classmethod
	def _quat_to_rotmat_ca(cls, quat_xyzw):
		ca = cls._get_casadi_module()
		x = quat_xyzw[0]
		y = quat_xyzw[1]
		z = quat_xyzw[2]
		w = quat_xyzw[3]
		r11 = 1.0 - 2.0 * (y * y + z * z)
		r12 = 2.0 * (x * y - z * w)
		r13 = 2.0 * (x * z + y * w)
		r21 = 2.0 * (x * y + z * w)
		r22 = 1.0 - 2.0 * (x * x + z * z)
		r23 = 2.0 * (y * z - x * w)
		r31 = 2.0 * (x * z - y * w)
		r32 = 2.0 * (y * z + x * w)
		r33 = 1.0 - 2.0 * (x * x + y * y)
		return ca.vertcat(
			ca.horzcat(r11, r12, r13),
			ca.horzcat(r21, r22, r23),
			ca.horzcat(r31, r32, r33),
		)

	def _state_to_output_ca(self, x):
		ca = self._get_casadi_module()
		pos = x[0:3]
		vel = x[7:10]
		yaw = self._quat_to_yaw_ca(x[3:7])
		return ca.vertcat(pos, vel, yaw)

	@classmethod
	def _quat_to_yaw_ca(cls, q):
		ca = cls._get_casadi_module()
		x = q[0]
		y = q[1]
		z = q[2]
		w = q[3]
		siny_cosp = 2.0 * (w * z + x * y)
		cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
		return ca.atan2(siny_cosp, cosy_cosp)

	def _estimate_constraint_violation(self, x0, u_seq):
		x = x0.copy()
		max_violation = 0.0
		for i in range(self.HORIZON):
			u = u_seq[i]
			thrust_v = max(0.0, self.MASS_NORMALIZED_THRUST_BOUNDS[0] - u[0], u[0] - self.MASS_NORMALIZED_THRUST_BOUNDS[1])
			rate_low = self.BODY_RATE_BOUNDS[:, 0] - u[1:4]
			rate_high = u[1:4] - self.BODY_RATE_BOUNDS[:, 1]
			rate_v = np.max(np.maximum(0.0, np.maximum(rate_low, rate_high)))

			x = self._predict_dynamics(x, u)
			vel_low = self.VEL_BOUNDS[:, 0] - x[7:10]
			vel_high = x[7:10] - self.VEL_BOUNDS[:, 1]
			vel_v = np.max(np.maximum(0.0, np.maximum(vel_low, vel_high)))

			max_violation = max(max_violation, thrust_v, rate_v, vel_v)
		return float(max_violation)

	def _check_ground_collision(self):
		return False
		if not hasattr(self, "DRONE_IDS"):
			return False
		if hasattr(self, "PLANE_ID"):
			contacts = p.getContactPoints(
				bodyA=int(self.DRONE_IDS[0]),
				bodyB=int(self.PLANE_ID),
				physicsClientId=self.CLIENT,
			)
			if len(contacts) > 0:
				return True

		# Fallback z-based collision check if no plane id is present.
		return bool(self.pos[0, 2] <= 0.02)

	@staticmethod
	def _quat_normalize(q):
		n = np.linalg.norm(q)
		if n < 1e-9:
			return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
		return q / n

	@staticmethod
	def _quat_to_rotmat(quat_xyzw):
		return np.array(p.getMatrixFromQuaternion(quat_xyzw), dtype=float).reshape(3, 3)

	@staticmethod
	def _quat_derivative_xyzw(q, w_body):
		x, y, z, w = q
		p_rate, q_rate, r_rate = w_body
		return 0.5 * np.array(
			[
				w * p_rate + y * r_rate - z * q_rate,
				w * q_rate + z * p_rate - x * r_rate,
				w * r_rate + x * q_rate - y * p_rate,
				-x * p_rate - y * q_rate - z * r_rate,
			],
			dtype=float,
		)

	@staticmethod
	def _world_to_body(v_world, quat_xyzw):
		rot = np.array(p.getMatrixFromQuaternion(quat_xyzw), dtype=float).reshape(3, 3)
		return rot.T @ v_world

	@staticmethod
	def _quat_to_yaw(quat_xyzw):
		return float(p.getEulerFromQuaternion(quat_xyzw)[2])


def _build_demo_trajectory(duration_sec=8.0, dt=0.02):
	"""Create a demo reference trajectory sampled at 50Hz from BackRollTemplate.

	Falls back to the old synthetic figure-8 if template generation fails.
	"""
	from gym_pybullet_drones.gateRL.waypoints.acro_templates import\
		BackRollTemplate, FrontRollTemplate, \
		SplitSLeftTemplate, SplitSRightTemplate, \
		BarrelRollRightTemplate, BarrelRollLeftTemplate

	from gym_pybullet_drones.sensori_agent.trajectory_generation import build_trajectory_from_template
	template = BackRollTemplate()
	traj_obj = build_trajectory_from_template(template, randomized=True)
	traj_obj, optimized_time, min_result = optimize_trj_time(
        traj_obj,
        time_penalty=np.array([100000 for seg in traj_obj._segments]),
        preserve_total_time=False,
        max_velocity=20,
        max_normalized_thrust=60,
        report_peaks=True,
    )
	duration_sec = sum(seg._duration for seg in traj_obj._segments)
	steps = int(round(duration_sec / dt))
	sampled = traj_obj.sample_full_state(sampling_rate=int(round(1.0 / dt)))
	traj = np.zeros((sampled["pos"].shape[0], 13), dtype=float)
	traj[:, 0:3] = sampled["pos"]
	traj[:, 3:7] = sampled["quat"]  # xyzw
	traj[:, 7:10] = sampled["vel"]
	# Use sampled roll/pitch/yaw rates as a practical body-rate proxy for demo.
	traj[:, 10:13] = sampled["rpy_rate"]
	if traj.shape[0] > steps:
		traj = traj[:steps]
	elif traj.shape[0] < steps:
		pad = np.repeat(traj[-1:, :], steps - traj.shape[0], axis=0)
		traj = np.vstack([traj, pad])
	return traj


def _plot_trajectory_pyplot(reference_traj, actual_positions, actual_quats=None, show=True, save_path=None, axis_stride=10, axis_scale=0.2):
	"""Visualize reference vs actual trajectory and their x/z body axes using pyplot."""
	import matplotlib.pyplot as plt

	ref = np.asarray(reference_traj[:, 0:3], dtype=float)
	ref_quat = np.asarray(reference_traj[:, 3:7], dtype=float)
	act = np.asarray(actual_positions, dtype=float)
	if act.ndim != 2 or act.shape[1] != 3:
		raise ValueError("actual_positions must be shape (N, 3)")
	if actual_quats is None:
		raise ValueError("actual_quats must be provided with shape (N, 4)")
	act_quat = np.asarray(actual_quats, dtype=float)
	if act_quat.ndim != 2 or act_quat.shape[1] != 4:
		raise ValueError("actual_quats must be shape (N, 4)")

	min_len = max(1, min(ref.shape[0], act.shape[0], ref_quat.shape[0], act_quat.shape[0]))
	ref = ref[:min_len]
	ref_quat = ref_quat[:min_len]
	act = act[:min_len]
	act_quat = act_quat[:min_len]
	t = np.arange(min_len)
	axis_idx = np.arange(0, min_len, max(1, int(axis_stride)))

	# Compute body x- and z-axes from quaternions.
	ref_x = np.zeros((min_len, 3), dtype=float)
	ref_z = np.zeros((min_len, 3), dtype=float)
	act_x = np.zeros((min_len, 3), dtype=float)
	act_z = np.zeros((min_len, 3), dtype=float)
	for i in range(min_len):
		ref_rot = np.array(p.getMatrixFromQuaternion(ref_quat[i]), dtype=float).reshape(3, 3)
		act_rot = np.array(p.getMatrixFromQuaternion(act_quat[i]), dtype=float).reshape(3, 3)
		ref_x[i], ref_z[i] = ref_rot[:, 0], ref_rot[:, 2]
		act_x[i], act_z[i] = act_rot[:, 0], act_rot[:, 2]

	fig = plt.figure(figsize=(12, 5))
	ax3d = fig.add_subplot(1, 2, 1, projection="3d")
	ax3d.plot(ref[:, 0], ref[:, 1], ref[:, 2], color="tab:orange", linewidth=2.0, label="reference")
	ax3d.plot(act[:, 0], act[:, 1], act[:, 2], color="tab:blue", linewidth=1.6, label="quadrotor")
	ax3d.quiver(
		ref[axis_idx, 0], ref[axis_idx, 1], ref[axis_idx, 2],
		ref_x[axis_idx, 0], ref_x[axis_idx, 1], ref_x[axis_idx, 2],
		length=float(axis_scale), normalize=True, color="tab:cyan", linewidth=1.0,
	)
	ax3d.quiver(
		ref[axis_idx, 0], ref[axis_idx, 1], ref[axis_idx, 2],
		ref_z[axis_idx, 0], ref_z[axis_idx, 1], ref_z[axis_idx, 2],
		length=float(axis_scale), normalize=True, color="tab:purple", linewidth=1.0,
	)
	ax3d.quiver(
		act[axis_idx, 0], act[axis_idx, 1], act[axis_idx, 2],
		act_x[axis_idx, 0], act_x[axis_idx, 1], act_x[axis_idx, 2],
		length=float(axis_scale), normalize=True, color="tab:green", linewidth=1.0,
	)
	ax3d.quiver(
		act[axis_idx, 0], act[axis_idx, 1], act[axis_idx, 2],
		act_z[axis_idx, 0], act_z[axis_idx, 1], act_z[axis_idx, 2],
		length=float(axis_scale), normalize=True, color="tab:red", linewidth=1.0,
	)
	ax3d.scatter(ref[0, 0], ref[0, 1], ref[0, 2], color="tab:green", s=30, label="start")
	ax3d.set_xlabel("x [m]")
	ax3d.set_ylabel("y [m]")
	ax3d.set_zlabel("z [m]")
	# Keep identical axis scale for x/y/z to avoid distorted geometry.
	all_pts = np.vstack((ref, act))
	mins = np.min(all_pts, axis=0)
	maxs = np.max(all_pts, axis=0)
	center = 0.5 * (mins + maxs)
	radius = 0.5 * np.max(maxs - mins)
	radius = max(radius, 1e-3)
	ax3d.set_xlim(center[0] - radius, center[0] + radius)
	ax3d.set_ylim(center[1] - radius, center[1] + radius)
	ax3d.set_zlim(center[2] - radius, center[2] + radius)
	ax3d.set_box_aspect((1, 1, 1))
	ax3d.set_title("3D Trajectory with X/Z Axes")
	ax3d.legend(loc="upper right")

	ax = fig.add_subplot(1, 2, 2)
	ax.plot(t, ref[:, 0], "--", color="tab:red", linewidth=1.2, label="x_ref")
	ax.plot(t, act[:, 0], color="tab:red", linewidth=1.2, label="x")
	ax.plot(t, ref[:, 1], "--", color="tab:green", linewidth=1.2, label="y_ref")
	ax.plot(t, act[:, 1], color="tab:green", linewidth=1.2, label="y")
	ax.plot(t, ref[:, 2], "--", color="tab:blue", linewidth=1.2, label="z_ref")
	ax.plot(t, act[:, 2], color="tab:blue", linewidth=1.2, label="z")
	ax.set_xlabel("sample index")
	ax.set_ylabel("position [m]")
	ax.set_title("Position Tracking")
	ax.grid(True, alpha=0.3)
	ax.legend(ncol=2, fontsize=8)

	plt.tight_layout()
	if save_path is not None and len(str(save_path)) > 0:
		fig.savefig(str(save_path), dpi=160)
		print(f"[DEMO] Saved trajectory plot to {save_path}")
	if show:
		plt.show()
	else:
		plt.close(fig)

def main():
	"""Run a minimal end-to-end MPC demo simulation."""
	import argparse

	parser = argparse.ArgumentParser(description="MPCControlEnv simple demonstration")
	parser.add_argument("--duration-sec", type=float, default=8.0, help="Demo duration in seconds")
	parser.add_argument("--gui", action="store_true", help="Enable PyBullet GUI")
	parser.add_argument("--pyb-freq", type=int, default=500, help="PyBullet frequency")
	parser.add_argument("--ctrl-freq", type=int, default=50, help="Control frequency (must be 50)")
	parser.add_argument("--no-show-plot", action="store_true", help="Do not display pyplot window")
	parser.add_argument("--plot-save-path", type=str, default="", help="Optional path to save trajectory plot image")
	parser.add_argument("--axis-stride", type=int, default=10, help="Plot every N-th axis sample")
	parser.add_argument("--axis-scale", type=float, default=0.2, help="Axis vector scale in meters")
	args = parser.parse_args()

	if args.ctrl_freq != 50:
		raise ValueError("This demo requires --ctrl-freq 50 to match 50Hz trajectory sampling")

	dt = 1.0 / float(args.ctrl_freq)
	traj = _build_demo_trajectory(duration_sec=args.duration_sec, dt=dt)

	env = MPCControlEnv(
		gui=bool(args.gui),
		record=False,
		ctrl_freq=args.ctrl_freq,
		pyb_freq=args.pyb_freq,
		episode_len_sec=args.duration_sec,
	)

	obs, info = env.reset(options={"trajectory": traj, "trajectory_dt": dt})
	_ = obs
	print("[DEMO] Reset complete")
	print(f"[DEMO] initial_status={info['mpc_status']} initial_error={info['state_error_norm']:.3f}")
	# print(f"[DEMO] backend={args.mpc_backend} horizon={args.horizon} ipopt_max_iter={args.ipopt_max_iter}")

	state_errors = []
	output_errors = []
	solver_successes = 0
	actual_positions = [env.pos[0].copy()]
	actual_quats = [env.quat[0].copy()]

	max_steps = len(traj)
	for k in range(max_steps):
		obs, rew, terminated, truncated, info = env.step(None)
		_ = obs, rew
		actual_positions.append(env.pos[0].copy())
		actual_quats.append(env.quat[0].copy())
		state_errors.append(info["state_error_norm"])
		output_errors.append(info["output_error_norm"])
		solver_successes += int(info["mpc_success"])

		if k % 50 == 0:
			print(
				f"[DEMO] step={k:04d} status={info['mpc_status']} "
				f"state_err={info['state_error_norm']:.3f} out_err={info['output_error_norm']:.3f}"
			)

		if terminated or truncated:
			print(f"[DEMO] finished early at step={k} terminated={terminated} truncated={truncated}")
			break

	env.close()

	run_steps = max(1, len(state_errors))
	print("[DEMO] Run summary")
	print(f"[DEMO] steps={run_steps}")
	print(f"[DEMO] solver_success_rate={solver_successes / run_steps:.3f}")
	print(f"[DEMO] mean_state_error={float(np.mean(state_errors)):.3f}")
	print(f"[DEMO] mean_output_error={float(np.mean(output_errors)):.3f}")

	_plot_trajectory_pyplot(
		reference_traj=traj,
		actual_positions=np.asarray(actual_positions, dtype=float),
		actual_quats=np.asarray(actual_quats, dtype=float),
		show=not args.no_show_plot,
		save_path=args.plot_save_path if args.plot_save_path else None,
		axis_stride=args.axis_stride,
		axis_scale=args.axis_scale,
	)


if __name__ == "__main__":
	main()
