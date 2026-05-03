import importlib

import numpy as np

from gym_pybullet_drones.sensori_agent.MPC.mpc_helpers import (
    quat_derivative_ca,
    quat_integrate_body_rate,
    quat_normalize,
    quat_to_rotmat,
    quat_to_rotmat_ca,
)
from gym_pybullet_drones.sensori_agent.trajectory.trajectory import Trajectory


class SimpleQuadrotorMPC:
    """Independent CasADi-based MPC controller for a quadrotor model.

    State uses the layout [p(3), q_xyzw(4), v(3), w_body(3)] and control uses
    [collective_thrust, p_rate, q_rate, r_rate].
    """

    STATE_DIM = 13
    CTRL_DIM = 4

    def __init__(
        self,
        ctrl_freq: int = 50,
        horizon: int = 10,
        horizon_dt: float = 0.05,
        max_thrust: float = 50.0,
        max_velocity: float = 30.0,
        max_roll_pitch_rate: float = 5.0 * np.pi,
        max_yaw_rate: float = 2.0 * np.pi,
        qx_weights=None,
        r_weights=None,
        rd_weights=None,
        ipopt_max_iter: int = 80,
        gravity: float = 9.81,
    ):
        if ctrl_freq <= 0:
            raise ValueError("ctrl_freq must be > 0")
        if horizon <= 0:
            raise ValueError("horizon must be > 0")

        self.CTRL_FREQ = int(ctrl_freq)
        self.MPC_DT = 1.0 / float(ctrl_freq)
        self.HORIZON = int(horizon)
        self.HORIZON_DT = self.MPC_DT if horizon_dt is None else float(horizon_dt)
        if self.HORIZON_DT <= 0.0:
            raise ValueError("horizon_dt must be > 0")

        self.G = float(gravity)
        self.IPOPT_MAX_ITER = int(ipopt_max_iter)

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
                [-float(max_velocity), float(max_velocity)],
                [-float(max_velocity), float(max_velocity)],
                [-float(max_velocity), float(max_velocity)],
            ],
            dtype=float,
        )

        self.Qx = np.diag(
            np.array(
                qx_weights
                if qx_weights is not None
                else [
					20.5,
					20.5,
					20.5,
					5.5,
					5.5,
					5.5,
					5.5,
					5.5,
					5.5,
					5.5,
					1.0,
					1.0,
					1.0,
				],
                dtype=float,
            )
        )
        self.R = np.diag(np.array(r_weights if r_weights is not None else [0, 0, 0, 0], dtype=float))
        self.Rd = np.diag(np.array(rd_weights if rd_weights is not None else [0, 1e-1, 1e-1, 1e-1], dtype=float))

        if self.Qx.shape != (self.STATE_DIM, self.STATE_DIM):
            raise ValueError("qx_weights must define a 13x13 diagonal matrix")
        if self.R.shape != (self.CTRL_DIM, self.CTRL_DIM):
            raise ValueError("r_weights must define a 4x4 diagonal matrix")
        if self.Rd.shape != (self.CTRL_DIM, self.CTRL_DIM):
            raise ValueError("rd_weights must define a 4x4 diagonal matrix")

        self.reference_trajectory = np.zeros((1, self.STATE_DIM), dtype=float)
        self.reference_dt = self.MPC_DT
        self.ref_step = 0

        self._u_prev = np.array([self.G, 0.0, 0.0, 0.0], dtype=float)
        self._u_warm = np.tile(self._u_prev, (self.HORIZON, 1))

        self._casadi_solver = None
        self._cached_lb = None
        self._cached_ub = None

        self._last_info = {
            "success": False,
            "status": "not_solved",
            "nit": 0,
            "cost": 0.0,
            "constraint_violation": 0.0,
        }

        self._build_casadi_solver()

    def reset(self, trajectory_obj: Trajectory, trajectory_sample_freq: float):
        """Reset controller state and load reference from a trajectory object."""
        if not hasattr(trajectory_obj, "sample_full_state"):
            raise ValueError("trajectory_obj must implement sample_full_state()")
        sample_freq = float(trajectory_sample_freq)
        if sample_freq <= 0.0:
            raise ValueError("trajectory_sample_freq must be > 0")

        self.reference_trajectory = self._sample_reference_from_trajectory_obj(trajectory_obj, sample_freq)
        self.reference_dt = 1.0 / sample_freq
        self.ref_step = 0

        self._u_prev = np.array([self.G, 0.0, 0.0, 0.0], dtype=float)
        self._u_warm = np.tile(self._u_prev, (self.HORIZON, 1))
        self._last_info = {
            "success": False,
            "status": "reset",
            "nit": 0,
            "cost": 0.0,
            "constraint_violation": 0.0,
        }

    def compute_control(self, x0, advance_reference: bool = True):
        """Solve MPC for the current state and return (u_cmd, info)."""
        x0 = np.asarray(x0, dtype=float).reshape(-1)
        if x0.shape[0] != self.STATE_DIM:
            raise ValueError("x0 must be a length-13 state vector")

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

            stats = self._casadi_solver.stats()
            status = str(stats.get("return_status", "unknown"))
            nit = int(stats.get("iter_count", 0))
            success = "success" in status.lower() or "solve_succeeded" in status.lower()
            cost = float(sol["f"])
        except Exception as exc:
            u0 = self._u_prev.copy()
            self._u_warm = np.tile(u0, (self.HORIZON, 1))
            status = f"casadi_exception:{type(exc).__name__}"
            nit = 0
            success = False
            cost = float(self._rollout_cost(x0, self._u_warm))

        u0_clipped = np.empty(4, dtype=float)
        u0_clipped[0] = float(np.clip(u0[0], self.MASS_NORMALIZED_THRUST_BOUNDS[0], self.MASS_NORMALIZED_THRUST_BOUNDS[1]))
        u0_clipped[1:4] = np.clip(u0[1:4], self.BODY_RATE_BOUNDS[:, 0], self.BODY_RATE_BOUNDS[:, 1])
        self._u_prev = u0_clipped.copy()

        constraint_violation = self._estimate_constraint_violation(x0, self._u_warm)
        self._last_info = {
            "success": bool(success),
            "status": status,
            "nit": int(nit),
            "cost": float(cost),
            "constraint_violation": float(constraint_violation),
        }

        if advance_reference:
            self.ref_step += int(1 * (self.MPC_DT/self.reference_dt))

        return u0_clipped, dict(self._last_info)

    def _sample_reference_from_trajectory_obj(self, trajectory_obj, sampling_rate):
        sampled = trajectory_obj.sample_full_state(sampling_rate=float(sampling_rate), include_terminal=True)
        required = ("pos", "quat", "vel")
        for key in required:
            if key not in sampled:
                raise ValueError(f"sample_full_state() output must contain '{key}'")

        pos = np.asarray(sampled["pos"], dtype=float)
        quat = np.asarray(sampled["quat"], dtype=float)
        vel = np.asarray(sampled["vel"], dtype=float)
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError("sampled['pos'] must be shape (N, 3)")
        if quat.ndim != 2 or quat.shape[1] != 4:
            raise ValueError("sampled['quat'] must be shape (N, 4)")
        if vel.ndim != 2 or vel.shape[1] != 3:
            raise ValueError("sampled['vel'] must be shape (N, 3)")

        if "body_rate" in sampled:
            w_ref = np.asarray(sampled["body_rate"], dtype=float)
        elif "ang_vel_body" in sampled:
            w_ref = np.asarray(sampled["ang_vel_body"], dtype=float)
        else:
            raise ValueError("sample_full_state() output must contain 'body_rate' or 'ang_vel_body'")
        if w_ref.ndim != 2 or w_ref.shape[1] != 3:
            raise ValueError("sampled angular-rate field must be shape (N, 3)")

        n = min(pos.shape[0], quat.shape[0], vel.shape[0], w_ref.shape[0])
        if n < 1:
            raise ValueError("sample_full_state() returned no samples")

        x_ref = np.zeros((n, self.STATE_DIM), dtype=float)
        x_ref[:, 0:3] = pos[:n]
        x_ref[:, 3:7] = np.array([quat_normalize(q) for q in quat[:n]], dtype=float)
        x_ref[:, 7:10] = np.clip(vel[:n], self.VEL_BOUNDS[:, 0], self.VEL_BOUNDS[:, 1])
        x_ref[:, 10:13] = np.clip(w_ref[:n], self.BODY_RATE_BOUNDS[:, 0], self.BODY_RATE_BOUNDS[:, 1])
        return x_ref

    def _build_casadi_solver(self):
        ca = self._get_casadi_module()
        nu = self.HORIZON * self.CTRL_DIM
        u_var = ca.SX.sym("u", nu)

        nx0 = self.STATE_DIM
        nxref = self.HORIZON * self.STATE_DIM
        nup = self.CTRL_DIM
        p_var = ca.SX.sym("p", nx0 + nxref + nup)

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

        opts = {
            "ipopt.print_level": 0,
            "print_time": False,
            "ipopt.max_iter": self.IPOPT_MAX_ITER,
            "ipopt.acceptable_tol": 1e-3,
            "ipopt.acceptable_iter": 3,
            "ipopt.sb": "yes",
        }
        self._casadi_solver = ca.nlpsol("simple_quad_mpc_solver", "ipopt", {"x": u_var, "p": p_var, "f": cost}, opts)

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
        refs = [self._get_horizon_ref_state(i) for i in range(self.HORIZON)]
        return np.concatenate([x0, np.concatenate(refs), self._u_prev])

    def _get_ref_state(self, k):
        idx = int(np.clip(k, 0, len(self.reference_trajectory) - 1))
        return self.reference_trajectory[idx].copy()

    def _get_horizon_ref_state(self, pred_idx):
        seconds_ahead = (int(pred_idx) + 1) * self.HORIZON_DT
        step_ahead = int(np.ceil(seconds_ahead / self.reference_dt))
        step_ahead = max(1, step_ahead)
        return self._get_ref_state(self.ref_step + step_ahead)

    def _rollout_cost(self, x0, u_seq):
        x = x0.copy()
        u_prev = self._u_prev.copy()
        cost = 0.0
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
        return float(cost)

    def _predict_dynamics(self, x, u):
        pos = x[0:3]
        quat = x[3:7]
        vel = x[7:10]
        dt = self.HORIZON_DT

        thrust = float(np.clip(u[0], self.MASS_NORMALIZED_THRUST_BOUNDS[0], self.MASS_NORMALIZED_THRUST_BOUNDS[1]))
        w_next = np.clip(u[1:4], self.BODY_RATE_BOUNDS[:, 0], self.BODY_RATE_BOUNDS[:, 1])

        quat_next = quat_integrate_body_rate(quat, w_next, dt)
        thrust_world = quat_to_rotmat(quat_next) @ np.array([0.0, 0.0, thrust], dtype=float)
        acc_world = thrust_world - np.array([0.0, 0.0, self.G], dtype=float)

        vel_next = vel + dt * acc_world
        vel_next = np.clip(vel_next, self.VEL_BOUNDS[:, 0], self.VEL_BOUNDS[:, 1])
        pos_next = pos + dt * vel_next
        return np.hstack([pos_next, quat_next, vel_next, w_next])

    def _predict_dynamics_ca(self, x, u):
        ca = self._get_casadi_module()
        pos = x[0:3]
        quat = x[3:7]
        vel = x[7:10]
        dt = self.HORIZON_DT

        thrust = ca.fmin(ca.fmax(u[0], self.MASS_NORMALIZED_THRUST_BOUNDS[0]), self.MASS_NORMALIZED_THRUST_BOUNDS[1])
        w_next = ca.vertcat(
            ca.fmin(ca.fmax(u[1], self.BODY_RATE_BOUNDS[0, 0]), self.BODY_RATE_BOUNDS[0, 1]),
            ca.fmin(ca.fmax(u[2], self.BODY_RATE_BOUNDS[1, 0]), self.BODY_RATE_BOUNDS[1, 1]),
            ca.fmin(ca.fmax(u[3], self.BODY_RATE_BOUNDS[2, 0]), self.BODY_RATE_BOUNDS[2, 1]),
        )

        q_dot = quat_derivative_ca(ca, quat, w_next)
        quat_next = quat + dt * q_dot
        quat_next = quat_next / (ca.sqrt(ca.sumsqr(quat_next)) + 1e-9)

        rot = quat_to_rotmat_ca(ca, quat_next)
        thrust_world = ca.mtimes(rot, ca.vertcat(0.0, 0.0, thrust))
        acc_world = thrust_world - ca.vertcat(0.0, 0.0, self.G)

        vel_next = vel + dt * acc_world
        vel_next = ca.vertcat(
            ca.fmin(ca.fmax(vel_next[0], self.VEL_BOUNDS[0, 0]), self.VEL_BOUNDS[0, 1]),
            ca.fmin(ca.fmax(vel_next[1], self.VEL_BOUNDS[1, 0]), self.VEL_BOUNDS[1, 1]),
            ca.fmin(ca.fmax(vel_next[2], self.VEL_BOUNDS[2, 0]), self.VEL_BOUNDS[2, 1]),
        )

        pos_next = pos + dt * vel_next
        return ca.vertcat(pos_next, quat_next, vel_next, w_next)

    def _estimate_constraint_violation(self, x0, u_seq):
        x = x0.copy()
        max_violation = 0.0
        for i in range(self.HORIZON):
            u = u_seq[i]
            thrust_v = max(0.0, self.MASS_NORMALIZED_THRUST_BOUNDS[0] - u[0], u[0] - self.MASS_NORMALIZED_THRUST_BOUNDS[1])

            rate_low = self.BODY_RATE_BOUNDS[:, 0] - u[1:4]
            rate_high = u[1:4] - self.BODY_RATE_BOUNDS[:, 1]
            rate_v = float(np.max(np.maximum(0.0, np.maximum(rate_low, rate_high))))

            x = self._predict_dynamics(x, u)
            vel_low = self.VEL_BOUNDS[:, 0] - x[7:10]
            vel_high = x[7:10] - self.VEL_BOUNDS[:, 1]
            vel_v = float(np.max(np.maximum(0.0, np.maximum(vel_low, vel_high))))

            max_violation = max(max_violation, float(thrust_v), rate_v, vel_v)
        return float(max_violation)

    @staticmethod
    def _get_casadi_module():
        try:
            return importlib.import_module("casadi")
        except Exception as exc:
            raise RuntimeError("casadi is required for this MPC controller. Install with 'pip install casadi'.") from exc
