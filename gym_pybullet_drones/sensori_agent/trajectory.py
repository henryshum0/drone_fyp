import numpy as np
import math
from scipy.linalg import block_diag
from scipy.optimize import minimize

class Node():
    # flat output of drone = [x, y, z, psi], 
    # order of position polynomial = 7, order of psi polynomial = 3
    # assume psi is always along the velocity direction
    def __init__(self, pos, psi, con_vel=None, con_acc=None):
        x = pos[0]
        y = pos[1]
        z = pos[2]
        psi = psi
        con_vx = con_vel[0] if con_vel is not None else np.nan
        con_vy = con_vel[1] if con_vel is not None else np.nan
        con_vz = con_vel[2] if con_vel is not None else np.nan
        con_ax = con_acc[0] if con_acc is not None else np.nan
        con_ay = con_acc[1] if con_acc is not None else np.nan
        con_az = con_acc[2] if con_acc is not None else np.nan

        self.con_x = (x, con_vx, con_ax)
        self.con_y = (y, con_vy, con_ay)
        self.con_z = (z, con_vz, con_az)
        self.con_psi = (psi, )

        self.n_pos_con = len(self.con_x)
        self.n_psi_con = len(self.con_psi)

    def A(self, t):
        A_x = []
        A_y = []
        A_z = []
        A_psi = []

        for derivative_order in range(self.n_pos_con):
            row = []
            for i in range(10):
                if (i < derivative_order):
                    row.append(0)
                else:
                    entry = t**(i - derivative_order) * math.factorial(i) / math.factorial(i - derivative_order)
                    row.append(entry)
            A_x.append(row)
            A_y.append(row)
            A_z.append(row)

        for derivative_order in range(self.n_psi_con):
            row = []
            for i in range(3):
                if (i < derivative_order):
                    row.append(0)
                else:
                    entry = t**(i - derivative_order) * math.factorial(i) / math.factorial(i - derivative_order)
                    row.append(entry)
            A_psi.append(row)

        return np.array(A_x), np.array(A_y), np.array(A_z), np.array(A_psi)


class Segment():
    def __init__(
        self,
        start_node: Node,
        end_node: Node,
        duration: float,
        pos_derivative_costs: dict[int, float] | None = None,
        psi_derivative_costs: dict[int, float] | None = None,
    ):
        self._start_node = start_node
        self._end_node = end_node
        self._duration = duration
        self._poly_pos_order = 10
        self._poly_psi_order = 3
        self._pos_derivative_costs = pos_derivative_costs or {0:150000, 1:1, 2:1, 3: 1, 4: 2.5}
        self._psi_derivative_costs = psi_derivative_costs or {0:3, 1: 5, 2: 1.0}
        self._get_b()
        self._get_A()
        self._get_Hessian()

    def build_new(self, duration):
        return Segment(
            start_node=self._start_node,
            end_node=self._end_node,
            duration=duration,
            pos_derivative_costs=self._pos_derivative_costs,
            psi_derivative_costs=self._psi_derivative_costs,
        )

    def H(self, tau):
        H_x = np.zeros((self._poly_pos_order, self._poly_pos_order))
        H_y = np.zeros((self._poly_pos_order, self._poly_pos_order))
        H_z = np.zeros((self._poly_pos_order, self._poly_pos_order))
        H_psi = np.zeros((self._poly_psi_order, self._poly_psi_order))

        def add_derivative_cost(H, poly_order, derivative_order, weight):
            if weight == 0.0:
                return
            if derivative_order < 0:
                return
            if derivative_order >= poly_order:
                return

            for i in range(derivative_order, poly_order):
                coeff_i = math.factorial(i) / math.factorial(i - derivative_order)
                for j in range(derivative_order, poly_order):
                    coeff_j = math.factorial(j) / math.factorial(j - derivative_order)
                    power = i + j - 2 * derivative_order + 1
                    H[i, j] += weight * coeff_i * coeff_j * (tau ** power) / power

        for derivative_order, weight in self._pos_derivative_costs.items():
            add_derivative_cost(H_x, self._poly_pos_order, int(derivative_order), float(weight))
            add_derivative_cost(H_y, self._poly_pos_order, int(derivative_order), float(weight))
            add_derivative_cost(H_z, self._poly_pos_order, int(derivative_order), float(weight))

        for derivative_order, weight in self._psi_derivative_costs.items():
            add_derivative_cost(H_psi, self._poly_psi_order, int(derivative_order), float(weight))

        return H_x, H_y, H_z, H_psi

    def _get_b(self):
        self.b_x = np.concatenate((self._start_node.con_x, self._end_node.con_x))
        self.b_y = np.concatenate((self._start_node.con_y, self._end_node.con_y))
        self.b_z = np.concatenate((self._start_node.con_z, self._end_node.con_z))
        self.b_psi = np.concatenate((self._start_node.con_psi, self._end_node.con_psi))

    def _get_A(self):
        A_x_left, A_y_left, A_z_left, A_psi_left = self._start_node.A(0)
        A_x_right, A_y_right, A_z_right, A_psi_right = self._end_node.A(self._duration)
        self.A_x = np.vstack((A_x_left, A_x_right))
        self.A_y = np.vstack((A_y_left, A_y_right))
        self.A_z = np.vstack((A_z_left, A_z_right))
        self.A_psi = np.vstack((A_psi_left, A_psi_right))
        
        
    def _get_Hessian(self):
        # Build Hessians from weighted integral costs of squared derivatives:
        # J = sum_k w_k * integral_0^T (d^k p / dt^k)^2 dt
        self.H_x, self.H_y, self.H_z, self.H_psi = self.H(self._duration)

    def solve(self):
        if np.isnan(self.b_x).any() or np.isnan(self.b_y).any() or np.isnan(self.b_z).any() or np.isnan(self.b_psi).any():
            raise ValueError("Segment.solve() expects fully specified b without NaN; run global b solve first.")

        self.coeffs_x = np.linalg.lstsq(self.A_x, self.b_x, rcond=None)[0]
        self.coeffs_y = np.linalg.lstsq(self.A_y, self.b_y, rcond=None)[0]
        self.coeffs_z = np.linalg.lstsq(self.A_z, self.b_z, rcond=None)[0]
        self.coeffs_psi = np.linalg.lstsq(self.A_psi, self.b_psi, rcond=None)[0]

        return self.coeffs_x, self.coeffs_y, self.coeffs_z, self.coeffs_psi
    

class Trajectory():
    def __init__(self, segments:list[Segment]):
        self._segments = segments
        self.b_x = np.concatenate([seg.b_x for seg in segments])
        self.b_y = np.concatenate([seg.b_y for seg in segments])
        self.b_z = np.concatenate([seg.b_z for seg in segments])
        self.b_psi = np.concatenate([seg.b_psi for seg in segments])
        self.A_x = block_diag(*[seg.A_x for seg in segments])
        self.A_y = block_diag(*[seg.A_y for seg in segments])
        self.A_z = block_diag(*[seg.A_z for seg in segments])
        self.A_psi = block_diag(*[seg.A_psi for seg in segments])
        self.H_x = block_diag(*[seg.H_x for seg in segments])
        self.H_y = block_diag(*[seg.H_y for seg in segments])
        self.H_z = block_diag(*[seg.H_z for seg in segments])
        self.H_psi = block_diag(*[seg.H_psi for seg in segments])
        self._find_C()
        self._solve_for_b_free()
        self._update_segments_b_from_global()
        self._solve_segments()
    
    def build_new(self, segment_times):
        new_segments = []
        for seg, new_time in zip(self._segments, segment_times):
            new_segments.append(seg.build_new(new_time))
        return Trajectory(new_segments)

    def _find_C(self):
        def rearrangement_matrix(b):
            known_idx = [i for i, val in enumerate(b) if not np.isnan(val)]
            free_idx = [i for i, val in enumerate(b) if np.isnan(val)]
            permutation = known_idx + free_idx
            C = np.eye(len(b))[permutation]
            return C
        
        self.C_x = rearrangement_matrix(self.b_x)
        self.C_y = rearrangement_matrix(self.b_y)
        self.C_z = rearrangement_matrix(self.b_z)
        self.C_psi = rearrangement_matrix(self.b_psi)

    def _solve_for_b_free(self):
        def build_R(C, A, H):
            A_pinv = np.linalg.pinv(A)
            return C @ A_pinv.T @ H @ A_pinv @ C.T

        def continuity_matrix(total_len, block_size, half_size):
            n_constraints = max(0, len(self._segments) - 1) * half_size
            E = np.zeros((n_constraints, total_len))
            row = 0
            for seg_idx in range(len(self._segments) - 1):
                right_start = seg_idx * block_size + half_size
                next_left_start = (seg_idx + 1) * block_size
                for d in range(half_size):
                    E[row, right_start + d] = 1.0
                    E[row, next_left_start + d] = -1.0
                    row += 1
            return E

        def solve_axis(b, C, R, block_size, half_size):
            known_mask = ~np.isnan(b)
            n_fixed = int(np.sum(known_mask))
            perm = np.argmax(C, axis=1)

            b_work = np.where(np.isnan(b), 0.0, b)
            b_perm = b_work[perm]
            b_f = b_perm[:n_fixed]

            n_total = b.shape[0]
            n_free = n_total - n_fixed
            if n_free <= 0:
                return b

            R_fp = R[:n_fixed, n_fixed:]
            R_pp = R[n_fixed:, n_fixed:]

            E_orig = continuity_matrix(n_total, block_size, half_size)
            E = E_orig @ C.T
            E_f = E[:, :n_fixed]
            E_p = E[:, n_fixed:]

            rhs_top = -R_fp.T @ b_f
            rhs_bottom = -E_f @ b_f

            if E_p.shape[0] > 0:
                KKT = np.block([
                    [R_pp, E_p.T],
                    [E_p, np.zeros((E_p.shape[0], E_p.shape[0]))],
                ])
                rhs = np.concatenate((rhs_top, rhs_bottom))
                try:
                    sol = np.linalg.solve(KKT, rhs)
                except np.linalg.LinAlgError:
                    sol = np.linalg.lstsq(KKT, rhs, rcond=None)[0]
                b_p = sol[:n_free]
            else:
                try:
                    b_p = np.linalg.solve(R_pp, rhs_top)
                except np.linalg.LinAlgError:
                    b_p = np.linalg.lstsq(R_pp, rhs_top, rcond=None)[0]

            b_perm[n_fixed:] = b_p
            return C.T @ b_perm

        R_x = build_R(self.C_x, self.A_x, self.H_x)
        R_y = build_R(self.C_y, self.A_y, self.H_y)
        R_z = build_R(self.C_z, self.A_z, self.H_z)
        R_psi = build_R(self.C_psi, self.A_psi, self.H_psi)

        self.b_x = solve_axis(self.b_x, self.C_x, R_x, len(self._segments[0].b_x), len(self._segments[0].b_x) // 2)
        self.b_y = solve_axis(self.b_y, self.C_y, R_y, len(self._segments[0].b_y), len(self._segments[0].b_y) // 2)
        self.b_z = solve_axis(self.b_z, self.C_z, R_z, len(self._segments[0].b_z), len(self._segments[0].b_z) // 2)
        self.b_psi = solve_axis(self.b_psi, self.C_psi, R_psi, len(self._segments[0].b_psi), len(self._segments[0].b_psi) // 2)

    def _update_segments_b_from_global(self):
        pos_offset = 0
        psi_offset = 0
        for seg in self._segments:
            pos_len = len(seg.b_x)
            psi_len = len(seg.b_psi)

            seg.b_x = self.b_x[pos_offset:pos_offset + pos_len].copy()
            seg.b_y = self.b_y[pos_offset:pos_offset + pos_len].copy()
            seg.b_z = self.b_z[pos_offset:pos_offset + pos_len].copy()
            seg.b_psi = self.b_psi[psi_offset:psi_offset + psi_len].copy()

            pos_offset += pos_len
            psi_offset += psi_len

    def _solve_segments(self):
        self.coeffs_x = []
        self.coeffs_y = []
        self.coeffs_z = []
        self.coeffs_psi = []
        for seg in self._segments:
            seg.solve()
            self.coeffs_x.append(seg.coeffs_x)
            self.coeffs_y.append(seg.coeffs_y)
            self.coeffs_z.append(seg.coeffs_z)
            self.coeffs_psi.append(seg.coeffs_psi)
        self.coeffs_x = np.concatenate(self.coeffs_x)
        self.coeffs_y = np.concatenate(self.coeffs_y)
        self.coeffs_z = np.concatenate(self.coeffs_z)
        self.coeffs_psi = np.concatenate(self.coeffs_psi)

    @staticmethod
    def _evaluate_poly(coeffs, t_values):
        powers = np.vander(t_values, N=len(coeffs), increasing=True)
        return powers @ coeffs

    @staticmethod
    def _evaluate_poly_derivative(coeffs, t_values, derivative_order=1):
        if derivative_order < 0:
            raise ValueError("derivative_order must be non-negative")
        if derivative_order == 0:
            return Trajectory._evaluate_poly(coeffs, t_values)

        d_coeffs = coeffs.astype(float).copy()
        for _ in range(derivative_order):
            if len(d_coeffs) <= 1:
                d_coeffs = np.array([0.0])
                break
            d_coeffs = np.array([i * d_coeffs[i] for i in range(1, len(d_coeffs))], dtype=float)

        powers = np.vander(t_values, N=len(d_coeffs), increasing=True)
        return powers @ d_coeffs

    @staticmethod
    def _set_equal_3d_axes(ax, x, y, z):
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)
        z_min, z_max = np.min(z), np.max(z)

        x_mid = 0.5 * (x_min + x_max)
        y_mid = 0.5 * (y_min + y_max)
        z_mid = 0.5 * (z_min + z_max)

        half_range = 0.5 * max(x_max - x_min, y_max - y_min, z_max - z_min)
        if half_range == 0:
            half_range = 1.0

        ax.set_xlim(x_mid - half_range, x_mid + half_range)
        ax.set_ylim(y_mid - half_range, y_mid + half_range)
        ax.set_zlim(z_mid - half_range, z_mid + half_range)
        ax.set_box_aspect((1, 1, 1))

    @staticmethod
    def _safe_normalize_rows(vectors, eps=1e-9):
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms < eps, 1.0, norms)
        return vectors / norms

    @staticmethod
    def _compute_body_axes(accel_vec, yaw, gravity_vector):
        thrust_vec = accel_vec - gravity_vector.reshape(1, 3)
        b3 = Trajectory._safe_normalize_rows(thrust_vec)

        c1 = np.column_stack((np.cos(yaw), np.sin(yaw), np.zeros_like(yaw)))
        b2 = np.cross(np.abs(b3), c1)
        b2_norm = np.linalg.norm(b2, axis=1)

        singular = b2_norm < 1e-9
        if np.any(singular):
            ref_x = np.array([1.0, 0.0, 0.0])
            ref_y = np.array([0.0, 1.0, 0.0])
            b3_s = b3[singular]

            use_y = np.abs(b3_s @ ref_x) > 0.9
            ref = np.tile(ref_x, (b3_s.shape[0], 1))
            ref[use_y] = ref_y

            b2_fallback = np.cross(b3_s, ref)
            b2[singular] = b2_fallback

        b2 = Trajectory._safe_normalize_rows(b2)
        b1 = np.cross(b2, b3)
        b1 = Trajectory._safe_normalize_rows(b1)
        b3 = Trajectory._safe_normalize_rows(b3)

        return b1, b2, b3

    @staticmethod
    def _quat_from_rpy(roll, pitch, yaw):
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)

        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        qw = cr * cp * cy + sr * sp * sy
        return np.column_stack((qx, qy, qz, qw))

    def sample_full_state(
        self,
        sampling_rate,
        gravity_vector=np.array([0.0, 0.0, -9.81]),
        include_terminal=True,
    ):
        if sampling_rate <= 0:
            raise ValueError("sampling_rate must be > 0")

        dt = 1.0 / float(sampling_rate)
        all_t = []
        all_x = []
        all_y = []
        all_z = []
        all_vx = []
        all_vy = []
        all_vz = []
        all_ax = []
        all_ay = []
        all_az = []
        all_jx = []
        all_jy = []
        all_jz = []
        all_yaw = []
        all_yaw_rate = []

        t_offset = 0.0
        for seg_idx, seg in enumerate(self._segments):
            n_samples = max(2, int(np.round(seg._duration * sampling_rate)) + 1)
            t_local = np.linspace(0.0, seg._duration, n_samples, endpoint=include_terminal)

            if seg_idx > 0 and t_local.size > 0:
                t_local = t_local[1:]

            if t_local.size == 0:
                t_offset += seg._duration
                continue

            all_t.append(t_offset + t_local)
            all_x.append(self._evaluate_poly(seg.coeffs_x, t_local))
            all_y.append(self._evaluate_poly(seg.coeffs_y, t_local))
            all_z.append(self._evaluate_poly(seg.coeffs_z, t_local))
            all_vx.append(self._evaluate_poly_derivative(seg.coeffs_x, t_local, derivative_order=1))
            all_vy.append(self._evaluate_poly_derivative(seg.coeffs_y, t_local, derivative_order=1))
            all_vz.append(self._evaluate_poly_derivative(seg.coeffs_z, t_local, derivative_order=1))
            all_ax.append(self._evaluate_poly_derivative(seg.coeffs_x, t_local, derivative_order=2))
            all_ay.append(self._evaluate_poly_derivative(seg.coeffs_y, t_local, derivative_order=2))
            all_az.append(self._evaluate_poly_derivative(seg.coeffs_z, t_local, derivative_order=2))
            all_jx.append(self._evaluate_poly_derivative(seg.coeffs_x, t_local, derivative_order=3))
            all_jy.append(self._evaluate_poly_derivative(seg.coeffs_y, t_local, derivative_order=3))
            all_jz.append(self._evaluate_poly_derivative(seg.coeffs_z, t_local, derivative_order=3))
            all_yaw.append(self._evaluate_poly(seg.coeffs_psi, t_local))
            all_yaw_rate.append(self._evaluate_poly_derivative(seg.coeffs_psi, t_local, derivative_order=1))

            t_offset += seg._duration

        if len(all_t) == 0:
            raise ValueError("No samples generated from trajectory")

        t = np.concatenate(all_t)
        x = np.concatenate(all_x)
        y = np.concatenate(all_y)
        z = np.concatenate(all_z)
        vx = np.concatenate(all_vx)
        vy = np.concatenate(all_vy)
        vz = np.concatenate(all_vz)
        ax = np.concatenate(all_ax)
        ay = np.concatenate(all_ay)
        az = np.concatenate(all_az)
        jx = np.concatenate(all_jx)
        jy = np.concatenate(all_jy)
        jz = np.concatenate(all_jz)
        yaw = np.concatenate(all_yaw)
        yaw_rate = np.concatenate(all_yaw_rate)

        pos = np.column_stack((x, y, z))
        vel = np.column_stack((vx, vy, vz))
        acc = np.column_stack((ax, ay, az))
        jerk = np.column_stack((jx, jy, jz))

        b1, b2, b3 = self._compute_body_axes(acc, yaw, gravity_vector)
        R = np.stack((b1, b2, b3), axis=2)

        rpy = np.zeros((len(t), 3))
        rpy[:, 0] = np.arctan2(R[:, 2, 1], R[:, 2, 2])
        rpy[:, 1] = np.arcsin(np.clip(-R[:, 2, 0], -1.0, 1.0))
        rpy[:, 2] = yaw

        if len(t) > 1:
            roll_unwrapped = np.unwrap(rpy[:, 0])
            pitch_unwrapped = np.unwrap(rpy[:, 1])
            yaw_unwrapped = np.unwrap(yaw)
            roll_rate = np.gradient(roll_unwrapped, t)
            pitch_rate = np.gradient(pitch_unwrapped, t)
            yaw_rate_num = np.gradient(yaw_unwrapped, t)
            rpy_rate = np.column_stack((roll_rate, pitch_rate, yaw_rate_num))
        else:
            rpy_rate = np.zeros((1, 3))

        quat = self._quat_from_rpy(rpy[:, 0], rpy[:, 1], rpy[:, 2])

        return {
            "sampling_rate": float(sampling_rate),
            "dt": dt,
            "t": t,
            "pos": pos,
            "vel": vel,
            "acc": acc,
            "jerk": jerk,
            "yaw": yaw,
            "yaw_rate": yaw_rate,
            "rpy": rpy,
            "rpy_rate": rpy_rate,
            "quat": quat,
            "body_x": b1,
            "body_y": b2,
            "body_z": b3,
            "full_state": np.hstack((pos, vel, rpy, rpy_rate)),
        }

    def visualize(
        self,
        num_points_per_segment=100,
        sampling_rate=None,
        show=True,
        ax=None,
        show_velocity=True,
        show_body_x=True,
        show_body_z=True,
        vector_stride=10,
        vector_scale=0.3,
        gravity_vector=np.array([0.0, 0.0, -9.81]),
    ):
        import matplotlib.pyplot as plt

        if sampling_rate is None:
            if num_points_per_segment < 2:
                raise ValueError("num_points_per_segment must be >= 2")
            durations = np.array([seg._duration for seg in self._segments], dtype=float)
            mean_duration = float(np.mean(durations)) if durations.size > 0 else 1.0
            mean_duration = max(mean_duration, 1e-9)
            sampling_rate = (num_points_per_segment - 1) / mean_duration

        sampled = self.sample_full_state(
            sampling_rate=sampling_rate,
            gravity_vector=gravity_vector,
            include_terminal=True,
        )

        x = sampled["pos"][:, 0]
        y = sampled["pos"][:, 1]
        z = sampled["pos"][:, 2]
        vx = sampled["vel"][:, 0]
        vy = sampled["vel"][:, 1]
        vz = sampled["vel"][:, 2]
        b1 = sampled["body_x"]
        b3 = sampled["body_z"]

        created_fig = False
        if ax is None:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
            created_fig = True
        else:
            fig = ax.figure

        ax.plot(x, y, z, linewidth=2)
        ax.scatter([x[0]], [y[0]], [z[0]], s=40)
        ax.scatter([x[-1]], [y[-1]], [z[-1]], s=40)

        sample_idx = np.arange(0, len(x), max(1, int(vector_stride)))

        if show_velocity:
            ax.quiver(
                x[sample_idx], y[sample_idx], z[sample_idx],
                vx[sample_idx], vy[sample_idx], vz[sample_idx],
                length=vector_scale,
                normalize=True,
                color='tab:green',
                linewidth=1.0,
            )

        if show_body_x:
            ax.quiver(
                x[sample_idx], y[sample_idx], z[sample_idx],
                b1[sample_idx, 0], b1[sample_idx, 1], b1[sample_idx, 2],
                length=vector_scale,
                normalize=True,
                color='tab:blue',
                linewidth=1.0,
            )

        if show_body_z:
            ax.quiver(
                x[sample_idx], y[sample_idx], z[sample_idx],
                b3[sample_idx, 0], b3[sample_idx, 1], b3[sample_idx, 2],
                length=vector_scale,
                normalize=True,
                color='tab:red',
                linewidth=1.0,
            )

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Trajectory (green: velocity, blue: body x-axis, red: body z-axis)')
        self._set_equal_3d_axes(ax, x, y, z)

        if show and created_fig:
            plt.show()

        return fig, ax, np.column_stack((x, y, z))

        
if __name__ == "__main__":
    node1 = Node(pos=[0, 0, 0], con_vel=[0, 0, 0], con_acc=[0, 0, 0], psi=np.pi)
    node2 = Node(pos=[10, 0, 0], psi=np.pi,)
    node3 = Node(pos=[12.5, 0, 2.5], psi=np.pi, )
    node4 = Node(pos=[10, 0, 5], psi=np.pi, con_vel=[-12, 0, 0])
    node5 = Node(pos=[8.5, 0, 2.5], psi=np.pi, )
    node6 = Node(pos=[10, 0, 0], con_vel=[0, 0, 0], con_acc=[0, 0, 0], psi=np.pi, )
    segment = Segment(node1, node2, duration=1)
    segment2 = Segment(node2, node3, duration=.25)
    segment3 = Segment(node3, node4, duration=.35)
    segment4 = Segment(node4, node5, duration=.35)
    segment5 = Segment(node5, node6, duration=1)
    segments = [segment, segment2, segment3, segment4, segment5]
    trajectory = Trajectory(segments)
    print(
        "max_velocity_x:", max([seg.b_x[1] for seg in trajectory._segments]),
        "max_velocity_y:", max([seg.b_y[1] for seg in trajectory._segments]),
        "max_velocity_z:", max([seg.b_z[1] for seg in trajectory._segments]),
        "max_acceleration_x:", max([seg.b_x[2] for seg in trajectory._segments]),
        "max_acceleration_y:", max([seg.b_y[2] for seg in trajectory._segments]),
        "max_acceleration_z:", max([seg.b_z[2] for seg in trajectory._segments]),
        "min_acceleration_x:", min([seg.b_x[2] for seg in trajectory._segments]),
        "min_acceleration_y:", min([seg.b_y[2] for seg in trajectory._segments]),
        "min_acceleration_z:", min([seg.b_z[2] for seg in trajectory._segments]),
        )
    trajectory.visualize(show=True)
    trajectory.build_new(segment_times=[1, 0.25, 0.35, 0.35, 1]).visualize(show=True)