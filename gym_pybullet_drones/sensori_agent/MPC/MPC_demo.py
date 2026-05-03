import time

import cv2
import numpy as np
import pybullet as p

from gym_pybullet_drones.sensori_agent.MPC.MPC_control_env import MPCControlEnv
from gym_pybullet_drones.sensori_agent.trajectory.trajectory import Node, Segment, Trajectory
from gym_pybullet_drones.sensori_agent.trajectory.trajectory_optimize import optimize_trj_time


WORKSPACE_MIN = 0.0
WORKSPACE_MAX = 10.0
TRAJ_SAMPLE_FREQ = 50.0
CANVAS_SIZE_PX = 860
CANVAS_MARGIN_PX = 70


def _add_button(name: str, client_id: int):
	return p.addUserDebugParameter(name, 0, 1, 0, physicsClientId=client_id)


def _button_pressed(button_id: int, prev_val: float, client_id: int):
	cur = p.readUserDebugParameter(button_id, physicsClientId=client_id)
	pressed = (cur - prev_val) > 0.5
	return pressed, cur


def _key_triggered(events, key_char: str):
	key_state = events.get(ord(key_char), 0)
	return (key_state & p.KEY_WAS_TRIGGERED) != 0


def _draw_workspace_grid(client_id: int):
	debug_ids = []
	c_grid = [0.45, 0.45, 0.45]
	c_axes = [1.0, 1.0, 1.0]
	c_border = [0.9, 0.9, 0.2]

	for v in np.linspace(WORKSPACE_MIN, WORKSPACE_MAX, 11):
		debug_ids.append(
			p.addUserDebugLine(
				[WORKSPACE_MIN, 0.0, float(v)],
				[WORKSPACE_MAX, 0.0, float(v)],
				c_grid,
				1.0,
				physicsClientId=client_id,
			)
		)
		debug_ids.append(
			p.addUserDebugLine(
				[float(v), 0.0, WORKSPACE_MIN],
				[float(v), 0.0, WORKSPACE_MAX],
				c_grid,
				1.0,
				physicsClientId=client_id,
			)
		)

	debug_ids.append(
		p.addUserDebugLine(
			[WORKSPACE_MIN, 0.0, WORKSPACE_MIN],
			[WORKSPACE_MAX, 0.0, WORKSPACE_MIN],
			c_border,
			3.0,
			physicsClientId=client_id,
		)
	)
	debug_ids.append(
		p.addUserDebugLine(
			[WORKSPACE_MIN, 0.0, WORKSPACE_MIN],
			[WORKSPACE_MIN, 0.0, WORKSPACE_MAX],
			c_border,
			3.0,
			physicsClientId=client_id,
		)
	)
	debug_ids.append(
		p.addUserDebugText(
			"(0,0)",
			[WORKSPACE_MIN, 0.0, WORKSPACE_MIN],
			c_axes,
			textSize=1.2,
			physicsClientId=client_id,
		)
	)
	debug_ids.append(
		p.addUserDebugText(
			"(10,10)",
			[WORKSPACE_MAX, 0.0, WORKSPACE_MAX],
			c_axes,
			textSize=1.2,
			physicsClientId=client_id,
		)
	)
	return debug_ids


def _remove_debug_items(item_ids, client_id: int):
	for item_id in item_ids:
		try:
			p.removeUserDebugItem(item_id, physicsClientId=client_id)
		except Exception:
			pass


def _world_to_canvas(x: float, z: float):
	range_px = CANVAS_SIZE_PX - 2 * CANVAS_MARGIN_PX
	px = int(np.round(CANVAS_MARGIN_PX + (x - WORKSPACE_MIN) / (WORKSPACE_MAX - WORKSPACE_MIN) * range_px))
	py = int(np.round(CANVAS_MARGIN_PX + (WORKSPACE_MAX - z) / (WORKSPACE_MAX - WORKSPACE_MIN) * range_px))
	return px, py


def _canvas_to_world(px: int, py: int):
	range_px = CANVAS_SIZE_PX - 2 * CANVAS_MARGIN_PX
	x = WORKSPACE_MIN + (float(px - CANVAS_MARGIN_PX) / float(range_px)) * (WORKSPACE_MAX - WORKSPACE_MIN)
	z = WORKSPACE_MAX - (float(py - CANVAS_MARGIN_PX) / float(range_px)) * (WORKSPACE_MAX - WORKSPACE_MIN)
	x = float(np.clip(x, WORKSPACE_MIN, WORKSPACE_MAX))
	z = float(np.clip(z, WORKSPACE_MIN, WORKSPACE_MAX))
	return x, z


def _draw_cv2_editor_canvas(waypoints, pending_pos, pending_dir, pending_acc_unconstrained, choose_dir_mode, acc_mag):
	img = np.full((CANVAS_SIZE_PX, CANVAS_SIZE_PX, 3), 245, dtype=np.uint8)
	range_px = CANVAS_SIZE_PX - 2 * CANVAS_MARGIN_PX

	for i in range(11):
		g = int(np.round(CANVAS_MARGIN_PX + i * range_px / 10.0))
		cv2.line(img, (CANVAS_MARGIN_PX, g), (CANVAS_MARGIN_PX + range_px, g), (210, 210, 210), 1)
		cv2.line(img, (g, CANVAS_MARGIN_PX), (g, CANVAS_MARGIN_PX + range_px), (210, 210, 210), 1)

	cv2.rectangle(
		img,
		(CANVAS_MARGIN_PX, CANVAS_MARGIN_PX),
		(CANVAS_MARGIN_PX + range_px, CANVAS_MARGIN_PX + range_px),
		(0, 0, 0),
		2,
	)
	cv2.putText(img, "(0,0)", (CANVAS_MARGIN_PX - 18, CANVAS_MARGIN_PX + range_px + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1)
	cv2.putText(
		img,
		"(10,10)",
		(CANVAS_MARGIN_PX + range_px - 28, CANVAS_MARGIN_PX - 12),
		cv2.FONT_HERSHEY_SIMPLEX,
		0.5,
		(20, 20, 20),
		1,
	)

	for i, wp in enumerate(waypoints):
		pos = np.asarray(wp["pos"], dtype=float)
		px, py = _world_to_canvas(float(pos[0]), float(pos[2]))
		color = (20, 180, 20) if i in (0, len(waypoints) - 1) else (50, 140, 230)
		cv2.circle(img, (px, py), 6, color, -1)
		cv2.putText(img, f"{i}", (px + 8, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

		acc = wp["acc"]
		if acc is None:
			cv2.putText(img, "a:NaN", (px + 8, py + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (70, 70, 70), 1)
		else:
			acc = np.asarray(acc, dtype=float)
			end = _world_to_canvas(float(pos[0] + 0.05 * acc[0]), float(pos[2] + 0.05 * acc[2]))
			cv2.arrowedLine(img, (px, py), end, (20, 120, 220), 2, tipLength=0.2)

	if pending_pos is not None:
		pp = np.asarray(pending_pos, dtype=float)
		px, py = _world_to_canvas(float(pp[0]), float(pp[2]))
		cv2.circle(img, (px, py), 8, (30, 180, 220), 2)
		cv2.putText(img, "pending", (px + 10, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 120, 140), 1)
		if pending_acc_unconstrained:
			if choose_dir_mode:
				cv2.putText(img, "dir mode: click to set accel direction", (px + 10, py + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1)
			else:
				cv2.putText(img, "acc=NaN default (press c to confirm)", (px + 10, py + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1)
		elif pending_dir is not None:
			dir_end = (int(px + 28 * pending_dir[0]), int(py - 28 * pending_dir[1]))
			cv2.arrowedLine(img, (px, py), dir_end, (20, 120, 220), 2, tipLength=0.25)

	msg1 = "Left click: set waypoint position"
	msg2 = "u:undo last/pending | f:finish (>=2) | q or ESC:abort"
	msg3 = f"Default acc is NaN. Press d then click to set accel direction, or c to confirm NaN. Magnitude={acc_mag:.1f} [0,30]"
	msg4 = "First and last waypoints enforce zero velocity; others have unconstrained velocity"
	cv2.putText(img, msg1, (18, CANVAS_SIZE_PX - 72), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 10, 10), 1)
	cv2.putText(img, msg2, (18, CANVAS_SIZE_PX - 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 10, 10), 1)
	cv2.putText(img, msg3, (18, CANVAS_SIZE_PX - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 10, 10), 1)
	cv2.putText(img, msg4, (18, CANVAS_SIZE_PX - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 10, 10), 1)
	return img


def _collect_waypoints_interactive(client_id: int):
	_ = client_id
	window = "MPC Waypoint Editor (x-z)"
	click_state = {"point": None}
	waypoints = []
	pending_pos = None
	pending_dir = None
	pending_acc_unconstrained = True
	choose_dir_mode = False

	def _on_mouse(event, x, y, flags, param):
		_ = flags, param
		if event != cv2.EVENT_LBUTTONDOWN:
			return
		x_world, z_world = _canvas_to_world(x, y)
		click_state["point"] = np.array([x_world, z_world], dtype=float)

	def _noop(_):
		return None

	cv2.namedWindow(window, cv2.WINDOW_NORMAL)
	cv2.resizeWindow(window, CANVAS_SIZE_PX, CANVAS_SIZE_PX)
	cv2.setMouseCallback(window, _on_mouse)
	cv2.createTrackbar("acc_magnitude", window, 8, 30, _noop)

	try:
		while True:
			acc_mag = float(cv2.getTrackbarPos("acc_magnitude", window))

			point = click_state["point"]
			if point is not None:
				click_state["point"] = None
				if pending_pos is None:
					pending_pos = np.array([point[0], 0.0, point[1]], dtype=float)
					pending_dir = None
					pending_acc_unconstrained = True
					choose_dir_mode = False
				else:
					if choose_dir_mode:
						dx = float(point[0] - pending_pos[0])
						dz = float(point[1] - pending_pos[2])
						nrm = float(np.hypot(dx, dz))
						if nrm > 1e-6:
							dir_unit = np.array([dx / nrm, dz / nrm], dtype=float)
							pending_dir = dir_unit
							acc = np.array([acc_mag * dir_unit[0], 0.0, acc_mag * dir_unit[1]], dtype=float)
							waypoints.append({"pos": pending_pos.copy(), "acc": acc})
							pending_pos = None
							pending_dir = None
							pending_acc_unconstrained = True
							choose_dir_mode = False

			img = _draw_cv2_editor_canvas(waypoints, pending_pos, pending_dir, pending_acc_unconstrained, choose_dir_mode, acc_mag)
			cv2.imshow(window, img)
			key = cv2.waitKey(20) & 0xFF

			if key in (27, ord("q")):
				return None

			if key == ord("u"):
				if pending_pos is not None:
					pending_pos = None
					pending_dir = None
					pending_acc_unconstrained = True
					choose_dir_mode = False
				elif len(waypoints) > 0:
					waypoints.pop()

			if key == ord("d") and pending_pos is not None:
				choose_dir_mode = True
				pending_acc_unconstrained = False
				pending_dir = None

			if key == ord("c") and pending_pos is not None and pending_acc_unconstrained:
				waypoints.append({"pos": pending_pos.copy(), "acc": None})
				pending_pos = None
				pending_dir = None
				pending_acc_unconstrained = True
				choose_dir_mode = False

			if key == ord("f") and pending_pos is None and len(waypoints) >= 2:
				return waypoints
	finally:
		cv2.destroyWindow(window)


def _build_trajectory_from_waypoints(waypoints):
	nodes = []
	for i, wp in enumerate(waypoints):
		pos = np.asarray(wp["pos"], dtype=float)
		acc = None if wp["acc"] is None else np.asarray(wp["acc"], dtype=float)

		if i < len(waypoints) - 1:
			nxt = np.asarray(waypoints[i + 1]["pos"], dtype=float)
			d = nxt - pos
			psi = float(np.arctan2(d[2], d[0]))
		elif i > 0:
			prv = np.asarray(waypoints[i - 1]["pos"], dtype=float)
			d = pos - prv
			psi = float(np.arctan2(d[2], d[0]))
		else:
			psi = 0.0

		con_vel = np.zeros(3, dtype=float) if i in (0, len(waypoints) - 1) else None
		nodes.append(Node(pos=pos, psi=psi, con_vel=con_vel, con_acc=acc))

	segments = []
	for i in range(len(nodes) - 1):
		p0 = np.asarray(waypoints[i]["pos"], dtype=float)
		p1 = np.asarray(waypoints[i + 1]["pos"], dtype=float)
		dist = float(np.linalg.norm(p1 - p0))
		duration = max(0.4, dist / 4.0)
		segments.append(Segment(nodes[i], nodes[i + 1], duration=duration))

	return Trajectory(segments)


def _draw_trajectory_preview(trj, color, text, client_id: int):
	ids = []
	sampled = trj.sample_full_state(sampling_rate=TRAJ_SAMPLE_FREQ, include_terminal=True)
	pos = np.asarray(sampled["pos"], dtype=float)
	for i in range(pos.shape[0] - 1):
		a = [float(pos[i, 0]), float(pos[i, 1]), float(pos[i, 2])]
		b = [float(pos[i + 1, 0]), float(pos[i + 1, 1]), float(pos[i + 1, 2])]
		ids.append(p.addUserDebugLine(a, b, color, 2.0, physicsClientId=client_id))

	if pos.shape[0] > 0:
		ids.append(
			p.addUserDebugText(
				text,
				[float(pos[0, 0]), float(pos[0, 1]), float(pos[0, 2] + 0.4)],
				color,
				textSize=1.2,
				physicsClientId=client_id,
			)
		)
	return ids


def _preview_and_confirm_trajectory(trj_raw, trj_opt, client_id: int):
	preview_ids = []
	try:
		preview_ids.extend(_draw_trajectory_preview(trj_raw, [1.0, 0.6, 0.0], "generated", client_id))
		preview_ids.extend(_draw_trajectory_preview(trj_opt, [0.0, 1.0, 0.2], "optimized", client_id))
		preview_ids.append(
			p.addUserDebugText(
				"Preview: orange=generated, green=optimized.",
				[WORKSPACE_MIN, 0.0, WORKSPACE_MAX + 1.2],
				[1.0, 1.0, 1.0],
				textSize=1.2,
				physicsClientId=client_id,
			)
		)
		preview_ids.append(
			p.addUserDebugText(
				"Press C to CONTINUE, A to ABORT and return to edit.",
				[WORKSPACE_MIN, 0.0, WORKSPACE_MAX + 0.8],
				[1.0, 1.0, 1.0],
				textSize=1.2,
				physicsClientId=client_id,
			)
		)

		while True:
			key_events = p.getKeyboardEvents(physicsClientId=client_id)
			if _key_triggered(key_events, "c") or _key_triggered(key_events, "C"):
				return True
			if _key_triggered(key_events, "a") or _key_triggered(key_events, "A"):
				return False
			time.sleep(0.02)
	finally:
		_remove_debug_items(preview_ids, client_id)


def _run_tracking(env: MPCControlEnv):
	client_id = env.CLIENT
	stop_btn = _add_button("STOP_RUN", client_id)
	prev_stop = p.readUserDebugParameter(stop_btn, physicsClientId=client_id)
	freq = env.MPC_FREQ
	try:
		max_steps = 1000
		for k in range(max_steps):
			start = time.time()
			_, _, terminated, truncated, info = env.step(None)
			if k % 50 == 0:
				print(
					f"[MPC_DEMO] step={k:04d} status={info['mpc_status']} "
					f"state_err={info['state_error_norm']:.3f} out_err={info['output_error_norm']:.3f}"
				)

			pressed_stop, prev_stop = _button_pressed(stop_btn, prev_stop, client_id)
			if pressed_stop:
				print("[MPC_DEMO] User stopped run")
				break
			if terminated or truncated:
				print(f"[MPC_DEMO] Finished: terminated={terminated} truncated={truncated}")
				break
			if (time.time() - start) < 1.0 / freq:
				time.sleep(1.0 / freq - (time.time() - start))
	finally:
		_remove_debug_items([stop_btn], client_id)


def _post_run_action_prompt(client_id: int):
	"""Return one of: 'repeat', 'edit', 'quit'."""
	prompt_ids = []
	try:
		prompt_ids.append(
			p.addUserDebugText(
				"Run complete. Press R to repeat last performance, E to edit waypoints, Q to quit.",
				[WORKSPACE_MIN, 0.0, WORKSPACE_MAX + 0.8],
				[1.0, 1.0, 1.0],
				textSize=1.2,
				physicsClientId=client_id,
			)
		)

		while True:
			key_events = p.getKeyboardEvents(physicsClientId=client_id)
			if _key_triggered(key_events, "r") or _key_triggered(key_events, "R"):
				return "repeat"
			if _key_triggered(key_events, "e") or _key_triggered(key_events, "E"):
				return "edit"
			if _key_triggered(key_events, "q") or _key_triggered(key_events, "Q"):
				return "quit"
			time.sleep(0.02)
	finally:
		_remove_debug_items(prompt_ids, client_id)


def main():
	env = MPCControlEnv(
		gui=True,
		record=False,
	)

	# Initialize env first so PyBullet GUI exists before waypoint input.
	env.reset()
	client_id = env.CLIENT

	p.resetDebugVisualizerCamera(
		cameraDistance=10.0,
		cameraYaw=0.0,
		cameraPitch=0.0,
		cameraTargetPosition=[5.0, 0.0, 5.0],
		physicsClientId=client_id,
	)

	static_grid_ids = _draw_workspace_grid(client_id)
	last_trj_opt = None

	try:
		while True:
			if last_trj_opt is None:
				waypoints = _collect_waypoints_interactive(client_id)
				if waypoints is None:
					print("[MPC_DEMO] Editor aborted")
					break

				trj_raw = _build_trajectory_from_waypoints(waypoints)
				time_penalty = np.array([200.0 for _ in trj_raw._segments], dtype=float)
				trj_opt, optimized_time, _ = optimize_trj_time(
					trj_raw,
					time_penalty=time_penalty,
					preserve_total_time=False,
					min_velocity=0.0,
					max_velocity=30.0,
					max_normalized_thrust=env.MASS_NORMALIZED_THRUST_BOUNDS[1],
					report_peaks=True,
				)
				print(f"[MPC_DEMO] optimized trajectory time={optimized_time}s")

				should_run = _preview_and_confirm_trajectory(trj_raw, trj_opt, client_id)
				if not should_run:
					continue
				last_trj_opt = trj_opt
			else:
				trj_opt = last_trj_opt

			env.reset(
				options={
					"trajectory_obj": trj_opt,
					"trajectory_sample_freq": TRAJ_SAMPLE_FREQ,
				}
			)
			print("[MPC_DEMO] Starting MPC tracking")
			_run_tracking(env)

			action = _post_run_action_prompt(client_id)
			if action == "repeat":
				print("[MPC_DEMO] Repeating last performance")
				continue
			if action == "edit":
				last_trj_opt = None
				print("[MPC_DEMO] Returning to waypoint editor")
				continue
			print("[MPC_DEMO] Exiting")
			break
	finally:
		_remove_debug_items(static_grid_ids, client_id)
		env.close()


if __name__ == "__main__":
	main()
