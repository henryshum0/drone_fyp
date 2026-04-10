import numpy as np

from gym_pybullet_drones.gateRL.waypoints.WaypointTemplate import *


def _default_spawn():
	return [
		{
			"pos": np.array([-2, 0.0, 0.0]),
			"vel": np.array([0.0, 0.0, 0.0]),
			"acc": np.array([0.0, 0.0, 0.0]),
			"rpy": np.array([0.0, 0.0, 0.0]),
			"next_waypoints": [0, 1],
		}
	]


def _with_entry_waypoint(waypoints_xyzs, waypoints_rpys, waypoints_normal_distr):
	"""Prepends a fixed entry gate before the main maneuver waypoints."""
	entry_xyz = np.array([[-1, 0.0, 0.0]])
	entry_rpy = np.array([[0.0, 0.0, 0.0]])
	entry_noise = np.array([[[0.0, 0.05], [0.0, 0.05], [0.0, 0.05]]])

	waypoints_xyzs = np.vstack([entry_xyz, waypoints_xyzs])
	waypoints_rpys = np.vstack([entry_rpy, waypoints_rpys])
	waypoints_normal_distr = np.vstack([entry_noise, waypoints_normal_distr])
	return waypoints_xyzs, waypoints_rpys, waypoints_normal_distr


def _fixed_rpy_choices(waypoints_rpys):
	"""Keep WaypointTemplate API but enforce deterministic RPYs per waypoint."""
	return [[np.array(rpy, dtype=float)] for rpy in waypoints_rpys]


class BackRollTemplate(WaypointTemplate):
	"""Back-roll pair: same XY, first waypoint lower than second."""

	def __init__(self):
		waypoints_xyzs = np.array([
			[0.8, 0.0, -0.35],
			[0.8, 0.0, 0.35],
		])
		waypoints_rpys = np.array([
			[0.0, 0.0, 0.0],
			[3.14159265, 0.0, 3.14159265],
		])
		waypoints_normal_distr = np.array([
			[[0.0, 0.1], [0.0, 0.1], [0.0, 0.10]],
			[[0.0, 0.1], [0.0, 0.1], [0.0, 0.10]],
		])
		waypoints_xyzs, waypoints_rpys, waypoints_normal_distr = _with_entry_waypoint(
			waypoints_xyzs,
			waypoints_rpys,
			waypoints_normal_distr,
		)
		waypoints_rpys_choices = _fixed_rpy_choices(waypoints_rpys)
		waypoints_scale = [0.9, 3]

		super().__init__(
			waypoints_xyzs=waypoints_xyzs,
			waypoints_rpys=waypoints_rpys,
			spawns=_default_spawn(),
			waypoints_normal_distr=waypoints_normal_distr,
			rpy_choices=waypoints_rpys_choices,
			waypoints_scale=waypoints_scale,
			max_dist=5,
			difficulty="easy",
			repeat=0,
			time_limit_sec=5,
		)


class FrontRollTemplate(WaypointTemplate):
	"""Front-roll pair: same XY, first waypoint higher than second."""

	def __init__(self):
		waypoints_xyzs = np.array([
			[0.8, 0.0, 0.35],
			[0.8, 0.0, -0.35],
		])
		waypoints_rpys = np.array([
			[3.14159265, 0.0, 0.0],
			[0.0, 0.0, 3.14159265],
		])
		waypoints_normal_distr = np.array([
			[[0.0, 0.1], [0.0, 0.1], [0.0, 0.10]],
			[[0.0, 0.1], [0.0, 0.1], [0.0, 0.10]],
		])
		waypoints_xyzs, waypoints_rpys, waypoints_normal_distr = _with_entry_waypoint(
			waypoints_xyzs,
			waypoints_rpys,
			waypoints_normal_distr,
		)
		waypoints_rpys_choices = _fixed_rpy_choices(waypoints_rpys)
		waypoints_scale = [0.9, 3]

		super().__init__(
			waypoints_xyzs=waypoints_xyzs,
			waypoints_rpys=waypoints_rpys,
			spawns=_default_spawn(),
			waypoints_normal_distr=waypoints_normal_distr,
			rpy_choices=waypoints_rpys_choices,
			waypoints_scale=waypoints_scale,
			max_dist=5,
			difficulty="easy",
			repeat=0,
			time_limit_sec=5,
		)


class SplitSLeftTemplate(WaypointTemplate):
	"""Split-S: invert, descend, and exit in the opposite heading."""

	def __init__(self):
		waypoints_xyzs = np.array([
			[0.0, 0.0, 0.],
			[1, 0.5, 0.],
			[1.5, 0.75 , -0.5],
			[-0.5, 0.75, -1.5],
		])
		waypoints_rpys = np.array([
			RPY_FRONT_UP,
			RPY_FRONT_LEFT,
			RPY_DOWN_BACK,
			RPY_BACK_UP,
		])
		waypoints_normal_distr = np.array([
			[[0.0, 0.1], [0.0, 0.1], [0.0, 0.1]],
			[[0.0, 0.15], [0.0, 0.15], [0.0, 0.10]],
			[[0.0, 0.20], [0.0, 0.12], [0.0, 0.10]],
			[[0.0, 0.20], [0.0, 0.12], [0.0, 0.10]],
		])
		waypoints_xyzs, waypoints_rpys, waypoints_normal_distr = _with_entry_waypoint(
			waypoints_xyzs,
			waypoints_rpys,
			waypoints_normal_distr,
		)
		waypoints_rpys_choices = _fixed_rpy_choices(waypoints_rpys)
		waypoints_scale = [0.85, 1.35]

		super().__init__(
			waypoints_xyzs=waypoints_xyzs,
			waypoints_rpys=waypoints_rpys,
			spawns=_default_spawn(),
			waypoints_normal_distr=waypoints_normal_distr,
			rpy_choices=waypoints_rpys_choices,
			waypoints_scale=waypoints_scale,
			max_dist=3,
			difficulty="easy",
			repeat=0,
			time_limit_sec=6,
		)
		
class SplitSRightTemplate(WaypointTemplate):
	"""Split-S: invert, descend, and exit in the opposite heading."""

	def __init__(self):
		waypoints_xyzs = np.array([
			[0.0, 0.0, 0.],
			[1, -0.5, 0.],
			[1.5, -0.75, -0.5],
			[-0.5, -0.75, -1.5],
		])
		waypoints_rpys = np.array([
			RPY_FRONT_UP,
			RPY_FRONT_RIGHT,
			RPY_DOWN_BACK,
			RPY_BACK_UP,
		])
		waypoints_normal_distr = np.array([
			[[0.0, 0.1], [0.0, 0.1], [0.0, 0.1]],
			[[0.0, 0.15], [0.0, 0.15], [0.0, 0.10]],
			[[0.0, 0.20], [0.0, 0.12], [0.0, 0.10]],
			[[0.0, 0.20], [0.0, 0.12], [0.0, 0.10]],
		])
		waypoints_xyzs, waypoints_rpys, waypoints_normal_distr = _with_entry_waypoint(
			waypoints_xyzs,
			waypoints_rpys,
			waypoints_normal_distr,
		)
		waypoints_rpys_choices = _fixed_rpy_choices(waypoints_rpys)
		waypoints_scale = [0.85, 1.35]

		super().__init__(
			waypoints_xyzs=waypoints_xyzs,
			waypoints_rpys=waypoints_rpys,
			spawns=_default_spawn(),
			waypoints_normal_distr=waypoints_normal_distr,
			rpy_choices=waypoints_rpys_choices,
			waypoints_scale=waypoints_scale,
			max_dist=3,
			difficulty="easy",
			repeat=0,
			time_limit_sec=6,
		)


class BarrelRollRightTemplate(WaypointTemplate):
	"""Barrel roll progression with forward travel through the roll."""

	def __init__(self):
		waypoints_xyzs = np.array([
			[0.0, 0.0, 0.0],
			[1, -0.5, 0.6],
			[1.5, -0.7, 1],
			[2.0, -0.5, 0.6],
			[4.0, 0.0, 0.0],
		])
		waypoints_rpys = np.array([
			RPY_FRONT_UP,
			RPY_FRONT_RIGHT + np.array([0.0, 0.0, 0.4]),
			RPY_FRONT_DOWN,
			RPY_FRONT_LEFT + np.array([0.0, 0.0, -0.4]),
			RPY_FRONT_UP,
		])
		waypoints_normal_distr = np.array([
			[[0.0, 0.1], [0.0, 0.1], [0.0, 0.1]],
			[[0.0, 0.12], [0.0, 0.12], [0.0, 0.08]],
			[[0.0, 0.12], [0.0, 0.12], [0.0, 0.08]],
			[[0.0, 0.12], [0.0, 0.12], [0.0, 0.08]],
			[[0.0, 0.12], [0.0, 0.12], [0.0, 0.08]],
		])
		waypoints_xyzs, waypoints_rpys, waypoints_normal_distr = _with_entry_waypoint(
			waypoints_xyzs,
			waypoints_rpys,
			waypoints_normal_distr,
		)
		waypoints_rpys_choices = _fixed_rpy_choices(waypoints_rpys)
		waypoints_scale = [0.85, 1.30]

		super().__init__(
			waypoints_xyzs=waypoints_xyzs,
			waypoints_rpys=waypoints_rpys,
			spawns=_default_spawn(),
			waypoints_normal_distr=waypoints_normal_distr,
			rpy_choices=waypoints_rpys_choices,
			waypoints_scale=waypoints_scale,
			max_dist=3,
			difficulty="easy",
			repeat=0,
			time_limit_sec=6,
		)
		
class BarrelRollLeftTemplate(WaypointTemplate):
	"""Barrel roll progression with forward travel through the roll."""

	def __init__(self):
		waypoints_xyzs = np.array([
			[0.0, 0.0, 0.0],
			[1, 0.5, 0.6],
			[1.5, 0.7, 1],
			[2.0, 0.5, 0.6],
			[4.0, 0.0, 0.0],
		])
		waypoints_rpys = np.array([
			RPY_FRONT_UP,
			RPY_FRONT_LEFT + np.array([0.0, 0.0, 0.4]),
			RPY_FRONT_DOWN,
			RPY_FRONT_RIGHT + np.array([0.0, 0.0, -0.4]),
			RPY_FRONT_UP,
		])
		waypoints_normal_distr = np.array([
			[[0.0, 0.1], [0.0, 0.1], [0.0, 0.1]],
			[[0.0, 0.12], [0.0, 0.12], [0.0, 0.08]],
			[[0.0, 0.12], [0.0, 0.12], [0.0, 0.08]],
			[[0.0, 0.12], [0.0, 0.12], [0.0, 0.08]],
			[[0.0, 0.12], [0.0, 0.12], [0.0, 0.08]],
		])
		waypoints_xyzs, waypoints_rpys, waypoints_normal_distr = _with_entry_waypoint(
			waypoints_xyzs,
			waypoints_rpys,
			waypoints_normal_distr,
		)
		waypoints_rpys_choices = _fixed_rpy_choices(waypoints_rpys)
		waypoints_scale = [0.85, 1.30]

		super().__init__(
			waypoints_xyzs=waypoints_xyzs,
			waypoints_rpys=waypoints_rpys,
			spawns=_default_spawn(),
			waypoints_normal_distr=waypoints_normal_distr,
			rpy_choices=waypoints_rpys_choices,
			waypoints_scale=waypoints_scale,
			max_dist=3,
			difficulty="easy",
			repeat=0,
			time_limit_sec=6,
		)


TRAIN_TEMPLATES2 = [
	BackRollTemplate,
	FrontRollTemplate,
	SplitSLeftTemplate,
	SplitSRightTemplate,
	BarrelRollRightTemplate,
	BarrelRollLeftTemplate,
]


def visualize_all_templates(randomized=False, show_orientation=True, cols=3):
	"""Visualize all templates in TRAIN_TEMPLATES2 in a single figure grid."""
	import math
	import matplotlib.pyplot as plt

	total = len(TRAIN_TEMPLATES2)
	cols = max(1, int(cols))
	rows = int(math.ceil(total / cols))
	fig = plt.figure(figsize=(5 * cols, 4.5 * rows))

	for i, template_cls in enumerate(TRAIN_TEMPLATES2):
		ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
		template = template_cls()
		template.visualize_waypoints(
			randomized=randomized,
			show_orientation=show_orientation,
			show_spawn=True,
			show=False,
			ax=ax,
			title=template_cls.__name__,
		)

	plt.tight_layout()
	plt.show()


if __name__ == "__main__":
	visualize_all_templates(randomized=True, show_orientation=True, cols=3)