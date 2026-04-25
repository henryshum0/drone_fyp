import numpy as np

from gym_pybullet_drones.gateRL.waypoints.WaypointTemplate import *


def _default_spawn():
	return [
		{
			"pos": np.array([-2, 0.0, 0.2]),
			"vel": np.array([0.0, 0.0, 0.0]),
			"acc": np.array([0.0, 0.0, 0.0]),
			"rpy": np.array([0.0, 0.0, 0.0]),
			"next_waypoints": [0, 1],
		}
	]


def _with_entry_waypoint(waypoints_xyzs, waypoints_rpys, waypoints_normal_distr):
	"""Prepends a fixed entry gate before the main maneuver waypoints."""
	entry_xyz = np.array([[-2, 0.0, 0.2]])
	entry_rpy = np.array([[0.0, 0.0, 0.0]])
	entry_speed = np.array([0.])
	entry_acceleration = [np.array([0., 0., 0.])]
	entry_duration = np.array([1])
	entry_noise = np.array([[[0.0, 0.05], [0.0, 0.05], [0.0, 0.05]]])

	waypoints_xyzs = np.vstack([entry_xyz, waypoints_xyzs])
	waypoints_rpys = np.vstack([entry_rpy, waypoints_rpys])
	waypoints_normal_distr = np.vstack([entry_noise, waypoints_normal_distr])
	return waypoints_xyzs, waypoints_rpys, waypoints_normal_distr, entry_speed, entry_acceleration, entry_duration


def _fixed_rpy_choices(waypoints_rpys):
	"""Keep WaypointTemplate API but enforce deterministic RPYs per waypoint."""
	return [[np.array(rpy, dtype=float)] for rpy in waypoints_rpys]


def _empty_waypoint_accelerations(waypoints_xyzs):
	n = int(np.asarray(waypoints_xyzs).shape[0])
	return np.full((n, 3), np.nan, dtype=float)

class HeartTemplate(WaypointTemplate):

	def __init__(self):
		waypoints_xyzs = np.array([
			[-2, 0.0, 1],
			[0., 0.0, 1.],
			[2, 0, 3],
			[1, 0.0, 4],
			[-0.5, 0, 3],
			[0, 0., 2],
			[0.5, 0, 3],
			[-1., 0.0, 4],
			[-2., 0.0, 3],
			[0., 0.0, 1.],
		])
		waypoints_rpys = np.array([
			[0.0, 0.0, 0],
			[0.0, 0.0, 0],
			[0.0, 0.0, 0],
			[0.0, 0.0, 0],
			[0.0, 0.0, 0],
			[0.0, 0.0, 0],
			[0.0, 0.0, 0],
			[0, np.pi, 0],
			[0.0, 0.0, 0],
			[0.0, 0.0, 0],
		])
		waypoints_speeds = np.array([
			0,
			None,
			None,
			None,
			None,
			None,
			None,
			None,
			None,
			5,
		])
		waypoints_accelerations = [
			None,
			None,
			None,
			np.array([0., 0., -10.]),
			np.array([10., 0., 0.]),
			np.array([0., 0., 10.]),
			np.array([-10., 0., 0.]),
			np.array([0., 0., -10.]),
			None,
			None,
		]
		waypoints_durations = np.array([
			0.,
			1,
			1,
			1,
			1,
			1,
			1,
			1,
			1,
			1
		])
		waypoints_normal_distr = np.array([
			[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
			[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
			[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
			[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
			[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
			[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
			[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
			[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
			[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
			[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
		])

		waypoints_rpys_choices = _fixed_rpy_choices(waypoints_rpys)
		waypoints_scale = [2,2]

		super().__init__(
			waypoints_xyzs=waypoints_xyzs,
			waypoints_rpys=waypoints_rpys,
			waypoints_speeds=waypoints_speeds,
			waypoints_durations=waypoints_durations,
			spawns=_default_spawn(),
			waypoints_normal_distr=waypoints_normal_distr,
			rpy_choices=waypoints_rpys_choices,
			waypoints_scale=waypoints_scale,
			max_dist=7,
			difficulty="easy",
			repeat=0,
			time_limit_sec=5,
		)
		self.waypoints_accelerations = waypoints_accelerations

class PowerloopTemplate(WaypointTemplate):
	"""Back-roll pair: same XY, first waypoint lower than second."""

	def __init__(self):
		waypoints_xyzs = np.array([
			[-2, 0.0, 1],
			[0., 0.0, 1.],
			[1, 0, 2],
			[0., 0.0, 3],
			[-1, 0., 2],
			[0., 0.0, 1],
			[2., 0.0, 1],
		])
		waypoints_rpys = np.array([
			[0.0, 0.0, 0],
			[0.0, 0.0, 0],
			[0.0, 0.0, 0],
			[0.0, 0.0, 0],
			[0, np.pi, 0],
			[0.0, 0.0, 0],
			[0.0, 0.0, 0],
		])
		waypoints_speeds = np.array([
			0,
			None,
			None,
			None,
			None,
			None,
			5,
		])
		waypoints_accelerations = [
			None,
			None,
			None,
			np.array([0., 0., -10.]),
			None,
			None,
			None,
		]
		waypoints_durations = np.array([
			0.,
			0.5,
			0.5,
			0.5,
			0.5,
			0.5,
			0.5
		])
		waypoints_normal_distr = np.array([
			[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
			[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
			[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
			[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
			[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
			[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
			[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
		])

		waypoints_rpys_choices = _fixed_rpy_choices(waypoints_rpys)
		waypoints_scale = [2,2]

		super().__init__(
			waypoints_xyzs=waypoints_xyzs,
			waypoints_rpys=waypoints_rpys,
			waypoints_speeds=waypoints_speeds,
			waypoints_durations=waypoints_durations,
			spawns=_default_spawn(),
			waypoints_normal_distr=waypoints_normal_distr,
			rpy_choices=waypoints_rpys_choices,
			waypoints_scale=waypoints_scale,
			max_dist=7,
			difficulty="easy",
			repeat=0,
			time_limit_sec=5,
		)
		self.waypoints_accelerations = waypoints_accelerations


class SplitSLeftTemplate(WaypointTemplate):
	"""Split-S: invert, descend, and exit in the opposite heading."""

	def __init__(self):
		waypoints_xyzs = np.array([
			[1.0, 0.0, 1.5],
			[2, -1, 1.5],
			[3., 1 , 1.5],
			[1, 0 , 0.5],
			
		])
		waypoints_speeds = np.array([
			None,
			None,
			10,
			None,
			
		])
		waypoints_durations = np.array([
			1.00,
			1.10,
			0.90,
			1.10,
			
		])
		waypoints_rpys = np.array([
			RPY_FRONT_UP,
			RPY_FRONT_UP,
			RPY_LEFT_BACK,
			RPY_BACK_UP,
			
		])
		waypoints_normal_distr = np.array([
			[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
			[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
			[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
			[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
			
		])
		waypoints_xyzs, waypoints_rpys, waypoints_normal_distr, entry_speed, _, entry_duration = _with_entry_waypoint(
			waypoints_xyzs,
			waypoints_rpys,
			waypoints_normal_distr,
		)
		waypoints_speeds = np.concatenate([entry_speed, waypoints_speeds])
		waypoints_durations = np.concatenate([entry_duration, waypoints_durations])
		waypoints_accelerations = _empty_waypoint_accelerations(waypoints_xyzs)
		waypoints_rpys_choices = _fixed_rpy_choices(waypoints_rpys)
		waypoints_scale = [1,2]

		super().__init__(
			waypoints_xyzs=waypoints_xyzs,
			waypoints_rpys=waypoints_rpys,
			waypoints_speeds=waypoints_speeds,
			waypoints_durations=waypoints_durations,
			spawns=_default_spawn(),
			waypoints_normal_distr=waypoints_normal_distr,
			rpy_choices=waypoints_rpys_choices,
			waypoints_scale=waypoints_scale,
			max_dist=7,
			difficulty="easy",
			repeat=0,
			time_limit_sec=7,
		)
		self.waypoints_accelerations = waypoints_accelerations
		
class SplitSRightTemplate(WaypointTemplate):
	"""Split-S: invert, descend, and exit in the opposite heading."""

	def __init__(self):
		waypoints_xyzs = np.array([
			[1.0, 0.0, 1.5],
			[2, 1, 1.5],
			[3., -1 , 1.5],
			[1, 0 , 0.5],
			
		])
		waypoints_speeds = np.array([
			None,
			None,
			10,
			None,
			
		])
		waypoints_durations = np.array([
			1.00,
			1.10,
			0.90,
			1.10,
			
		])
		waypoints_rpys = np.array([
			RPY_FRONT_UP,
			RPY_FRONT_UP,
			RPY_RIGHT_BACK,
			RPY_BACK_UP,
			
		])
		waypoints_normal_distr = np.array([
			[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
			[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
			[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
			[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
			
		])
		waypoints_xyzs, waypoints_rpys, waypoints_normal_distr, entry_speed, _, entry_duration = _with_entry_waypoint(
			waypoints_xyzs,
			waypoints_rpys,
			waypoints_normal_distr,
		)
		waypoints_speeds = np.concatenate([entry_speed, waypoints_speeds])
		waypoints_durations = np.concatenate([entry_duration, waypoints_durations])
		waypoints_accelerations = _empty_waypoint_accelerations(waypoints_xyzs)
		waypoints_rpys_choices = _fixed_rpy_choices(waypoints_rpys)
		waypoints_scale = [2, 3]

		super().__init__(
			waypoints_xyzs=waypoints_xyzs,
			waypoints_rpys=waypoints_rpys,
			waypoints_speeds=waypoints_speeds,
			waypoints_durations=waypoints_durations,
			spawns=_default_spawn(),
			waypoints_normal_distr=waypoints_normal_distr,
			rpy_choices=waypoints_rpys_choices,
			waypoints_scale=waypoints_scale,
			max_dist=7,
			difficulty="easy",
			repeat=0,
			time_limit_sec=7,
		)
		self.waypoints_accelerations = waypoints_accelerations

class BarrelRollLeftTemplate(WaypointTemplate):
	"""Barrel roll progression with forward travel through the roll."""

	def __init__(self):
		waypoints_xyzs = np.array([
			[1.0, 0., 1],
			[2, 1., 2.5],
			[2.5, 2., 1],
			[5, 2., 1],

		])
		waypoints_speeds = np.array([
			None,
			None,
			None,
			10,
		])
		waypoints_durations = np.array([
			1.00,
			0.2,
			1,
			1.0
		])
		waypoints_accelerations = [
			None,
			np.array([0., 0., -20.]),
			None,
			None,
		]
		waypoints_rpys = np.array([
			RPY_FRONT_UP,
			RPY_FRONT_DOWN,
			RPY_FRONT_UP,
			RPY_FRONT_UP,

		])
		waypoints_normal_distr = np.array([
			[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
			[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
			[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
			[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
		])
		waypoints_xyzs, waypoints_rpys, waypoints_normal_distr, entry_speed, entry_acceleration, entry_duration = _with_entry_waypoint(
			waypoints_xyzs,
			waypoints_rpys,
			waypoints_normal_distr,
		)
		waypoints_speeds = np.concatenate([entry_speed, waypoints_speeds])
		waypoints_durations = np.concatenate([entry_duration, waypoints_durations])
		waypoints_accelerations = entry_acceleration + waypoints_accelerations
		waypoints_rpys_choices = _fixed_rpy_choices(waypoints_rpys)
		waypoints_scale = [1,2]

		super().__init__(
			waypoints_xyzs=waypoints_xyzs,
			waypoints_rpys=waypoints_rpys,
			waypoints_speeds=waypoints_speeds,
			waypoints_durations=waypoints_durations,
			spawns=_default_spawn(),
			waypoints_normal_distr=waypoints_normal_distr,
			rpy_choices=waypoints_rpys_choices,
			waypoints_scale=waypoints_scale,
			max_dist=7,
			difficulty="easy",
			repeat=0,
			time_limit_sec=7,
		)
		self.waypoints_accelerations = waypoints_accelerations

class BarrelRollRightTemplate(WaypointTemplate):
	"""Barrel roll progression with forward travel through the roll."""

	def __init__(self):
		waypoints_xyzs = np.array([
			[1.0, 0., 1],
			[2, -1., 2.5],
			[2.5, -2., 1],
			[5, -2., 1],

		])
		waypoints_speeds = np.array([
			None,
			None,
			None,
			10,
		])
		waypoints_durations = np.array([
			1.00,
			0.2,
			1,
			1.0
		])
		waypoints_accelerations = [
			None,
			np.array([0., 0., -20.]),
			None,
			None,
		]
		waypoints_rpys = np.array([
			RPY_FRONT_UP,
			RPY_FRONT_DOWN,
			RPY_FRONT_UP,
			RPY_FRONT_UP,

		])
		waypoints_normal_distr = np.array([
			[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
			[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
			[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
			[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
		])
		waypoints_xyzs, waypoints_rpys, waypoints_normal_distr, entry_speed, entry_acceleration, entry_duration = _with_entry_waypoint(
			waypoints_xyzs,
			waypoints_rpys,
			waypoints_normal_distr,
		)
		waypoints_speeds = np.concatenate([entry_speed, waypoints_speeds])
		waypoints_durations = np.concatenate([entry_duration, waypoints_durations])
		waypoints_accelerations = entry_acceleration + waypoints_accelerations
		waypoints_rpys_choices = _fixed_rpy_choices(waypoints_rpys)
		waypoints_scale = [1,2]

		super().__init__(
			waypoints_xyzs=waypoints_xyzs,
			waypoints_rpys=waypoints_rpys,
			waypoints_speeds=waypoints_speeds,
			waypoints_durations=waypoints_durations,
			spawns=_default_spawn(),
			waypoints_normal_distr=waypoints_normal_distr,
			rpy_choices=waypoints_rpys_choices,
			waypoints_scale=waypoints_scale,
			max_dist=7,
			difficulty="easy",
			repeat=0,
			time_limit_sec=7,
		)
		self.waypoints_accelerations = waypoints_accelerations
		



TRAIN_TEMPLATES2 = [
	PowerloopTemplate,
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