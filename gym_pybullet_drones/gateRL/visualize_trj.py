import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_trajectory_csv(csv_path):
	"""Load timestamp and drone position columns from a mocap CSV file."""
	timestamps, x, y, z = [], [], [], []
	with open(csv_path, "r", newline="") as f:
		reader = csv.DictReader(f)
		required_cols = ["timestamp", "drone_x", "drone_y", "drone_z"]
		missing = [c for c in required_cols if c not in reader.fieldnames]
		if missing:
			raise ValueError(f"Missing required column(s): {missing}")

		for row in reader:
			timestamps.append(float(row["timestamp"]))
			x.append(float(row["drone_x"]))
			y.append(float(row["drone_y"]))
			z.append(float(row["drone_z"]))

	t = np.asarray(timestamps, dtype=np.float64)
	if t.size == 0:
		raise ValueError("CSV has no data rows")

	# Convert usec timestamp to seconds elapsed from start.
	t = (t - t[0]) / 1e6
	return t, np.asarray(x), np.asarray(y), np.asarray(z)


def _downsample(*arrays, stride=1):
	if stride <= 1:
		return arrays
	return tuple(arr[::stride] for arr in arrays)


def plot_trajectory(t, x, y, z, title="Trajectory", save_path=None):
	"""Plot 3D trajectory and 2D projections."""
	fig = plt.figure(figsize=(12, 9))

	ax3d = fig.add_subplot(2, 2, 1, projection="3d")
	ax_xy = fig.add_subplot(2, 2, 2)
	ax_xz = fig.add_subplot(2, 2, 3)
	ax_yz = fig.add_subplot(2, 2, 4)

	# 3D path
	ax3d.plot(x, y, z, linewidth=1.2, color="tab:blue")
	ax3d.scatter([x[0]], [y[0]], [z[0]], color="green", s=35, label="start")
	ax3d.scatter([x[-1]], [y[-1]], [z[-1]], color="red", s=35, label="end")
	ax3d.set_xlabel("x [m]")
	ax3d.set_ylabel("y [m]")
	ax3d.set_zlabel("z [m]")
	ax3d.set_title("3D")
	ax3d.legend(loc="best")
	ax3d.grid(True, alpha=0.3)

	# 2D projections
	ax_xy.plot(x, y, color="tab:orange", linewidth=1.1)
	ax_xy.set_xlabel("x [m]")
	ax_xy.set_ylabel("y [m]")
	ax_xy.set_title("XY")
	ax_xy.grid(True, alpha=0.3)

	ax_xz.plot(x, z, color="tab:purple", linewidth=1.1)
	ax_xz.set_xlabel("x [m]")
	ax_xz.set_ylabel("z [m]")
	ax_xz.set_title("XZ")
	ax_xz.grid(True, alpha=0.3)

	ax_yz.plot(y, z, color="tab:brown", linewidth=1.1)
	ax_yz.set_xlabel("y [m]")
	ax_yz.set_ylabel("z [m]")
	ax_yz.set_title("YZ")
	ax_yz.grid(True, alpha=0.3)

	fig.suptitle(title)
	fig.tight_layout()

	if save_path is not None:
		save_path = Path(save_path)
		save_path.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(save_path, dpi=180)
		print(f"Saved trajectory figure: {save_path}")

	plt.show()


def main():
	default_csv = Path(__file__).resolve().parent / "data" / "piloted" / "flight-01p-ellipse" / "csv_raw" / "mocap_flight-01p-ellipse.csv"

	parser = argparse.ArgumentParser(description="Visualize drone trajectory from mocap CSV")
	parser.add_argument("--csv", type=str, default=str(default_csv), help="Path to input CSV")
	parser.add_argument("--stride", type=int, default=10, help="Downsample stride for plotting")
	parser.add_argument("--save", type=str, default=None, help="Optional output image path")
	args = parser.parse_args()

	csv_path = Path(args.csv)
	t, x, y, z = load_trajectory_csv(csv_path)
	t, x, y, z = _downsample(t, x, y, z, stride=max(1, args.stride))

	title = f"Drone Trajectory\n{csv_path.name} | N={len(x)}"
	plot_trajectory(t, x, y, z, title=title, save_path=args.save)


if __name__ == "__main__":
	main()
