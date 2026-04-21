from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np

try:
	import cv2
except ImportError as exc:  # pragma: no cover
	raise ImportError(
		"vins_frontend requires OpenCV. Install with: pip install opencv-python"
	) from exc


@dataclass
class FrontendConfig:
	max_features: int = 200
	min_features: int = 120
	quality_level: float = 0.01
	min_distance: float = 25.0
	block_size: int = 3
	clahe: bool = True
	clahe_clip_limit: float = 2.0
	clahe_tile_grid_size: tuple[int, int] = (8, 8)
	lk_win_size: tuple[int, int] = (21, 21)
	lk_max_level: int = 3
	lk_iters: int = 30
	lk_eps: float = 0.01
	f_ransac_reproj_threshold: float = 1.5
	f_ransac_confidence: float = 0.99
	f_ransac_max_iters: int = 500
	min_parallax_px: float = 10.0


@dataclass
class FrontendOutput:
	timestamp: float
	image_size: tuple[int, int]
	is_initialized: bool
	is_keyframe: bool
	num_tracked: int
	num_new: int
	points: np.ndarray
	undistorted_points: np.ndarray
	feature_ids: np.ndarray
	track_lengths: np.ndarray
	id_to_point: Dict[int, np.ndarray]


class VinsMonoFrontend:
	"""A compact VINS-MONO-style visual frontend.

	Pipeline per frame:
	1) Preprocess image (grayscale + optional CLAHE)
	2) Track existing points using pyramidal LK optical flow
	3) Reject outliers using RANSAC fundamental matrix
	4) Keep longest tracks first and detect new corners to replenish
	5) Compute undistorted normalized points and keyframe decision
	"""

	def __init__(
		self,
		K: np.ndarray,
		dist_coeffs: Optional[np.ndarray] = None,
		config: Optional[FrontendConfig] = None,
	):
		self.config = config if config is not None else FrontendConfig()

		self.K = np.asarray(K, dtype=np.float64)
		if self.K.shape != (3, 3):
			raise ValueError(f"K must be shape (3, 3), got {self.K.shape}")

		if dist_coeffs is None:
			self.dist_coeffs = np.zeros((5, 1), dtype=np.float64)
		else:
			dc = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1)
			if dc.size not in (4, 5, 8):
				raise ValueError("dist_coeffs must have 4, 5, or 8 elements")
			self.dist_coeffs = dc.reshape(-1, 1)

		self._clahe = None
		if self.config.clahe:
			self._clahe = cv2.createCLAHE(
				clipLimit=self.config.clahe_clip_limit,
				tileGridSize=self.config.clahe_tile_grid_size,
			)

		self.reset()

	def reset(self):
		self.prev_gray: Optional[np.ndarray] = None
		self.prev_points: Optional[np.ndarray] = None
		self.prev_timestamp: Optional[float] = None
		self.prev_ids: np.ndarray = np.empty((0,), dtype=np.int64)
		self.prev_track_lengths: np.ndarray = np.empty((0,), dtype=np.int32)
		self._next_id: int = 0

	def process(self, image: np.ndarray, timestamp: float) -> FrontendOutput:
		gray = self._preprocess(image)
		h, w = gray.shape

		if self.prev_gray is None or self.prev_points is None or len(self.prev_points) == 0:
			points = self._detect_new_features(gray, np.empty((0, 2), dtype=np.float32), self.config.max_features)
			ids = self._allocate_ids(len(points))
			lengths = np.ones((len(points),), dtype=np.int32)
			output = self._make_output(timestamp, (h, w), True, False, len(points), len(points), points, ids, lengths)
			self._cache_frame(gray, points, ids, lengths, timestamp)
			return output

		tracked_points, tracked_ids, tracked_lengths = self._track_features(self.prev_gray, gray)
		tracked_points, tracked_ids, tracked_lengths = self._reject_outliers_fundamental(
			self.prev_points,
			tracked_points,
			tracked_ids,
			tracked_lengths,
		)

		tracked_points, tracked_ids, tracked_lengths = self._sort_by_track_length(
			tracked_points,
			tracked_ids,
			tracked_lengths,
		)

		num_to_add = max(0, self.config.max_features - len(tracked_points))
		new_points = np.empty((0, 2), dtype=np.float32)
		if num_to_add > 0:
			new_points = self._detect_new_features(gray, tracked_points, num_to_add)

		if len(new_points) > 0:
			new_ids = self._allocate_ids(len(new_points))
			new_lengths = np.ones((len(new_points),), dtype=np.int32)
			points = np.vstack((tracked_points, new_points)).astype(np.float32)
			ids = np.hstack((tracked_ids, new_ids)).astype(np.int64)
			lengths = np.hstack((tracked_lengths, new_lengths)).astype(np.int32)
		else:
			points, ids, lengths = tracked_points, tracked_ids, tracked_lengths

		is_keyframe = self._compute_keyframe(tracked_points, tracked_ids)
		output = self._make_output(
			timestamp,
			(h, w),
			True,
			is_keyframe,
			len(tracked_points),
			len(new_points),
			points,
			ids,
			lengths,
		)
		self._cache_frame(gray, points, ids, lengths, timestamp)
		return output

	def _preprocess(self, image: np.ndarray) -> np.ndarray:
		img = np.asarray(image)
		if img.ndim == 3 and img.shape[2] == 3:
			gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		elif img.ndim == 2:
			gray = img
		else:
			raise ValueError(f"Expected image shape (H, W) or (H, W, 3), got {img.shape}")

		if gray.dtype != np.uint8:
			gray = np.clip(gray, 0, 255).astype(np.uint8)

		if self._clahe is not None:
			gray = self._clahe.apply(gray)
		return gray

	def _track_features(self, prev_gray: np.ndarray, gray: np.ndarray):
		if len(self.prev_points) == 0:
			return (
				np.empty((0, 2), dtype=np.float32),
				np.empty((0,), dtype=np.int64),
				np.empty((0,), dtype=np.int32),
			)

		lk_params = dict(
			winSize=self.config.lk_win_size,
			maxLevel=self.config.lk_max_level,
			criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, self.config.lk_iters, self.config.lk_eps),
		)
		prev_pts_cv = self.prev_points.reshape(-1, 1, 2).astype(np.float32)
		next_pts_cv, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts_cv, None, **lk_params)

		if next_pts_cv is None or status is None:
			return (
				np.empty((0, 2), dtype=np.float32),
				np.empty((0,), dtype=np.int64),
				np.empty((0,), dtype=np.int32),
			)

		next_pts = next_pts_cv.reshape(-1, 2)
		status = status.reshape(-1).astype(bool)

		h, w = gray.shape
		in_bounds = (
			(next_pts[:, 0] >= 0.0)
			& (next_pts[:, 0] < w)
			& (next_pts[:, 1] >= 0.0)
			& (next_pts[:, 1] < h)
		)
		keep = status & in_bounds

		points = next_pts[keep].astype(np.float32)
		ids = self.prev_ids[keep]
		lengths = self.prev_track_lengths[keep] + 1
		return points, ids, lengths

	def _reject_outliers_fundamental(
		self,
		prev_pts: np.ndarray,
		curr_pts: np.ndarray,
		curr_ids: np.ndarray,
		curr_lengths: np.ndarray,
	):
		if len(curr_pts) < 8:
			return curr_pts, curr_ids, curr_lengths

		id_to_prev = {int(fid): prev_pts[i] for i, fid in enumerate(self.prev_ids)}
		paired_prev = np.array([id_to_prev[int(fid)] for fid in curr_ids], dtype=np.float32)

		keep_finite = np.isfinite(paired_prev).all(axis=1) & np.isfinite(curr_pts).all(axis=1)
		if not np.all(keep_finite):
			paired_prev = paired_prev[keep_finite]
			curr_pts = curr_pts[keep_finite]
			curr_ids = curr_ids[keep_finite]
			curr_lengths = curr_lengths[keep_finite]

		if len(curr_pts) < 8:
			return curr_pts, curr_ids, curr_lengths

		paired_prev = np.ascontiguousarray(paired_prev.reshape(-1, 2), dtype=np.float32)
		curr_pts = np.ascontiguousarray(curr_pts.reshape(-1, 2), dtype=np.float32)

		try:
			F, mask = cv2.findFundamentalMat(
				paired_prev,
				curr_pts,
				method=cv2.FM_RANSAC,
				ransacReprojThreshold=self.config.f_ransac_reproj_threshold,
				confidence=self.config.f_ransac_confidence,
				maxIters=self.config.f_ransac_max_iters,
			)
		except cv2.error:
			return curr_pts, curr_ids, curr_lengths
		if F is None or mask is None:
			return curr_pts, curr_ids, curr_lengths

		inlier = mask.reshape(-1).astype(bool)
		return curr_pts[inlier], curr_ids[inlier], curr_lengths[inlier]

	def _sort_by_track_length(self, points: np.ndarray, ids: np.ndarray, lengths: np.ndarray):
		if len(points) == 0:
			return points, ids, lengths
		order = np.argsort(-lengths)
		return points[order], ids[order], lengths[order]

	def _detect_new_features(self, gray: np.ndarray, existing_points: np.ndarray, max_new: int) -> np.ndarray:
		if max_new <= 0:
			return np.empty((0, 2), dtype=np.float32)

		mask = np.full(gray.shape, 255, dtype=np.uint8)
		radius = max(5, int(self.config.min_distance))
		for pxy in existing_points:
			x = int(round(float(pxy[0])))
			y = int(round(float(pxy[1])))
			cv2.circle(mask, (x, y), radius, 0, -1)

		pts = cv2.goodFeaturesToTrack(
			gray,
			maxCorners=max_new,
			qualityLevel=self.config.quality_level,
			minDistance=self.config.min_distance,
			mask=mask,
			blockSize=self.config.block_size,
			useHarrisDetector=False,
		)
		if pts is None:
			return np.empty((0, 2), dtype=np.float32)
		return pts.reshape(-1, 2).astype(np.float32)

	def _undistort_points(self, points: np.ndarray) -> np.ndarray:
		if len(points) == 0:
			return np.empty((0, 2), dtype=np.float32)
		pts = points.reshape(-1, 1, 2).astype(np.float32)
		undist = cv2.undistortPoints(pts, self.K, self.dist_coeffs)
		return undist.reshape(-1, 2).astype(np.float32)

	def _compute_keyframe(self, tracked_points: np.ndarray, tracked_ids: np.ndarray) -> bool:
		if len(tracked_points) == 0 or len(self.prev_points) == 0:
			return True

		if len(tracked_ids) == 0:
			return True

		prev_map = {int(fid): self.prev_points[i] for i, fid in enumerate(self.prev_ids)}
		curr_map = {int(fid): tracked_points[i] for i, fid in enumerate(tracked_ids)}
		ids = [fid for fid in curr_map.keys() if fid in prev_map]
		if len(ids) == 0:
			return True

		prev_xy = np.array([prev_map[fid] for fid in ids], dtype=np.float32)
		curr_xy = np.array([curr_map[fid] for fid in ids], dtype=np.float32)

		parallax = np.linalg.norm(curr_xy - prev_xy, axis=1)
		if len(parallax) == 0:
			return True

		median_parallax = float(np.median(parallax))
		low_track_count = len(tracked_points) < self.config.min_features
		return low_track_count or median_parallax >= self.config.min_parallax_px

	def _allocate_ids(self, n: int) -> np.ndarray:
		if n <= 0:
			return np.empty((0,), dtype=np.int64)
		ids = np.arange(self._next_id, self._next_id + n, dtype=np.int64)
		self._next_id += n
		return ids

	def _make_output(
		self,
		timestamp: float,
		image_size: tuple[int, int],
		is_initialized: bool,
		is_keyframe: bool,
		num_tracked: int,
		num_new: int,
		points: np.ndarray,
		ids: np.ndarray,
		lengths: np.ndarray,
	) -> FrontendOutput:
		undist = self._undistort_points(points)
		id_to_point = {int(fid): points[i].copy() for i, fid in enumerate(ids)}
		return FrontendOutput(
			timestamp=float(timestamp),
			image_size=image_size,
			is_initialized=is_initialized,
			is_keyframe=bool(is_keyframe),
			num_tracked=int(num_tracked),
			num_new=int(num_new),
			points=points.copy(),
			undistorted_points=undist,
			feature_ids=ids.copy(),
			track_lengths=lengths.copy(),
			id_to_point=id_to_point,
		)

	def _cache_frame(self, gray: np.ndarray, points: np.ndarray, ids: np.ndarray, lengths: np.ndarray, timestamp: float):
		self.prev_gray = gray.copy()
		self.prev_points = points.astype(np.float32, copy=True)
		self.prev_ids = ids.astype(np.int64, copy=True)
		self.prev_track_lengths = lengths.astype(np.int32, copy=True)
		self.prev_timestamp = float(timestamp)


def _build_default_camera_matrix(width: int, height: int) -> np.ndarray:
	# A generic pinhole model for visualization/demo purposes.
	fx = 0.9 * float(width)
	fy = 0.9 * float(width)
	cx = 0.5 * float(width)
	cy = 0.5 * float(height)
	return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def _draw_feature_points(image_bgr: np.ndarray, output: FrontendOutput) -> np.ndarray:
	canvas = image_bgr.copy()
	for pxy, track_len in zip(output.points, output.track_lengths):
		x = int(round(float(pxy[0])))
		y = int(round(float(pxy[1])))
		color = (0, 220, 0) if int(track_len) > 1 else (0, 200, 255)
		cv2.circle(canvas, (x, y), 2, color, -1)

	text = (
		f"tracked={output.num_tracked} new={output.num_new} "
		f"keyframe={int(output.is_keyframe)}"
	)
	cv2.putText(canvas, text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
	cv2.putText(canvas, text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
	return canvas


def _draw_optical_flow(image_bgr: np.ndarray, prev_output: Optional[FrontendOutput], curr_output: FrontendOutput) -> np.ndarray:
	canvas = image_bgr.copy()
	if prev_output is None:
		cv2.putText(canvas, "optical flow unavailable on first frame", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
		cv2.putText(canvas, "optical flow unavailable on first frame", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
		return canvas

	prev_map = prev_output.id_to_point
	for fid, p1 in curr_output.id_to_point.items():
		if fid not in prev_map:
			continue
		p0 = prev_map[fid]
		x0, y0 = int(round(float(p0[0]))), int(round(float(p0[1])))
		x1, y1 = int(round(float(p1[0]))), int(round(float(p1[1])))
		cv2.arrowedLine(canvas, (x0, y0), (x1, y1), (0, 180, 255), 1, tipLength=0.25)
		cv2.circle(canvas, (x1, y1), 2, (255, 100, 0), -1)

	return canvas


def run_vins_frontend_image_sequence_demo(
	image_dir: str,
	output_dir: Optional[str] = None,
	fps: float = 30.0,
	max_frames: int = 120,
) -> dict:
	"""Run a short VINS frontend demo on an image folder and save visualizations.

	Creates two output streams:
	- `features`: frame overlays with current tracked/new points
	- `flow`: frame overlays with feature motion arrows between consecutive frames
	"""
	image_path = Path(image_dir)
	if not image_path.exists() or not image_path.is_dir():
		raise ValueError(f"image_dir does not exist or is not a directory: {image_dir}")

	frame_paths = sorted(image_path.glob("*.png"))
	if len(frame_paths) == 0:
		raise ValueError(f"No PNG frames found in: {image_dir}")

	max_frames = int(max_frames)
	if max_frames <= 0:
		raise ValueError("max_frames must be > 0")
	fps = float(fps)
	if fps <= 0.0:
		raise ValueError("fps must be > 0")

	frame_paths = frame_paths[:max_frames]
	first_bgr = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
	if first_bgr is None:
		raise RuntimeError(f"Failed to read image: {frame_paths[0]}")
	h, w = first_bgr.shape[:2]

	K = _build_default_camera_matrix(w, h)
	frontend = VinsMonoFrontend(K=K, dist_coeffs=np.zeros((5, 1), dtype=np.float64))

	if output_dir is None:
		output_path = image_path / "vins_demo_outputs"
	else:
		output_path = Path(output_dir)
	features_dir = output_path / "features"
	flow_dir = output_path / "flow"
	features_dir.mkdir(parents=True, exist_ok=True)
	flow_dir.mkdir(parents=True, exist_ok=True)

	prev_output: Optional[FrontendOutput] = None
	keyframe_count = 0
	for i, frame_path in enumerate(frame_paths):
		bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
		if bgr is None:
			continue
		rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
		timestamp = float(i) / fps
		output = frontend.process(rgb, timestamp)

		if output.is_keyframe:
			keyframe_count += 1

		feature_vis = _draw_feature_points(bgr, output)
		flow_vis = _draw_optical_flow(bgr, prev_output, output)

		frame_name = frame_path.name
		cv2.imwrite(str(features_dir / frame_name), feature_vis)
		cv2.imwrite(str(flow_dir / frame_name), flow_vis)

		prev_output = output

	return {
		"num_input_frames": len(frame_paths),
		"num_processed_frames": len(list(features_dir.glob("*.png"))),
		"keyframe_count": int(keyframe_count),
		"features_dir": str(features_dir),
		"flow_dir": str(flow_dir),
	}


if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description="Run a short VINS frontend image-sequence demo")
	parser.add_argument(
		"--image-dir",
		type=str,
		default="/home/henryshum0/drone_fyp/gym_pybullet_drones/examples/results/drone_camera_04.16.2026_22.45.53",
		help="Folder containing ordered PNG frames",
	)
	parser.add_argument(
		"--output-dir",
		type=str,
		default="",
		help="Optional output folder (default: <image-dir>/vins_demo_outputs)",
	)
	parser.add_argument("--fps", type=float, default=30.0, help="Image sequence frame rate")
	parser.add_argument("--max-frames", type=int, default=120, help="Maximum number of frames to process")
	args = parser.parse_args()

	result = run_vins_frontend_image_sequence_demo(
		image_dir=args.image_dir,
		output_dir=args.output_dir if len(args.output_dir) > 0 else None,
		fps=args.fps,
		max_frames=args.max_frames,
	)

	print("[VINS DEMO] completed")
	for key, value in result.items():
		print(f"[VINS DEMO] {key}: {value}")
