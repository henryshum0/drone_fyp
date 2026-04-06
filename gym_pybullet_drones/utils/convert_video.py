from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _natural_sort_key(path: Path):
	return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def _collect_images(input_dir: Path, extension: str | None) -> tuple[list[Path], str]:
	if extension is not None:
		ext = extension.lower()
		if not ext.startswith("."):
			ext = f".{ext}"
		if ext not in SUPPORTED_EXTENSIONS:
			raise ValueError(
				f"Unsupported extension '{extension}'. Choose one of: {sorted(SUPPORTED_EXTENSIONS)}"
			)
	else:
		candidates = [
			file.suffix.lower()
			for file in input_dir.iterdir()
			if file.is_file() and file.suffix.lower() in SUPPORTED_EXTENSIONS
		]
		if not candidates:
			raise FileNotFoundError(f"No images found in '{input_dir}'.")
		ext = max(set(candidates), key=candidates.count)

	images = sorted(
		[
			file
			for file in input_dir.iterdir()
			if file.is_file() and file.suffix.lower() == ext
		],
		key=_natural_sort_key,
	)

	if not images:
		raise FileNotFoundError(f"No '*{ext}' images found in '{input_dir}'.")

	return images, ext


def images_to_video(
	input_dir: str | Path,
	fps: int,
	output_path: str | Path | None = None,
	extension: str | None = None,
) -> Path:
	input_path = Path(input_dir).expanduser().resolve()
	if not input_path.exists() or not input_path.is_dir():
		raise FileNotFoundError(f"Input directory does not exist: '{input_path}'")

	if fps <= 0:
		raise ValueError("fps must be greater than 0")

	images, ext = _collect_images(input_path, extension)

	if output_path is None:
		output_file = input_path / f"{input_path.name}.mp4"
	else:
		output_file = Path(output_path).expanduser().resolve()
		output_file.parent.mkdir(parents=True, exist_ok=True)

	with tempfile.TemporaryDirectory(prefix="frames_to_video_") as tmp:
		tmp_dir = Path(tmp)
		for index, source in enumerate(images, start=1):
			linked_name = tmp_dir / f"frame_{index:06d}{ext}"
			try:
				os.symlink(source, linked_name)
			except OSError:
				shutil.copy2(source, linked_name)

		ffmpeg_cmd = [
			"ffmpeg",
			"-y",
			"-framerate",
			str(fps),
			"-i",
			str(tmp_dir / f"frame_%06d{ext}"),
			"-c:v",
			"libx264",
			"-pix_fmt",
			"yuv420p",
			str(output_file),
		]

		try:
			subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		except FileNotFoundError as exc:
			raise RuntimeError("ffmpeg is not installed or not available on PATH.") from exc
		except subprocess.CalledProcessError as exc:
			error_message = exc.stderr.decode("utf-8", errors="ignore")
			raise RuntimeError(f"ffmpeg failed while creating video:\n{error_message}") from exc

	return output_file


def _build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Convert images in a folder to an MP4 video.")
	parser.add_argument("input_dir", help="Folder containing image frames")
	parser.add_argument("--fps", type=int, required=True, help="Frames per second for the output video")
	parser.add_argument("--output", default=None, help="Output video path (default: <input_dir>/<folder_name>.mp4)")
	parser.add_argument(
		"--ext",
		default=None,
		help="Image extension to use, e.g. png or jpg (default: auto-detect most common extension)",
	)
	return parser


def main() -> None:
	args = _build_parser().parse_args()
	video_path = images_to_video(
		input_dir=args.input_dir,
		fps=args.fps,
		output_path=args.output,
		extension=args.ext,
	)
	print(f"Video created: {video_path}")


if __name__ == "__main__":
	main()
