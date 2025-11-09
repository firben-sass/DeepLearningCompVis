"""Utility to convert DRIVE dataset .tif images to .jpg files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

from PIL import Image


DEFAULT_DATASET_ROOT = Path(__file__).resolve().parent / "data" / "DRIVE"


def iter_tif_files(root: Path) -> Iterable[Path]:
	"""Yield all .tif files under the provided root directory."""

	if not root.exists():
		return []

	for extension in ("*.tif", "*.TIF", "*.tiff", "*.TIFF"):
		yield from root.rglob(extension)


def convert_tif_to_jpg(tif_path: Path, *, overwrite: bool, quality: int) -> str:
	"""Convert a single .tif file to .jpg and return the outcome string."""

	jpg_path = tif_path.with_suffix(".jpg")

	if not overwrite and jpg_path.exists():
		return "skipped"

	try:
		with Image.open(tif_path) as image:
			# DRIVE images are RGB; convert() ensures consistent 3-channel output.
			rgb_image = image.convert("RGB")
			rgb_image.save(jpg_path, format="JPEG", quality=quality)
	except (OSError, ValueError) as exc:  # Pillow raises OSError on decode errors.
		print(f"[ERROR] Failed to convert {tif_path}: {exc}", file=sys.stderr)
		return "failed"

	return "converted"


def main(argv: list[str] | None = None) -> int:
	parser = argparse.ArgumentParser(
		description="Convert DRIVE dataset .tif images to .jpg files."
	)
	parser.add_argument(
		"--dataset-root",
		type=Path,
		default=DEFAULT_DATASET_ROOT,
		help=(
			"Path to the DRIVE dataset root. Defaults to 'data/DRIVE' relative to "
			"this script."
		),
	)
	parser.add_argument(
		"--overwrite",
		action="store_true",
		help="Overwrite existing .jpg files if they are already present.",
	)
	parser.add_argument(
		"--quality",
		type=int,
		default=95,
		help="JPEG quality (1-95). Defaults to 95.",
	)

	args = parser.parse_args(argv)

	dataset_root = args.dataset_root.resolve()

	if not dataset_root.exists():
		print(f"[ERROR] Dataset root not found: {dataset_root}", file=sys.stderr)
		return 1

	tif_files = sorted(
		{path for path in iter_tif_files(dataset_root) if path.is_file()}
	)

	if not tif_files:
		print(f"[WARN] No .tif files found under {dataset_root}")
		return 0

	stats = {"converted": 0, "skipped": 0, "failed": 0}

	print(f"[INFO] Converting {len(tif_files)} .tif files under {dataset_root}...")

	for tif_path in tif_files:
		outcome = convert_tif_to_jpg(
			tif_path, overwrite=args.overwrite, quality=args.quality
		)

		stats[outcome] += 1

	print(
		"[INFO] Done. Converted {converted} file(s), skipped {skipped} file(s), "
		"failed {failed} file(s).".format(**stats)
	)

	return 0


if __name__ == "__main__":
	raise SystemExit(main())

