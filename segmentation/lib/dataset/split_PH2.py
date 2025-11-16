#!/usr/bin/env python3
"""Utility script for splitting the PH2 dataset into train/test/validate sets."""

from __future__ import annotations

import argparse
import math
import random
import shutil
from pathlib import Path
from typing import Dict, List


DEFAULT_BASE_DIR = (
	Path(__file__).resolve().parents[2]
	/ "data"
	/ "PH2_Dataset_images"
)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Split PH2 cases into train/test/validate folders (60/20/20).",
	)
	parser.add_argument(
		"--base-dir",
		type=Path,
		default=DEFAULT_BASE_DIR,
		help="Root directory containing IMD* case folders (default: %(default)s)",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=None,
		help="Destination directory for the split dataset. Defaults to <base>/PH2_split.",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=42,
		help="Seed for the RNG controlling the split ordering (default: %(default)s)",
	)
	parser.add_argument(
		"--force",
		action="store_true",
		help="Remove output directory if it already exists.",
	)
	parser.add_argument(
		"--dry-run",
		action="store_true",
		help="Show the split summary without copying any files.",
	)
	parser.add_argument(
		"--mode",
		choices=("copy", "symlink", "move"),
		default="copy",
		help="Transfer mode for case folders (default: %(default)s)",
	)
	return parser.parse_args()


def collect_cases(base_dir: Path) -> List[Path]:
	if not base_dir.is_dir():
		raise FileNotFoundError(f"Base directory does not exist: {base_dir}")
	cases = [p for p in base_dir.iterdir() if p.is_dir() and p.name.upper().startswith("IMD")]
	if not cases:
		raise RuntimeError(f"No IMD* case folders found in {base_dir}")
	cases.sort()
	return cases


def compute_counts(total: int, ratios: List[float]) -> List[int]:
	if total <= 0:
		raise ValueError("Cannot compute split counts for an empty dataset.")
	raw = [total * r for r in ratios]
	base_counts = [math.floor(value) for value in raw]
	remainder = total - sum(base_counts)
	fractions = sorted(
		((raw[i] - base_counts[i], i) for i in range(len(ratios))),
		reverse=True,
	)
	for idx in range(remainder):
		_, split_idx = fractions[idx % len(ratios)]
		base_counts[split_idx] += 1
	return base_counts


def prepare_output_dir(output_dir: Path, force: bool) -> None:
	if output_dir.exists():
		if not force:
			raise FileExistsError(
				f"Output directory {output_dir} already exists. Use --force to overwrite."
			)
		shutil.rmtree(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)


def transfer_case(case_dir: Path, destination: Path, mode: str) -> None:
	if destination.exists():
		raise FileExistsError(f"Destination already exists: {destination}")
	if mode == "copy":
		shutil.copytree(case_dir, destination)
	elif mode == "symlink":
		destination.symlink_to(case_dir, target_is_directory=True)
	elif mode == "move":
		shutil.move(str(case_dir), destination)
	else:
		raise ValueError(f"Unsupported transfer mode: {mode}")


def main() -> None:
	args = parse_args()
	base_dir = args.base_dir.expanduser().resolve()
	output_dir = (
		args.output_dir.expanduser().resolve() if args.output_dir else base_dir / "PH2_split"
	)

	if output_dir == base_dir:
		raise ValueError("Output directory must differ from the base directory.")

	cases = collect_cases(base_dir)
	rng = random.Random(args.seed)
	rng.shuffle(cases)

	splits = ("train", "test", "validate")
	ratios = (0.6, 0.2, 0.2)
	counts = compute_counts(len(cases), list(ratios))

	summary: Dict[str, List[Path]] = {}
	offset = 0
	for split_name, count in zip(splits, counts):
		summary[split_name] = cases[offset : offset + count]
		offset += count

	if args.dry_run:
		print(f"Found {len(cases)} cases under {base_dir}")
		for split_name in splits:
			print(f"- {split_name}: {len(summary[split_name])} cases")
		print("Dry run requested; no files were copied.")
		return

	prepare_output_dir(output_dir, args.force)
	for split_name in splits:
		split_dir = output_dir / split_name
		split_dir.mkdir(parents=True, exist_ok=True)
		for case_dir in summary[split_name]:
			destination = split_dir / case_dir.name
			transfer_case(case_dir, destination, args.mode)

	print(
		f"Split {len(cases)} cases into train/test/validate at {output_dir} using {args.mode} mode."
	)


if __name__ == "__main__":
	main()
