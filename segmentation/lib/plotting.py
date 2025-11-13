"""Utility functions for visualising training progress."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt


def plot_training_curves(
	loss_history: Mapping[str, Sequence[float]],
	score_history: Mapping[str, Sequence[float]],
	output_dir: str | Path | None = None,
	show: bool = False,
) -> dict[str, Path | None]:
	"""Plot loss curves and validation metric traces across epochs.

	Args:
		loss_history: Mapping that must contain ``train`` and ``val`` keys whose
			values are per-epoch loss sequences.
		score_history: Mapping from metric name to the corresponding validation
			scores recorded after each epoch.
		output_dir: Directory where the figures should be saved. Directories are
			created if missing. When ``None`` the figures are not written to
			disk.
		show: When ``True`` the figures are displayed via ``plt.show()``.

	Returns:
		A dictionary with the paths of the saved figures (``loss`` and
		``metrics``) or ``None`` when a figure was not written to disk.

	Raises:
		ValueError: If the provided histories are inconsistent.
	"""

	if "train" not in loss_history or "val" not in loss_history:
		raise ValueError("loss_history must contain 'train' and 'val' sequences")

	train_losses = list(loss_history["train"])
	val_losses = list(loss_history["val"])

	if len(train_losses) != len(val_losses):
		raise ValueError("Train and validation loss histories must be the same length")

	epochs = list(range(1, len(train_losses) + 1))

	# Validate metric histories to align with number of epochs.
	for metric_name, values in score_history.items():
		if len(values) != len(epochs):
			raise ValueError(
				f"Metric '{metric_name}' history length ({len(values)}) does not match number of epochs ({len(epochs)})"
			)

	loss_fig, loss_ax = plt.subplots(figsize=(8, 5))
	loss_ax.plot(epochs, train_losses, label="Train", color="steelblue", linewidth=2)
	loss_ax.plot(epochs, val_losses, label="Validation", color="tomato", linewidth=2)
	loss_ax.set_title("Training and Validation Loss")
	loss_ax.set_xlabel("Epoch")
	loss_ax.set_ylabel("Loss")
	loss_ax.legend()
	loss_ax.grid(alpha=0.3)
	loss_fig.tight_layout()

	metrics_fig = None
	metrics_ax = None
	if score_history:
		metrics_fig, metrics_ax = plt.subplots(figsize=(8, 5))
		for metric_name, values in score_history.items():
			metrics_ax.plot(epochs, values, label=metric_name, linewidth=2)
		metrics_ax.set_title("Validation Metrics")
		metrics_ax.set_xlabel("Epoch")
		metrics_ax.set_ylabel("Score")
		metrics_ax.legend()
		metrics_ax.grid(alpha=0.3)
		metrics_fig.tight_layout()

	saved_paths: dict[str, Path | None] = {"loss": None, "metrics": None}

	if output_dir is not None:
		output_root = Path(output_dir)
		output_root.mkdir(parents=True, exist_ok=True)

		loss_path = output_root / "loss_curve.png"
		if loss_path.exists():
			loss_path.unlink()
		loss_fig.savefig(loss_path, bbox_inches="tight")
		saved_paths["loss"] = loss_path

		if metrics_fig is not None:
			metrics_path = output_root / "metrics_curve.png"
			if metrics_path.exists():
				metrics_path.unlink()
			metrics_fig.savefig(metrics_path, bbox_inches="tight")
			saved_paths["metrics"] = metrics_path

	if show:
		plt.show()
	else:
		# Free up resources when figures are not displayed interactively.
		plt.close(loss_fig)
		if metrics_fig is not None:
			plt.close(metrics_fig)

	return saved_paths

