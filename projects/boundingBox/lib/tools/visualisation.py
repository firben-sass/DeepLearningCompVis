"""Utility helpers for drawing and visualising bounding boxes."""

from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt

from .images import extract_image_patches
from .load_bounding_boxes import load_bounding_boxes

Box = Sequence[int]


def _ensure_box(box: Box) -> Tuple[int, int, int, int]:
	"""Cast box coordinates to ints and ensure correct length."""
	if len(box) != 4:
		raise ValueError(f"Bounding box must have 4 values, got {len(box)}")
	x1, y1, x2, y2 = (int(coord) for coord in box)
	return x1, y1, x2, y2


def _prepare_save_path(path_like) -> Path:
	"""Ensure the save path has a suffix and parent directory."""
	path = Path(path_like)
	if not path.suffix:
		path = path.with_suffix(".png")
	path.parent.mkdir(parents=True, exist_ok=True)
	return path


def visualize_bboxes_with_patches(
	image,
	boxes: Iterable[Box],
	probabilities: Iterable[float],
	output_path: str,
	patches_path: Optional[str] = None,
	gt_xml_path: Optional[str] = None,
	gt_color: Tuple[int, int, int] = (255, 0, 0),
	color: Tuple[int, int, int] = (0, 255, 0),
	thickness: int = 2,
	show_indices: bool = True,
	figsize: Tuple[int, int] = (10, 10),
):
	"""Draw bounding boxes on an image and show the associated patches.

	Args:
		image: numpy array in BGR format (as loaded by cv2.imread).
		boxes: Iterable of (x1, y1, x2, y2) coordinates.
		probabilities: Iterable containing probability for each box.
		output_path: File path to save the annotated image.
		patches_path: Optional path for saving extracted patches (defaults to
		              `<output_path>_patches`).
		gt_xml_path: Optional path to an XML file with ground-truth boxes.
		gt_color: BGR color for ground-truth rectangles.
		color: BGR color for rectangles.
		thickness: Rectangle line thickness.
		show_indices: Whether to annotate boxes with their index.
		figsize: Matplotlib figure size for the boxed image.
	"""

	box_list = list(boxes)
	prob_list = list(probabilities)
	if not box_list:
		raise ValueError("No bounding boxes provided for visualization.")
	if len(box_list) != len(prob_list):
		raise ValueError("Number of probabilities must match number of boxes.")

	bbox_save_path = _prepare_save_path(output_path)
	patches_save_path = _prepare_save_path(
		patches_path
		if patches_path is not None
		else bbox_save_path.with_name(f"{bbox_save_path.stem}_patches{bbox_save_path.suffix}")
	)

	annotated = image.copy()
	gt_boxes: Sequence[Box] = []
	if gt_xml_path:
		try:
			resize_target = (image.shape[1], image.shape[0])
			gt_boxes = load_bounding_boxes(gt_xml_path, resize_target)
		except Exception as exc:
			print(f"Warning: could not load ground-truth boxes from '{gt_xml_path}': {exc}")
			gt_boxes = []
	for idx, box in enumerate(box_list):
		x1, y1, x2, y2 = _ensure_box(box)
		cv2.rectangle(annotated, (x1, y1), (x2, y2), color=color, thickness=thickness)
		label = f"{idx}: {prob_list[idx]:.2f}" if show_indices else f"{prob_list[idx]:.2f}"
		cv2.putText(
			annotated,
			label,
			(x1, max(y1 - 5, 10)),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.5,
			color,
			1,
			cv2.LINE_AA,
		)

	for box in gt_boxes:
		x1, y1, x2, y2 = _ensure_box(box)
		cv2.rectangle(annotated, (x1, y1), (x2, y2), color=gt_color, thickness=max(1, thickness - 1))

	# Convert BGR (OpenCV) to RGB for matplotlib display.
	annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

	fig1 = plt.figure(figsize=figsize)
	ax1 = fig1.add_subplot(111)
	ax1.imshow(annotated_rgb)
	ax1.axis("off")
	ax1.set_title("Bounding Boxes")
	fig1.savefig(bbox_save_path, bbox_inches="tight")
	plt.close(fig1)

	patches = extract_image_patches(image, box_list)
	if not patches:
		raise ValueError("No valid patches extracted for visualization.")

	cols = min(4, len(patches))
	rows = (len(patches) + cols - 1) // cols
	fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
	axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

	for ax, patch in zip(axes, patches):
		patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
		ax.imshow(patch_rgb)
		ax.axis("off")

	# Hide unused subplots if any.
	for ax in axes[len(patches) :]:
		ax.axis("off")

	fig.suptitle("Extracted Patches")
	fig.savefig(patches_save_path, bbox_inches="tight")
	plt.close(fig)
	print(f"Saved annotated image to {bbox_save_path}")