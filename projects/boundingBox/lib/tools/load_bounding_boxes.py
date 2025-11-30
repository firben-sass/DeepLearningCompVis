"""Utilities for reading bounding-box annotations from XML files."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence, Tuple
import xml.etree.ElementTree as ET

Box = Tuple[int, int, int, int]
_COORD_TAGS: Sequence[str] = ("xmin", "ymin", "xmax", "ymax")


def _coerce_int(value: str, tag: str, xml_path: Path) -> int:

	try:
		return int(float(value))
	except (TypeError, ValueError) as exc:  # pragma: no cover - defensive branch
		raise ValueError(
			f"Could not parse coordinate '{tag}' in '{xml_path}'"
		) from exc


def _get_original_size(root: ET.Element, xml_path: Path) -> Tuple[int, int]:
	"""Extract original (width, height) from the XML metadata."""
	size_node = root.find("size")
	if size_node is None:
		raise ValueError(f"Missing 'size' entry in '{xml_path}'")
	width_text = size_node.findtext("width")
	height_text = size_node.findtext("height")
	if width_text is None or height_text is None:
		raise ValueError(f"Missing width/height entries in 'size' for '{xml_path}'")
	orig_width = _coerce_int(width_text, "width", xml_path)
	orig_height = _coerce_int(height_text, "height", xml_path)
	return orig_width, orig_height


def _normalize_resize_target(resize_img: Sequence[int]) -> Tuple[int, int]:
	"""Validate and normalize the resize target tuple."""
	if len(resize_img) != 2:
		raise ValueError("'resize_img' must contain exactly two values: (width, height)")
	try:
		width = int(resize_img[0])
		height = int(resize_img[1])
	except (TypeError, ValueError) as exc:  # pragma: no cover - defensive branch
		raise ValueError("'resize_img' values must be integers") from exc
	if width <= 0 or height <= 0:
		raise ValueError("'resize_img' dimensions must be positive")
	return width, height


def load_bounding_boxes(
	xml_file: Path | str,
	resize_img: Sequence[int] | None = None,
) -> List[Box]:

	path = Path(xml_file)
	if not path.exists():
		raise FileNotFoundError(f"XML file not found: {path}")

	root = ET.parse(path).getroot()
	original_size: Tuple[int, int] | None = None
	if resize_img is not None:
		original_size = _get_original_size(root, path)
		target_size = _normalize_resize_target(resize_img)
		scale_x = target_size[0] / original_size[0]
		scale_y = target_size[1] / original_size[1]
	boxes: List[Box] = []

	for bndbox in root.findall(".//bndbox"):
		coords = []
		for tag in _COORD_TAGS:
			coord_text = bndbox.findtext(tag)
			if coord_text is None:
				raise ValueError(f"Missing '{tag}' entry for a bounding box in '{path}'")
			coords.append(_coerce_int(coord_text, tag, path))
		x1, y1, x2, y2 = coords
		if x2 < x1:
			x1, x2 = x2, x1
		if y2 < y1:
			y1, y2 = y2, y1
		if resize_img is not None and original_size is not None:
			x1 = int(round(x1 * scale_x))
			y1 = int(round(y1 * scale_y))
			x2 = int(round(x2 * scale_x))
			y2 = int(round(y2 * scale_y))
		boxes.append((x1, y1, x2, y2))

	if not boxes:
		raise ValueError(f"No bounding boxes found in '{path}'")

	return boxes


__all__: Iterable[str] = ("load_bounding_boxes",)
