import os
import glob
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image


def _natural_key(path: str) -> List[object]:
    """Return a sort key that keeps numbered files in numeric order."""
    basename = os.path.basename(path)
    return [int(chunk) if chunk.isdigit() else chunk.lower() for chunk in re.split(r"(\d+)", basename)]


class BoundingBoxDataset(torch.utils.data.Dataset):
    """Dataset that loads RGB images and Pascal VOC style XML annotations."""

    _IMAGE_PATTERNS: Sequence[str] = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")

    def __init__(
        self,
        split: str = "train",
        transform: Optional[Callable] = None,
        label_transform: Optional[Callable] = None,
        data_root: Optional[Path] = None,
        split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: int = 1337,
    ) -> None:
        if isinstance(split, bool):
            split = "train" if split else "test"
        if not isinstance(split, str):
            raise TypeError("split must be a bool or a string.")

        split = split.lower()
        if split == "val":
            split = "validate"

        valid_splits = ("train", "validate", "test")
        if split not in valid_splits:
            raise ValueError(f"Unsupported split '{split}'. Choose from {valid_splits}.")

        if data_root is None:
            data_root = Path(__file__).resolve().parents[2] / "data"
        self.data_root = Path(data_root)
        self.image_dir = self.data_root / "images"
        self.annotation_dir = self.data_root / "annotations"
        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"No image directory found at {self.image_dir}.")
        if not self.annotation_dir.is_dir():
            raise FileNotFoundError(f"No annotation directory found at {self.annotation_dir}.")

        self.transform = transform
        self.target_transform = label_transform
        self.split = split
        self.split_ratios = split_ratios
        self.seed = seed
        self.class_to_idx: Dict[str, int] = {"pothole": 1}

        all_samples = self._collect_samples()
        if not all_samples:
            raise FileNotFoundError(f"No paired image/XML files found under {self.data_root}.")

        split_mapping = self._build_split_mapping(all_samples)
        self.samples: List[Tuple[str, str]] = split_mapping.get(split, [])
        if not self.samples:
            raise RuntimeError(
                f"Requested split '{split}' is empty. Provide a split file or adjust split ratios."
            )

    def _collect_samples(self) -> List[Tuple[str, str]]:
        image_paths: List[str] = []
        for pattern in self._IMAGE_PATTERNS:
            image_paths.extend(glob.glob(str(self.image_dir / pattern)))
        image_paths = sorted(set(image_paths), key=_natural_key)

        samples: List[Tuple[str, str]] = []
        missing_xml: List[str] = []
        for image_path in image_paths:
            stem = Path(image_path).stem
            xml_path = self.annotation_dir / f"{stem}.xml"
            if xml_path.is_file():
                samples.append((image_path, str(xml_path)))
            else:
                missing_xml.append(stem)

        if missing_xml:
            missing_list = ", ".join(missing_xml[:5])
            extra = "" if len(missing_xml) <= 5 else f" (+{len(missing_xml) - 5} more)"
            raise FileNotFoundError(
                f"Missing XML annotations for: {missing_list}{extra}. Please ensure every image has a matching XML file."
            )
        return samples

    def _build_split_mapping(self, samples: Sequence[Tuple[str, str]]) -> Dict[str, List[Tuple[str, str]]]:
        split_file = self.data_root / f"{self.split}.txt"
        if split_file.is_file():
            requested = self._load_split_from_file(split_file)
            lookup = {Path(img).stem: (img, ann) for img, ann in samples}
            resolved = []
            unknown: List[str] = []
            for stem in requested:
                entry = lookup.get(stem)
                if entry:
                    resolved.append(entry)
                else:
                    unknown.append(stem)
            if unknown:
                missing = ", ".join(unknown[:5])
                extra = "" if len(unknown) <= 5 else f" (+{len(unknown) - 5} more)"
                raise FileNotFoundError(
                    f"Split file {split_file} references unknown samples: {missing}{extra}."
                )
            return {self.split: resolved}

        train_ratio, val_ratio, test_ratio = self.split_ratios
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError("Split ratios must sum to 1.0.")

        rng = np.random.RandomState(self.seed)
        shuffled = list(samples)
        rng.shuffle(shuffled)

        total = len(shuffled)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        train_split = shuffled[:train_end]
        val_split = shuffled[train_end:val_end]
        test_split = shuffled[val_end:]
        return {"train": train_split, "validate": val_split, "test": test_split}

    @staticmethod
    def _load_split_from_file(path: Path) -> List[str]:
        with path.open("r", encoding="utf-8") as handle:
            return [line.strip() for line in handle if line.strip()]

    @staticmethod
    def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
        array = np.array(image, dtype=np.float32)
        if array.ndim == 2:
            array = array[None, :, :]
        else:
            array = array.transpose(2, 0, 1)
        return torch.from_numpy(array / 255.0)

    def _parse_annotation(self, annotation_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        tree = ET.parse(annotation_path)
        root = tree.getroot()

        boxes: List[List[float]] = []
        labels: List[int] = []
        for obj in root.findall("object"):
            name = obj.findtext("name", default="pothole").strip().lower()
            bbox = obj.find("bndbox")
            if bbox is None:
                continue
            xmin = float(bbox.findtext("xmin", default="0"))
            ymin = float(bbox.findtext("ymin", default="0"))
            xmax = float(bbox.findtext("xmax", default="0"))
            ymax = float(bbox.findtext("ymax", default="0"))
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_to_idx.get(name, 1))

        if boxes:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
        return boxes_tensor, labels_tensor

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        image_path, annotation_path = self.samples[idx]

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        boxes, labels = self._parse_annotation(annotation_path)
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor(idx, dtype=torch.int64),
            "iscrowd": torch.zeros(labels.shape[0], dtype=torch.int64),
        }
        if boxes.numel():
            widths = boxes[:, 2] - boxes[:, 0]
            heights = boxes[:, 3] - boxes[:, 1]
            target["area"] = widths * heights
        else:
            target["area"] = torch.zeros((0,), dtype=torch.float32)

        target["size"] = torch.tensor([image.height, image.width], dtype=torch.int64)
        target["image_path"] = image_path
        target["annotation_path"] = annotation_path

        if self.transform is not None:
            transformed = self.transform(image)
        else:
            transformed = self._pil_to_tensor(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return transformed, target
    

class PotholeDataset(torch.utils.data.Dataset):
    """Dataset for pothole detection (training and validation)"""
    
    def _init_(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir: Root directory containing train/val/test folders
            split: 'train' or 'val'
            transform: Optional transform to be applied on images
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # Paths to positive and negative samples
        self.pothole_dir = os.path.join(root_dir, split, 'potholes')
        self.background_dir = os.path.join(root_dir, split, 'background')
        
        # Load file paths and labels
        self.samples = []
        
        # Add pothole samples (label = 1)
        if os.path.exists(self.pothole_dir):
            pothole_files = [f for f in os.listdir(self.pothole_dir) if f.endswith('.png')]
            for fname in pothole_files:
                self.samples.append((os.path.join(self.pothole_dir, fname), 1))
        
        # Add background samples (label = 0)
        if os.path.exists(self.background_dir):
            background_files = [f for f in os.listdir(self.background_dir) if f.endswith('.png')]
            for fname in background_files:
                self.samples.append((os.path.join(self.background_dir, fname), 0))
        
        print(f"{split.upper()} dataset: {len(self.samples)} samples")
        print(f"  - Potholes: {sum(1 for _, label in self.samples if label == 1)}")
        print(f"  - Background: {sum(1 for _, label in self.samples if label == 0)}")
    
    def _len_(self):
        return len(self.samples)
    
    def _getitem_(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label
