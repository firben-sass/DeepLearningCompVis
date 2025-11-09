"""Utility to summarize segmentation datasets using torch DataLoaders."""

import argparse
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

try:
    from segmentation.lib.dataset.Datasets import CMP, DRIVE, PH2
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Unable to import dataset definitions. Ensure PYTHONPATH includes repository root.") from exc


def _build_dataset(name: str, split: str) -> Dataset:
    key = name.lower()
    split = split.lower()

    if key == "drive":
        return DRIVE(split=split)

    if key == "ph2":
        return PH2(split=split)

    if key == "cmp":
        if split not in {"train", "test"}:
            raise ValueError(f"Split '{split}' is not supported for dataset '{name}'.")
        return CMP(train=(split == "train"))

    raise ValueError(f"Unknown dataset '{name}'. Available options: ['cmp', 'drive', 'ph2']")


def _describe_loader(loader: DataLoader, track_label_values: bool = True) -> Dict[str, object]:
    stats: Dict[str, object] = {
        "image_count": 0,
        "label_count": 0,
        "image_shapes": set(),
        "label_shapes": set(),
        "image_dtypes": set(),
        "label_dtypes": set(),
        "image_range": None,
        "label_range": None,
        "label_unique": set() if track_label_values else None,
    }

    max_label_values = 32

    for images, labels in loader:
        images = images.detach()
        stats["image_count"] += images.size(0)
        stats["image_dtypes"].add(str(images.dtype))
        stats["image_shapes"].add(tuple(images.shape[1:]))
        batch_min = images.min().item()
        batch_max = images.max().item()
        if stats["image_range"] is None:
            stats["image_range"] = [batch_min, batch_max]
        else:
            stats["image_range"][0] = min(stats["image_range"][0], batch_min)
            stats["image_range"][1] = max(stats["image_range"][1], batch_max)

        if labels is None:
            continue

        labels = labels.detach()
        stats["label_count"] += labels.size(0)
        stats["label_dtypes"].add(str(labels.dtype))
        stats["label_shapes"].add(tuple(labels.shape[1:]))

        batch_min = labels.min().item()
        batch_max = labels.max().item()
        if stats["label_range"] is None:
            stats["label_range"] = [batch_min, batch_max]
        else:
            stats["label_range"][0] = min(stats["label_range"][0], batch_min)
            stats["label_range"][1] = max(stats["label_range"][1], batch_max)

        if stats["label_unique"] is not None:
            unique_vals = torch.unique(labels)
            stats["label_unique"].update(float(val) for val in unique_vals.cpu())
            if len(stats["label_unique"]) > max_label_values:
                stats["label_unique"] = None

    return stats


def _estimate_file_counts(dataset: Dataset) -> Tuple[Optional[int], Optional[int]]:
    image_count = None
    label_count = None

    if hasattr(dataset, "image_paths"):
        image_paths = getattr(dataset, "image_paths")
        if isinstance(image_paths, Sequence):
            image_count = len(image_paths)

    if hasattr(dataset, "label_paths"):
        label_paths = getattr(dataset, "label_paths")
        if isinstance(label_paths, Sequence):
            label_count = len(label_paths)

    if image_count is None and hasattr(dataset, "samples"):
        samples = getattr(dataset, "samples")
        if isinstance(samples, Sequence):
            image_count = len(samples)
            if samples and isinstance(samples[0], Sequence) and len(samples[0]) > 1:
                label_count = len(samples)

    return image_count, label_count


def _format_shape(shape: Tuple[int, ...]) -> str:
    return "x".join(str(dim) for dim in shape)


def _print_summary(dataset_name: str, split: str, dataset: Dataset, stats: Dict[str, object]) -> None:
    image_count, label_count = _estimate_file_counts(dataset)

    print(f"Dataset: {dataset_name.upper()} ({split})")
    print(f"  Samples reported by dataset: {len(dataset)}")
    if image_count is not None:
        print(f"  Image files found: {image_count}")
    if label_count is not None:
        print(f"  Label files found: {label_count}")

    image_shapes = ", ".join(sorted(_format_shape(shape) for shape in stats["image_shapes"])) or "unknown"
    print(f"  Image tensor shapes (C×H×W): {image_shapes}")
    print(f"  Image tensor dtypes: {', '.join(sorted(stats['image_dtypes'])) or 'unknown'}")
    if stats["image_range"] is not None:
        img_min, img_max = stats["image_range"]
        print(f"  Image value range: [{img_min:.4f}, {img_max:.4f}]")

    if stats["label_count"]:
        label_shapes = ", ".join(sorted(_format_shape(shape) for shape in stats["label_shapes"])) or "unknown"
        print(f"  Label tensor shapes (C×H×W): {label_shapes}")
        print(f"  Label tensor dtypes: {', '.join(sorted(stats['label_dtypes'])) or 'unknown'}")
        if stats["label_range"] is not None:
            lbl_min, lbl_max = stats["label_range"]
            print(f"  Label value range: [{lbl_min:.4f}, {lbl_max:.4f}]")
        if stats["label_unique"] is None:
            print("  Unique label values: more than tracked threshold")
        else:
            sorted_vals = sorted(stats["label_unique"])
            formatted_vals = ", ".join(f"{val:.4f}" for val in sorted_vals)
            print(f"  Unique label values (sampled): {formatted_vals or 'none'}")
    else:
        print("  Labels: none detected (dataset may be unlabeled)")


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize segmentation datasets using DataLoaders.")
    parser.add_argument(
        "--dataset",
        default="all",
        choices=["all", "drive", "ph2", "cmp"],
        help="Dataset to summarize (default: all).",
    )
    parser.add_argument(
        "--split",
        default="all",
        choices=["train", "validate", "test", "all"],
        help="Dataset split to inspect (default: all).",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="DataLoader batch size (default: 1).")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader workers (default: 0).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)

    dataset_names: List[str]
    if args.dataset == "all":
        dataset_names = ["drive", "ph2", "cmp"]
    else:
        dataset_names = [args.dataset]

    splits: List[str]
    if args.split == "all":
        splits = ["train", "validate", "test"]
    else:
        splits = [args.split]

    for dataset_name in dataset_names:
        for split in splits:
            try:
                dataset = _build_dataset(dataset_name, split)
            except (FileNotFoundError, RuntimeError) as err:
                print(f"Dataset: {dataset_name.upper()} ({split})")
                print(f"  Skipped: {err}")
                print()
                continue

            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
            )

            stats = _describe_loader(loader)
            _print_summary(dataset_name, split, dataset, stats)
            print()


if __name__ == "__main__":
    main()
