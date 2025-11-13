"""Self-contained 3D CNN training module for video classification on HPC clusters."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch
from torch import nn
from torch.cuda import amp
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.io import read_video


@dataclass
class DataConfig:
    train_manifest: Path
    val_manifest: Path
    video_root: Path
    num_classes: int
    clip_len: int = 16
    frame_rate: Optional[int] = None
    batch_size: int = 8
    num_workers: int = 4
    pin_memory: bool = True


class VideoClipDataset(Dataset):
    """Dataset that loads clips from video files listed in a CSV manifest."""

    def __init__(
        self,
        manifest_file: Path,
        video_root: Path,
        clip_len: int,
        frame_rate: Optional[int] = None,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        self.video_root = video_root
        self.clip_len = clip_len
        self.frame_rate = frame_rate
        self.transform = transform
        self.samples: List[Tuple[Path, int]] = []

        with open(manifest_file, "r", newline="") as f:
            reader = csv.DictReader(f)
            if "video" not in reader.fieldnames or "label" not in reader.fieldnames:
                raise ValueError("Manifest must contain 'video' and 'label' columns")
            for row in reader:
                video_path = video_root / row["video"]
                label = int(row["label"])
                self.samples.append((video_path, label))

    def __len__(self) -> int:
        return len(self.samples)

    def _sample_indices(self, num_frames: int) -> torch.Tensor:
        if num_frames >= self.clip_len:
            return torch.linspace(0, num_frames - 1, steps=self.clip_len).long()
        repeats = math.ceil(self.clip_len / max(num_frames, 1))
        tiled = torch.arange(num_frames).repeat(repeats)
        return tiled[: self.clip_len]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        video_path, label = self.samples[idx]
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        video, _, info = read_video(
            str(video_path),
            pts_unit="sec",
            output_format="TCHW",
            stream="video",
        )

        if video.dtype != torch.float32:
            video = video.float()

        # read_video returns T x C x H x W when using TCHW
        clip = video.permute(1, 0, 2, 3)

        if self.frame_rate is not None and info.get("video_fps"):
            step = max(int(info["video_fps"] // self.frame_rate), 1)
            clip = clip[:, ::step]

        frame_count = clip.shape[1]
        if frame_count == 0:
            raise RuntimeError(f"Video {video_path} contains no frames")

        indices = self._sample_indices(frame_count)
        clip = clip[:, indices]
        clip = clip / 255.0

        if self.transform:
            clip = self.transform(clip)

        return clip, label


class Simple3DCNN(nn.Module):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((2, 2, 2)),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((2, 2, 2)),
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


class VideoClassificationTrainer:
    """Encapsulates dataloaders, model, and training loop for 3D CNNs."""

    def __init__(
        self,
        data_cfg: DataConfig,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 30,
        mixed_precision: bool = True,
        grad_clip: Optional[float] = 1.0,
        output_dir: Optional[Path] = None,
    ) -> None:
        self.data_cfg = data_cfg
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.grad_clip = grad_clip
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mixed_precision = mixed_precision and self.device.type == "cuda"
        self.scaler = amp.GradScaler(enabled=self.mixed_precision)
        self.output_dir = (output_dir or Path.cwd() / "checkpoints").resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history = {"epoch": [], "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

        self.model: nn.Module = Simple3DCNN(3, data_cfg.num_classes)
        self.model.to(self.device)

        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs
        )

        self.train_loader = self._build_loader(data_cfg.train_manifest, shuffle=True)
        self.val_loader = self._build_loader(data_cfg.val_manifest, shuffle=False)

    def _default_transforms(self, train: bool) -> Callable[[torch.Tensor], torch.Tensor]:
        if train:
            augment = transforms.Compose(
                (
                    transforms.RandomResizedCrop(size=112, scale=(0.8, 1.0), antialias=True),
                    transforms.RandomHorizontalFlip(),
                )
            )
        else:
            augment = transforms.Compose(
                (
                    transforms.Resize(size=128, antialias=True),
                    transforms.CenterCrop(112),
                )
            )

        normalize = transforms.Normalize(
            mean=(0.43216, 0.394666, 0.37645), std=(0.22803, 0.22145, 0.216989)
        )

        def apply_all(clip: torch.Tensor) -> torch.Tensor:
            clip = clip.permute(1, 0, 2, 3)
            frames = [normalize(augment(frame)) for frame in clip]
            return torch.stack(frames).permute(1, 0, 2, 3)

        return apply_all

    def _build_loader(self, manifest: Path, shuffle: bool) -> DataLoader:
        transform = self._default_transforms(train=shuffle)
        dataset = VideoClipDataset(
            manifest_file=manifest,
            video_root=self.data_cfg.video_root,
            clip_len=self.data_cfg.clip_len,
            frame_rate=self.data_cfg.frame_rate,
            transform=transform,
        )
        return DataLoader(
            dataset,
            batch_size=self.data_cfg.batch_size,
            shuffle=shuffle,
            num_workers=self.data_cfg.num_workers,
            pin_memory=self.data_cfg.pin_memory and self.device.type == "cuda",
        )

    def train(self) -> None:
        best_acc = 0.0
        for epoch in range(1, self.epochs + 1):
            train_loss, train_acc = self._train_one_epoch(epoch)
            val_loss, val_acc = self._validate()
            self.scheduler.step()

            if val_acc > best_acc:
                best_acc = val_acc
                self._save_checkpoint(epoch, best=True)

            print(
                f"Epoch {epoch:03d}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"train_acc={train_acc:.2%} val_acc={val_acc:.2%}"
            )
            self._save_checkpoint(epoch, best=False)
            
            # record
            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)

            # persist to CSV for reproducibility
            metrics_csv = self.output_dir / "metrics.csv"
            write_header = not metrics_csv.exists()
            with open(metrics_csv, "a", newline="") as f:
                w = csv.writer(f)
                if write_header:
                    w.writerow(["epoch","train_loss","val_loss","train_acc","val_acc"])
                w.writerow([epoch, train_loss, val_loss, train_acc, val_acc])
        
        self._render_plots()



    def _render_plots(self) -> None:
        try:
            # safe for headless HPC nodes
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            epochs = self.history["epoch"]

            # Loss
            plt.figure()
            plt.plot(epochs, self.history["train_loss"], label="train")
            plt.plot(epochs, self.history["val_loss"], label="val")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Loss vs. Epochs")
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.output_dir / "loss.png", dpi=150)

            # Accuracy
            plt.figure()
            plt.plot(epochs, [x * 100 for x in self.history["train_acc"]], label="train")
            plt.plot(epochs, [x * 100 for x in self.history["val_acc"]], label="val")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy (%)")
            plt.title("Accuracy vs. Epochs")
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.output_dir / "accuracy.png", dpi=150)

            print(f"Saved plots to {self.output_dir}")
        except Exception as e:
            print(f"Plotting failed: {e}")



    def _train_one_epoch(self, epoch: int) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for step, (clips, labels) in enumerate(self.train_loader, start=1):
            clips = clips.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with amp.autocast(enabled=self.mixed_precision):
                logits = self.model(clips)
                loss = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()

            if self.grad_clip is not None:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item() * labels.size(0)
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            if step % 10 == 0:
                print(
                    f"Epoch {epoch} Step {step}: loss={loss.item():.4f} "
                    f"acc={(correct / total):.2%}"
                )

        return total_loss / total, correct / total

    def _validate(self) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for clips, labels in self.val_loader:
                clips = clips.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                logits = self.model(clips)
                loss = self.criterion(logits, labels)
                total_loss += loss.item() * labels.size(0)
                predictions = logits.argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        return total_loss / total, correct / total

    def _save_checkpoint(self, epoch: int, best: bool) -> None:
        ckpt = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
        }
        tag = "best" if best else f"epoch{epoch:03d}"
        torch.save(ckpt, self.output_dir / f"checkpoint_{tag}.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a 3D CNN for video classification")
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--clip-len", type=int, default=16)
    parser.add_argument("--frame-rate", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=Path("./checkpoints"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.backends.cudnn.benchmark = True

    data_cfg = DataConfig(
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        video_root=args.video_root,
        num_classes=args.num_classes,
        clip_len=args.clip_len,
        frame_rate=args.frame_rate,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    trainer = VideoClassificationTrainer(
        data_cfg=data_cfg,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        mixed_precision=not args.no_amp,
        grad_clip=args.grad_clip,
        output_dir=args.output_dir,
    )
    trainer.train()


if __name__ == "__main__":
    main()
