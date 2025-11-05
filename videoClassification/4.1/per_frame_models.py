"""Per-frame CNN models for video classification experiments."""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms as T
from tqdm.auto import tqdm

from datasets import FrameImageDataset, FrameVideoDataset


class SimplePerFrameCNN(nn.Module):
	"""Lightweight CNN for per-frame classification."""

	def __init__(self, in_channels: int = 3, num_classes: int = 10, model_path: Optional[str] = None, map_location: Optional[str] = None):
		super().__init__()

		# Stem keeps spatial dims manageable while extracting low-level features.
		self.stem = nn.Sequential(
			nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(32),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2),
		)

		self.features = nn.Sequential(
			nn.Conv2d(32, 64, kernel_size=3, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2),
			nn.Conv2d(64, 128, kernel_size=3, padding=1),
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True),
			nn.AdaptiveAvgPool2d((1, 1)),
		)

		self.classifier = nn.Linear(128, num_classes)

		if model_path:
			self.load_weights(model_path, map_location=map_location)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""Return per-frame logits given batched frames."""

		x = self.stem(x)
		x = self.features(x)
		x = torch.flatten(x, 1)
		return self.classifier(x)

	def extract_features(self, x: torch.Tensor) -> torch.Tensor:
		"""Expose pooled feature vector for downstream heads."""

		x = self.stem(x)
		x = self.features(x)
		return torch.flatten(x, 1)

	def infer(self, frames: torch.Tensor) -> int:
		"""
		Classify a video by averaging per-frame probabilities and taking the highest score.
		Args:
			frames (torch.Tensor): Tensor of shape (N, C, H, W) where N is number of frames.
		Returns:
			int: The predicted class index based on averaged probabilities across frames.
		"""
		super().eval()
		with torch.no_grad():
			logits = self.forward(frames)  # (N, num_classes)
			probs = F.softmax(logits, dim=1)
			avg_probs = probs.mean(dim=0)  # (num_classes,)
			predicted_class = torch.argmax(avg_probs).item()
		return predicted_class
	
	def eval_videos(self, videos: list, labels: list) -> float:
		"""
		Evaluate the model on a list of videos and their true classes.
		Args:
			videos (list of torch.Tensor): Each tensor is (N, C, H, W) for a video.
			labels (list of int): True class indices for each video.
		Returns:
			float: Test accuracy (between 0 and 1).
		"""
		assert len(videos) == len(labels), "Number of videos and labels must match."
		correct = 0
		total = len(videos)
		for video, label in zip(videos, labels):
			pred = self.infer(video)
			if pred == label:
				correct += 1
		return correct / total if total > 0 else 0.0

	def eval(self, *args, **kwargs):
		"""Extend nn.Module.eval with dataset-based evaluation when arguments are provided."""
		if not args and 'root_dir' not in kwargs and 'split' not in kwargs:
			return super().eval(*args, **kwargs)
		return self._eval_dataset(*args, **kwargs)

	def _eval_dataset(
		self,
		root_dir: str,
		split: str = 'test',
		transform: Optional[T.Compose] = None,
		batch_size: int = 1,
		num_workers: int = 0,
		stack_frames: bool = True,
		limit: Optional[int] = None,
		show_progress: bool = True,
	) -> float:
		"""Evaluate accuracy on a dataset of videos drawn from ``FrameVideoDataset``."""
		super().eval()
		if transform is None:
			transform = T.Compose([T.Resize((64, 64)), T.ToTensor()])
		dataset = FrameVideoDataset(
			root_dir=root_dir,
			split=split,
			transform=transform,
			stack_frames=stack_frames,
		)
		if limit is not None:
			limit = min(limit, len(dataset))
			dataset = torch.utils.data.Subset(dataset, list(range(limit)))
		loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
		videos: List[torch.Tensor] = []
		labels: List[int] = []
		iterator = loader
		if show_progress:
			iterator = tqdm(loader, desc='Collecting videos', leave=False)
		for batch_frames, batch_labels in iterator:
			for frames_tensor, label_tensor in zip(batch_frames, batch_labels):
				if stack_frames:
					frames_tensor = frames_tensor.permute(1, 0, 2, 3)
				else:
					if isinstance(frames_tensor, (list, tuple)):
						frames_tensor = torch.stack(list(frames_tensor))
				videos.append(frames_tensor)
				labels.append(int(label_tensor.item()))
				if limit is not None and len(videos) >= limit:
					break
			if limit is not None and len(videos) >= limit:
				break
		device = next(self.parameters()).device
		videos = [video.to(device) for video in videos]
		return self.eval_videos(videos, labels)

	def load_weights(self, weights_path: str, map_location: Optional[str] = None) -> None:
		"""Load model weights from disk, defaulting to the local models directory."""
		checkpoint_path = Path(weights_path)
		if not checkpoint_path.is_file():
			checkpoint_path = Path(__file__).resolve().parent / 'models' / weights_path
		if not checkpoint_path.is_file():
			raise FileNotFoundError(f"Could not find weights file at {weights_path} or {checkpoint_path}")
		state = torch.load(checkpoint_path, map_location=map_location or 'cpu')
		if isinstance(state, dict) and 'state_dict' in state:
			state = state['state_dict']
		self.load_state_dict(state)


if __name__ == '__main__':
	root_dir = "projects/videoClassification/data"
	model = SimplePerFrameCNN(model_path="per_frame_cnn.pth")
	accuracy = model.eval(root_dir=root_dir, batch_size=16, num_workers=4)
	print(f"Test Accuracy: {accuracy * 100:.2f}%")