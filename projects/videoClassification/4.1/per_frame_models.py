"""Per-frame CNN models for video classification experiments."""

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms as T
from tqdm.auto import tqdm

from datasets import FrameImageDataset


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
		Classify a video by classifying each frame and returning the most frequent class.
		Args:
			frames (torch.Tensor): Tensor of shape (N, C, H, W) where N is number of frames.
		Returns:
			int: The class index that most frames are classified as.
		"""
		super().eval()
		with torch.no_grad():
			logits = self.forward(frames)  # (N, num_classes)
			preds = torch.argmax(logits, dim=1)  # (N,)
			# Count occurrences of each class
			values, counts = preds.unique(return_counts=True)
			most_common = values[counts.argmax()].item()
		return most_common
	
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

	def eval(self, root_dir: str, batch_size: int = 32, num_workers: int = 2, device: Optional[str] = None) -> float:
		"""
		Evaluate the model on a test dataset using FrameImageDataset.
		Args:
			root_dir (str): Path to the root directory containing data (should have frames/ and metadata/).
			batch_size (int): Batch size for DataLoader.
			num_workers (int): Number of workers for DataLoader.
			device (str, optional): Device to run evaluation on. If None, uses 'cuda' if available.
		Returns:
			float: Test accuracy (between 0 and 1).
		"""
		if device is None:
			device = 'cuda' if torch.cuda.is_available() else 'cpu'
		self.to(device)
		super().eval()
        
		transform = T.Compose([
			T.Resize((64, 64)),
			T.ToTensor()
		])
		test_dataset = FrameImageDataset(root_dir=root_dir, split='test', transform=transform)
		test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
		correct = 0
		total = 0
		with torch.no_grad():
			for frames, labels in test_loader:
				frames = frames.to(device)
				labels = labels.to(device)
				outputs = self.forward(frames)
				preds = torch.argmax(outputs, dim=1)
				correct += (preds == labels).sum().item()
				total += labels.size(0)
		return correct / total if total > 0 else 0.0

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
	root_dir = "../data"
	model = SimplePerFrameCNN("per_frame_cnn.pth")
	accuracy = model.eval(root_dir=root_dir, batch_size=16, num_workers=4)
	print(f"Test Accuracy: {accuracy * 100:.2f}%")