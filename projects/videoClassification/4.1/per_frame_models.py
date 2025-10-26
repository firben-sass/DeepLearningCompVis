"""Per-frame CNN models for video classification experiments."""

import argparse
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

	def __init__(self, in_channels: int = 3, num_classes: int = 10):
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
		self.eval()
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