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
	"""Lightweight CNN for per-frame feature extraction."""

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

	def infer(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		"""Return logits and probabilities for convenience."""

		logits = self.forward(x)
		probs = F.softmax(logits, dim=-1)
		return logits, probs