from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable, List, Sequence, Union

import torch
from torch import nn
import torch.nn.functional as F


class SimplePerFrameCNN(nn.Module):
	"""
	A compact CNN for per-frame image classification into 10 classes.

	Inputs
	- forward(image):
		image: torch.Tensor
			Either a 3D tensor of shape (C, H, W) representing a single image
			or a 4D tensor of shape (N, C, H, W) representing a batch.
			Dtype should be float32 in range [0, 1] or [0, 255].

	Outputs
	- forward -> torch.Tensor: logits of shape (N, 10)

	Notes
	- The network uses AdaptiveAvgPool2d to support arbitrary input spatial sizes.
	- If a single (C, H, W) image is provided, it is internally batched to (1, C, H, W).
	"""

	def __init__(self, num_classes: int = 10, in_channels: int = 3, device: torch.device | None = None):
		super().__init__()

		self.num_classes = num_classes
		self.in_channels = in_channels

		# Small but effective feature extractor
		self.features = nn.Sequential(
			nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
			nn.BatchNorm2d(32),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2),  # H/2, W/2

			nn.Conv2d(32, 64, kernel_size=3, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2),  # H/4, W/4

			nn.Conv2d(64, 128, kernel_size=3, padding=1),
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True),
		)

		# Global pooling to be input-size agnostic
		self.gap = nn.AdaptiveAvgPool2d((1, 1))

		# Classifier head
		self.classifier = nn.Sequential(
			nn.Flatten(),
			nn.Dropout(p=0.2),
			nn.Linear(128, 64),
			nn.ReLU(inplace=True),
			nn.Dropout(p=0.2),
			nn.Linear(64, num_classes),
		)

		if device is None:
			device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.device = device
		self.to(self.device)

	def _ensure_batched(self, x: torch.Tensor) -> torch.Tensor:
		"""Ensure input is 4D (N, C, H, W). Accepts (C, H, W) or (N, C, H, W)."""
		if x.dim() == 3:
			x = x.unsqueeze(0)
		if x.dim() != 4:
			raise ValueError(f"Expected tensor of shape (C,H,W) or (N,C,H,W), got {tuple(x.shape)}")
		return x

	def _ensure_float(self, x: torch.Tensor) -> torch.Tensor:
		"""Cast to float32 and normalize if values look like [0, 255]."""
		if not torch.is_floating_point(x):
			x = x.float()
		# Heuristic normalization: if max > 1.5, assume 0..255 range
		max_val = x.max()
		if torch.isfinite(max_val) and max_val > 1.5:
			x = x / 255.0
		return x

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""
		Forward pass producing logits over 10 classes.

		Parameters
		- x: torch.Tensor of shape (C,H,W) or (N,C,H,W)

		Returns
		- logits: torch.Tensor of shape (N, num_classes)
		"""
		x = self._ensure_float(x)
		x = self._ensure_batched(x)
		x = x.to(self.device, non_blocking=True)

		feats = self.features(x)
		pooled = self.gap(feats)
		logits = self.classifier(pooled)
		return logits

	@torch.inference_mode()
	def predict_class(self, image: torch.Tensor) -> int:
		"""Predict the single most likely class index for one image tensor (C,H,W)."""
		logits = self.forward(image)
		pred = logits.argmax(dim=-1).item()
		return int(pred)

	@torch.inference_mode()
	def most_predicted_class(self, images: Sequence[torch.Tensor]) -> int:
		"""
		Run the forward pass for each image and return the most predicted class (majority vote).

		Parameters
		- images: sequence of torch.Tensors, each of shape (C,H,W)

		Returns
		- class_idx (int): the class with the highest vote count. Ties are broken by
		  the highest cumulative confidence (softmax max probability) across frames.
		"""
		if len(images) == 0:
			raise ValueError("'images' must be a non-empty sequence of image tensors")

		vote_counts: Counter[int] = Counter()
		confidence_sum: defaultdict[int, float] = defaultdict(float)

		for img in images:
			logits = self.forward(img)  # (1, num_classes)
			probs = F.softmax(logits, dim=-1)
			conf, pred = probs.max(dim=-1)
			cls = int(pred.item())
			vote_counts[cls] += 1
			confidence_sum[cls] += float(conf.item())

		# Majority vote
		if not vote_counts:
			raise RuntimeError("No predictions could be made from the provided images")

		# Determine highest count
		max_count = max(vote_counts.values())
		candidates = [c for c, k in vote_counts.items() if k == max_count]

		if len(candidates) == 1:
			return candidates[0]

		# Tie-breaker: highest cumulative confidence
		best_cls = max(candidates, key=lambda c: confidence_sum[c])
		return int(best_cls)


__all__ = ["SimplePerFrameCNN", "PerFrameTrainer"]


class PerFrameTrainer:
	"""
	Trainer that treats each frame in a video independently for supervision.

	Expected dataloader/yielding format per iteration:
	- (video, label)
	  where video is a tensor of shape (T, C, H, W) OR (B, T, C, H, W)
			label is an int/tensor (for (T, ...)) or shape (B,) for (B, T, ...)

	Behavior
	- Performs a separate forward/backward/step for each frame with the same label.
	- Tracks average loss and per-frame accuracy across the epoch.
	"""

	def __init__(
		self,
		model: SimplePerFrameCNN,
		optimizer: torch.optim.Optimizer | None = None,
		criterion: nn.Module | None = None,
		device: torch.device | None = None,
		grad_clip_norm: float | None = None,
		batch_frames: bool = True,
		frame_batch_size: int = 32,
		eval_frame_batch_size: int = 128,
	) -> None:
		self.model = model
		self.device = device or getattr(model, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
		self.model.to(self.device)

		self.criterion = criterion or nn.CrossEntropyLoss()
		self.optimizer = optimizer or torch.optim.Adam(self.model.parameters(), lr=1e-3)
		self.grad_clip_norm = grad_clip_norm
		self.batch_frames = batch_frames
		self.frame_batch_size = int(frame_batch_size)
		self.eval_frame_batch_size = int(eval_frame_batch_size)

	def _yield_frames_and_labels(self, video: torch.Tensor, label: Union[int, torch.Tensor]):
		"""Yield (frame, label) pairs from either (T,C,H,W) or (B,T,C,H,W)."""
		if video.dim() == 4:  # (T, C, H, W)
			T = video.size(0)
			for t in range(T):
				yield video[t], label
		elif video.dim() == 5:  # (B, T, C, H, W)
			B, T = video.size(0), video.size(1)
			if torch.is_tensor(label) and label.dim() == 0:
				labels = [label.item()] * B
			elif torch.is_tensor(label):
				labels = label.tolist()
			else:
				labels = label
			for b in range(B):
				for t in range(T):
					yield video[b, t], labels[b]
		else:
			raise ValueError(f"Expected video of shape (T,C,H,W) or (B,T,C,H,W), got {tuple(video.shape)}")

	def train_epoch(self, dataloader, log_interval: int = 50) -> dict:
		self.model.train()
		total_loss = 0.0
		total_frames = 0
		correct_frames = 0

		for step, batch in enumerate(dataloader):
			if isinstance(batch, (list, tuple)) and len(batch) == 2:
				video, label = batch
			else:
				raise ValueError("Each batch must be a (video, label) pair")

			# Ensure tensor types
			if torch.is_tensor(video):
				pass
			else:
				video = torch.as_tensor(video)

			if not self.batch_frames:
				# Per-frame step (baseline, slower)
				for frame, lbl in self._yield_frames_and_labels(video, label):
					target = torch.as_tensor(lbl, device=self.device, dtype=torch.long).view(1)

					self.optimizer.zero_grad(set_to_none=True)
					logits = self.model(frame)  # (1, num_classes)
					loss = self.criterion(logits, target)
					loss.backward()

					if self.grad_clip_norm is not None:
						nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

					self.optimizer.step()

					total_loss += float(loss.item())
					total_frames += 1
					pred = logits.argmax(dim=-1)
					correct_frames += int((pred == target).sum().item())
			else:
				# Batched frames for efficiency
				frames_buf: list[torch.Tensor] = []
				targets_buf: list[torch.Tensor] = []

				def _flush():
					if not frames_buf:
						return 0, 0.0, 0
					frames = torch.stack(frames_buf, dim=0).to(self.device, non_blocking=True)
					targets = torch.stack(targets_buf, dim=0).to(self.device)

					self.optimizer.zero_grad(set_to_none=True)
					logits = self.model(frames)  # (N, num_classes)
					loss = self.criterion(logits, targets)
					loss.backward()

					if self.grad_clip_norm is not None:
						nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

					self.optimizer.step()

					pred = logits.argmax(dim=-1)
					correct = int((pred == targets).sum().item())
					N = targets.numel()
					# accumulate sum of per-frame loss (criterion default mean)
					loss_sum = float(loss.item()) * N

					frames_buf.clear()
					targets_buf.clear()
					return N, loss_sum, correct

				for frame, lbl in self._yield_frames_and_labels(video, label):
					target = torch.as_tensor(lbl, dtype=torch.long)
					frames_buf.append(frame)
					targets_buf.append(target)
					if len(frames_buf) >= self.frame_batch_size:
						N, loss_sum, correct = _flush()
						total_frames += N
						total_loss += loss_sum
						correct_frames += correct

				# Flush remainder
				N, loss_sum, correct = _flush()
				total_frames += N
				total_loss += loss_sum
				correct_frames += correct

			if log_interval and total_frames and (step + 1) % log_interval == 0:
				avg_loss = total_loss / total_frames
				acc = correct_frames / total_frames
				# Optional: print progress — kept minimal to avoid noisy output
				# print(f"Step {step+1}: avg_loss={avg_loss:.4f}, acc_frame={acc:.4f}")

		avg_loss = total_loss / max(1, total_frames)
		acc_frame = correct_frames / max(1, total_frames)
		return {"loss": avg_loss, "acc_frame": acc_frame, "frames": total_frames}

	@torch.inference_mode()
	def evaluate(self, dataloader) -> dict:
		self.model.eval()
		total_loss = 0.0
		total_frames = 0
		correct_frames = 0
		correct_videos_majority = 0
		total_videos = 0

		for batch in dataloader:
			if isinstance(batch, (list, tuple)) and len(batch) == 2:
				video, label = batch
			else:
				raise ValueError("Each batch must be a (video, label) pair")

			# Materialize frames and labels in order
			if video.dim() == 4:  # (T, C, H, W)
				Tvid = video.size(0)
				frames = [video[t] for t in range(Tvid)]
				labels = [label] * Tvid
				total_videos += 1
			elif video.dim() == 5:  # (B, T, C, H, W)
				B, Tvid = video.size(0), video.size(1)
				if torch.is_tensor(label) and label.dim() == 0:
					label_list = [label.item()] * B
				elif torch.is_tensor(label):
					label_list = label.tolist()
				else:
					label_list = label
				frames = [video[b, t] for b in range(B) for t in range(Tvid)]
				labels = [label_list[b] for b in range(B) for _ in range(Tvid)]
				total_videos += B
			else:
				raise ValueError(f"Expected video of shape (T,C,H,W) or (B,T,C,H,W), got {tuple(video.shape)}")

			# Batched evaluation over frames for speed
			frame_preds: list[int] = []
			start = 0
			while start < len(frames):
				end = min(start + self.eval_frame_batch_size, len(frames))
				chunk = torch.stack(frames[start:end], dim=0).to(self.device, non_blocking=True)
				target_chunk = torch.as_tensor(labels[start:end], device=self.device, dtype=torch.long)
				logits = self.model(chunk)
				loss = self.criterion(logits, target_chunk)
				# accumulate sum of per-frame loss
				total_loss += float(loss.item()) * (end - start)
				total_frames += (end - start)
				preds = logits.argmax(dim=-1)
				correct_frames += int((preds == target_chunk).sum().item())
				frame_preds.extend([int(p) for p in preds.tolist()])
				start = end

			# Majority vote per video
			if video.dim() == 4:
				majority = Counter(frame_preds).most_common(1)[0][0]
				true_label = int(labels[0])
				correct_videos_majority += int(majority == true_label)
			else:
				for i in range(0, len(frame_preds), Tvid):
					majority = Counter(frame_preds[i : i + Tvid]).most_common(1)[0][0]
					true_label = int(labels[i])
					correct_videos_majority += int(majority == true_label)

		avg_loss = total_loss / max(1, total_frames)
		acc_frame = correct_frames / max(1, total_frames)
		acc_video_majority = correct_videos_majority / max(1, total_videos)
		return {"loss": avg_loss, "acc_frame": acc_frame, "acc_video_majority": acc_video_majority, "frames": total_frames, "videos": total_videos}


if __name__ == "__main__":
	# Minimal smoke test
	model = SimplePerFrameCNN(num_classes=10)
	img = torch.randn(3, 128, 128)  # single image
	logits = model(img)
	print("Logits shape (single):", logits.shape)
	preds = model.most_predicted_class([torch.randn(3, 128, 128) for _ in range(5)])
	print("Most predicted class (dummy data):", preds)

