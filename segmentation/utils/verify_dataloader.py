#!/usr/bin/env python3
"""Utility script for sanity-checking segmentation dataloaders."""

import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image

from lib.dataset.Datasets import DRIVE, PH2


def make_transforms(image_size=256):
	image_transform = transforms.Compose([
		transforms.Resize((image_size, image_size)),
		transforms.ToTensor(),
	])
	label_transform = transforms.Compose([
		transforms.Resize((image_size, image_size), interpolation=Image.NEAREST),
		transforms.ToTensor(),
	])
	return image_transform, label_transform


def prepare_loader(dataset_cls, image_transform, label_transform, batch_size=4, *, split='train'):
	dataset = dataset_cls(
		split=split,
		transform=image_transform,
		label_transform=label_transform,
	)
	return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)


def tensor_to_numpy_image(tensor):
	tensor = tensor.detach().cpu()
	if tensor.ndim == 3:
		if tensor.shape[0] == 1:
			return tensor.squeeze(0).numpy()
		if tensor.shape[0] == 3:
			return tensor.permute(1, 2, 0).numpy()
		return torch.argmax(tensor, dim=0).numpy()
	if tensor.ndim == 2:
		return tensor.numpy()
	raise ValueError(f"Unexpected tensor shape for visualization: {tuple(tensor.shape)}")


def plot_batch(images, labels, title, save_path=None):
	batch_size = images.size(0)
	fig, axes = plt.subplots(2, batch_size, figsize=(4 * batch_size, 8))
	if batch_size == 1:
		axes = axes.reshape(2, 1)

	for idx in range(batch_size):
		img_np = tensor_to_numpy_image(images[idx])
		mask_np = tensor_to_numpy_image(labels[idx])

		axes[0, idx].imshow(img_np, cmap="gray" if img_np.ndim == 2 else None)
		axes[0, idx].axis("off")
		axes[0, idx].set_title(f"Sample {idx + 1}")

		axes[1, idx].imshow(mask_np, cmap="gray")
		axes[1, idx].axis("off")

	fig.suptitle(title)
	plt.tight_layout()
	if save_path:
		fig.savefig(save_path, bbox_inches="tight")
	plt.show()
	plt.close(fig)


def main():
	batch_size = 4
	image_size = 256
	image_transform, label_transform = make_transforms(image_size)

	save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs", "verify_dataloader"))
	os.makedirs(save_dir, exist_ok=True)

	drive_loader = prepare_loader(DRIVE, image_transform, label_transform, batch_size, split='train')
	ph2_loader = prepare_loader(PH2, image_transform, label_transform, batch_size, split='train')

	drive_images, drive_labels = next(iter(drive_loader))
	drive_path = os.path.join(save_dir, "drive_samples.png")
	plot_batch(drive_images, drive_labels, "DRIVE Training Samples", save_path=drive_path)

	ph2_images, ph2_labels = next(iter(ph2_loader))
	ph2_path = os.path.join(save_dir, "ph2_samples.png")
	plot_batch(ph2_images, ph2_labels, "PH2 Training Samples", save_path=ph2_path)


if __name__ == "__main__":
	main()

