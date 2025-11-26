"""Train per-frame CNN using the shared train() helper."""

import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import T as T

from lib.model.mobilnet import ProposalClassifier
from lib.dataset.Datasets import PotholeDataset
from train import train

def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser()
	parser.add_argument("--root_dir", default="../data")
	parser.add_argument("--batch_size", type=int, default=64)
	parser.add_argument("--epochs", type=int, default=500)
	parser.add_argument("--lr", type=float, default=1e-3)
	parser.add_argument("--device", default="cuda")
	parser.add_argument("--save_name", default="per_frame_cnn")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	device = torch.device(args.device if torch.cuda.is_available() else "cpu")

	# Training T with augmentation
	train_transform = T.Compose([
		T.ToTensor(),
		T.Normalize(mean=[0.485, 0.456, 0.406], 
							std=[0.229, 0.224, 0.225])
	])

	# Validation T (no augmentation)
	val_transform = T.Compose([
		T.ToTensor(),
		T.Normalize(mean=[0.485, 0.456, 0.406], 
							std=[0.229, 0.224, 0.225])
	])

	train_dataset = PotholeDataset(root_dir=args.root_dir, split="train", transform=train_transform)
	val_dataset = PotholeDataset(root_dir=args.root_dir, split="val", transform=val_transform)

	train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
	val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

	model = ProposalClassifier(num_classes=len(set(train_dataset.df["label"].tolist())))
	model.to(device)

	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

	train(
		model,
		optimizer,
		num_epochs=args.epochs,
		save_name_model=args.save_name,
		train_loader=train_loader,
		val_loader=val_loader,
		trainset=train_dataset,
		valset=val_dataset,
		device=device,
	)


if __name__ == "__main__":
	main()
