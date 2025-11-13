import argparse
import os
import numpy as np
import glob
from pathlib import Path
import PIL.Image as Image

# pip install torchsummary
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models
from torchsummary import summary
import torch.optim as optim
from time import time
from lib.model.EncDecModel import EncDec
from lib.model.DilatedNetModel import DilatedNet
from lib.model.UNetModel import UNet, UNet2
from lib.losses import BCELoss, DiceLoss, FocalLoss, BCELoss_TotalVariation, CrossEntropySegmentationLoss
from lib.dataset.Datasets import PhC, CMP, DRIVE, PH2
from lib.plotting import plot_training_curves
from measure import (
    accuracy,
    dice_overlap,
    intersection_over_union,
    sensitivity,
    specificity,
)

def parse_args():
    parser = argparse.ArgumentParser(description="Train a segmentation model.")
    parser.add_argument(
        "--run-name",
        required=True,
        help="Unique name for this training run (used for saving artifacts).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs to execute.",
    )
    parser.add_argument(
        "--dataset",
        type=lambda value: value.upper(),
        choices=["PH2", "DRIVE"],
        default="PH2",
        help="Dataset to use for training and evaluation.",
    )
    return parser.parse_args()


# Dataset
size = 256
train_transform = transforms.Compose([
    transforms.Resize((size, size)),
    transforms.ToTensor(),
])
test_transform = transforms.Compose([
    transforms.Resize((size, size)),
    transforms.ToTensor(),
])
label_transform = transforms.Compose([
    transforms.Resize((size, size), interpolation=Image.NEAREST),
    transforms.ToTensor(),
])

batch_size = 6


def main():
    args = parse_args()

    run_name = Path(args.run_name).name
    if not run_name:
        raise ValueError("Run name resolves to an empty path component.")

    if args.epochs <= 0:
        raise ValueError("Epochs must be a positive integer.")

    dataset_map = {
        "PH2": PH2,
        "DRIVE": DRIVE,
    }
    dataset_cls = dataset_map[args.dataset]

    # Prepare datasets and loaders based on the requested dataset.
    trainset = dataset_cls(split="train", transform=train_transform, label_transform=label_transform)
    testset = dataset_cls(split="test", transform=test_transform, label_transform=label_transform)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=3)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=3)

    print(f"Loaded {len(trainset)} {args.dataset} training images")
    print(f"Loaded {len(testset)} {args.dataset} test images")

    # Training setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # model = EncDec().to(device)
    # model = UNet().to(device)  # TODO
    model = UNet2().to(device)  # TODO
    # model = DilatedNet().to(device)  # TODO
    summary(model, (3, 256, 256))
    learning_rate = 0.001
    opt = optim.Adam(model.parameters(), learning_rate)

    # loss_fn = BCELoss()
    # loss_fn = DiceLoss()  # TODO
    # loss_fn = FocalLoss()  # TODO
    # loss_fn = BCELoss_TotalVariation()  # TODO
    loss_fn = BCELoss()
    epochs = args.epochs

    metric_fns = {
        "Dice": dice_overlap,
        "IoU": intersection_over_union,
        "Accuracy": accuracy,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
    }
    metric_history = {name: [] for name in metric_fns}
    train_losses = []
    val_losses = []

    model.train()  # train mode
    for epoch in range(epochs):
        tic = time()
        print(f"* Epoch {epoch + 1}/{epochs}")

        epoch_train_loss = 0.0
        for X_batch, y_true in train_loader:
            X_batch = X_batch.to(device)
            y_true = y_true.to(device)

            opt.zero_grad()

            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_true)
            loss.backward()
            opt.step()

            epoch_train_loss += loss.item()

        epoch_train_loss /= max(len(train_loader), 1)
        train_losses.append(epoch_train_loss)

        model.eval()

        epoch_val_loss = 0.0
        metrics_sums = {name: 0.0 for name in metric_fns}
        with torch.no_grad():
            for X_val, y_val in test_loader:
                X_val = X_val.to(device)
                y_val = y_val.to(device)

                logits = model(X_val)
                val_loss = loss_fn(logits, y_val)
                epoch_val_loss += val_loss.item()

                predictions = (torch.sigmoid(logits) > 0.5).float()
                for name, fn in metric_fns.items():
                    metrics_sums[name] += fn(predictions, y_val)

        num_val_batches = max(len(test_loader), 1)
        epoch_val_loss /= num_val_batches
        val_losses.append(epoch_val_loss)

        for name in metric_fns:
            metric_value = metrics_sums[name] / num_val_batches
            metric_history[name].append(metric_value)

        print(f" - train_loss: {epoch_train_loss:.4f} | val_loss: {epoch_val_loss:.4f}")
        for name in metric_fns:
            print(f"   {name}: {metric_history[name][-1]:.4f}")

        model.train()

    artifacts_root = Path(__file__).resolve().parent
    model_dir = artifacts_root / "saved_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{run_name}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Training has finished! Model saved to {model_path}")

    plots_dir = artifacts_root / "outputs" / "training_curves" / run_name
    plot_training_curves(
        {"train": train_losses, "val": val_losses},
        metric_history,
        output_dir=plots_dir,
        show=False,
    )
    print(f"Training curves written to {plots_dir}")


if __name__ == "__main__":
    main()
