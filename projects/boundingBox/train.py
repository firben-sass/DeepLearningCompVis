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
from lib.losses import BCELoss, DiceLoss, FocalLoss, BCELoss_TotalVariation
from lib.dataset.Datasets import PhC, CMP, DRIVE, PH2
from lib.plotting import plot_training_curves
from utils.evaluation import evaluate_model
from utils.report_writer import write_metrics_report
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
        default="test_run",
        help="Unique name for this training run (used for saving artifacts).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs to execute.",
    )
    parser.add_argument(
        "--dataset",
        type=lambda value: value.upper(),
        choices=["PH2", "DRIVE"],
        default="PH2",
        help="Dataset to use for training and evaluation.",
    )
    parser.add_argument(
        "--model",
        type=lambda value: value.lower(),
        choices=["encdec", "unet", "unet2", "dilatednet"],
        default="unet2",
        help="Segmentation architecture to train (case-insensitive).",
    )
    parser.add_argument(
        "--loss",
        type=lambda value: value.lower(),
        choices=[
            "bce",
            "dice",
            "focal",
            "bce_tv",
        ],
        default="dice",
        help="Loss function to optimise (case-insensitive).",
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

    model_map = {
        "encdec": EncDec,
        "unet": UNet,
        "unet2": UNet2,
        "dilatednet": DilatedNet,
    }
    model_cls = model_map[args.model]

    loss_map = {
        "bce": BCELoss,
        "dice": DiceLoss,
        "focal": FocalLoss,
        "bce_tv": BCELoss_TotalVariation,
    }
    loss_cls = loss_map[args.loss]

    # Prepare datasets and loaders based on the requested dataset.
    trainset = dataset_cls(split="train", transform=train_transform, label_transform=label_transform)
    valset = dataset_cls(split="validate", transform=test_transform, label_transform=label_transform)
    testset = dataset_cls(split="test", transform=test_transform, label_transform=label_transform)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=3)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=3)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=3)

    print(f"Loaded {len(trainset)} {args.dataset} training images")
    print(f"Loaded {len(valset)} {args.dataset} validation images")
    print(f"Loaded {len(testset)} {args.dataset} test images")

    # Training setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Using model: {model_cls.__name__}")
    model = model_cls().to(device)
    summary(model, (3, size, size))
    learning_rate = 0.001
    opt = optim.Adam(model.parameters(), learning_rate)

    loss_fn = loss_cls()
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

    artifacts_root = Path(__file__).resolve().parent
    model_dir = artifacts_root / "saved_models"
    checkpoint_dir = model_dir / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model.train()  # train mode
    checkpoint_records = []
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

        epoch_train_loss /= len(train_loader)
        train_losses.append(epoch_train_loss)

        epoch_val_loss, epoch_metrics = evaluate_model(
            model=model,
            data_loader=val_loader,
            loss_fn=loss_fn,
            metric_fns=metric_fns,
            device=device,
        )
        val_losses.append(epoch_val_loss)
        for name, metric_value in epoch_metrics.items():
            metric_history[name].append(metric_value)

        print(f" - train_loss: {epoch_train_loss:.4f} | val_loss: {epoch_val_loss:.4f}")
        for name in metric_fns:
            print(f"   {name}: {metric_history[name][-1]:.4f}")

        if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
            checkpoint_path = checkpoint_dir / f"epoch_{epoch + 1:03d}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            checkpoint_records.append({
                "epoch": epoch + 1,
                "val_loss": epoch_val_loss,
                "path": checkpoint_path,
            })
            print(f"   Saved checkpoint to {checkpoint_path}")

    print(f"Training has finished! Models saved to {checkpoint_dir}")

    if not checkpoint_records:
        raise RuntimeError("No checkpoints were saved; cannot select best model.")

    best_checkpoint = min(checkpoint_records, key=lambda record: record["val_loss"])
    print(
        "Best checkpoint by validation loss: "
        f"epoch {best_checkpoint['epoch']} with val_loss {best_checkpoint['val_loss']:.4f}"
    )

    model.load_state_dict(torch.load(best_checkpoint["path"], map_location=device))

    test_loss, test_metrics = evaluate_model(
        model=model,
        data_loader=test_loader,
        loss_fn=loss_fn,
        metric_fns=metric_fns,
        device=device,
    )
    print(f"Test loss: {test_loss:.4f}")
    for name, value in test_metrics.items():
        print(f"Test {name}: {value:.4f}")

    plots_dir = artifacts_root / "outputs" / "training_curves" / run_name
    plot_training_curves(
        {"train": train_losses, "val": val_losses},
        metric_history,
        output_dir=plots_dir,
        show=False,
    )
    print(f"Training curves written to {plots_dir}")

    metadata = [
        ("Dataset", args.dataset),
        ("Model", model_cls.__name__),
        ("Loss", loss_cls.__name__),
        ("Epochs", epochs),
        ("Best checkpoint epoch", best_checkpoint["epoch"]),
        ("Best checkpoint val_loss", f"{best_checkpoint['val_loss']:.4f}"),
    ]
    report_path = write_metrics_report(
        output_dir=plots_dir,
        run_name=run_name,
        metadata=metadata,
        test_loss=test_loss,
        test_metrics=test_metrics,
    )
    print(f"Test metrics written to {report_path}")


if __name__ == "__main__":
    main()
