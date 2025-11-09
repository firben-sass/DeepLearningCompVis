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

# Dataset
size = 256
train_transform = transforms.Compose([transforms.Resize((size, size)),
                                    transforms.ToTensor()])
test_transform = transforms.Compose([transforms.Resize((size, size)),
                                    transforms.ToTensor()])
label_transform = transforms.Compose([
    transforms.Resize((size, size), interpolation=Image.NEAREST),
    transforms.ToTensor()
])

batch_size = 6
# trainset = PhC(train=True, transform=train_transform)
# train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
#                           num_workers=3)
# testset = PhC(train=False, transform=test_transform)
# test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False,
#                           num_workers=3)
# trainset = CMP(train=True, transform=train_transform, label_transform=label_transform, num_classes=12)
# train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
#                           num_workers=3)
# testset = CMP(train=False, transform=test_transform, label_transform=label_transform, num_classes=12)
# test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False,
#                           num_workers=3)
trainset = PH2(split='train', transform=train_transform, label_transform=label_transform)
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                          num_workers=3)
testset = PH2(split='test', transform=test_transform, label_transform=label_transform)
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False,
                          num_workers=3)
# IMPORTANT NOTE: There is no validation set provided here, but don't forget to
# have one for the project

print(f"Loaded {len(trainset)} training images")
print(f"Loaded {len(testset)} test images")

# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')
#model = EncDec().to(device)
#model = UNet().to(device) # TODO
model = UNet2().to(device) # TODO
#model = DilatedNet().to(device) # TODO
summary(model, (3, 256, 256))
learning_rate = 0.001
opt = optim.Adam(model.parameters(), learning_rate)

#loss_fn = BCELoss()
#loss_fn = DiceLoss() # TODO
#loss_fn = FocalLoss() # TODO
# loss_fn = BCELoss_TotalVariation() # TODO
loss_fn = BCELoss()
epochs = 5

# Training loop
X_test, Y_test = next(iter(test_loader))
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
    print(f'* Epoch {epoch+1}/{epochs}')

    epoch_train_loss = 0.0
    for X_batch, y_true in train_loader:
        X_batch = X_batch.to(device)
        y_true = y_true.to(device)

        # set parameter gradients to zero
        opt.zero_grad()

        # forward
        y_pred = model(X_batch)
        # IMPORTANT NOTE: Check whether y_pred is normalized or unnormalized
        # and whether it makes sense to apply sigmoid or softmax.
        loss = loss_fn(y_pred, y_true)  # forward-pass
        loss.backward()  # backward-pass
        opt.step()  # update weights

        # calculate metrics to show the user
        epoch_train_loss += loss.item()

    epoch_train_loss /= max(len(train_loader), 1)
    train_losses.append(epoch_train_loss)

    # IMPORTANT NOTE: It is a good practice to check performance on a
    # validation set after each epoch.
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

    print(f' - train_loss: {epoch_train_loss:.4f} | val_loss: {epoch_val_loss:.4f}')
    for name in metric_fns:
        print(f'   {name}: {metric_history[name][-1]:.4f}')

    model.train()

# Save the model
torch.save(model.state_dict(), "/work3/s204164/DeepLearningCompVis/segmentation/saved_models/model.pth")
print("Training has finished!")

plots_dir = Path(__file__).resolve().parent / "outputs" / "training_curves"
plot_training_curves(
    {"train": train_losses, "val": val_losses},
    metric_history,
    output_dir=plots_dir,
    show=False,
)
