import os
import numpy as np
import glob
import argparse
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
from lib.model.UNetModel import UNet2
from lib.losses import PointClickLoss
from lib.dataset.dataset_points import PH2Dataset, create_dataloaders, create_ph2_splits
from lib.plotting import plot_training_curves
from measure import (
    accuracy,
    dice_overlap,
    intersection_over_union,
    sensitivity,
    specificity,
)

# ============================================================================
# Parse Command Line Arguments
# ============================================================================
parser = argparse.ArgumentParser(description='Train UNet model with point-click supervision')
parser.add_argument('--run_name', type=str, required=True,
                    help='Name for this training run (creates output folder)')
parser.add_argument('--learning_rate', '--lr', type=float, default=0.001,
                    help='Learning rate (default: 0.001)')
parser.add_argument('--batch_size', '--bs', type=int, default=6,
                    help='Batch size (default: 6)')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs (default: 100)')
parser.add_argument('--image_size', '--size', type=int, default=256,
                    help='Image size (default: 256)')
parser.add_argument('--num_positive_points', type=int, default=5,
                    help='Number of positive points (default: 5)')
parser.add_argument('--num_negative_points', type=int, default=5,
                    help='Number of negative points (default: 5)')
parser.add_argument('--sampling_strategy', type=str, default='random',
                    help='Sampling Strategy: random, boundary, center_biased, mixed (default: random)')
parser.add_argument('--random_seed', type=int, default=42,
                    help='Random seed (default: 42)')

args = parser.parse_args()

# ============================================================================
# Create Output Directory
# ============================================================================
output_dir = Path(__file__).resolve().parent / "outputs" / args.run_name
output_dir.mkdir(parents=True, exist_ok=True)
print(f"\n{'='*80}")
print(f"RUN NAME: {args.run_name}")
print(f"OUTPUT DIRECTORY: {output_dir}")
print(f"{'='*80}\n")

# Save configuration
config_file = output_dir / "config.txt"
with open(config_file, 'w') as f:
    f.write(f"Training Configuration\n")
    f.write(f"{'='*50}\n")
    f.write(f"Run Name:           {args.run_name}\n")
    f.write(f"Learning Rate:      {args.learning_rate}\n")
    f.write(f"Batch Size:         {args.batch_size}\n")
    f.write(f"Epochs:             {args.epochs}\n")
    f.write(f"Image Size:         {args.image_size}\n")
    f.write(f"Positive Points:    {args.num_positive_points}\n")
    f.write(f"Negative Points:    {args.num_negative_points}\n")
    f.write(f"Sampling Strategy:    {args.sampling_strategy}\n")
    f.write(f"Random Seed:        {args.random_seed}\n")
    f.write(f"{'='*50}\n")
print(f"Configuration saved to: {config_file}\n")

# ============================================================================
# Create Dataloaders
# ============================================================================
train_loader, val_loader, test_loader = create_dataloaders(
    batch_size=args.batch_size,
    num_positive_points=args.num_positive_points,
    num_negative_points=args.num_negative_points,
    random_seed=args.random_seed,
    sampling_strategy=args.sampling_strategy,
    image_size= args.image_size,
)

# ============================================================================
# Training Setup
# ============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')
model = UNet2().to(device)
summary(model, (3, args.image_size, args.image_size))

opt = optim.Adam(model.parameters(), args.learning_rate)
loss_fn = PointClickLoss()

# Metric functions
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

# ============================================================================
# Training Loop
# ============================================================================
print(f"\n{'='*80}")
print(f"STARTING TRAINING")
print(f"{'='*80}\n")

model.train()  # train mode
for epoch in range(args.epochs):
    tic = time()
    print(f'* Epoch {epoch+1}/{args.epochs}')

    epoch_train_loss = 0.0
    for batch in train_loader:
        # Extract data from dictionary
        X_batch = batch['image'].to(device)
        pos_points = batch['positive_points'].to(device)
        neg_points = batch['negative_points'].to(device)

        # set parameter gradients to zero
        opt.zero_grad()

        # forward
        y_pred = model(X_batch)  # Shape: (B, 1, H, W)
        
        # Use point-click loss
        loss = loss_fn(y_pred, pos_points, neg_points)
        loss.backward()  # backward-pass
        opt.step()  # update weights

        # calculate metrics to show the user
        epoch_train_loss += loss.item()

    epoch_train_loss /= max(len(train_loader), 1)
    train_losses.append(epoch_train_loss)

    # Validation
    model.eval()

    epoch_val_loss = 0.0
    metrics_sums = {name: 0.0 for name in metric_fns}
    
    with torch.no_grad():
        for batch in val_loader:
            X_val = batch['image'].to(device)
            y_val = batch['mask'].to(device)  # Ground truth mask
            pos_points = batch['positive_points'].to(device)
            neg_points = batch['negative_points'].to(device)

            logits = model(X_val)  # Shape: (B, 1, H, W)
            
            # Calculate point-click loss
            val_loss = loss_fn(logits, pos_points, neg_points)
            epoch_val_loss += val_loss.item()

            # Calculate metrics using full mask
            # Add channel dimension to y_val if needed
            if y_val.dim() == 3:
                y_val = y_val.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)
            
            predictions = (torch.sigmoid(logits) > 0.5).float()
            
            for name, fn in metric_fns.items():
                metrics_sums[name] += fn(predictions, y_val)

    num_val_batches = max(len(val_loader), 1)
    epoch_val_loss /= num_val_batches
    val_losses.append(epoch_val_loss)

    for name in metric_fns:
        metric_value = metrics_sums[name] / num_val_batches
        metric_history[name].append(metric_value)

    toc = time()
    print(f' - train_loss: {epoch_train_loss:.4f} | val_loss: {epoch_val_loss:.4f} | time: {toc-tic:.2f}s')
    for name in metric_fns:
        print(f'   {name}: {metric_history[name][-1]:.4f}')

    model.train()

# ============================================================================
# Save Model and Training Curves
# ============================================================================
model_path = output_dir / "model.pth"
torch.save(model.state_dict(), model_path)
print(f"\n{'='*80}")
print(f"Training has finished!")
print(f"Model saved to: {model_path}")
print(f"{'='*80}\n")

plot_training_curves(
    {"train": train_losses, "val": val_losses},
    metric_history,
    output_dir=output_dir,
    show=False,
)
print(f"Training curves saved to: {output_dir}")

# ============================================================================
# Test Set Evaluation
# ============================================================================
print("\n" + "="*80)
print("EVALUATING ON TEST SET")
print("="*80)

model.eval()
test_metrics_sums = {name: 0.0 for name in metric_fns}
all_predictions = []
all_masks = []
all_images = []
all_ids = []

with torch.no_grad():
    for batch in test_loader:
        X_test = batch['image'].to(device)
        y_test = batch['mask'].to(device)
        image_ids = batch['image_id']
        
        # Get predictions
        logits = model(X_test)
        predictions = (torch.sigmoid(logits) > 0.5).float()
        
        # Add channel dimension to y_test if needed
        if y_test.dim() == 3:
            y_test = y_test.unsqueeze(1)
        
        # Calculate metrics
        for name, fn in metric_fns.items():
            test_metrics_sums[name] += fn(predictions, y_test)
        
        # Store for visualization
        all_predictions.append(predictions.cpu())
        all_masks.append(y_test.cpu())
        all_images.append(X_test.cpu())
        all_ids.extend(image_ids)

# Calculate average metrics
num_test_batches = max(len(test_loader), 1)
test_metrics = {name: test_metrics_sums[name] / num_test_batches 
                for name in metric_fns}

print("\nTest Set Metrics:")
print("-" * 40)
for name, value in test_metrics.items():
    print(f"{name:15s}: {value:.4f}")

# ============================================================================
# Calculate Overall Test Metrics
# ============================================================================
import matplotlib.pyplot as plt

print("\n" + "="*80)
print("OVERALL TEST METRICS")
print("="*80)

# Concatenate all batches
all_predictions = torch.cat(all_predictions, dim=0)
all_masks = torch.cat(all_masks, dim=0)
all_images = torch.cat(all_images, dim=0)

# Calculate overall metrics on all predictions
overall_metrics = {}
for name, fn in metric_fns.items():
    metric_value = fn(all_predictions, all_masks)
    # Handle both tensor and float returns
    if isinstance(metric_value, torch.Tensor):
        overall_metrics[name] = metric_value.item()
    else:
        overall_metrics[name] = metric_value

# Print overall metrics
print(f"\nOverall Metrics (on {len(all_predictions)} images):")
print(f"  Dice:        {overall_metrics['Dice']:.4f}")
print(f"  IoU:         {overall_metrics['IoU']:.4f}")
print(f"  Accuracy:    {overall_metrics['Accuracy']:.4f}")
print(f"  Sensitivity: {overall_metrics['Sensitivity']:.4f}")
print(f"  Specificity: {overall_metrics['Specificity']:.4f}")

# Save metrics to file
metrics_file = output_dir / "test_metrics.txt"
with open(metrics_file, 'w') as f:
    f.write(f"Test Set Metrics\n")
    f.write(f"{'='*50}\n")
    f.write(f"Number of images: {len(all_predictions)}\n\n")
    f.write(f"Dice:        {overall_metrics['Dice']:.4f}\n")
    f.write(f"IoU:         {overall_metrics['IoU']:.4f}\n")
    f.write(f"Accuracy:    {overall_metrics['Accuracy']:.4f}\n")
    f.write(f"Sensitivity: {overall_metrics['Sensitivity']:.4f}\n")
    f.write(f"Specificity: {overall_metrics['Specificity']:.4f}\n")
    f.write(f"{'='*50}\n")
print(f"\nTest metrics saved to: {metrics_file}")

# ============================================================================
# Visualize Example Predictions
# ============================================================================
print("\n" + "="*80)
print("VISUALIZING EXAMPLE PREDICTIONS")
print("="*80)

# Select 4 random examples
num_examples = min(4, len(all_predictions))
indices = np.random.choice(len(all_predictions), size=num_examples, replace=False)

fig, axes = plt.subplots(num_examples, 4, figsize=(16, 4*num_examples))
if num_examples == 1:
    axes = axes.reshape(1, -1)

for i, idx in enumerate(indices):
    # Get data
    image = all_images[idx].permute(1, 2, 0).numpy()  # (C, H, W) -> (H, W, C)
    mask = all_masks[idx].squeeze().numpy()  # (1, H, W) -> (H, W)
    pred = all_predictions[idx].squeeze().numpy()  # (1, H, W) -> (H, W)
    image_id = all_ids[idx]
    
    # Calculate metrics for this specific image
    pred_tensor = all_predictions[idx:idx+1]
    mask_tensor = all_masks[idx:idx+1]
    
    img_metrics = {}
    for name, fn in metric_fns.items():
        metric_value = fn(pred_tensor, mask_tensor)
        # Handle both tensor and float returns
        if isinstance(metric_value, torch.Tensor):
            img_metrics[name] = metric_value.item()
        else:
            img_metrics[name] = metric_value
    
    # Plot original image
    axes[i, 0].imshow(image)
    axes[i, 0].set_title(f'Image: {image_id}')
    axes[i, 0].axis('off')
    
    # Plot ground truth mask
    axes[i, 1].imshow(mask, cmap='gray')
    axes[i, 1].set_title('Ground Truth')
    axes[i, 1].axis('off')
    
    # Plot prediction
    axes[i, 2].imshow(pred, cmap='gray')
    axes[i, 2].set_title('Prediction')
    axes[i, 2].axis('off')
    
    # Plot overlay
    axes[i, 3].imshow(image)
    # Show ground truth in green, prediction in red, overlap in yellow
    overlay = np.zeros((*mask.shape, 4))
    overlay[mask == 1] = [0, 1, 0, 0.4]  # Green for ground truth
    overlay[pred == 1] = [1, 0, 0, 0.4]  # Red for prediction
    overlap = (mask == 1) & (pred == 1)
    overlay[overlap] = [1, 1, 0, 0.5]  # Yellow for overlap
    axes[i, 3].imshow(overlay)
    axes[i, 3].set_title(f"Overlay\nDice: {img_metrics['Dice']:.3f} | IoU: {img_metrics['IoU']:.3f}")
    axes[i, 3].axis('off')

plt.tight_layout()
output_path = output_dir / "test_predictions.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nVisualization saved to: {output_path}")
plt.close()

print("\n" + "="*80)
print("EVALUATION COMPLETE")
print(f"All outputs saved to: {output_dir}")
print("="*80)