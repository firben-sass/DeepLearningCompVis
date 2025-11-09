import os
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import random


class PH2Dataset(Dataset):
    """PH2 Dataset loader with support for point-click annotations."""
    
    def __init__(
        self,
        image_ids: List[str],
        root_dir: str = "/dtu/datasets1/02516/PH2_Dataset_images/",
        split: str = "train",
        num_positive_points: int = 5,
        num_negative_points: int = 5,
        transform=None,
    ):
        """
        Args:
            image_ids: List of image IDs (e.g., ['IMD002', 'IMD003', ...])
            root_dir: Root directory of PH2 dataset
            split: 'train', 'val', or 'test'
            num_positive_points: Number of positive (foreground) points to sample
            num_negative_points: Number of negative (background) points to sample
            transform: Optional transform to apply to images
        """
        self.image_ids = image_ids
        self.root_dir = Path(root_dir)
        self.split = split
        self.num_positive_points = num_positive_points
        self.num_negative_points = num_negative_points
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.image_ids)
    
    def _sample_points_from_mask(
        self, 
        mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample positive and negative points from a binary mask.
        
        Args:
            mask: Binary mask (H, W) with values 0 or 255
            
        Returns:
            positive_points: Array of shape (num_positive_points, 2) with (x, y) coordinates
            negative_points: Array of shape (num_negative_points, 2) with (x, y) coordinates
        """
        # Convert mask to binary
        binary_mask = (mask > 127).astype(np.uint8)
        
        # Get coordinates of positive and negative pixels
        positive_coords = np.argwhere(binary_mask == 1)  # (y, x) format
        negative_coords = np.argwhere(binary_mask == 0)
        
        # Sample points
        if len(positive_coords) >= self.num_positive_points:
            pos_indices = np.random.choice(
                len(positive_coords), 
                size=self.num_positive_points, 
                replace=False
            )
            positive_points = positive_coords[pos_indices]
        else:
            # If not enough positive pixels, sample with replacement
            pos_indices = np.random.choice(
                len(positive_coords), 
                size=self.num_positive_points, 
                replace=True
            )
            positive_points = positive_coords[pos_indices]
        
        if len(negative_coords) >= self.num_negative_points:
            neg_indices = np.random.choice(
                len(negative_coords), 
                size=self.num_negative_points, 
                replace=False
            )
            negative_points = negative_coords[neg_indices]
        else:
            # If not enough negative pixels, sample with replacement
            neg_indices = np.random.choice(
                len(negative_coords), 
                size=self.num_negative_points, 
                replace=True
            )
            negative_points = negative_coords[neg_indices]
        
        # Convert from (y, x) to (x, y) format
        positive_points = positive_points[:, [1, 0]]
        negative_points = negative_points[:, [1, 0]]
        
        return positive_points, negative_points
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a sample from the dataset.
        
        Returns:
            Dictionary containing:
                - 'image': Tensor of shape (C, H, W), normalized to [0, 1]
                - 'mask': Tensor of shape (H, W), binary values {0, 1}
                - 'positive_points': Tensor of shape (N_pos, 2) with (x, y) coordinates
                - 'negative_points': Tensor of shape (N_neg, 2) with (x, y) coordinates
                - 'image_id': String identifier
        """
        image_id = self.image_ids[idx]
        
        # Construct paths
        image_dir = self.root_dir / image_id / f"{image_id}_Dermoscopic_Image"
        mask_dir = self.root_dir / image_id / f"{image_id}_lesion"
        
        image_path = image_dir / f"{image_id}.bmp"
        mask_path = mask_dir / f"{image_id}_lesion.bmp"
        
        # Load image and mask
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # Grayscale

        #target_size = (576, 768)
        target_size = (128, 128)
        image = image.resize(target_size[::-1], Image.BILINEAR)
        mask = mask.resize(target_size[::-1], Image.NEAREST)

        
        # Convert to numpy for processing
        image_np = np.array(image)
        mask_np = np.array(mask)
        
        # Sample point clicks for all splits
        positive_points, negative_points = self._sample_points_from_mask(mask_np)
        
        # Apply transforms if provided (to PIL images)
        if self.transform:
            image = self.transform(image)
            # If transform returns tensor, convert back to numpy for consistency
            if isinstance(image, torch.Tensor):
                image_np = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            else:
                image_np = np.array(image)
        
        # Convert to tensors
        # Image: (H, W, C) -> (C, H, W), normalize to [0, 1]
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
        
        # Mask: (H, W), binarize to {0, 1}
        mask_tensor = torch.from_numpy((mask_np > 127).astype(np.float32))
        
        # Points: already numpy arrays, convert to tensors
        positive_points_tensor = torch.from_numpy(positive_points).float()
        negative_points_tensor = torch.from_numpy(negative_points).float()
        
        # Prepare output dictionary
        sample = {
            'image_id': image_id,
            'image': image_tensor,
            'mask': mask_tensor,
            'positive_points': positive_points_tensor,
            'negative_points': negative_points_tensor,
        }
        
        return sample


class PointClickLoss(nn.Module):
    """
    Loss function for point-click based segmentation.
    
    This loss samples the predicted segmentation mask at the locations
    specified by positive and negative point clicks, and computes
    binary cross-entropy loss to ensure the model predicts high probability
    at positive points and low probability at negative points.
    """
    
    def __init__(self, reduction='mean'):
        """
        Args:
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.reduction = reduction
        
    def forward(
        self, 
        pred_mask: torch.Tensor,
        positive_points: torch.Tensor,
        negative_points: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute point-click loss.
        
        Args:
            pred_mask: Predicted segmentation mask of shape (B, 1, H, W) or (B, H, W)
                      Values should be logits (pre-sigmoid) or probabilities
            positive_points: Positive point coordinates of shape (B, N_pos, 2)
                           Format: (x, y) coordinates
            negative_points: Negative point coordinates of shape (B, N_neg, 2)
                           Format: (x, y) coordinates
                           
        Returns:
            Loss value
        """
        # Ensure pred_mask is (B, 1, H, W)
        if pred_mask.dim() == 3:
            pred_mask = pred_mask.unsqueeze(1)
        
        batch_size = pred_mask.shape[0]
        device = pred_mask.device
        
        # Get mask dimensions
        _, _, H, W = pred_mask.shape
        
        # Sample predictions at positive points
        positive_losses = []
        for b in range(batch_size):
            pos_pts = positive_points[b]  # (N_pos, 2)
            
            # Normalize coordinates to [-1, 1] for grid_sample
            normalized_pts = pos_pts.clone().float()
            normalized_pts[:, 0] = 2.0 * pos_pts[:, 0] / (W - 1) - 1.0  # x
            normalized_pts[:, 1] = 2.0 * pos_pts[:, 1] / (H - 1) - 1.0  # y
            
            # Reshape for grid_sample: (1, N_pos, 1, 2)
            grid = normalized_pts.unsqueeze(0).unsqueeze(2)
            
            # Sample from prediction mask
            sampled = F.grid_sample(
                pred_mask[b:b+1], 
                grid, 
                mode='bilinear', 
                align_corners=True
            )  # (1, 1, N_pos, 1)
            
            sampled = sampled.squeeze()  # (N_pos,)
            
            # Compute BCE loss (target = 1 for positive points)
            pos_loss = F.binary_cross_entropy_with_logits(
                sampled, 
                torch.ones_like(sampled),
                reduction='none'
            )
            positive_losses.append(pos_loss.mean())
        
        # Sample predictions at negative points
        negative_losses = []
        for b in range(batch_size):
            neg_pts = negative_points[b]  # (N_neg, 2)
            
            # Normalize coordinates to [-1, 1] for grid_sample
            normalized_pts = neg_pts.clone().float()
            normalized_pts[:, 0] = 2.0 * neg_pts[:, 0] / (W - 1) - 1.0  # x
            normalized_pts[:, 1] = 2.0 * neg_pts[:, 1] / (H - 1) - 1.0  # y
            
            # Reshape for grid_sample: (1, N_neg, 1, 2)
            grid = normalized_pts.unsqueeze(0).unsqueeze(2)
            
            # Sample from prediction mask
            sampled = F.grid_sample(
                pred_mask[b:b+1], 
                grid, 
                mode='bilinear', 
                align_corners=True
            )  # (1, 1, N_neg, 1)
            
            sampled = sampled.squeeze()  # (N_neg,)
            
            # Compute BCE loss (target = 0 for negative points)
            neg_loss = F.binary_cross_entropy_with_logits(
                sampled, 
                torch.zeros_like(sampled),
                reduction='none'
            )
            negative_losses.append(neg_loss.mean())
        
        # Combine losses
        positive_loss = torch.stack(positive_losses)
        negative_loss = torch.stack(negative_losses)
        
        total_loss = (positive_loss + negative_loss) / 2.0
        
        if self.reduction == 'mean':
            return total_loss.mean()
        elif self.reduction == 'sum':
            return total_loss.sum()
        else:
            return total_loss


def create_ph2_splits(
    root_dir: str = "/dtu/datasets1/02516/PH2_Dataset_images/",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Create train/val/test splits for PH2 dataset.
    
    Args:
        root_dir: Root directory of PH2 dataset
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        test_ratio: Proportion of data for testing
        random_seed: Random seed for reproducibility
        
    Returns:
        train_ids, val_ids, test_ids: Lists of image IDs for each split
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"
    
    # Get all image IDs
    root_path = Path(root_dir)
    image_ids = sorted([
        d.name for d in root_path.iterdir() 
        if d.is_dir() and d.name.startswith('IMD')
    ])
    
    print(f"Found {len(image_ids)} images in dataset")
    
    # Set random seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # First split: train vs (val + test)
    train_ids, temp_ids = train_test_split(
        image_ids,
        train_size=train_ratio,
        random_state=random_seed,
        shuffle=True
    )
    
    # Second split: val vs test
    val_size = val_ratio / (val_ratio + test_ratio)
    val_ids, test_ids = train_test_split(
        temp_ids,
        train_size=val_size,
        random_state=random_seed,
        shuffle=True
    )
    
    print(f"Split sizes - Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")
    
    return train_ids, val_ids, test_ids


def create_dataloaders(
    root_dir: str = "/dtu/datasets1/02516/PH2_Dataset_images/",
    batch_size: int = 8,
    num_positive_points: int = 5,
    num_negative_points: int = 5,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    num_workers: int = 4,
    transform=None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for train, validation, and test sets.
    
    Args:
        root_dir: Root directory of PH2 dataset
        batch_size: Batch size for DataLoaders
        num_positive_points: Number of positive points per sample
        num_negative_points: Number of negative points per sample
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        test_ratio: Proportion of data for testing
        random_seed: Random seed for reproducibility
        num_workers: Number of worker processes for data loading
        transform: Optional transforms to apply
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # Create splits
    train_ids, val_ids, test_ids = create_ph2_splits(
        root_dir, train_ratio, val_ratio, test_ratio, random_seed
    )
    
    # Create datasets
    train_dataset = PH2Dataset(
        train_ids, root_dir, split='train',
        num_positive_points=num_positive_points,
        num_negative_points=num_negative_points,
        transform=transform
    )
    
    val_dataset = PH2Dataset(
        val_ids, root_dir, split='val',
        num_positive_points=num_positive_points,
        num_negative_points=num_negative_points,
        transform=transform
    )
    
    test_dataset = PH2Dataset(
        test_ids, root_dir, split='test',
        num_positive_points=num_positive_points,
        num_negative_points=num_negative_points,
        transform=transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    
    size = 128

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=4,
        num_positive_points=5,
        num_negative_points=5,
        random_seed=42,
        # transform= transforms.Compose([transforms.Resize((size, size)),
        #                             transforms.ToTensor(),
        #                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    )
    
    # Test data loading
    print("\n=== Training Data Sample ===")
    train_batch = next(iter(train_loader))
    print(f"Image IDs: {train_batch['image_id']}")
    print(f"Positive points shape: {train_batch['positive_points'].shape}")
    print(f"Negative points shape: {train_batch['negative_points'].shape}")
    print(f"Mask type: {type(train_batch['mask'])}")
    
    # Print actual point coordinates for first sample
    print("\n=== Point Coordinates for First Sample ===")
    print(f"Image ID: {train_batch['image_id'][0]}")
    print(f"\nPositive points (x, y):")
    for i, pt in enumerate(train_batch['positive_points'][0]):
        print(f"  Point {i+1}: ({pt[0]:.1f}, {pt[1]:.1f})")
    
    print(f"\nNegative points (x, y):")
    for i, pt in enumerate(train_batch['negative_points'][0]):
        print(f"  Point {i+1}: ({pt[0]:.1f}, {pt[1]:.1f})")
    
    # Visualize the points on the image and mask
    print("\n=== Creating Visualization ===")
    
    # Get first sample - convert from tensor back to numpy for visualization
    img = (train_batch['image'][0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    mask = train_batch['mask'][0].numpy()
    pos_pts = train_batch['positive_points'][0].numpy()
    neg_pts = train_batch['negative_points'][0].numpy()
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot original image
    axes[0].imshow(img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Plot mask
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Ground Truth Mask')
    axes[1].axis('off')
    
    # Plot image with point annotations
    axes[2].imshow(img)
    axes[2].scatter(pos_pts[:, 0], pos_pts[:, 1], c='green', s=100, 
                    marker='o', edgecolors='white', linewidths=2, label='Positive')
    axes[2].scatter(neg_pts[:, 0], neg_pts[:, 1], c='red', s=100, 
                    marker='x', linewidths=3, label='Negative')
    axes[2].set_title('Point Click Labels')
    axes[2].legend(loc='upper right')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('point_labels_visualization.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to 'point_labels_visualization.png'")
    
    # Print statistics about mask coverage
    print("\n=== Mask Statistics ===")
    binary_mask = mask.astype(np.uint8)
    total_pixels = binary_mask.size
    foreground_pixels = np.sum(binary_mask)
    background_pixels = total_pixels - foreground_pixels
    
    print(f"Image shape: {img.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Total pixels: {total_pixels:,}")
    print(f"Foreground pixels: {foreground_pixels:,} ({100*foreground_pixels/total_pixels:.2f}%)")
    print(f"Background pixels: {background_pixels:,} ({100*background_pixels/total_pixels:.2f}%)")
    
    # Verify points are correctly placed
    print("\n=== Verifying Point Placement ===")
    print("Checking if positive points are on foreground (mask value = 1):")
    for i, pt in enumerate(pos_pts):
        x, y = int(pt[0]), int(pt[1])
        mask_value = mask[y, x]  # Note: mask indexing is [y, x]
        print(f"  Positive point {i+1}: mask[{y}, {x}] = {mask_value} {'✓' if mask_value == 1 else '✗'}")
    
    print("\nChecking if negative points are on background (mask value = 0):")
    for i, pt in enumerate(neg_pts):
        x, y = int(pt[0]), int(pt[1])
        mask_value = mask[y, x]
        print(f"  Negative point {i+1}: mask[{y}, {x}] = {mask_value} {'✓' if mask_value == 0 else '✗'}")
    
    # Test loss functions
    print("\n=== Testing Loss Functions ===")
    
    # Create dummy predictions (logits)
    B, H, W = 4, 256, 256
    pred_mask = torch.randn(B, 1, H, W)  # Random logits
    
    # Convert loaded points to tensors
    pos_points = train_batch['positive_points'].float()  # (B, N_pos, 2)
    neg_points = train_batch['negative_points'].float()  # (B, N_neg, 2)
    
    # Test point-click loss only
    point_loss_fn = PointClickLoss()
    loss = point_loss_fn(pred_mask, pos_points, neg_points)
    print(f"Point-click loss: {loss.item():.4f}")
