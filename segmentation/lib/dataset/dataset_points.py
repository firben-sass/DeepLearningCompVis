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
        sampling_strategy: str = "random",
        transform=None,
    ):
        """
        Args:
            image_ids: List of image IDs (e.g., ['IMD002', 'IMD003', ...])
            root_dir: Root directory of PH2 dataset
            split: 'train', 'val', or 'test'
            num_positive_points: Number of positive (foreground) points to sample
            num_negative_points: Number of negative (background) points to sample
            sampling_strategy: Strategy for sampling points:
                - 'random': Uniform random sampling (default)
                - 'boundary': Sample near object boundaries
                - 'center_biased': Sample more from center of object/background
                - 'mixed': Combination of random and boundary sampling
            transform: Optional transform to apply to images
        """
        self.image_ids = image_ids
        self.root_dir = Path(root_dir)
        self.split = split
        self.num_positive_points = num_positive_points
        self.num_negative_points = num_negative_points
        self.sampling_strategy = sampling_strategy
        self.transform = transform
        
        # Validate sampling strategy
        valid_strategies = ['random', 'boundary', 'center_biased', 'mixed']
        if sampling_strategy not in valid_strategies:
            raise ValueError(f"sampling_strategy must be one of {valid_strategies}")
        
    def __len__(self) -> int:
        return len(self.image_ids)
    
    def _get_boundary_mask(self, binary_mask: np.ndarray, dilation_size: int = 5) -> np.ndarray:
        """
        Get boundary region of the mask using morphological operations.
        
        Args:
            binary_mask: Binary mask (H, W) with values 0 or 1
            dilation_size: Size of dilation kernel for boundary width
            
        Returns:
            boundary_mask: Binary mask with 1s at boundary regions
        """
        from scipy import ndimage
        
        # Dilate the mask
        dilated = ndimage.binary_dilation(binary_mask, iterations=dilation_size)
        # Erode the mask
        eroded = ndimage.binary_erosion(binary_mask, iterations=dilation_size)
        
        # Boundary is the difference
        boundary = dilated.astype(np.uint8) - eroded.astype(np.uint8)
        
        return boundary
    
    def _get_distance_transform(self, binary_mask: np.ndarray) -> np.ndarray:
        """
        Compute distance transform for center-biased sampling.
        
        Args:
            binary_mask: Binary mask (H, W)
            
        Returns:
            distance_map: Distance from nearest boundary
        """
        from scipy import ndimage
        
        # Distance transform gives distance to nearest zero pixel
        distance = ndimage.distance_transform_edt(binary_mask)
        
        return distance
    
    def _sample_points_random(
        self, 
        mask: np.ndarray,
        num_points: int,
        target_value: int
    ) -> np.ndarray:
        """
        Random uniform sampling strategy.
        
        Args:
            mask: Binary mask
            num_points: Number of points to sample
            target_value: Value to sample (0 or 1)
            
        Returns:
            points: Array of shape (num_points, 2) with (x, y) coordinates
        """
        coords = np.argwhere(mask == target_value)  # (N, 2) in (y, x) format
        
        if len(coords) >= num_points:
            indices = np.random.choice(len(coords), size=num_points, replace=False)
        else:
            indices = np.random.choice(len(coords), size=num_points, replace=True)
        
        points = coords[indices][:, [1, 0]]  # Convert to (x, y)
        return points
    
    def _sample_points_boundary(
        self, 
        mask: np.ndarray,
        num_points: int,
        target_value: int
    ) -> np.ndarray:
        """
        Boundary-focused sampling strategy.
        Samples points near the boundary of the segmentation mask.
        
        This is useful because:
        - Boundaries are the hardest to segment correctly
        - Provides stronger supervision where it matters most
        - Mimics how humans would click on uncertain regions
        
        Args:
            mask: Binary mask
            num_points: Number of points to sample
            target_value: Value to sample (0 or 1)
            
        Returns:
            points: Array of shape (num_points, 2) with (x, y) coordinates
        """
        if target_value == 1:  # Positive points - sample near foreground boundary
            boundary = self._get_boundary_mask(mask, dilation_size=5)
            # Get foreground pixels near boundary
            valid_region = (mask == 1) & (boundary == 1)
            
            if valid_region.sum() < num_points // 2:
                # If not enough boundary pixels, fall back to random foreground
                valid_region = (mask == 1)
        else:  # Negative points - sample near background boundary
            boundary = self._get_boundary_mask(mask, dilation_size=5)
            # Get background pixels near boundary
            valid_region = (mask == 0) & (boundary == 1)
            
            if valid_region.sum() < num_points // 2:
                # If not enough boundary pixels, fall back to random background
                valid_region = (mask == 0)
        
        coords = np.argwhere(valid_region)  # (N, 2) in (y, x) format
        
        if len(coords) >= num_points:
            indices = np.random.choice(len(coords), size=num_points, replace=False)
        else:
            indices = np.random.choice(len(coords), size=num_points, replace=True)
        
        points = coords[indices][:, [1, 0]]  # Convert to (x, y)
        return points
    
    def _sample_points_center_biased(
        self, 
        mask: np.ndarray,
        num_points: int,
        target_value: int
    ) -> np.ndarray:
        """
        Center-biased sampling strategy.
        Samples more points from the center of regions (weighted by distance from boundary).
        
        This is useful because:
        - Center points are less ambiguous
        - Provides reliable supervision for core object regions
        - Mimics intuitive clicking behavior
        
        Args:
            mask: Binary mask
            num_points: Number of points to sample
            target_value: Value to sample (0 or 1)
            
        Returns:
            points: Array of shape (num_points, 2) with (x, y) coordinates
        """
        coords = np.argwhere(mask == target_value)  # (N, 2) in (y, x) format
        
        if len(coords) == 0:
            # Fallback to random if no valid coords
            return self._sample_points_random(mask, num_points, target_value)
        
        # Compute distance transform for weighting
        distance_map = self._get_distance_transform(mask == target_value)
        
        # Get distances at valid coordinates
        distances = distance_map[coords[:, 0], coords[:, 1]]
        
        # Weights proportional to distance (prefer center points)
        weights = distances + 1e-6  # Add small epsilon to avoid zero weights
        weights = weights / weights.sum()
        
        # Sample with probability proportional to distance from boundary
        if len(coords) >= num_points:
            indices = np.random.choice(len(coords), size=num_points, replace=False, p=weights)
        else:
            indices = np.random.choice(len(coords), size=num_points, replace=True, p=weights)
        
        points = coords[indices][:, [1, 0]]  # Convert to (x, y)
        return points
    
    def _sample_points_mixed(
        self, 
        mask: np.ndarray,
        num_points: int,
        target_value: int
    ) -> np.ndarray:
        """
        Mixed sampling strategy.
        Combines random, boundary, and center-biased sampling.
        
        This provides:
        - Diverse supervision across the entire region
        - Both easy (center) and hard (boundary) examples
        - Good generalization
        
        Distribution:
        - 50% random
        - 25% boundary
        - 25% center-biased
        
        Args:
            mask: Binary mask
            num_points: Number of points to sample
            target_value: Value to sample (0 or 1)
            
        Returns:
            points: Array of shape (num_points, 2) with (x, y) coordinates
        """
        n_random = num_points // 2
        n_boundary = num_points // 4
        n_center = num_points - n_random - n_boundary
        
        points_list = []
        
        if n_random > 0:
            points_list.append(self._sample_points_random(mask, n_random, target_value))
        if n_boundary > 0:
            points_list.append(self._sample_points_boundary(mask, n_boundary, target_value))
        if n_center > 0:
            points_list.append(self._sample_points_center_biased(mask, n_center, target_value))
        
        points = np.vstack(points_list)
        
        # Shuffle to mix the different sampling strategies
        np.random.shuffle(points)
        
        return points
    
    def _sample_points_from_mask(
        self, 
        mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample positive and negative points from a binary mask using the specified strategy.
        
        Args:
            mask: Binary mask (H, W) with values 0 or 255
            
        Returns:
            positive_points: Array of shape (num_positive_points, 2) with (x, y) coordinates
            negative_points: Array of shape (num_negative_points, 2) with (x, y) coordinates
        """
        # Convert mask to binary
        binary_mask = (mask > 127).astype(np.uint8)
        
        # Select sampling function based on strategy
        if self.sampling_strategy == 'random':
            sample_fn = self._sample_points_random
        elif self.sampling_strategy == 'boundary':
            sample_fn = self._sample_points_boundary
        elif self.sampling_strategy == 'center_biased':
            sample_fn = self._sample_points_center_biased
        elif self.sampling_strategy == 'mixed':
            sample_fn = self._sample_points_mixed
        
        # Sample positive and negative points
        positive_points = sample_fn(binary_mask, self.num_positive_points, target_value=1)
        negative_points = sample_fn(binary_mask, self.num_negative_points, target_value=0)
        
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
    sampling_strategy: str = "random",
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
        transform=transform,
        sampling_strategy= sampling_strategy
    )
    
    val_dataset = PH2Dataset(
        val_ids, root_dir, split='val',
        num_positive_points=num_positive_points,
        num_negative_points=num_negative_points,
        transform=transform,
        sampling_strategy= sampling_strategy
    )
    
    test_dataset = PH2Dataset(
        test_ids, root_dir, split='test',
        num_positive_points=num_positive_points,
        num_negative_points=num_negative_points,
        transform=transform,
        sampling_strategy= sampling_strategy
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
        sampling_strategy='random',
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
