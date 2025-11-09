# Here, you can load the predicted segmentation masks, and evaluate the
# performance metrics (accuracy, etc.)

import torch


def _flatten_masks(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reshape batched masks to (B, C, N) for metric computation."""
    if pred.shape != target.shape:
        raise ValueError("pred and target must have the same shape")

    if pred.ndim != 4:
        raise ValueError("pred and target must have shape (B, C, H, W)")

    batch, channels = pred.shape[:2]
    pred_flat = pred.float().view(batch, channels, -1)
    target_flat = target.float().view(batch, channels, -1)
    return pred_flat, target_flat
    

def dice_overlap(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> float:
    """Compute the mean Dice Overlap for batched multi-class segmentation masks.

    Args:
        pred (torch.Tensor): Predicted segmentation mask of shape (B, C, H, W).
        target (torch.Tensor): Ground truth segmentation mask with the same shape as
            ``pred``.
        eps (float): Small value to avoid division by zero.

    Returns:
        float: Mean Dice score averaged over batch and channels.
    """
    pred_flat, target_flat = _flatten_masks(pred, target)

    intersection = (pred_flat * target_flat).sum(dim=2)
    dice_scores = (2.0 * intersection + eps) / (
        pred_flat.sum(dim=2) + target_flat.sum(dim=2) + eps
    )

    return dice_scores.mean().item()


def intersection_over_union(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> float:
    """Compute the mean IoU for batched multi-class segmentation masks."""
    pred_flat, target_flat = _flatten_masks(pred, target)

    intersection = (pred_flat * target_flat).sum(dim=2)
    union = pred_flat.sum(dim=2) + target_flat.sum(dim=2) - intersection
    iou_scores = (intersection + eps) / (union + eps)
    return iou_scores.mean().item()


def accuracy(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> float:
    """Compute mean pixel accuracy for batched multi-class segmentation masks."""
    pred_flat, target_flat = _flatten_masks(pred, target)

    tp = (pred_flat * target_flat).sum(dim=2)
    tn = ((1.0 - pred_flat) * (1.0 - target_flat)).sum(dim=2)
    total = pred_flat.shape[2]
    accuracy_scores = (tp + tn + eps) / (total + eps)
    return accuracy_scores.mean().item()


def sensitivity(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> float:
    """Compute mean sensitivity (recall) for batched multi-class masks."""
    pred_flat, target_flat = _flatten_masks(pred, target)

    tp = (pred_flat * target_flat).sum(dim=2)
    fn = ((1.0 - pred_flat) * target_flat).sum(dim=2)
    sens_scores = (tp + eps) / (tp + fn + eps)
    return sens_scores.mean().item()


def specificity(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> float:
    """Compute mean specificity for batched multi-class segmentation masks."""
    pred_flat, target_flat = _flatten_masks(pred, target)

    tn = ((1.0 - pred_flat) * (1.0 - target_flat)).sum(dim=2)
    fp = (pred_flat * (1.0 - target_flat)).sum(dim=2)
    spec_scores = (tn + eps) / (tn + fp + eps)
    return spec_scores.mean().item()