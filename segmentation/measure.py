# Here, you can load the predicted segmentation masks, and evaluate the
# performance metrics (accuracy, etc.)

import torch

def dice_overlap(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    """
    Compute the Dice Overlap between predicted and target segmentation masks.
    Args:
        pred (torch.Tensor): Predicted segmentation mask of shape (H, W).
        target (torch.Tensor): Ground truth segmentation mask of shape (H, W).
        eps (float): Small value to avoid division by zero.
    Returns:
        float: Dice Overlap score.
    """
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)

    intersection = (pred_flat * target_flat).sum()
    dice_score = (2.0 * intersection + eps) / (pred_flat.sum() + target_flat.sum() + eps)

    return dice_score.item()

def 