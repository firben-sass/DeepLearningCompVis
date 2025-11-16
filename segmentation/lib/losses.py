import torch
import torch.nn as nn
import torch.nn.functional as F


# class BCELoss(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, y_pred, y_true):
#         loss = -torch.mean(y_true * torch.log(torch.sigmoid(y_pred)) + \
#                            (1 - y_true) * torch.log(1 - torch.sigmoid(y_pred)))
#         return loss

# class DiceLoss(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, y_pred, y_true):
#         return 1 - (torch.mean(2*y_pred*y_true) + 1) / (torch.mean(y_pred+y_true) + 1)

# class FocalLoss(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, y_pred, y_true):
#         gamma = 2.0
#         return -torch.sum((1-torch.sigmoid(y_pred))**gamma * y_true * torch.log(torch.sigmoid(y_pred)) + \
#                           (1 - y_true) * torch.log(1 - torch.sigmoid(y_pred)))

# class BCELoss_TotalVariation(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, y_pred, y_true):
#         alpha = 0.1
#         loss = -torch.mean(y_true * torch.log(torch.sigmoid(y_pred)) + \
#                            (1 - y_true) * torch.log(1 - torch.sigmoid(y_pred)))
#         right_shift = y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1]
#         down_shift = y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]
#         regularization = torch.sum(torch.abs(right_shift)) + torch.sum(torch.abs(down_shift))
#         return loss + alpha*regularization

class BCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        loss = torch.mean(y_pred - y_true*y_pred + torch.log(1 + torch.exp(-y_pred)))
        return loss

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-7):
        super().__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        # Apply sigmoid to predictions to get probabilities
        y_pred = torch.sigmoid(y_pred)
        
        # Flatten tensors
        y_pred_flat = y_pred.view(-1)
        y_true_flat = y_true.view(-1)
        
        # Calculate Dice coefficient
        intersection = torch.sum(y_pred_flat * y_true_flat)
        dice_coeff = (2.0 * intersection + self.smooth) / (torch.sum(y_pred_flat) + torch.sum(y_true_flat) + self.smooth)
        
        # Return Dice loss (1 - Dice coefficient)
        return 1 - dice_coeff

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        # Apply sigmoid to get probabilities
        p = torch.sigmoid(y_pred)
        
        # Calculate binary cross entropy
        bce_loss = -(y_true * torch.log(p + 1e-7) + (1 - y_true) * torch.log(1 - p + 1e-7))
        
        # Calculate focal weight
        p_t = y_true * p + (1 - y_true) * (1 - p)
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        
        # Apply focal weight
        focal_loss = focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss

class BCELoss_TotalVariation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        loss = torch.mean(y_pred - y_true*y_pred + torch.log(1 + torch.exp(-y_pred)))

        p = torch.sigmoid(y_pred)

        # pytorch tensors are in the format (batch, channel, height, width)

        # vertical variation
        # vertical diff: p[i+1,j] - p[i,j]  for all n,c
        tv_h = torch.abs(p[:, :, :-1, :] - p[:, :, 1:, :])
        # horizontal variation
        # horizontal diff: p[i,j+1] - p[i,j]  for all
        tv_w = torch.abs(p[:, :, :, :-1] - p[:, :, :, 1:])

        regularization = torch.sum(tv_h) + torch.sum(tv_w)

        return loss + 0.1*regularization

# class CrossEntropySegmentationLoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.criterion = nn.CrossEntropyLoss()

#     def forward(self, y_pred, y_true):
#         if y_true.dim() == y_pred.dim():
#             y_true = torch.argmax(y_true, dim=1)
#         return self.criterion(y_pred, y_true.long())

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