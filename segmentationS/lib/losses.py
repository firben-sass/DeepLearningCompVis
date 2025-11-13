import torch
import torch.nn as nn

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

