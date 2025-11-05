import torch.nn as nn
import torch


class BCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        loss = torch.mean(y_pred - y_true*y_pred + torch.log(1 + torch.exp(-y_pred)))
        return loss

class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        return 1 - (torch.mean(2*y_pred*y_true) + 1) / (torch.mean(y_pred+y_true) + 1)

class FocalLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        return -torch.sum((1-torch.sigmoid(y_pred))**2 * y_true * torch.log(torch.sigmoid(y_pred)) + \
                          (1 - y_true) * torch.log(1 - torch.sigmoid(y_pred)))

class BCELoss_TotalVariation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        loss = torch.mean(y_pred - y_true*y_pred + torch.log(1 + torch.exp(-y_pred)))
        right_shift = y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1]
        down_shift = y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]
        regularization = torch.sum(torch.abs(right_shift)) + torch.sum(torch.abs(down_shift))
        return loss + 0.1*regularization


class CrossEntropySegmentationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, y_pred, y_true):
        if y_true.dim() == y_pred.dim():
            y_true = torch.argmax(y_true, dim=1)
        return self.criterion(y_pred, y_true.long())