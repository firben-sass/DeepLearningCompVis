import torch
import torch.nn as nn
import torch.nn.functional as F

class DilatedNet(nn.Module):
    """A simple dilated convolutional network for image segmentation"""
    
    def __init__(self, n_channels=3, n_classes=1):
        super(DilatedNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        # Initial convolution
        self.conv1 = nn.Conv2d(n_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Dilated convolutions with different dilation rates
        self.dconv1 = nn.Conv2d(64, 128, kernel_size=3, padding=2, dilation=2)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.dconv2 = nn.Conv2d(128, 256, kernel_size=3, padding=4, dilation=4)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.dconv3 = nn.Conv2d(256, 256, kernel_size=3, padding=8, dilation=8)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Decoder
        self.up_conv1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        
        self.up_conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(64)
        
        # Final output layer
        self.out_conv = nn.Conv2d(64, n_classes, kernel_size=1)
        
    def forward(self, x):
        # Encoder with dilated convolutions
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.dconv1(x)))
        x = F.relu(self.bn3(self.dconv2(x)))
        x = F.relu(self.bn4(self.dconv3(x)))
        
        # Decoder
        x = F.relu(self.bn5(self.up_conv1(x)))
        x = F.relu(self.bn6(self.up_conv2(x)))
        
        # Output
        x = self.out_conv(x)
        
        return x
