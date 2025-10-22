import torch
import torch.nn as nn

class EarlyFusion(nn.Module):
        # Early fusion merges time steps (10 frames * 3 RGB channels = 30 input channels)
    def __init__(self, num_classes=10, in_channels=30, p_dropout=0.25):
        super(EarlyFusion, self).__init__()
        # Shared 2D CNN Backbone for individual frame feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # Downsample (H/2, W/2)
            nn.Dropout2d(p_dropout),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # Downsample (H/4, W/4)
            nn.Dropout2d(p_dropout),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # Downsample (H/8, W/8)
            nn.Dropout2d(p_dropout),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  # Output shape: (512, 1, 1)
        )

        # Fully Connected Layer for aggregated features
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        """
        Forward pass for EarlyFusion.
        :param x: Tensor of shape (batch_size, C, T, H, W)
                  where C = number of channels, T = number of frames,
                  H = height, W = width
        :return: Logits of shape (batch_size, num_classes)
        """
        # Combine temporal and channel dimensions for early fusion: (C, T, H, W) -> (C*T, H, W)
        if len(x.size()) == 5:
            batch_size, C, T, H, W = x.size()
            x = x.view(batch_size, C * T, H, W)  # Early fusion by stacking channels
        else:
            batch_size, C,  H, W = x.size()
            x = x.view(batch_size, C, H, W)  # Early fusion by stacking channels


        # Pass through the backbone
        x = self.feature_extractor(x)  # Shape: (batch_size, 256, 1, 1)

        # Pass through the fully connected layers
        x = self.fc(x)  # Shape: (batch_size, num_classes)

        return x