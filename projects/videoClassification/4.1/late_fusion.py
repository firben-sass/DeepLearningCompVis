import torch
import torch.nn as nn

class LateFusion(nn.Module):
    def __init__(self, num_classes=10, in_channels=3, p_dropout=0.25):
        super(LateFusion, self).__init__()
        # Shared 2D CNN Backbone for individual frame feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # Downsample (H/2, W/2)
            nn.Dropout2d(p_dropout),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # Downsample (H/4, W/4)
            nn.Dropout2d(p_dropout),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  # Output shape: (256, 1, 1)
        )

        # Fully Connected Layer for aggregated features
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        """
        Forward pass for LateFusion.
        :param x: Tensor of shape (batch_size, T, C, H, W)
                  where T = number of frames, C = channels, H = height, W = width.
        :return: Logits of shape (batch_size, num_classes)
        """
        batch_size, C, T, H, W = x.size()

        # Extract features for each frame independently
        frame_features = []
        for t in range(T):
            frame = x[:, :, t]  # Shape: (batch_size, C, H, W)
            features = self.feature_extractor(frame)  # Shape: (batch_size, 256, 1, 1)
            frame_features.append(features.squeeze(-1).squeeze(-1))  # Shape: (batch_size, 256)

        # Stack frame features and aggregate (e.g., average pooling)
        frame_features = torch.stack(frame_features, dim=1)  # Shape: (batch_size, T, 256)
        video_features = frame_features.mean(dim=1)  # Temporal average pooling: Shape: (batch_size, 256)

        # Classification
        logits = self.fc(video_features)  # Shape: (batch_size, num_classes)
        return logits