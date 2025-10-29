import torch
import torch.nn as nn
from torchvision.models import resnet18

class OpticalStream(nn.Module):
    def __init__(self, num_classes, p_dropout=0.25):
        super(OpticalStream, self).__init__()
        self.convolutional = nn.Sequential(
            nn.Conv2d(3, 96, 7, stride=2, padding=3),  #conv1
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),

            nn.Conv2d(96, 256, 5, stride=2, padding=2),#conv2
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),

            nn.Conv2d(256, 512, 3, stride=1, padding=1),#conv3
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.25),

            nn.Conv2d(512, 512, 3, stride=1, padding=1),#conv4
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.25),

            nn.Conv2d(512, 512, 3, stride=1, padding=1),#conv5
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),

            
        )
        # Replace giant FC layers with Global Average Pooling + compact FC head
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fully_connected = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.convolutional(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)         # Flatten to [batch, 512]
        x = self.fully_connected(x)
        return x


