import torch
import torch.nn as nn

class TemporalStream(nn.Module):
    def __init__(self, num_classes, num_channels=18):
        super(TemporalStream, self).__init__()
        self.convolutional = nn.Sequential(
            nn.Conv2d(num_channels, 96, 7, stride=2, padding=3),  #conv1
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), 

            nn.Conv2d(96, 256, 5, stride=2, padding=2),#conv2
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3, stride=1, padding=1),#conv3
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),#conv4
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),#conv5
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.fully_connected = nn.Sequential(
            nn.Linear(7*7*512, 4096), #full6
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 2048), #full7
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, num_classes)
        )
    def forward(self, x):
        x = self.convolutional(x)
        # Reshape
        x = x.view(x.size(0), -1)
        x = self.fully_connected(x)
        return x
