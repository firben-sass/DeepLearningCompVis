import torch.nn as nn
import torchvision.models as models

class OpticalStream(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        base = models.resnet18(pretrained=pretrained)
        self.features = nn.Sequential(*list(base.children())[:-1])  # remove classifier
        self.fc = nn.Linear(base.fc.in_features, 256)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
