
import torch.nn as nn
from torchvision import models

class ProposalClassifier(nn.Module):
    """
    Remember, N+1 classes!! (N objects + background) for us it is 1+1=2 classes
    """
    def _init_(self, num_classes, pretrained=True):
        super(ProposalClassifier, self)._init_()
        mobilenet = models.mobilenet_v3_small(pretrained=pretrained)

        # Remove the final classification layer
        self.features = mobilenet.features
        self.avgpool = mobilenet.avgpool

        # Add custom classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(576, 256),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x
