import os
import torch
import torch.nn as nn
from torchvision import models


class ProposalClassifier(nn.Module):
    """
    Remember, N+1 classes!! (N objects + background) for us it is 1+1=2 classes
    """
    def __init__(self, num_classes, pretrained=True):
        super(ProposalClassifier, self).__init__()
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


def load_proposal_classifier(weights_path, num_classes=2, device=None, pretrained=False, strict=True):
    """Instantiate ProposalClassifier and load saved weights from ``weights_path``."""
    if not weights_path or not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Cannot load ProposalClassifier weights, file not found: {weights_path}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    model = ProposalClassifier(num_classes=num_classes, pretrained=pretrained).to(device)

    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=strict)
    return model
