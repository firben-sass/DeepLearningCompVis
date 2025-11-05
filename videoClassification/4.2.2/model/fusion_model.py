
import torch
import torch.nn as nn

class TwoStreamFusion(nn.Module):
    """
    Minimal logits-level fusion of two precomputed logits tensors.
    Provides a learnable scalar weight alpha in [0,1].
    """
    def __init__(self, num_classes: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.num_classes = num_classes

    def forward(self, logits_rgb: torch.Tensor, logits_flow: torch.Tensor) -> torch.Tensor:
        a = torch.clamp(self.alpha, 0.0, 1.0)
        return a * logits_rgb + (1.0 - a) * logits_flow
