from temporal_model import TemporalStream
from optical_model import OpticalStream
import torch.nn as nn

class DualStreamNetwork(nn.Module):
    def __init__(self, num_classes):
        super(DualStreamNetwork, self).__init__()
        self.optical = OpticalStream(num_classes)
        self.motion = TemporalStream(num_classes)

    def forward(self, rgb, flow):
        rgb_out = self.optical(rgb)  # (B, num_classes)
        flow_out = self.motion(flow) # (B, num_classes)

        # Average predictions
        out = (rgb_out + flow_out) / 2
        return out
