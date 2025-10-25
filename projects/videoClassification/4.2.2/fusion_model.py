import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoStreamFusion(nn.Module):
    def __init__(self, optical_model, temporal_model, num_classes, fusion_type='concat'):
        super(TwoStreamFusion, self).__init__()
        self.optical_model = optical_model
        self.temporal_model = temporal_model
        self.fusion_type = fusion_type

        # Remove the final classification layers — we only need features
        self.optical_model.fully_connected = self.optical_model.fully_connected[:-1]
        self.temporal_model.fully_connected = self.temporal_model.fully_connected[:-1]

        # Fusion layer
        if fusion_type == 'concat':
            fusion_dim = 2048 * 2
        elif fusion_type == 'sum':
            fusion_dim = 2048
        else:
            raise ValueError("fusion_type must be 'concat' or 'sum'")

        # Final classification head
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, rgb_input, flow_input):
        # Extract features from both streams
        rgb_feat = self.optical_model(rgb_input)
        flow_feat = self.temporal_model(flow_input)

        if self.fusion_type == 'concat':
            fused = torch.cat((rgb_feat, flow_feat), dim=1)
        elif self.fusion_type == 'sum':
            fused = rgb_feat + flow_feat
        else:
            raise ValueError("Invalid fusion type")

        out = self.classifier(fused)
        return out
