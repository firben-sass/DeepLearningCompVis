import torch
import torch.nn as nn


class DilatedConvBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float = 0.0):
        super().__init__()
        padding = dilation
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=padding, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=padding, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)
        out = out + x
        return self.act(out)


class DilatedNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Dilated backbone keeps spatial size while expanding receptive field.
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        dilations = [2, 4, 8, 16]
        self.blocks = nn.ModuleList(
            [DilatedConvBlock(64, dilation=d, dropout=0.1 if d >= 8 else 0.0) for d in dilations]
        )

        fused_channels = 64 * (len(self.blocks) + 1)
        self.fusion = nn.Sequential(
            nn.Conv2d(fused_channels, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        features = []
        out = self.stem(x)
        features.append(out)

        for block in self.blocks:
            out = block(out)
            features.append(out)

        out = torch.cat(features, dim=1)
        out = self.fusion(out)
        out = self.head(out)
        return out
