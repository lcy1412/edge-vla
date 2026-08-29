"""
Lightweight Vision Encoder for EdgeVLA
Optimized for low-latency inference on embedded NPUs (Ascend 310B / ARM aarch64).
Supports MobileNetV4 / ConvNeXt-Femto style efficient inverted bottleneck blocks.
"""

import torch
import torch.nn as nn
import torchvision.models as models

class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=4):
        super().__init__()
        self.stride = stride
        hidden_dim = int(in_channels * expand_ratio)
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNAct(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0))
        # Depthwise
        layers.append(ConvBNAct(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim))
        # Pointwise-linear
        layers.append(nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)

class LightweightVisionEncoder(nn.Module):
    """
    Compact CNN-based spatial vision encoder yielding feature tokens.
    Input: (B, 3, 224, 224)
    Output: (B, num_tokens, feature_dim) or (B, feature_dim)
    """
    def __init__(self, embed_dim=256, variant="mobilenet_v4_small"):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Stem
        self.stem = nn.Sequential(
            ConvBNAct(3, 32, kernel_size=3, stride=2, padding=1),   # 112x112
            ConvBNAct(32, 32, kernel_size=3, stride=1, padding=1),
        )
        
        # Stage 1 (56x56)
        self.stage1 = nn.Sequential(
            InvertedResidual(32, 48, stride=2, expand_ratio=2),
            InvertedResidual(48, 48, stride=1, expand_ratio=2),
        )
        
        # Stage 2 (28x28)
        self.stage2 = nn.Sequential(
            InvertedResidual(48, 96, stride=2, expand_ratio=3),
            InvertedResidual(96, 96, stride=1, expand_ratio=3),
        )
        
        # Stage 3 (14x14)
        self.stage3 = nn.Sequential(
            InvertedResidual(96, 160, stride=2, expand_ratio=4),
            InvertedResidual(160, 160, stride=1, expand_ratio=4),
            InvertedResidual(160, 160, stride=1, expand_ratio=4),
        )
        
        # Stage 4 (7x7)
        self.stage4 = nn.Sequential(
            InvertedResidual(160, 256, stride=2, expand_ratio=4),
            InvertedResidual(256, 256, stride=1, expand_ratio=4),
        )
        
        self.proj = nn.Conv2d(256, embed_dim, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x, return_spatial_tokens=False):
        """
        x: (B, 3, 224, 224)
        """
        feat = self.stem(x)
        feat = self.stage1(feat)
        feat = self.stage2(feat)
        feat = self.stage3(feat)
        feat = self.stage4(feat)      # (B, 256, 7, 7)
        feat = self.proj(feat)        # (B, embed_dim, 7, 7)

        if return_spatial_tokens:
            # Flatten to (B, 49, embed_dim)
            B, C, H, W = feat.shape
            tokens = feat.flatten(2).permute(0, 2, 1)
            return tokens
        else:
            # Pooled vector (B, embed_dim)
            pooled = self.pool(feat).flatten(1)
            return pooled

if __name__ == "__main__":
    model = LightweightVisionEncoder(embed_dim=256)
    dummy_img = torch.randn(2, 3, 224, 224)
    out = model(dummy_img)
    print(f"Vision Encoder Output Shape: {out.shape}")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {num_params / 1e6:.2f} M")
