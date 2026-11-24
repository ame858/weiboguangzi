"""Pretrained hybrid CNN / SegFormer / Transformer-bottleneck backbones for SNL CenterDetector."""
import math

import torch
from torch import nn
from torch.nn import functional as F

BACKBONES = ("tiny", "mit-b0", "mit-b1", "resnet34-fpn", "resnet34-transformer")

_ENCODER_SPECS = {
    "mit-b0": ("mit_b0", False),
    "mit-b1": ("mit_b1", False),
    "resnet34-fpn": ("resnet34", False),
    "resnet34-transformer": ("resnet34", True),
}


def _conv_block(c_in, c_out, stride=1):
    groups = 8 if c_out % 8 == 0 else (4 if c_out % 4 == 0 else 1)
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False),
        nn.GroupNorm(groups, c_out),
        nn.SiLU(),
    )


def _sinusoidal_2d_pos_embed(height, width, dim, device, dtype):
    if dim % 4 != 0:
        raise ValueError(f"2D sinusoidal positional embedding dim must be divisible by 4, got {dim}")
    quarter = dim // 4
    omega = torch.exp(torch.arange(quarter, device=device, dtype=torch.float32) * (-math.log(10000.0) / quarter))
    y = torch.arange(height, device=device, dtype=torch.float32)[:, None] * omega[None, :]
    x = torch.arange(width, device=device, dtype=torch.float32)[:, None] * omega[None, :]
    pos_y = torch.cat((y.sin(), y.cos()), dim=1)[:, None, :].expand(height, width, dim // 2)
    pos_x = torch.cat((x.sin(), x.cos()), dim=1)[None, :, :].expand(height, width, dim // 2)
    return torch.cat((pos_y, pos_x), dim=-1).reshape(1, height * width, dim).to(dtype=dtype)


class TransformerBottleneck(nn.Module):
    """Pre-Norm 2D Transformer bottleneck at stride-32 for global road/scene context."""

    def __init__(self, dim=64, heads=4, layers=2, dropout=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        batch, dim, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        pos = _sinusoidal_2d_pos_embed(height, width, dim, x.device, x.dtype)
        attended = self.norm(self.encoder(tokens + pos))
        return x + attended.transpose(1, 2).reshape(batch, dim, height, width)


class PretrainedFPNBackbone(nn.Module):
    """Drop-in stride-2 feature extractor backed by SMP SegFormer (MiT) or ResNet34 + FPN."""

    def __init__(self, backbone, in_channels, *, pretrained=True, line_context=None, fpn_dim=64):
        super().__init__()
        if backbone not in _ENCODER_SPECS:
            raise ValueError(f"Unsupported backbone '{backbone}'. Choose from {BACKBONES}.")
        try:
            import segmentation_models_pytorch as smp
        except ImportError as exc:
            raise ImportError(
                f"Backbone '{backbone}' requires segmentation-models-pytorch (see requirements-unet.txt)."
            ) from exc

        encoder_name, use_transformer = _ENCODER_SPECS[backbone]
        weights = "imagenet" if pretrained else None
        self.encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=5,
            weights=weights,
        )
        channels = tuple(self.encoder.out_channels)
        # channels = (in_channels, c_s2, c_s4, c_s8, c_s16, c_s32)
        self.has_encoder_s2 = channels[1] > 0
        if self.has_encoder_s2:
            self.fine_proj = nn.Sequential(
                nn.Conv2d(channels[1], 32, 1, bias=False),
                nn.GroupNorm(8, 32),
                nn.SiLU(),
            )
            self.fine_stem = None
        else:
            # MiT (SegFormer) starts at stride 4; keep a lightweight stride-2 stem for small shadows.
            self.fine_proj = None
            self.fine_stem = nn.Sequential(
                _conv_block(in_channels, 24, stride=1),
                _conv_block(24, 32, stride=2),
                _conv_block(32, 32, stride=1),
            )

        self.lat5 = nn.Conv2d(channels[5], fpn_dim, 1)
        self.lat4 = nn.Conv2d(channels[4], fpn_dim, 1)
        self.lat3 = nn.Conv2d(channels[3], fpn_dim, 1)
        self.lat2 = nn.Conv2d(channels[2], fpn_dim, 1)
        self.bottleneck = TransformerBottleneck(dim=fpn_dim, heads=4, layers=2) if use_transformer else None
        self.smooth4 = _conv_block(fpn_dim, fpn_dim)
        self.smooth3 = _conv_block(fpn_dim, fpn_dim)
        self.smooth2 = _conv_block(fpn_dim, 48)
        self.line_context = line_context
        self.fuse_fine = nn.Sequential(_conv_block(80, 48), _conv_block(48, 32))

    def forward(self, x):
        height, width = x.shape[-2:]
        pad_h, pad_w = -height % 32, -width % 32
        padded = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate") if (pad_h or pad_w) else x

        feats = self.encoder(padded)
        fine = self.fine_proj(feats[1]) if self.has_encoder_s2 else self.fine_stem(padded)

        p5 = self.lat5(feats[5])
        if self.bottleneck is not None:
            p5 = self.bottleneck(p5)
        p4 = self.smooth4(self.lat4(feats[4]) + F.interpolate(p5, size=feats[4].shape[-2:], mode="bilinear", align_corners=False))
        p3 = self.smooth3(self.lat3(feats[3]) + F.interpolate(p4, size=feats[3].shape[-2:], mode="bilinear", align_corners=False))
        p2 = self.smooth2(self.lat2(feats[2]) + F.interpolate(p3, size=feats[2].shape[-2:], mode="bilinear", align_corners=False))
        if self.line_context is not None:
            p2 = self.line_context(p2)

        up_fine = F.interpolate(p2, size=fine.shape[-2:], mode="bilinear", align_corners=False)
        fused = self.fuse_fine(torch.cat((fine, up_fine), dim=1))
        return fused[..., :(height + 1) // 2, :(width + 1) // 2]
