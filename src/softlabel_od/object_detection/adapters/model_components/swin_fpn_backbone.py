from collections import OrderedDict
import torch
import torch.nn as nn
import torchvision
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool

_SWIN_FACTORY = {
    "swin_t":    (torchvision.models.swin_t,    torchvision.models.Swin_T_Weights.IMAGENET1K_V1),
    "swin_s":    (torchvision.models.swin_s,    torchvision.models.Swin_S_Weights.IMAGENET1K_V1),
    "swin_b":    (torchvision.models.swin_b,    torchvision.models.Swin_B_Weights.IMAGENET1K_V1),
    "swin_v2_t": (torchvision.models.swin_v2_t, torchvision.models.Swin_V2_T_Weights.IMAGENET1K_V1),
    "swin_v2_s": (torchvision.models.swin_v2_s, torchvision.models.Swin_V2_S_Weights.IMAGENET1K_V1),
    "swin_v2_b": (torchvision.models.swin_v2_b, torchvision.models.Swin_V2_B_Weights.IMAGENET1K_V1),
}

# Per-stage channel dims (C0..C3) for each variant
_SWIN_CHANNELS = {
    "swin_t":    [96, 192, 384, 768],
    "swin_s":    [96, 192, 384, 768],
    "swin_b":    [128, 256, 512, 1024],
    "swin_v2_t": [96, 192, 384, 768],
    "swin_v2_s": [96, 192, 384, 768],
    "swin_v2_b": [128, 256, 512, 1024],
}


class _SwinBody(nn.Module):
    """
    Runs torchvision Swin's `features` and returns the 4 stage outputs in NCHW.

    torchvision Swin is channels-last (NHWC) internally; FPN/Conv2d need NCHW,
    so we permute each stage output.

    `features` is a flat 8-module Sequential:
        [0] patch embed        -> stride 4,  C0  (NHWC)
        [1] stage 0 blocks     -> stride 4,  C0
        [2] PatchMergingV2     -> stride 8,  C1
        [3] stage 1 blocks     -> stride 8,  C1
        [4] PatchMergingV2     -> stride 16, C2
        [5] stage 2 blocks     -> stride 16, C2
        [6] PatchMergingV2     -> stride 32, C3
        [7] stage 3 blocks     -> stride 32, C3
    We hook the post-stage-block outputs (indices 1, 3, 5, 7).
    """

    def __init__(self, swin_features: nn.Sequential):
        super().__init__()
        self.features = swin_features

    def forward(self, x):
        # x: (B, 3, H, W) NCHW
        f = self.features
        x = f[0](x)              # patch embed -> (B, H/4,  W/4,  C0) NHWC
        x = f[1](x)              # stage 0 blocks
        out0 = x.permute(0, 3, 1, 2).contiguous()
        x = f[2](x)              # PatchMerging -> stride 8
        x = f[3](x)              # stage 1 blocks
        out1 = x.permute(0, 3, 1, 2).contiguous()
        x = f[4](x)              # PatchMerging -> stride 16
        x = f[5](x)              # stage 2 blocks
        out2 = x.permute(0, 3, 1, 2).contiguous()
        x = f[6](x)              # PatchMerging -> stride 32
        x = f[7](x)              # stage 3 blocks
        out3 = x.permute(0, 3, 1, 2).contiguous()
        return OrderedDict([("0", out0), ("1", out1), ("2", out2), ("3", out3)])


class SwinFPNBackbone(nn.Module):
    def __init__(self, body, fpn, out_channels):
        super().__init__()
        self.body = body
        self.fpn = fpn
        self.out_channels = out_channels

    def forward(self, x):
        return self.fpn(self.body(x))


def swin_fpn_backbone(backbone_name, pretrained=True, out_channels=256):
    """
    Swin + FPN backbone compatible with torchvision FasterRCNN.

    Returns a module whose forward returns an OrderedDict of 5 FPN levels
    ('0','1','2','3','pool') at strides 4,8,16,32,64 with `out_channels` channels.
    """
    if backbone_name not in _SWIN_FACTORY:
        raise ValueError(f"Unsupported Swin variant: {backbone_name}")
    factory_fn, weights_enum = _SWIN_FACTORY[backbone_name]
    weights = weights_enum if pretrained else None
    swin = factory_fn(weights=weights)

    body = _SwinBody(swin.features)
    fpn = FeaturePyramidNetwork(
        in_channels_list=_SWIN_CHANNELS[backbone_name],
        out_channels=out_channels,
        extra_blocks=LastLevelMaxPool(),
    )
    return SwinFPNBackbone(body, fpn, out_channels)