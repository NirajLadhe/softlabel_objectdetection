"""
CLIP RN50 backbone + open-vocabulary classifier for the LVIS probabilistic detector.

Why this exists
---------------
LVIS is long-tailed: ~88% of classes get ~0 AP because they have too few training
examples for a learned 825-way linear classifier to discriminate them from 824
others. The fix is *vision-language* knowledge the small dataset cannot provide:

  1. ``CLIPRN50FPNBackbone`` — wraps the CLIP RN50 (ModifiedResNet) visual trunk
     into a torchvision-compatible FPN backbone. CLIP's image-text pretraining gives
     features that already "know" rare categories.
  2. ``OpenVocabProbabilisticPredictor`` — replaces the learned linear classifier
     with a cosine-similarity head against frozen CLIP *text embeddings* of the class
     names. A class with zero training boxes can still be recognised because its text
     embedding sits in the same space as the visual features. The GMM box-regression
     branch (the bbox_stds source for PDQ) is left untouched.
  3. ``build_lvis_text_embeddings`` — encodes the 824 LVIS class names with the CLIP
     text encoder, in the dataset's index order (sorted names -> idx = pos + 1).

CLIP RN50 stage geometry (verified at 800px input):
    stem -> /4 (64ch) ; layer1 C2 /4 256 ; layer2 C3 /8 512 ;
    layer3 C4 /16 1024 ; layer4 C5 /32 2048   -> identical layout to torchvision
    ResNet, so FeaturePyramidNetwork([256,512,1024,2048]) + LastLevelMaxPool yields
    the same 5 levels as ``resnet_fpn_backbone`` (P2..P6), matching the LVIS anchors.

The CLIP weights expect CLIP image normalization, NOT ImageNet — the caller must set
FasterRCNN ``image_mean``/``image_std`` to ``CLIP_PIXEL_MEAN``/``CLIP_PIXEL_STD``.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import FeaturePyramidNetwork
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

# open_clip is imported lazily inside the builders so importing this module never
# forces the dependency unless a CLIP backbone is actually requested.

# CLIP RN50 visual output dim and pixel statistics (OpenAI weights).
CLIP_EMBED_DIM = 1024
CLIP_PIXEL_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_PIXEL_STD = (0.26862954, 0.26130258, 0.27577711)

# We use the quickgelu variant so the *text* encoder matches the OpenAI pretrained
# tag exactly (the visual ModifiedResNet is ReLU-based, so it is unaffected, but the
# weights are the same checkpoint either way).
CLIP_MODEL_NAME = "RN50-quickgelu"
CLIP_PRETRAINED = "openai"

# Prompt ensemble — averaged then renormalized (standard CLIP zero-shot practice).
_DEFAULT_TEMPLATES = (
    "a photo of a {}.",
    "a photo of the {}.",
    "there is a {} in the scene.",
    "a {}.",
)


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------
class _CLIPModifiedResNetBody(nn.Module):
    """Runs the CLIP ModifiedResNet stem + layer1..4, returns C2..C5 as a dict."""

    def __init__(self, visual: nn.Module):
        super().__init__()
        self.visual = visual

    def _stem(self, x: torch.Tensor) -> torch.Tensor:
        v = self.visual
        if hasattr(v, "stem"):
            return v.stem(x)
        # Fallback for open_clip versions without a .stem() method.
        for conv, bn, act in (
            (v.conv1, v.bn1, v.act1),
            (v.conv2, v.bn2, v.act2),
            (v.conv3, v.bn3, v.act3),
        ):
            x = act(bn(conv(x)))
        return v.avgpool(x)

    def forward(self, x: torch.Tensor) -> "OrderedDict[str, torch.Tensor]":
        v = self.visual
        x = self._stem(x)
        c2 = v.layer1(x)
        c3 = v.layer2(c2)
        c4 = v.layer3(c3)
        c5 = v.layer4(c4)
        return OrderedDict([("0", c2), ("1", c3), ("2", c4), ("3", c5)])


class CLIPRN50FPNBackbone(nn.Module):
    """CLIP RN50 visual trunk + FPN, drop-in for torchvision FasterRCNN."""

    def __init__(self, visual: nn.Module, out_channels: int = 256):
        super().__init__()
        self.body = _CLIPModifiedResNetBody(visual)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[256, 512, 1024, 2048],
            out_channels=out_channels,
            extra_blocks=LastLevelMaxPool(),
        )
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> "OrderedDict[str, torch.Tensor]":
        return self.fpn(self.body(x))


def build_clip_rn50_fpn_backbone(out_channels: int = 256) -> CLIPRN50FPNBackbone:
    """Create the CLIP RN50 FPN backbone with OpenAI-pretrained visual weights."""
    import open_clip

    clip_model = open_clip.create_model(CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED)
    visual = clip_model.visual
    # Drop the attention-pool head — we only need the convolutional feature maps.
    if hasattr(visual, "attnpool"):
        visual.attnpool = nn.Identity()
    return CLIPRN50FPNBackbone(visual, out_channels=out_channels)


# ---------------------------------------------------------------------------
# Open-vocabulary classifier (cosine sim vs CLIP text embeddings)
# ---------------------------------------------------------------------------
class OpenVocabProbabilisticPredictor(nn.Module):
    """Probabilistic predictor with an open-vocab cosine-similarity classifier.

    Classification: project RoI features into CLIP space, L2-normalize, and take the
    temperature-scaled cosine similarity against frozen text embeddings (index >=1)
    plus a single learnable background embedding (index 0).

    Regression: identical 8-values-per-class GMM head as
    ``ProbabilisticFastRCNNPredictor`` so bbox means + log-vars (PDQ) are unchanged.

    The ``text_embeddings`` buffer is created empty and filled at fit-start via
    :meth:`set_text_embeddings` once the dataset's class order is known.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        clip_dim: int = CLIP_EMBED_DIM,
        log_var_bias: float = -4.0,
        logit_scale_init: float = 100.0,
        learnable_logit_scale: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes  # includes background at index 0
        self.clip_dim = clip_dim

        # RoI feature -> CLIP space
        self.proj = nn.Linear(in_channels, clip_dim)
        nn.init.normal_(self.proj.weight, std=0.01)
        nn.init.zeros_(self.proj.bias)

        # Learnable background embedding (index 0) — no text for "background".
        self.bg_embed = nn.Parameter(torch.randn(clip_dim) * 0.01)

        # Temperature (CLIP convention: log space, clamped to <= log(100)).
        log_scale = math.log(logit_scale_init)
        if learnable_logit_scale:
            self.logit_scale = nn.Parameter(torch.tensor(log_scale, dtype=torch.float32))
        else:
            self.register_buffer("logit_scale", torch.tensor(log_scale, dtype=torch.float32))

        # Frozen text embeddings for the (num_classes - 1) foreground classes.
        self.register_buffer(
            "text_embeddings", torch.zeros(num_classes - 1, clip_dim)
        )
        self._text_ready = False

        # GMM regression head — identical to ProbabilisticFastRCNNPredictor.
        self.bbox_pred_gmm = nn.Linear(in_channels, num_classes * 8)
        nn.init.normal_(self.bbox_pred_gmm.weight, std=0.01)
        nn.init.constant_(self.bbox_pred_gmm.bias, 0)
        with torch.no_grad():
            bias = self.bbox_pred_gmm.bias.view(num_classes, 8)
            bias[:, 4:] = log_var_bias
            self.bbox_pred_gmm.bias.copy_(bias.view(-1))

    @torch.no_grad()
    def set_text_embeddings(self, embeddings: torch.Tensor) -> None:
        """Install L2-normalized foreground text embeddings ([num_classes-1, clip_dim])."""
        if embeddings.shape != self.text_embeddings.shape:
            raise ValueError(
                f"text embedding shape {tuple(embeddings.shape)} != "
                f"expected {tuple(self.text_embeddings.shape)}"
            )
        self.text_embeddings.copy_(F.normalize(embeddings, dim=-1).to(self.text_embeddings))
        self._text_ready = True

    def _classifier_weight(self) -> torch.Tensor:
        bg = F.normalize(self.bg_embed, dim=-1).unsqueeze(0)          # [1, D]
        fg = F.normalize(self.text_embeddings, dim=-1)                # [C-1, D]
        return torch.cat([bg, fg], dim=0)                            # [C, D], idx0=bg

    def forward(self, x: torch.Tensor):
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)

        region = F.normalize(self.proj(x), dim=-1)                   # [N, D]
        weight = self._classifier_weight()                          # [C, D]
        scale = self.logit_scale.clamp(max=math.log(100.0)).exp()
        scores = scale * (region @ weight.t())                      # [N, C]

        gmm_params = self.bbox_pred_gmm(x)
        return scores, gmm_params


# ---------------------------------------------------------------------------
# Text embeddings
# ---------------------------------------------------------------------------
def clean_lvis_name(name: str) -> str:
    """Turn an LVIS class id into natural language.

    e.g. ``train_(railroad_vehicle)`` -> ``train railroad vehicle``.
    """
    text = name.replace("_", " ").replace("(", " ").replace(")", " ")
    return " ".join(text.split())


@torch.no_grad()
def build_lvis_text_embeddings(
    class_names: Sequence[str],
    device: Optional[torch.device] = None,
    templates: Optional[Sequence[str]] = None,
) -> torch.Tensor:
    """Encode foreground class names into CLIP text embeddings (prompt-ensembled).

    Args:
        class_names: foreground names in index order (idx 1..C-1 -> positions 0..C-2).
        device:      device to run the CLIP text encoder on.
        templates:   prompt templates; defaults to a 4-prompt ensemble.

    Returns:
        Tensor [len(class_names), CLIP_EMBED_DIM], L2-normalized.
    """
    import open_clip

    device = device or torch.device("cpu")
    templates = list(templates or _DEFAULT_TEMPLATES)

    clip_model = open_clip.create_model(CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED)
    clip_model = clip_model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)

    embeddings: List[torch.Tensor] = []
    for name in class_names:
        clean = clean_lvis_name(name)
        prompts = tokenizer([t.format(clean) for t in templates]).to(device)
        feats = clip_model.encode_text(prompts)                     # [T, D]
        feats = F.normalize(feats, dim=-1).mean(dim=0)              # ensemble
        embeddings.append(F.normalize(feats, dim=-1))

    return torch.stack(embeddings, dim=0)
