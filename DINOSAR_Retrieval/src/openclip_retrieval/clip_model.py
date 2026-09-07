"""Dual-tower CLIP model: swapped vision tower + open_clip text tower.

The vision tower's pooled feature is projected to the shared 512-d embedding
space by a bias-free linear head; the text tower projects with its checkpoint
initialized text_projection. Both sides are L2-normalized on request, and the
learnable temperature is maintained in fp32 and clamped to [0, ln(100)].
"""

import math

import torch
from torch import nn

from .config import ExperimentConfig
from .text_tower import load_text_tower
from .towers import load_tower

LOGIT_SCALE_MAX = math.log(100.0)


class ClipModel(nn.Module):
    def __init__(self, vision_tower, text_tower, embed_dim: int = 512):
        super().__init__()
        self.tower = vision_tower
        self.text = text_tower
        self.visual_projection = nn.Linear(vision_tower.feature_dim, embed_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))

    def clamp_logit_scale(self) -> None:
        with torch.no_grad():
            self.logit_scale.clamp_(0.0, LOGIT_SCALE_MAX)

    def encode_image(self, images: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        features = self.visual_projection(self.tower(images))
        if normalize:
            features = features / features.norm(dim=-1, keepdim=True)
        return features

    def encode_text(self, tokens: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        features = self.text(tokens)
        if normalize:
            features = features / features.norm(dim=-1, keepdim=True)
        return features

    def forward(self, images: torch.Tensor, tokens: torch.Tensor):
        self.clamp_logit_scale()
        return self.encode_image(images), self.encode_text(tokens), self.logit_scale

    def set_grad_checkpointing(self, enable: bool) -> None:
        self.tower.set_grad_checkpointing(enable)
        self.text.set_grad_checkpointing(enable)


def build_clip_model(config: ExperimentConfig) -> ClipModel:
    vision_tower = load_tower(config.vision)
    text_tower, _ = load_text_tower(config.text)
    model = ClipModel(vision_tower, text_tower, embed_dim=config.text.output_dim)
    model.clamp_logit_scale()
    return model


def load_clip_checkpoint(model: ClipModel, path) -> dict:
    """Load a training checkpoint written by scripts/train_clip.py.

    weights_only=False because the payload embeds RNG states beyond plain
    tensors; these files are produced by this project.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    return checkpoint
