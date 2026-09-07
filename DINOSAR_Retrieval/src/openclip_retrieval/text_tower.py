"""CLIP text tower extracted from an open_clip ViT-B-32 checkpoint.

The tower is the open_clip TextTransformer with the checkpoint-matching
activation (plain GELU for laion2b_s34b_b79k), so the text-half keys
strict-load unchanged.
"""

import torch
from torch import nn

from .config import TextTowerConfig


def build_text_tower(config: TextTowerConfig):
    from open_clip.transformer import QuickGELU, TextTransformer

    act_layer = QuickGELU if config.quick_gelu else nn.GELU
    return TextTransformer(
        context_length=config.context_length,
        vocab_size=config.vocab_size,
        width=config.width,
        heads=config.heads,
        layers=config.layers,
        output_dim=config.output_dim,
        act_layer=act_layer,
    )


def extract_text_state_dict(path) -> tuple[dict, torch.Tensor]:
    """Return the text-half state dict and the checkpoint's logit_scale."""
    full = torch.load(path, map_location="cpu", weights_only=True)
    text_state = {key: value for key, value in full.items() if not key.startswith("visual.")}
    logit_scale = text_state.pop("logit_scale", None)
    if logit_scale is None:
        raise KeyError(f"{path} has no logit_scale tensor")
    return text_state, logit_scale


def load_text_tower(config: TextTowerConfig):
    tower = build_text_tower(config)
    text_state, _ = extract_text_state_dict(config.checkpoint)
    incompatible = tower.load_state_dict(text_state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"strict text-tower load failed from {config.checkpoint}: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    return tower, len(text_state)
