"""Symmetric InfoNCE (CLIP) loss with optional false-negative masking."""

import torch
import torch.nn.functional as F


def clip_loss(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    logit_scale: torch.Tensor,
    image_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    logits = logit_scale.exp() * image_features @ text_features.t()
    labels = torch.arange(logits.shape[0], device=logits.device)
    if image_ids is not None:
        same_image = image_ids.unsqueeze(0) == image_ids.unsqueeze(1)
        false_negative = same_image & ~torch.eye(len(image_ids), dtype=torch.bool, device=logits.device)
        logits = logits.masked_fill(false_negative, float("-inf"))
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
