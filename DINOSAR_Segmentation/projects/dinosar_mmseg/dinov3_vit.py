"""MMSeg backbone wrapper for DINOSAR / DINOv3 ViT checkpoints.

The original AIR-PolSAR-Seg gray config registers ``DINOv3VisionTransformer``.
That class previously lived in the in-house fine-tuning codebase used before this release.
This module restores the same MMSeg type name and constructor, using the
DINOSAR ViT-B/16 implementation (DINOv3 architecture, 1-channel SAR, 4
storage tokens, RoPE) plus the shared FPN adapter used by the other ViT
segmentors in this workspace.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmseg.registry import MODELS
from torch.utils.checkpoint import checkpoint as grad_checkpoint


class Norm2d(nn.Module):
    """LayerNorm over channels for BCHW feature maps, shared by the ViT adapters."""

    def __init__(self, embed_dim):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        return x.permute(0, 3, 1, 2).contiguous()


def _resolve_dinosar_repo() -> Path:
    env = os.environ.get("DINOSAR_ROOT")
    candidates = []
    if env:
        candidates.append(Path(env))
    candidates.append(Path(__file__).resolve().parents[3] / "DINOSAR")
    for path in candidates:
        if (path / "dinosar" / "model.py").is_file():
            return path
    raise FileNotFoundError(
        "Cannot find the DINOSAR repository. Set DINOSAR_ROOT or place DINOSAR next to DINOSAR_Segmentation."
    )


_DINOSAR_REPO = _resolve_dinosar_repo()
if str(_DINOSAR_REPO) not in sys.path:
    sys.path.insert(0, str(_DINOSAR_REPO))

from dinosar.model import VisionTransformer  # noqa: E402


def _unwrap_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint format: {type(checkpoint)}")
    tensor_values = [value for value in checkpoint.values() if torch.is_tensor(value)]
    if tensor_values and len(tensor_values) == len(checkpoint):
        return dict(checkpoint)
    for key in ("model", "state_dict", "teacher", "student"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            nested = value.get("backbone")
            if isinstance(nested, dict) and any(torch.is_tensor(item) for item in nested.values()):
                return dict(nested)
            if any(torch.is_tensor(item) for item in value.values()):
                return dict(value)
    raise TypeError(f"Unsupported checkpoint layout: keys={list(checkpoint.keys())}")


def _strip_prefix(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    return {key[len(prefix) :] if key.startswith(prefix) else key: value for key, value in state_dict.items()}


@MODELS.register_module()
class DINOv3VisionTransformer(BaseModule):
    """MMSeg encoder that exposes DINOv3/DINOSAR multi-scale ViT features."""

    def __init__(
        self,
        img_size=512,
        patch_size=16,
        in_chans=1,
        input_adapter="none",
        drop_path_rate=0.0,
        out_indices=(3, 5, 7, 11),
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        use_checkpoint=False,
        n_storage_tokens=4,
        fp32_attention=True,
        pretrained=None,
        pos_embed_rope_base=100.0,
        pos_embed_rope_rescale_coords=2.0,
        layerscale_init=1.0e-5,
        mask_k_bias=True,
        untie_global_and_local_cls_norm=True,
        init_cfg=None,
        **kwargs,
    ):
        super().__init__(init_cfg=init_cfg)
        del kwargs
        if input_adapter not in (None, "none"):
            raise ValueError(f"Unsupported input_adapter: {input_adapter}")
        if patch_size != 16:
            raise ValueError(f"The ViT FPN adapter currently expects patch_size=16, got {patch_size}")

        self.out_indices = tuple(out_indices)
        self.use_checkpoint = bool(use_checkpoint)
        self.fp32_attention = bool(fp32_attention)
        self.embed_dim = embed_dim
        self.num_features = embed_dim
        self.patch_size = patch_size
        self.pretrained = pretrained

        self.backbone = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            ffn_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_path_rate=drop_path_rate,
            num_register_tokens=n_storage_tokens,
            layerscale_init=layerscale_init,
            mask_k_bias=mask_k_bias,
            untie_global_and_local_cls_norm=untie_global_and_local_cls_norm,
            pos_embed_rope_base=pos_embed_rope_base,
            pos_embed_rope_rescale_coords=pos_embed_rope_rescale_coords,
        )
        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            Norm2d(embed_dim),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
        )
        self.fpn2 = nn.Sequential(nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2))
        self.fpn3 = nn.Identity()
        self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Load in init_weights(), not here. MMSeg EncoderDecoder calls
        # init_weights() after construction; VisionTransformer.init_weights()
        # would otherwise re-randomize the encoder and wipe the checkpoint.

    def init_weights(self):
        if self.pretrained:
            self._load_pretrained(self.pretrained)
            return
        self.backbone.init_weights()

    def _load_pretrained(self, pretrained: str) -> None:
        checkpoint_data = torch.load(pretrained, map_location="cpu")
        state_dict = _strip_prefix(_unwrap_state_dict(checkpoint_data), "module.")
        state_dict = _strip_prefix(state_dict, "backbone.")
        model_state = self.backbone.state_dict()
        loaded_state = {}
        loaded = 0
        skipped = 0
        for key, value in state_dict.items():
            if key in model_state and model_state[key].shape == value.shape:
                loaded_state[key] = value
                loaded += 1
            else:
                skipped += 1
        missing, unexpected = self.backbone.load_state_dict(loaded_state, strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected DINOv3 keys after load: {unexpected[:20]}")
        unused_missing = [key for key in missing if not key.startswith(("fpn",))]
        if unused_missing:
            raise RuntimeError(f"Missing DINOv3 encoder keys after load: {unused_missing[:20]}")
        print(f"Loaded DINOv3 pretrained weights: loaded={loaded}, skipped={skipped}")

    def forward(self, x: torch.Tensor):
        if self.fp32_attention and x.dtype != torch.float32:
            x = x.float()

        if self.use_checkpoint and self.training:
            outputs = []
            tokens, (height, width) = self.backbone.prepare_tokens_with_masks(x)
            rope = self.backbone.rope_embed(height=height, width=width)
            for index, block in enumerate(self.backbone.blocks):
                tokens = grad_checkpoint(block, tokens, rope, use_reentrant=False)
                if index in self.out_indices:
                    outputs.append(self.backbone.norm(tokens))
            batch_size, _, image_height, image_width = x.shape
            patch_tokens = [output[:, self.backbone.num_register_tokens + 1 :] for output in outputs]
            features = [
                token.reshape(batch_size, image_height // self.patch_size, image_width // self.patch_size, -1)
                .permute(0, 3, 1, 2)
                .contiguous()
                for token in patch_tokens
            ]
        else:
            features = list(
                self.backbone.get_intermediate_layers(
                    x,
                    n=self.out_indices,
                    reshape=True,
                    return_class_token=False,
                    return_extra_tokens=False,
                    norm=True,
                )
            )

        ops = (self.fpn1, self.fpn2, self.fpn3, self.fpn4)
        if len(features) != len(ops):
            raise RuntimeError(f"FPN expects {len(ops)} feature maps, got {len(features)}")
        return tuple(op(feat) for op, feat in zip(ops, features))
