"""Vision tower wrappers: checkpoint unwrapping, strict loading, pooling.

Each tower owns one architecture family and knows the checkpoint prefix used
by its weights file. Loading is strict: after unwrapping the fvcore shell and
stripping the prefix, both missing and unexpected keys must be empty.
"""

import functools
from collections.abc import Mapping

import torch
from torch import nn

from . import dinov3_model
from .config import Dinov3TowerConfig, VisionTowerConfig


def unwrap_state_dict(raw: object, prefix: str) -> dict:
    if not isinstance(raw, Mapping):
        raise TypeError(f"unsupported checkpoint payload: {type(raw)!r}")
    state_dict = dict(raw)
    if "model" in state_dict and isinstance(state_dict["model"], Mapping):
        shell_keys = [key for key in state_dict if key != "model"]
        if any(not isinstance(state_dict[key], Mapping) for key in shell_keys):
            state_dict = state_dict["model"]
    if prefix:
        stripped = {
            key.removeprefix(prefix): value
            for key, value in state_dict.items()
            if key.startswith(prefix)
        }
        if len(stripped) != len(state_dict):
            raise ValueError("checkpoint mixes prefixed and unprefixed keys")
        state_dict = stripped
    return state_dict


class VisionTower(nn.Module):
    pooling = "cls"

    def __init__(self):
        super().__init__()
        self.backbone = None

    @property
    def feature_dim(self) -> int:
        raise NotImplementedError

    def _load_into_backbone(self, state_dict: Mapping, path) -> int:
        incompatible = self.backbone.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                f"strict load failed for {type(self).__name__} from {path}: "
                f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
            )
        return len(state_dict)

    def load_checkpoint(self, path, checkpoint_prefix: str) -> int:
        raw = torch.load(path, map_location="cpu", weights_only=True)
        state_dict = unwrap_state_dict(raw, checkpoint_prefix)
        return self._load_into_backbone(state_dict, path)

    def set_grad_checkpointing(self, enable: bool) -> None:
        raise NotImplementedError

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class Dinov3Tower(VisionTower):
    def __init__(self, config: Dinov3TowerConfig):
        super().__init__()
        self.pooling = config.pooling
        self.backbone = dinov3_model.vit_base(
            patch_size=config.patch_size,
            in_chans=config.channels,
            embed_dim=config.embed_dim,
            depth=config.depth,
            num_heads=config.num_heads,
            num_register_tokens=config.num_register_tokens,
            layerscale_init=config.layerscale_init,
            mask_k_bias=config.mask_k_bias,
            untie_global_and_local_cls_norm=config.untie_global_and_local_cls_norm,
        )

    @property
    def feature_dim(self) -> int:
        return self.backbone.embed_dim

    def set_grad_checkpointing(self, enable: bool) -> None:
        self.backbone.grad_checkpointing = enable

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(images)
        if self.pooling == "cls":
            return features["x_norm_clstoken"]
        return features["x_norm_patchtokens"].mean(dim=1)


@functools.singledispatch
def build_tower(config: VisionTowerConfig) -> VisionTower:
    raise TypeError(f"no tower registered for config {type(config).__name__}")


@build_tower.register
def _(config: Dinov3TowerConfig) -> Dinov3Tower:
    return Dinov3Tower(config)


def load_tower(config: VisionTowerConfig) -> VisionTower:
    tower = build_tower(config)
    tower.load_checkpoint(config.checkpoint, config.checkpoint_prefix)
    return tower
