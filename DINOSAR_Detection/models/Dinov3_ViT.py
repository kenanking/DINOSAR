from __future__ import annotations

from typing import Dict, Union

import torch
import torch.nn as nn

from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone import Backbone
from dinov3.hub.backbones import Weights, _make_dinov3_vit


ARCH_DEFAULTS = {
    "vit_small": {
        "compact_arch_name": "vits",
        "hash": "08c60483",
        "embed_dim": 384,
        "depth": 12,
        "num_heads": 6,
        "ffn_ratio": 4,
        "ffn_layer": "mlp",
        "n_storage_tokens": 4,
    },
    "vit_base": {
        "compact_arch_name": "vitb",
        "hash": "73cec8be",
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "ffn_ratio": 4,
        "ffn_layer": "mlp",
        "n_storage_tokens": 4,
    },
    "vit_large": {
        "compact_arch_name": "vitl",
        "hash": "8aa4cbdd",
        "embed_dim": 1024,
        "depth": 24,
        "num_heads": 16,
        "ffn_ratio": 4,
        "ffn_layer": "mlp",
        "n_storage_tokens": 4,
    },
    "vit_huge": {
        "compact_arch_name": "vithplus",
        "hash": "7c1da9a5",
        "embed_dim": 1280,
        "depth": 32,
        "num_heads": 20,
        "ffn_ratio": 6,
        "ffn_layer": "swiglu",
        "n_storage_tokens": 4,
    },
    "vit_7b": {
        "compact_arch_name": "vit7b",
        "hash": "a955f4ea",
        "embed_dim": 4096,
        "depth": 40,
        "num_heads": 32,
        "ffn_ratio": 3,
        "ffn_layer": "swiglu64",
        "n_storage_tokens": 4,
    },
}


class DINOv3ViTBackbone(Backbone):
    def __init__(
        self,
        arch: str = "vit_base",
        out_feature: str = "last_feat",
        img_size: int = 224,
        in_chans: int = 3,
        n_storage_tokens: int | None = None,
        pretrained: bool = False,
        **kwargs,
    ):
        super().__init__()

        # Backward-compatible alias for existing single-channel configs.
        if arch == "vit_base_single_channel_without_storage_tokens":
            arch = "vit_base"
            in_chans = 1
            n_storage_tokens = 0

        self.model = make_dinov3_vit(
            arch=arch,
            img_size=img_size,
            in_chans=in_chans,
            n_storage_tokens=n_storage_tokens,
            pretrained=pretrained,
            **kwargs,
        )
        self.patch_size = self.model.patch_size

        self._out_feature_channels = {out_feature: self.model.embed_dim}
        self._out_feature_strides = {out_feature: self.patch_size}
        self._out_features = [out_feature]

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        outs = self.model.get_intermediate_layers(
            x,
            n=1,
            reshape=True,
            return_class_token=False,
            return_extra_tokens=False,
            norm=True,
        )
        return {self._out_features[0]: outs[0]}

    def output_shape(self) -> dict[str, ShapeSpec]:
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name],
            )
            for name in self._out_features
        }


class MultiScaleDINOv3ViTBackbone(DINOv3ViTBackbone):
    """DINOv3 ViT backbone with opt-in UPer-style multi-scale outputs."""

    def __init__(
        self,
        *,
        out_indices: tuple[int, int, int, int] = (3, 5, 7, 11),
        out_features: tuple[str, str, str, str] = ("p2", "p3", "p4", "p5"),
        **kwargs,
    ):
        super().__init__(**kwargs)
        if len(out_indices) != 4 or len(out_features) != 4:
            raise ValueError("MultiScaleDINOv3ViTBackbone expects four output indices/features.")
        self.out_indices = tuple(out_indices)
        self._out_features = list(out_features)
        self._out_feature_channels = {name: self.model.embed_dim for name in self._out_features}
        self._out_feature_strides = dict(zip(self._out_features, (4, 8, 16, 32)))
        self.multiscale_ops = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ConvTranspose2d(self.model.embed_dim, self.model.embed_dim, kernel_size=2, stride=2),
                    nn.GroupNorm(32, self.model.embed_dim),
                    nn.GELU(),
                    nn.ConvTranspose2d(self.model.embed_dim, self.model.embed_dim, kernel_size=2, stride=2),
                ),
                nn.ConvTranspose2d(self.model.embed_dim, self.model.embed_dim, kernel_size=2, stride=2),
                nn.Identity(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ]
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        outs = self.model.get_intermediate_layers(
            x,
            n=self.out_indices,
            reshape=True,
            return_class_token=False,
            return_extra_tokens=False,
            norm=True,
        )
        return {
            name: op(feature)
            for name, op, feature in zip(self._out_features, self.multiscale_ops, outs)
        }


def make_dinov3_vit(
    *,
    arch: str,
    img_size: int = 224,
    in_chans: int = 3,
    n_storage_tokens: int | None = None,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.LVD1689M,
    check_hash: bool = False,
    **kwargs,
):
    defaults = ARCH_DEFAULTS.get(arch, ARCH_DEFAULTS["vit_base"])
    if n_storage_tokens is None:
        n_storage_tokens = defaults["n_storage_tokens"]
    kwargs.setdefault("hash", defaults["hash"])
    kwargs["version"] = None

    return _make_dinov3_vit(
        img_size=img_size,
        patch_size=16,
        in_chans=in_chans,
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_rescale_coords=2,
        pos_embed_rope_dtype="fp32",
        embed_dim=defaults["embed_dim"],
        depth=defaults["depth"],
        num_heads=defaults["num_heads"],
        ffn_ratio=defaults["ffn_ratio"],
        qkv_bias=(arch != "vit_7b"),
        drop_path_rate=0.0,
        layerscale_init=1.0e-05,
        norm_layer="layernormbf16",
        ffn_layer=defaults["ffn_layer"],
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=n_storage_tokens,
        mask_k_bias=True,
        pretrained=pretrained,
        weights=weights,
        compact_arch_name=defaults["compact_arch_name"],
        check_hash=check_hash,
        **kwargs,
    )


def dinov3_vits16(
    *,
    img_size=224,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.LVD1689M,
    check_hash: bool = False,
    **kwargs,
):
    return make_dinov3_vit(
        arch="vit_small",
        img_size=img_size,
        pretrained=pretrained,
        weights=weights,
        check_hash=check_hash,
        **kwargs,
    )


def dinov3_vitb16(
    *,
    img_size=224,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.LVD1689M,
    check_hash: bool = False,
    **kwargs,
):
    return make_dinov3_vit(
        arch="vit_base",
        img_size=img_size,
        pretrained=pretrained,
        weights=weights,
        check_hash=check_hash,
        **kwargs,
    )


def dinov3_vitb16_single_channel_without_storage_tokens(
    *,
    img_size=224,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.LVD1689M,
    check_hash: bool = False,
    **kwargs,
):
    return make_dinov3_vit(
        arch="vit_base",
        img_size=img_size,
        in_chans=1,
        n_storage_tokens=0,
        pretrained=pretrained,
        weights=weights,
        check_hash=check_hash,
        **kwargs,
    )


def dinov3_vitl16(
    *,
    img_size=224,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.LVD1689M,
    check_hash: bool = False,
    **kwargs,
):
    untie_global_and_local_cls_norm = False
    if weights == Weights.SAT493M:
        kwargs.setdefault("hash", "eadcf0ff")
        untie_global_and_local_cls_norm = True
    kwargs.setdefault("untie_global_and_local_cls_norm", untie_global_and_local_cls_norm)
    return make_dinov3_vit(
        arch="vit_large",
        img_size=img_size,
        pretrained=pretrained,
        weights=weights,
        check_hash=check_hash,
        **kwargs,
    )


def dinov3_vits16plus(
    *,
    img_size=224,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.LVD1689M,
    check_hash: bool = False,
    **kwargs,
):
    return _make_dinov3_vit(
        img_size=img_size,
        patch_size=16,
        in_chans=kwargs.pop("in_chans", 3),
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_rescale_coords=2,
        pos_embed_rope_dtype="fp32",
        embed_dim=384,
        depth=12,
        num_heads=6,
        ffn_ratio=6,
        qkv_bias=True,
        drop_path_rate=0.0,
        layerscale_init=1.0e-05,
        norm_layer="layernormbf16",
        ffn_layer="swiglu",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=kwargs.pop("n_storage_tokens", 4),
        mask_k_bias=True,
        pretrained=pretrained,
        weights=weights,
        compact_arch_name="vitsplus",
        check_hash=check_hash,
        version=None,
        hash=kwargs.pop("hash", "4057cbaa"),
        **kwargs,
    )


def dinov3_vitl16plus(
    *,
    img_size=224,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.LVD1689M,
    check_hash: bool = False,
    **kwargs,
):
    return _make_dinov3_vit(
        img_size=img_size,
        patch_size=16,
        in_chans=kwargs.pop("in_chans", 3),
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_rescale_coords=2,
        pos_embed_rope_dtype="fp32",
        embed_dim=1024,
        depth=24,
        num_heads=16,
        ffn_ratio=6.0,
        qkv_bias=True,
        drop_path_rate=0.0,
        layerscale_init=1.0e-05,
        norm_layer="layernormbf16",
        ffn_layer="swiglu",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=kwargs.pop("n_storage_tokens", 4),
        mask_k_bias=True,
        pretrained=pretrained,
        weights=weights,
        compact_arch_name="vitlplus",
        check_hash=check_hash,
        hash=kwargs.pop("hash", "46503df0"),
        **kwargs,
    )


def dinov3_vith16plus(
    *,
    img_size=224,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.LVD1689M,
    check_hash: bool = False,
    **kwargs,
):
    return make_dinov3_vit(
        arch="vit_huge",
        img_size=img_size,
        pretrained=pretrained,
        weights=weights,
        check_hash=check_hash,
        **kwargs,
    )


def dinov3_vit7b16(
    *,
    img_size=224,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.LVD1689M,
    check_hash: bool = False,
    **kwargs,
):
    if weights == Weights.SAT493M:
        kwargs.setdefault("hash", "a6675841")
    kwargs.setdefault("untie_global_and_local_cls_norm", True)
    return make_dinov3_vit(
        arch="vit_7b",
        img_size=img_size,
        pretrained=pretrained,
        weights=weights,
        check_hash=check_hash,
        **kwargs,
    )


def get_vit_lr_decay_rate_for_dinov3(name, lr_decay_rate=1.0, num_layers=12):
    """
    Calculate lr decay rate for DINOv3 ViT parameters.
    """
    layer_id = num_layers + 1
    if name.startswith("backbone"):
        if ".pos_embed" in name or ".patch_embed" in name or ".rope_embed" in name:
            layer_id = 0
        elif ".blocks." in name and ".residual." not in name:
            layer_id = int(name[name.find(".blocks.") :].split(".")[2]) + 1

    return lr_decay_rate ** (num_layers + 1 - layer_id)
