import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmseg.registry import MODELS


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
        "Cannot find the DINOSAR repository. Set DINOSAR_ROOT, or place this "
        "repository as a sibling of the DINOSAR pretraining repository."
    )


DINOSAR_REPO = _resolve_dinosar_repo()
if str(DINOSAR_REPO) not in sys.path:
    sys.path.insert(0, str(DINOSAR_REPO))

from dinosar.model import load_backbone_from_config  # noqa: E402


class Norm2d(nn.Module):
    """LayerNorm over channels for BCHW feature maps, shared by the ViT adapters."""

    def __init__(self, embed_dim):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


@MODELS.register_module()
class DINOSARViT(BaseModule):
    """MMSeg backbone adapter for DINOSAR ViT checkpoints."""

    def __init__(
        self,
        arch="vit_base",
        patch_size=16,
        img_size=512,
        in_chans=1,
        checkpoint_in_chans=None,
        patch_embed_adapter="repeat_mean",
        out_indices=(3, 5, 7, 11),
        checkpoint=None,
        checkpoint_type="backbone",
        with_fpn=True,
        num_register_tokens=4,
        drop_path_rate=0.2,
        layerscale_init=1.0e-5,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        mask_k_bias=True,
        untie_cls_and_patch_norms=False,
        untie_global_and_local_cls_norm=True,
        pos_embed_rope_base=100.0,
        pos_embed_rope_rescale_coords=2.0,
        frozen=False,
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        self.out_indices = tuple(out_indices)
        self.frozen = frozen
        checkpoint_in_chans = in_chans if checkpoint_in_chans is None else checkpoint_in_chans

        model_config = dict(
            arch=arch,
            patch_size=patch_size,
            img_size=img_size,
            in_chans=checkpoint_in_chans,
            num_register_tokens=num_register_tokens,
            drop_path_rate=drop_path_rate,
            layerscale_init=layerscale_init,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            ffn_bias=ffn_bias,
            mask_k_bias=mask_k_bias,
            untie_cls_and_patch_norms=untie_cls_and_patch_norms,
            untie_global_and_local_cls_norm=untie_global_and_local_cls_norm,
            pos_embed_rope_base=pos_embed_rope_base,
            pos_embed_rope_rescale_coords=pos_embed_rope_rescale_coords,
        )
        if checkpoint:
            self.backbone = load_backbone_from_config(
                model_config,
                checkpoint,
                checkpoint_type=checkpoint_type,
                initialize_weights=False,
                strict=True,
            )
        else:
            from dinosar.model import DINO  # noqa: WPS433

            self.backbone = DINO.build_backbone_from_config(model_config, initialize_weights=True)

        if in_chans != checkpoint_in_chans:
            self._adapt_patch_embed_input_channels(in_chans, patch_embed_adapter)

        self.embed_dim = self.backbone.embed_dim
        self.num_features = self.embed_dim
        self.with_fpn = with_fpn
        if self.with_fpn:
            if patch_size != 16:
                raise ValueError(f"The ViT FPN adapter currently expects patch_size=16, got {patch_size}")
            self.fpn_ops = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.ConvTranspose2d(self.embed_dim, self.embed_dim, kernel_size=2, stride=2),
                        Norm2d(self.embed_dim),
                        nn.GELU(),
                        nn.ConvTranspose2d(self.embed_dim, self.embed_dim, kernel_size=2, stride=2),
                    ),
                    nn.Sequential(
                        nn.ConvTranspose2d(self.embed_dim, self.embed_dim, kernel_size=2, stride=2),
                    ),
                    nn.Identity(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                ]
            )

        if self.frozen:
            self.backbone.eval()
            self.backbone.requires_grad_(False)

    def _adapt_patch_embed_input_channels(self, in_chans, adapter):
        patch_embed = self.backbone.patch_embed
        old_proj = patch_embed.proj
        old_in_chans = old_proj.in_channels
        if adapter != "repeat_mean":
            raise ValueError(f"Unsupported patch_embed_adapter: {adapter}")
        if old_in_chans != 1:
            raise ValueError(
                f"repeat_mean adapter expects a 1-channel checkpoint patch embedding, got {old_in_chans}"
            )

        new_proj = nn.Conv2d(
            in_chans,
            old_proj.out_channels,
            kernel_size=old_proj.kernel_size,
            stride=old_proj.stride,
            padding=old_proj.padding,
            dilation=old_proj.dilation,
            groups=old_proj.groups,
            bias=old_proj.bias is not None,
            padding_mode=old_proj.padding_mode,
            device=old_proj.weight.device,
            dtype=old_proj.weight.dtype,
        )
        with torch.no_grad():
            new_proj.weight.copy_(old_proj.weight.repeat(1, in_chans, 1, 1) / float(in_chans))
            if old_proj.bias is not None:
                new_proj.bias.copy_(old_proj.bias)

        patch_embed.proj = new_proj
        patch_embed.in_chans = in_chans

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.backbone.eval()
        return self

    def forward(self, x):
        outputs = self.backbone.get_intermediate_layers(
            x,
            n=self.out_indices,
            reshape=True,
            return_class_token=False,
            return_extra_tokens=False,
            norm=True,
        )
        if self.with_fpn:
            if len(outputs) != len(self.fpn_ops):
                raise RuntimeError(f"FPN expects {len(self.fpn_ops)} feature maps, got {len(outputs)}")
            outputs = [op(feat) for op, feat in zip(self.fpn_ops, outputs)]
        return tuple(outputs)
