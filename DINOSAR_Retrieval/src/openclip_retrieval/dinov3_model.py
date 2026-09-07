"""DINOv3-family vision transformer backbone.

Vendored from DINOSAR (dinosar/model.py), trimmed to the single-view encoder
path used as a CLIP vision tower: multi-crop list forwards, DINO heads, and
training-checkpoint helpers are dropped. Parameter and buffer names are kept
identical so upstream DINOv3/DINOSAR checkpoints strict-load unchanged.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


def _pair(value):
    if isinstance(value, tuple):
        return value
    return (value, value)


def _rotate_half(x):
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _apply_rope(x, sin, cos):
    return (x * cos) + (_rotate_half(x) * sin)


@dataclass(frozen=True)
class VisionTransformerSpec:
    embed_dim: int
    depth: int
    num_heads: int
    ffn_ratio: float = 4.0


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5):
        super().__init__()
        self.init_values = init_values
        self.gamma = nn.Parameter(torch.full((dim,), init_values))

    def forward(self, x):
        return x * self.gamma


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=1, embed_dim=768):
        super().__init__()
        image_size = _pair(img_size)
        patch_size = _pair(patch_size)
        grid_size = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])

        self.img_size = image_size
        self.patch_size = patch_size
        self.patches_resolution = grid_size
        self.num_patches = grid_size[0] * grid_size[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        height, width = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x).reshape(-1, height, width, self.embed_dim)


class RopePositionEmbedding(nn.Module):
    """Axial rotary position embedding; frequencies kept in fp32."""

    def __init__(self, embed_dim, *, num_heads, base=100.0, rescale_coords=None):
        super().__init__()
        if embed_dim % (4 * num_heads) != 0:
            raise ValueError("embed_dim must be divisible by 4 * num_heads for RoPE.")

        self.head_dim = embed_dim // num_heads
        self.base = base
        self.rescale_coords = rescale_coords
        self.register_buffer("periods", torch.empty(self.head_dim // 4, dtype=torch.float32), persistent=True)
        self.reset_parameters()

    def reset_parameters(self):
        exponents = 2 * torch.arange(self.head_dim // 4, dtype=torch.float32) / (self.head_dim // 2)
        periods = self.base**exponents
        self.periods.copy_(periods.to(device=self.periods.device))

    def _build_coords(self, height, width):
        kwargs = {"device": self.periods.device, "dtype": torch.float32}
        coords_h = torch.arange(0.5, height, **kwargs) / height
        coords_w = torch.arange(0.5, width, **kwargs) / width
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1).flatten(0, 1)
        coords = 2.0 * coords - 1.0

        if self.training and self.rescale_coords is not None:
            log_rescale = math.log(self.rescale_coords)
            coords *= torch.empty(1, **kwargs).uniform_(-log_rescale, log_rescale).exp()
        return coords

    def forward(self, *, height, width):
        coords = self._build_coords(height, width)
        angles = 2 * math.pi * coords[:, :, None] / self.periods[None, None, :]
        angles = angles.flatten(1, 2).tile(2)
        return torch.sin(angles), torch.cos(angles)


class LinearKMaskedBias(nn.Linear):
    """QKV linear that zeros the K-projection bias, matching DINOv2/v3."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.bias is not None:
            if self.out_features % 3 != 0:
                raise ValueError("Masked QKV bias expects out_features divisible by 3.")
            self.register_buffer("bias_mask", torch.full_like(self.bias, fill_value=math.nan))

    def forward(self, x):
        bias = self.bias
        if bias is not None:
            bias = bias * self.bias_mask.to(dtype=bias.dtype)
        return F.linear(x, self.weight, bias)


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, act_layer=nn.GELU, bias=True):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features, bias=bias)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Attention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias, proj_bias, proj_drop=0.0, mask_k_bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        linear_cls = LinearKMaskedBias if mask_k_bias else nn.Linear
        self.qkv = linear_cls(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def _apply_rope(self, q, k, rope):
        sin, cos = rope
        sin = sin.to(dtype=q.dtype)
        cos = cos.to(dtype=q.dtype)
        prefix = q.shape[-2] - sin.shape[-2]
        q_prefix, q_tokens = q[:, :, :prefix], q[:, :, prefix:]
        k_prefix, k_tokens = k[:, :, :prefix], k[:, :, prefix:]
        q = torch.cat((q_prefix, _apply_rope(q_tokens, sin, cos)), dim=-2)
        k = torch.cat((k_prefix, _apply_rope(k_tokens, sin, cos)), dim=-2)
        return q, k

    def _reshape_qkv(self, qkv):
        batch, tokens, _ = qkv.shape
        qkv = qkv.reshape(batch, tokens, 3, self.num_heads, qkv.shape[-1] // (3 * self.num_heads))
        q, k, v = torch.unbind(qkv, dim=2)
        return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    def forward(self, x, rope=None):
        q, k, v = self._reshape_qkv(self.qkv(x))
        if rope is not None:
            q, k = self._apply_rope(q, k, rope)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(x.shape[0], -1, self.num_heads * self.head_dim)
        return self.proj_drop(self.proj(x))


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        ffn_ratio=4.0,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=lambda dim: nn.LayerNorm(dim, eps=1e-6),
        init_values=None,
        mask_k_bias=False,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            proj_drop=drop,
            mask_k_bias=mask_k_bias,
        )
        self.ls1 = LayerScale(dim, init_values) if init_values else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=int(dim * ffn_ratio),
            act_layer=act_layer,
            bias=ffn_bias,
        )
        self.ls2 = LayerScale(dim, init_values) if init_values else nn.Identity()
        self.sample_drop_ratio = drop_path

    def _forward_tensor(self, x, rope=None):
        batch_size = x.shape[0]
        subset_size = max(int(batch_size * (1 - self.sample_drop_ratio)), 1)
        scale = batch_size / subset_size

        if self.training and self.sample_drop_ratio > 0.0:
            indices_1 = torch.randperm(batch_size, device=x.device)[:subset_size]
            attn_residual = self.attn(self.norm1(x[indices_1]), rope=rope)
            x = torch.index_add(x, dim=0, source=self.ls1(attn_residual), index=indices_1, alpha=scale)

            indices_2 = torch.randperm(batch_size, device=x.device)[:subset_size]
            mlp_residual = self.mlp(self.norm2(x[indices_2]))
            return torch.index_add(x, dim=0, source=self.ls2(mlp_residual), index=indices_2, alpha=scale)

        x = x + self.ls1(self.attn(self.norm1(x), rope=rope))
        return x + self.ls2(self.mlp(self.norm2(x)))

    def forward(self, x, rope=None):
        return self._forward_tensor(x, rope=rope)


class VisionTransformer(nn.Module):
    """DINOv3-compatible ViT encoder with register tokens and RoPE."""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=1,
        embed_dim=768,
        depth=12,
        num_heads=12,
        ffn_ratio=4.0,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=lambda dim: nn.LayerNorm(dim, eps=1e-6),
        pos_embed_rope_base=100.0,
        num_register_tokens=0,
        layerscale_init=None,
        mask_k_bias=False,
        untie_global_and_local_cls_norm=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_features = embed_dim
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.num_register_tokens = num_register_tokens
        self.untie_global_and_local_cls_norm = untie_global_and_local_cls_norm
        self.grad_checkpointing = False

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim))
        self.storage_tokens = (
            nn.Parameter(torch.empty(1, num_register_tokens, embed_dim)) if num_register_tokens > 0 else None
        )
        self.mask_token = nn.Parameter(torch.empty(1, embed_dim))
        self.rope_embed = RopePositionEmbedding(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=pos_embed_rope_base,
        )
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    ffn_ratio=ffn_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    drop=drop_rate,
                    drop_path=drop_path_rate,
                    norm_layer=norm_layer,
                    init_values=layerscale_init,
                    mask_k_bias=mask_k_bias,
                )
                for _ in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.local_cls_norm = norm_layer(embed_dim) if untie_global_and_local_cls_norm else None

    def prepare_tokens(self, x):
        x = self.patch_embed(x)
        batch_size, height, width, _ = x.shape
        x = x.flatten(1, 2)

        cls_token = self.cls_token + 0 * self.mask_token
        if self.storage_tokens is None:
            storage_tokens = cls_token.new_empty(1, 0, self.embed_dim)
        else:
            storage_tokens = self.storage_tokens

        x = torch.cat(
            [cls_token.expand(batch_size, -1, -1), storage_tokens.expand(batch_size, -1, -1), x],
            dim=1,
        )
        return x, (height, width)

    def _normalize_outputs(self, tokens):
        split_index = self.num_register_tokens + 1
        if self.untie_global_and_local_cls_norm:
            cls_storage = self.norm(tokens[:, :split_index])
            return cls_storage[:, 0], self.norm(tokens[:, split_index:])
        normed = self.norm(tokens)
        return normed[:, 0], normed[:, split_index:]

    def forward_features(self, x):
        x, (height, width) = self.prepare_tokens(x)
        rope = self.rope_embed(height=height, width=width)
        for block in self.blocks:
            if self.grad_checkpointing and self.training and torch.is_grad_enabled():
                x = checkpoint(block, x, rope, use_reentrant=False)
            else:
                x = block(x, rope=rope)
        cls_token, patch_tokens = self._normalize_outputs(x)
        return {"x_norm_clstoken": cls_token, "x_norm_patchtokens": patch_tokens}

    def forward(self, x):
        return self.forward_features(x)["x_norm_clstoken"]


def _build_model(spec, *, patch_size=16, in_chans=1, **kwargs):
    kwargs.setdefault("embed_dim", spec.embed_dim)
    kwargs.setdefault("depth", spec.depth)
    kwargs.setdefault("num_heads", spec.num_heads)
    kwargs.setdefault("ffn_ratio", spec.ffn_ratio)
    return VisionTransformer(patch_size=patch_size, in_chans=in_chans, **kwargs)


def vit_base(patch_size=16, in_chans=1, **kwargs):
    return _build_model(
        VisionTransformerSpec(embed_dim=768, depth=12, num_heads=12),
        patch_size=patch_size,
        in_chans=in_chans,
        **kwargs,
    )


BACKBONES = {"vit_base": vit_base}
