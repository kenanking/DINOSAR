"""DINOv3 model definitions for DINOSAR."""

import math
from collections import OrderedDict
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------


def _pair(value):
    if isinstance(value, tuple):
        return value
    return (value, value)


def _pack_token_list(x_list):
    shapes = [x.shape for x in x_list]
    lengths = [math.prod(shape[:-1]) for shape in shapes]
    flat = torch.cat([x.reshape(-1, x.shape[-1]) for x in x_list], dim=0)
    return flat, shapes, lengths


def _unpack_token_list(flat, shapes, lengths):
    chunks = torch.split(flat, lengths, dim=0)
    return [chunk.reshape(*shape[:-1], flat.shape[-1]) for chunk, shape in zip(chunks, shapes)]


def _apply_to_token_list(fn, x_list):
    flat, shapes, lengths = _pack_token_list(x_list)
    return _unpack_token_list(fn(flat), shapes, lengths)


def _rotate_half(x):
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _apply_rope(x, sin, cos):
    return (x * cos) + (_rotate_half(x) * sin)


@dataclass(frozen=True)
class VisionTransformerSpec:
    """Architectural hyperparameters for a ViT variant (e.g. Small, Base, Large)."""

    embed_dim: int
    depth: int
    num_heads: int
    ffn_ratio: float = 4.0


class LayerScale(nn.Module):
    """Per-channel residual scaling."""

    def __init__(self, dim, init_values=1e-5, device=None):
        super().__init__()
        self.init_values = init_values
        self.gamma = nn.Parameter(torch.full((dim,), init_values, device=device))

    def reset_parameters(self):
        self.gamma.data.fill_(self.init_values)

    def forward(self, x):
        return x * self.gamma


class PatchEmbed(nn.Module):
    """Image to patch embedding."""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=1,
        embed_dim=768,
        norm_layer=None,
        flatten_embedding=True,
        device=None,
    ):
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
        self.flatten_embedding = flatten_embedding

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, device=device)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()

        if in_chans == 1:
            self.proj.weight.register_hook(lambda grad: grad.view(-1).view(grad.shape))

    def reset_parameters(self):
        bound = math.sqrt(1 / (self.in_chans * self.patch_size[0] * self.patch_size[1]))
        nn.init.uniform_(self.proj.weight, -bound, bound)
        if self.proj.bias is not None:
            nn.init.uniform_(self.proj.bias, -bound, bound)

    def forward(self, x):
        x = self.proj(x)
        height, width = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        if self.flatten_embedding:
            return x
        return x.reshape(-1, height, width, self.embed_dim)


class RopePositionEmbedding(nn.Module):
    """Axial rotary position embedding for 2D patch grids.

    Frequencies are always computed and stored in fp32 for numerical stability.
    Downstream callers cast the returned sin/cos to match q/k dtype.
    """

    def __init__(
        self,
        embed_dim,
        *,
        num_heads,
        base=100.0,
        rescale_coords=None,
        device=None,
    ):
        super().__init__()
        if embed_dim % (4 * num_heads) != 0:
            raise ValueError("embed_dim must be divisible by 4 * num_heads for RoPE.")

        self.head_dim = embed_dim // num_heads
        self.base = base
        self.rescale_coords = rescale_coords
        self.register_buffer(
            "periods", torch.empty(self.head_dim // 4, dtype=torch.float32, device=device), persistent=True
        )
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
    """QKV linear layer that zeros the K-projection bias to match DINOv2/v3 conventions.

    A ``bias_mask`` buffer selectively disables the bias for the middle third of
    ``out_features`` (the K-projection), leaving Q and V biases active.
    """

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
    """Transformer feed-forward network."""

    def __init__(
        self,
        in_features,
        hidden_features,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
        bias=True,
        device=None,
    ):
        super().__init__()
        out_features = in_features if out_features is None else out_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias, device=device)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias, device=device)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        return self.drop2(x)

    def forward_list(self, x_list):
        return _apply_to_token_list(self.forward, x_list)


class Attention(nn.Module):
    """Multi-head self-attention with optional RoPE and masked K-bias."""

    def __init__(
        self,
        dim,
        num_heads,
        qkv_bias,
        proj_bias,
        attn_drop=0.0,
        proj_drop=0.0,
        mask_k_bias=False,
        device=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        linear_cls = LinearKMaskedBias if mask_k_bias else nn.Linear
        self.qkv = linear_cls(dim, dim * 3, bias=qkv_bias, device=device)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias, device=device)
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
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
        x = x.transpose(1, 2).reshape(x.shape[0], -1, self.num_heads * self.head_dim)
        x = self.proj(x)
        return self.proj_drop(x)

    def forward_list(self, x_list, rope_list):
        x_flat, shapes, lengths = _pack_token_list(x_list)
        qkv_list = _unpack_token_list(self.qkv(x_flat), shapes, lengths)
        outputs: list[Tensor] = []
        for qkv, rope in zip(qkv_list, rope_list):
            q, k, v = self._reshape_qkv(qkv)
            if rope is not None:
                q, k = self._apply_rope(q, k, rope)
            x = F.scaled_dot_product_attention(q, k, v)
            outputs.append(x.transpose(1, 2).reshape(x.shape[0], -1, self.num_heads * self.head_dim))
        return _unpack_token_list(self.proj(_pack_token_list(outputs)[0]), shapes, lengths)


class Block(nn.Module):
    """Transformer block used by the DINOv3-compatible backbone."""

    def __init__(
        self,
        dim,
        num_heads,
        ffn_ratio=4.0,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=lambda dim: nn.LayerNorm(dim, eps=1e-6),
        init_values=None,
        mask_k_bias=False,
        device=None,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            mask_k_bias=mask_k_bias,
            device=device,
        )
        self.ls1 = LayerScale(dim, init_values, device=device) if init_values else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=int(dim * ffn_ratio),
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
            device=device,
        )
        self.ls2 = LayerScale(dim, init_values, device=device) if init_values else nn.Identity()
        self.sample_drop_ratio = drop_path

    @staticmethod
    def _select_rope(
        rope,
        indices,
    ):
        if rope is None:
            return None
        sin, cos = rope
        if sin.ndim == 4:
            return sin[indices], cos[indices]
        return sin, cos

    def _forward_tensor(self, x, rope=None):
        batch_size = x.shape[0]
        subset_size = max(int(batch_size * (1 - self.sample_drop_ratio)), 1)
        scale = batch_size / subset_size

        if self.training and self.sample_drop_ratio > 0.0:
            indices_1 = torch.randperm(batch_size, device=x.device)[:subset_size]
            attn_input = self.norm1(x[indices_1])
            attn_rope = self._select_rope(rope, indices_1)
            attn_residual = self.attn(attn_input, rope=attn_rope)
            x = torch.index_add(x, dim=0, source=self.ls1(attn_residual), index=indices_1, alpha=scale)

            indices_2 = torch.randperm(batch_size, device=x.device)[:subset_size]
            mlp_input = self.norm2(x[indices_2])
            mlp_residual = self.mlp(mlp_input)
            return torch.index_add(x, dim=0, source=self.ls2(mlp_residual), index=indices_2, alpha=scale)

        x = x + self.ls1(self.attn(self.norm1(x), rope=rope))
        return x + self.ls2(self.mlp(self.norm2(x)))

    def _forward_list(
        self,
        x_list,
        rope_list=None,
    ):
        rope_list = [None] * len(x_list) if rope_list is None else rope_list
        batch_sizes = [x.shape[0] for x in x_list]
        subset_sizes = [max(int(batch_size * (1 - self.sample_drop_ratio)), 1) for batch_size in batch_sizes]
        scales = [batch_size / subset_size for batch_size, subset_size in zip(batch_sizes, subset_sizes)]

        if self.training and self.sample_drop_ratio > 0.0:
            indices_1 = [
                torch.randperm(batch_size, device=x.device)[:subset_size]
                for x, batch_size, subset_size in zip(x_list, batch_sizes, subset_sizes)
            ]
            attn_inputs = [x[index] for x, index in zip(x_list, indices_1)]
            attn_inputs = _apply_to_token_list(self.norm1, attn_inputs)
            attn_ropes = [self._select_rope(rope, index) for rope, index in zip(rope_list, indices_1)]
            attn_residuals = self.attn.forward_list(attn_inputs, attn_ropes)
            x_list = [
                torch.index_add(x, dim=0, source=self.ls1(residual), index=index, alpha=scale)
                for x, residual, index, scale in zip(x_list, attn_residuals, indices_1, scales)
            ]

            indices_2 = [
                torch.randperm(batch_size, device=x.device)[:subset_size]
                for x, batch_size, subset_size in zip(x_list, batch_sizes, subset_sizes)
            ]
            mlp_inputs = [x[index] for x, index in zip(x_list, indices_2)]
            mlp_inputs = _apply_to_token_list(self.norm2, mlp_inputs)
            mlp_residuals = self.mlp.forward_list(mlp_inputs)
            return [
                torch.index_add(x, dim=0, source=self.ls2(residual), index=index, alpha=scale)
                for x, residual, index, scale in zip(x_list, mlp_residuals, indices_2, scales)
            ]

        return [self._forward_tensor(x, rope=rope) for x, rope in zip(x_list, rope_list)]

    def forward(
        self,
        x,
        rope=None,
    ):
        if isinstance(x, list):
            assert rope is None or isinstance(rope, list)
            return self._forward_list(x, rope_list=rope)
        assert rope is None or isinstance(rope, tuple)
        return self._forward_list([x], rope_list=[rope])[0]


class VisionTransformer(nn.Module):
    """Minimal DINOv3-compatible ViT backbone for SAR images."""

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
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=lambda dim: nn.LayerNorm(dim, eps=1e-6),
        pos_embed_rope_base=100.0,
        pos_embed_rope_rescale_coords=None,
        num_register_tokens=0,
        layerscale_init=None,
        mask_k_bias=False,
        untie_cls_and_patch_norms=False,
        untie_global_and_local_cls_norm=False,
        device=None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_features = embed_dim
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.num_register_tokens = num_register_tokens
        self.untie_cls_and_patch_norms = untie_cls_and_patch_norms
        self.untie_global_and_local_cls_norm = untie_global_and_local_cls_norm

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            flatten_embedding=False,
        )
        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim, device=device))
        self.storage_tokens = (
            nn.Parameter(torch.empty(1, num_register_tokens, embed_dim, device=device))
            if num_register_tokens > 0
            else None
        )
        self.mask_token = nn.Parameter(torch.empty(1, embed_dim, device=device))
        self.rope_embed = RopePositionEmbedding(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=pos_embed_rope_base,
            rescale_coords=pos_embed_rope_rescale_coords,
            device=device,
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
                    attn_drop=attn_drop_rate,
                    drop_path=drop_path_rate,
                    norm_layer=norm_layer,
                    init_values=layerscale_init,
                    mask_k_bias=mask_k_bias,
                    device=device,
                )
                for _ in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.cls_norm = norm_layer(embed_dim) if untie_cls_and_patch_norms else None
        self.local_cls_norm = norm_layer(embed_dim) if untie_global_and_local_cls_norm else None
        self.head = nn.Identity()

    def init_weights(self):
        self.rope_embed.reset_parameters()
        nn.init.normal_(self.cls_token, std=0.02)
        if self.storage_tokens is not None:
            nn.init.normal_(self.storage_tokens, std=0.02)
        nn.init.zeros_(self.mask_token)

        for module in self.modules():
            if isinstance(module, PatchEmbed):
                module.reset_parameters()
            elif isinstance(module, LayerScale):
                module.reset_parameters()
            elif isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                if hasattr(module, "bias_mask") and module.bias_mask is not None:
                    output_dim = module.out_features
                    module.bias_mask.fill_(1)
                    module.bias_mask[output_dim // 3 : 2 * output_dim // 3].fill_(0)
            elif isinstance(module, nn.LayerNorm):
                module.reset_parameters()

    def prepare_tokens_with_masks(self, x, masks=None):
        x = self.patch_embed(x)
        batch_size, height, width, _ = x.shape
        x = x.flatten(1, 2)

        cls_token = self.cls_token if masks is not None else self.cls_token + 0 * self.mask_token
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)

        if self.storage_tokens is None:
            storage_tokens = cls_token.new_empty(1, 0, self.embed_dim)
        else:
            storage_tokens = self.storage_tokens

        x = torch.cat(
            [cls_token.expand(batch_size, -1, -1), storage_tokens.expand(batch_size, -1, -1), x],
            dim=1,
        )
        return x, (height, width)

    def _normalize_outputs(self, tokens, input_index):
        split_index = self.num_register_tokens + 1
        if self.untie_cls_and_patch_norms or self.untie_global_and_local_cls_norm:
            if self.training and self.untie_global_and_local_cls_norm and input_index == 1:
                assert self.local_cls_norm is not None
                cls_storage = self.local_cls_norm(tokens[:, :split_index])
            elif self.untie_cls_and_patch_norms:
                cls_norm = self.norm if self.cls_norm is None else self.cls_norm
                cls_storage = cls_norm(tokens[:, :split_index])
            else:
                cls_storage = self.norm(tokens[:, :split_index])
            patch_tokens = self.norm(tokens[:, split_index:])
            return cls_storage, patch_tokens

        normed = self.norm(tokens)
        return normed[:, :split_index], normed[:, split_index:]

    def forward_features_list(
        self,
        x_list,
        masks_list,
        hidden_distill_blocks: tuple[int, ...] | None = None,
        *,
        hidden_distill_first_view_only: bool = False,
    ):
        token_batches: list[Tensor] = []
        spatial_shapes: list[tuple[int, int]] = []
        for x, masks in zip(x_list, masks_list):
            tokens, spatial_shape = self.prepare_tokens_with_masks(x, masks)
            token_batches.append(tokens)
            spatial_shapes.append(spatial_shape)

        ropes = [self.rope_embed(height=height, width=width) for height, width in spatial_shapes]
        hidden_indices = set(hidden_distill_blocks) if hidden_distill_blocks is not None else None
        hidden_states: list[list[Tensor]] = []
        for block_idx, block in enumerate(self.blocks):
            token_batches = block(token_batches, ropes)
            if hidden_indices is not None and block_idx in hidden_indices:
                if hidden_distill_first_view_only:
                    hidden_states.append([token_batches[0]])
                else:
                    hidden_states.append(list(token_batches))

        outputs: list[dict[str, Tensor]] = []
        for index, (tokens, masks) in enumerate(zip(token_batches, masks_list)):
            cls_storage, patch_tokens = self._normalize_outputs(tokens, index)
            outputs.append(
                {
                    "x_norm_clstoken": cls_storage[:, 0],
                    "x_norm_regtokens": cls_storage[:, 1:],
                    "x_storage_tokens": cls_storage[:, 1:],
                    "x_norm_patchtokens": patch_tokens,
                    "x_prenorm": tokens,
                    "masks": masks,
                }
            )
        if hidden_distill_blocks is not None:
            return outputs, hidden_states
        return outputs

    def forward_features(
        self,
        x,
        masks=None,
        hidden_distill_blocks: tuple[int, ...] | None = None,
        *,
        hidden_distill_first_view_only: bool = False,
    ):
        if isinstance(x, list):
            masks_list = [None] * len(x) if masks is None else masks
            assert isinstance(masks_list, list)
            return self.forward_features_list(
                x,
                masks_list,
                hidden_distill_blocks,
                hidden_distill_first_view_only=hidden_distill_first_view_only,
            )
        result = self.forward_features_list(
            [x],
            [masks],
            hidden_distill_blocks,
            hidden_distill_first_view_only=hidden_distill_first_view_only,
        )
        if hidden_distill_blocks is not None:
            outputs, hidden_states = result
            return outputs[0], hidden_states
        return result[0]

    def forward(
        self,
        x,
        masks=None,
        is_training=False,
        hidden_distill_blocks: tuple[int, ...] | None = None,
        *,
        hidden_distill_first_view_only: bool = False,
    ):
        outputs = self.forward_features(
            x,
            masks=masks,
            hidden_distill_blocks=hidden_distill_blocks,
            hidden_distill_first_view_only=hidden_distill_first_view_only,
        )
        if hidden_distill_blocks is not None:
            outputs, hidden_states = outputs
        else:
            hidden_states = None
        if is_training:
            if hidden_states is not None:
                return outputs, hidden_states
            return outputs
        if isinstance(outputs, list):
            return [self.head(output["x_norm_clstoken"]) for output in outputs]
        return self.head(outputs["x_norm_clstoken"])

    def get_intermediate_layers(
        self,
        x,
        *,
        n=1,
        reshape=False,
        return_class_token=False,
        return_extra_tokens=False,
        norm=True,
    ):
        batch_size, _, image_height, image_width = x.shape
        x, (height, width) = self.prepare_tokens_with_masks(x)
        rope = self.rope_embed(height=height, width=width)
        blocks_to_take = range(len(self.blocks) - n, len(self.blocks)) if isinstance(n, int) else n
        outputs: list[Tensor] = []

        for index, block in enumerate(self.blocks):
            x = block(x, rope=rope)
            if index in blocks_to_take:
                outputs.append(self.norm(x) if norm else x)

        class_tokens = [output[:, 0] for output in outputs]
        extra_tokens = [output[:, 1 : self.num_register_tokens + 1] for output in outputs]
        patch_tokens = [output[:, self.num_register_tokens + 1 :] for output in outputs]

        if reshape:
            patch_tokens = [
                output.reshape(batch_size, image_height // self.patch_size, image_width // self.patch_size, -1)
                .permute(0, 3, 1, 2)
                .contiguous()
                for output in patch_tokens
            ]

        results: list[Tensor | tuple[Tensor, ...]] = list(patch_tokens)
        if return_class_token and return_extra_tokens:
            results = list(zip(patch_tokens, class_tokens, extra_tokens))
        elif return_class_token:
            results = list(zip(patch_tokens, class_tokens))
        elif return_extra_tokens:
            results = list(zip(patch_tokens, extra_tokens))
        return tuple(results)


def _build_model(
    spec,
    *,
    patch_size=16,
    in_chans=1,
    initialize_weights=True,
    **kwargs,
):
    model = VisionTransformer(
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=spec.embed_dim,
        depth=spec.depth,
        num_heads=spec.num_heads,
        ffn_ratio=spec.ffn_ratio,
        **kwargs,
    )
    if initialize_weights:
        model.init_weights()
    return model


def vit_small(
    patch_size=16,
    in_chans=1,
    initialize_weights=True,
    **kwargs,
):
    return _build_model(
        VisionTransformerSpec(embed_dim=384, depth=12, num_heads=6),
        patch_size=patch_size,
        in_chans=in_chans,
        initialize_weights=initialize_weights,
        **kwargs,
    )


def vit_base(
    patch_size=16,
    in_chans=1,
    initialize_weights=True,
    **kwargs,
):
    return _build_model(
        VisionTransformerSpec(embed_dim=768, depth=12, num_heads=12),
        patch_size=patch_size,
        in_chans=in_chans,
        initialize_weights=initialize_weights,
        **kwargs,
    )


def vit_large(
    patch_size=16,
    in_chans=1,
    initialize_weights=True,
    **kwargs,
):
    return _build_model(
        VisionTransformerSpec(embed_dim=1024, depth=24, num_heads=16),
        patch_size=patch_size,
        in_chans=in_chans,
        initialize_weights=initialize_weights,
        **kwargs,
    )


# ---------------------------------------------------------------------------

BACKBONES = {
    "vit_small": vit_small,
    "vit_base": vit_base,
    "vit_large": vit_large,
}


def _load_checkpoint(path):
    """Load and validate a checkpoint dict from disk."""

    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint format: {type(checkpoint)!r}")
    return checkpoint


def _extract_flat_state_dict(checkpoint):
    """Extract a flat tensor state dict from a backbone checkpoint."""

    if all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
        return checkpoint

    model_state = checkpoint.get("model")
    if isinstance(model_state, dict) and all(isinstance(v, torch.Tensor) for v in model_state.values()):
        return model_state

    raise TypeError(f"Unsupported backbone checkpoint layout: keys={list(checkpoint.keys())!r}")


def _apply_backbone_weights(backbone, state_dict, *, strict=True):
    """Load checkpoint weights into a backbone with minimal compatibility handling."""

    state_dict = OrderedDict(state_dict)
    ignored_keys: list[str] = []

    if backbone.storage_tokens is None and "storage_tokens" in state_dict:
        state_dict.pop("storage_tokens")
        ignored_keys.append("storage_tokens")

    result = backbone.load_state_dict(state_dict, strict=False)
    if strict and (result.missing_keys or result.unexpected_keys):
        raise RuntimeError(
            f"Incompatible backbone checkpoint: missing_keys={result.missing_keys!r}, "
            f"unexpected_keys={result.unexpected_keys!r}"
        )

    return result, ignored_keys


def extract_backbone_state_dict(checkpoint):
    """Extract the teacher backbone state dict from a training checkpoint."""

    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, Mapping):
        raise TypeError("Checkpoint missing 'model_state_dict' key.")

    prefix = "teacher.backbone."
    extracted = OrderedDict(
        (name[len(prefix) :].removeprefix("_orig_mod."), value)
        for name, value in state_dict.items()
        if name.startswith(prefix)
    )
    if not extracted:
        raise TypeError("No teacher backbone weights found in checkpoint.")
    return extracted


def load_backbone_from_config(
    model_config,
    checkpoint_path,
    *,
    checkpoint_type="backbone",
    initialize_weights=False,
    strict=True,
):
    """Build a backbone from config and load either backbone or training weights."""

    backbone = DINO.build_backbone_from_config(model_config, initialize_weights=initialize_weights)
    checkpoint = _load_checkpoint(Path(checkpoint_path))

    if checkpoint_type == "backbone":
        state_dict = _extract_flat_state_dict(checkpoint)
    elif checkpoint_type == "training":
        state_dict = extract_backbone_state_dict(checkpoint)
    else:
        raise ValueError(f"Unsupported checkpoint type: {checkpoint_type!r}")

    _apply_backbone_weights(backbone, state_dict, strict=strict)
    return backbone


class DINOHead(nn.Module):
    """Project CLS features into the DINO prototype space."""

    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_dim=2048,
        bottleneck_dim=256,
        mlp_bias=True,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=mlp_bias),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim, bias=mlp_bias),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim, bias=mlp_bias),
        )
        self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)

    def init_weights(self):
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        if x.dim() == 3:
            x = x[:, 0]
        x = self.mlp(x)
        eps = 1e-6 if x.dtype == torch.float16 else 1e-12
        x = F.normalize(x, dim=-1, p=2, eps=eps)
        return self.last_layer(x)


class DINO(nn.Module):
    """Minimal DINO model composed of a backbone and projection head."""

    def __init__(
        self,
        backbone,
        out_dim,
        hidden_dim=2048,
        bottleneck_dim=256,
    ):
        super().__init__()
        hidden_dim = 2048 if hidden_dim is None else hidden_dim
        self.backbone = backbone
        self.head = DINOHead(
            in_dim=backbone.embed_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
        )
        self.embed_dim = backbone.embed_dim

    def forward_features(self, x):
        return self.backbone.forward_features(x)

    def forward(self, x):
        return self.head(self.backbone(x))

    @staticmethod
    def build_backbone(
        arch,
        patch_size,
        in_chans,
        initialize_weights=True,
        **kwargs,
    ):
        try:
            builder = BACKBONES[arch]
        except KeyError as error:
            raise ValueError(f"Unknown backbone architecture: {arch}") from error
        return builder(patch_size=patch_size, in_chans=in_chans, initialize_weights=initialize_weights, **kwargs)

    @classmethod
    def build_backbone_from_config(
        cls,
        model_config,
        *,
        initialize_weights=True,
    ):
        """Build a backbone from a resolved model config mapping."""

        kwargs = dict(model_config)
        kwargs.pop("init_checkpoint", None)
        return cls.build_backbone(initialize_weights=initialize_weights, **kwargs)


# ---------------------------------------------------------------------------


class StudentTeacherWrapper(nn.Module):
    """Wrap two DINO models for EMA teacher pretraining."""

    def __init__(
        self,
        student,
        teacher,
        momentum=0.996,
        num_global_crops=2,
        hidden_distill_blocks: tuple[int, ...] | None = None,
        hidden_distill_use_predictor=False,
    ):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.momentum = momentum
        self.num_global_crops = num_global_crops
        self.hidden_distill_blocks = tuple(hidden_distill_blocks or ())
        self.hidden_distill_use_predictor = bool(hidden_distill_use_predictor)
        self.hidden_distill_predictors = nn.ModuleDict()
        if self.hidden_distill_use_predictor:
            embed_dim = self.student.backbone.embed_dim
            for block in self.hidden_distill_blocks:
                predictor = nn.Linear(embed_dim, embed_dim, bias=False)
                nn.init.eye_(predictor.weight)
                self.hidden_distill_predictors[str(block)] = predictor
        self._ema_param_lists: tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]] | None = None
        if not hasattr(self.student, "ibot_head"):
            self.student.ibot_head = deepcopy(self.student.head)
        if not hasattr(self.teacher, "ibot_head"):
            self.teacher.ibot_head = deepcopy(self.teacher.head)
        self.teacher.load_state_dict(self.student.state_dict())
        for parameter in self.teacher.parameters():
            parameter.requires_grad = False
        self.teacher.eval()

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        return self

    @torch.no_grad()
    def update_teacher(self, momentum=None):
        momentum = self.momentum if momentum is None else momentum
        if self._ema_param_lists is None:
            student_param_list = list(self.student.parameters())
            teacher_param_list = list(self.teacher.parameters())
            self._ema_param_lists = (student_param_list, teacher_param_list)
        else:
            student_param_list, teacher_param_list = self._ema_param_lists
        torch._foreach_mul_(teacher_param_list, momentum)
        torch._foreach_add_(teacher_param_list, student_param_list, alpha=1 - momentum)

    def _apply_hidden_distill_predictors(self, hidden_stack, hidden_distill_blocks):
        if not self.hidden_distill_use_predictor:
            return hidden_stack
        if tuple(hidden_distill_blocks) != self.hidden_distill_blocks:
            raise ValueError(
                "hidden distillation predictor blocks must match configured blocks: "
                f"{tuple(hidden_distill_blocks)} vs {self.hidden_distill_blocks}"
            )
        return [
            [self.hidden_distill_predictors[str(block)](layer_tensor) for layer_tensor in layer_views]
            for block, layer_views in zip(hidden_distill_blocks, hidden_stack)
        ]

    def forward(
        self,
        views,
        masks=None,
        mask_indices_list=None,
        hidden_distill_blocks: tuple[int, ...] | None = None,
    ):
        batch_size = views[0].shape[0]
        global_views = views[: self.num_global_crops]
        local_views = views[self.num_global_crops :]

        student_global_batch = torch.cat(global_views, dim=0)
        student_inputs = [student_global_batch]
        if local_views:
            student_inputs.append(torch.cat(local_views, dim=0))
        student_masks = [None] * len(student_inputs) if masks is None else masks
        if len(student_masks) != len(student_inputs):
            raise ValueError("Student masks must align with the number of student input groups.")
        if hidden_distill_blocks is not None:
            student_outputs, student_hidden_stack = self.student.backbone(
                student_inputs,
                masks=student_masks,
                is_training=True,
                hidden_distill_blocks=hidden_distill_blocks,
                hidden_distill_first_view_only=True,
            )
        else:
            student_outputs = self.student.backbone(student_inputs, masks=student_masks, is_training=True)
            student_hidden_stack = None
        assert isinstance(student_outputs, list)

        student_global_features = student_outputs[0]
        student_local_features = student_outputs[1] if len(student_outputs) > 1 else None
        student_cls_batches = [student_global_features["x_norm_clstoken"]]
        if student_local_features is not None:
            student_cls_batches.append(student_local_features["x_norm_clstoken"])
        student_head_input = torch.cat(student_cls_batches, dim=0)
        student_head_output = self.student.head(student_head_input)
        student_sizes = [tensor.shape[0] for tensor in student_cls_batches]
        student_global_logits, *student_local_logits = torch.split(student_head_output, student_sizes, dim=0)
        student_global_cls = student_global_features["x_norm_clstoken"].unflatten(
            0, (self.num_global_crops, batch_size)
        )
        student_global_patch = student_global_features["x_norm_patchtokens"].unflatten(
            0, (self.num_global_crops, batch_size)
        )
        if student_local_logits:
            student_local = {
                "cls_pre_head": student_local_features["x_norm_clstoken"].unflatten(0, (len(local_views), batch_size)),
                "patch_pre_head": student_local_features["x_norm_patchtokens"].unflatten(
                    0, (len(local_views), batch_size)
                ),
                "cls_after_head": student_local_logits[0].unflatten(0, (len(local_views), batch_size)),
            }
        else:
            student_local = None

        student_global_output: dict[str, torch.Tensor] = {
            "cls_pre_head": student_global_cls,
            "patch_pre_head": student_global_patch,
            "cls_after_head": student_global_logits.unflatten(0, (self.num_global_crops, batch_size)),
        }
        if mask_indices_list is not None:
            student_masked_patch_pre_head = torch.index_select(
                student_global_features["x_norm_patchtokens"].flatten(0, 1),
                dim=0,
                index=mask_indices_list,
            )
            student_global_output["masked_patch_pre_head"] = student_masked_patch_pre_head
            student_global_output["masked_patch_after_head"] = self.student.ibot_head(student_masked_patch_pre_head)

        split_index = self.student.backbone.num_register_tokens + 1
        if hidden_distill_blocks is not None:
            student_hidden_stack = self._apply_hidden_distill_predictors(student_hidden_stack, hidden_distill_blocks)
            student_global_output["hidden_states"] = student_hidden_stack
            student_global_output["hidden_num_extra_tokens"] = split_index

        with torch.no_grad():
            if hidden_distill_blocks is not None:
                teacher_features, teacher_hidden_stack = self.teacher.backbone(
                    student_global_batch,
                    masks=None,
                    is_training=True,
                    hidden_distill_blocks=hidden_distill_blocks,
                )
            else:
                teacher_features = self.teacher.backbone(student_global_batch, masks=None, is_training=True)
                teacher_hidden_stack = None
            assert isinstance(teacher_features, dict)
            teacher_global_output: dict[str, torch.Tensor] = {
                "cls_pre_head": teacher_features["x_norm_clstoken"].unflatten(0, (self.num_global_crops, batch_size)),
                "patch_pre_head": teacher_features["x_norm_patchtokens"].unflatten(
                    0, (self.num_global_crops, batch_size)
                ),
                "cls_after_head": self.teacher.head(teacher_features["x_norm_clstoken"]).unflatten(
                    0, (self.num_global_crops, batch_size)
                ),
            }
            if mask_indices_list is not None:
                teacher_masked_patch_pre_head = torch.index_select(
                    teacher_features["x_norm_patchtokens"].flatten(0, 1),
                    dim=0,
                    index=mask_indices_list,
                )
                teacher_global_output["masked_patch_pre_head"] = teacher_masked_patch_pre_head
                teacher_global_output["masked_patch_after_head"] = self.teacher.ibot_head(teacher_masked_patch_pre_head)

            if hidden_distill_blocks is not None:
                teacher_global_output["hidden_states"] = teacher_hidden_stack
                teacher_global_output["hidden_num_extra_tokens"] = split_index

        return {
            "student_global": student_global_output,
            "teacher_global": teacher_global_output,
            "student_local": student_local,
        }

    @classmethod
    def from_backbones(
        cls,
        student,
        teacher=None,
        out_dim=65536,
        hidden_dim=None,
        bottleneck_dim=256,
        momentum=0.996,
        hidden_distill_blocks: tuple[int, ...] | None = None,
        hidden_distill_use_predictor=False,
    ):
        teacher = deepcopy(student) if teacher is None else teacher
        student_model = DINO(student, out_dim=out_dim, hidden_dim=hidden_dim, bottleneck_dim=bottleneck_dim)
        teacher_model = DINO(teacher, out_dim=out_dim, hidden_dim=hidden_dim, bottleneck_dim=bottleneck_dim)
        student_model.head.init_weights()
        return cls(
            student=student_model,
            teacher=teacher_model,
            momentum=momentum,
            hidden_distill_blocks=hidden_distill_blocks,
            hidden_distill_use_predictor=hidden_distill_use_predictor,
        )
