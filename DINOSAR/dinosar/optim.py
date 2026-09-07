"""Optimizer and parameter-grouping utilities for DINOSAR."""

from typing import Any

import torch.optim as optim

from dinosar.distributed import unwrap_model


def _get_vit_lr_decay_rate(name, lr_decay_rate, num_layers):
    """Return layer-wise LR decay scale for a backbone parameter name."""

    layer_id = num_layers + 1
    if name.startswith("backbone"):
        if any(t in name for t in (".pos_embed", ".patch_embed", ".mask_token", ".cls_token", ".storage_tokens")):
            layer_id = 0
        elif ".blocks." in name and ".residual." not in name:
            layer_id = int(name[name.find(".blocks.") :].split(".")[2]) + 1
    return lr_decay_rate ** (num_layers + 1 - layer_id)


def _fuse_param_groups(param_groups):
    fused: dict[tuple[float, float, bool], dict[str, Any]] = {}
    for pg in param_groups:
        key = (float(pg["lr_multiplier"]), float(pg["wd_multiplier"]), bool(pg["is_last_layer"]))
        if key not in fused:
            fused[key] = {"params": [], "lr_multiplier": key[0], "wd_multiplier": key[1], "is_last_layer": key[2]}
        fused[key]["params"].append(pg["params"])
    return list(fused.values())


def build_optimizer(
    model,
    lr=1e-4,
    weight_decay=0.04,
    betas=(0.9, 0.999),
    layerwise_decay=0.95,
    patch_embed_lr_mult=0.1,
    dino_head_wd_multiplier=1.0,
    head_betas=None,
    head_lr_multiplier=1.0,
):
    """Build an AdamW optimizer with DINOv3-style param groups."""

    model = unwrap_model(model)
    num_layers = len(model.student.backbone.blocks)

    backbone_groups: list[dict[str, Any]] = []
    head_groups: list[dict[str, Any]] = []

    for raw_name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        name = raw_name.replace("module.", "").replace("_orig_mod.", "")
        if name.startswith("student."):
            name = name[len("student.") :]

        is_head = ".head." in name or name.startswith("head.") or "ibot_head" in name
        lr_mult = _get_vit_lr_decay_rate(name, layerwise_decay, num_layers)
        if "patch_embed" in name:
            lr_mult *= patch_embed_lr_mult
        if is_head:
            lr_mult *= head_lr_multiplier

        is_hidden_distill_predictor = name.startswith("hidden_distill_predictors.")
        use_wd = parameter.ndim > 1 and not (
            name.endswith("bias")
            or "norm" in name
            or "gamma" in name
            or "fourier_w" in name
            or is_hidden_distill_predictor
        )
        wd_mult = 1.0 if use_wd else 0.0
        if is_head:
            wd_mult *= dino_head_wd_multiplier

        group = {
            "params": parameter,
            "lr_multiplier": lr_mult,
            "wd_multiplier": wd_mult,
            "is_last_layer": "last_layer" in name,
        }
        (head_groups if is_head else backbone_groups).append(group)

    fused_backbone = _fuse_param_groups(backbone_groups)
    fused_head = _fuse_param_groups(head_groups)

    all_groups = fused_backbone + fused_head
    on_cuda = any(p.device.type == "cuda" for g in all_groups for p in g["params"])

    if head_betas is not None:
        for g in fused_head:
            g["betas"] = head_betas

    return optim.AdamW(all_groups, lr=lr, weight_decay=weight_decay, betas=betas, fused=on_cuda)


def build_finetune_param_groups(backbone, head, weight_decay, layer_decay):
    """Build parameter groups with BEiT-style layer-wise lr decay for finetuning.

    Supports backbones that expose ``blocks`` either directly or nested under a
    ``model`` attribute (e.g. a backbone wrapper module).
    """

    inner = backbone
    prefix = ""
    if not hasattr(inner, "blocks") and hasattr(inner, "model") and hasattr(inner.model, "blocks"):
        inner = backbone.model
        prefix = "model."

    num_layers = len(inner.blocks) + 1
    layer_scales = [layer_decay ** (num_layers - i) for i in range(num_layers + 1)]

    no_wd_names: set[str] = set()
    if hasattr(inner, "no_weight_decay"):
        no_wd_names = {f"{prefix}{n}" for n in inner.no_weight_decay()}

    def _get_layer_id(name):
        name_inner = name[len(prefix) :] if name.startswith(prefix) else name
        if name_inner in ("cls_token", "pos_embed"):
            return 0
        if name_inner.startswith("patch_embed"):
            return 0
        if name_inner.startswith("blocks."):
            return int(name_inner.split(".")[1]) + 1
        return num_layers

    param_groups: dict[str, dict] = {}
    for n, p in backbone.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or n in no_wd_names:
            g_decay, wd = "no_decay", 0.0
        else:
            g_decay, wd = "decay", weight_decay
        layer_id = _get_layer_id(n)
        key = f"backbone_layer_{layer_id}_{g_decay}"
        if key not in param_groups:
            param_groups[key] = {"lr_scale": layer_scales[layer_id], "weight_decay": wd, "params": [], "role": "backbone"}
        param_groups[key]["params"].append(p)

    head_decay = [p for _, p in head.named_parameters() if p.requires_grad and p.ndim > 1]
    head_no_decay = [p for _, p in head.named_parameters() if p.requires_grad and p.ndim <= 1]
    if head_decay:
        param_groups["head_decay"] = {
            "lr_scale": 1.0,
            "weight_decay": weight_decay,
            "params": head_decay,
            "role": "head",
        }
    if head_no_decay:
        param_groups["head_no_decay"] = {
            "lr_scale": 1.0,
            "weight_decay": 0.0,
            "params": head_no_decay,
            "role": "head",
        }

    return list(param_groups.values())
