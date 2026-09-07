"""Few-shot fine-tuning utilities for DINOSAR evaluation."""

from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from dinosar.common import cosine_schedule
from dinosar.distributed import get_runtime_device, set_seed
from dinosar.evaluate import (
    ResizeAndPad,
    _build_eval_transform_from_config,
    _evaluate_predictions,
    _write_metrics,
    build_classification_datasets,
    build_eval_loader,
    build_few_shot_subset,
    build_protocol_train_val_subsets,
    load_eval_backbone_from_config,
)
from dinosar.optim import build_finetune_param_groups


def _infer_few_shot_protocol(dataset_root):
    name = Path(dataset_root).name.lower().replace("-", "_")
    if name in {"mstar", "mstar_soc", "mstar_soc10"}:
        return "mstar"
    if name in {"fusar_ship", "new_fusar"}:
        return "fusar_ship"
    if name == "sar_acd":
        return "sar_acd"
    raise ValueError(f"Unsupported few-shot benchmark root: {dataset_root!r}")


def _resolve_protocol_split_dirs(config, protocol):
    root = Path(config.data.root)
    configured_train = root / config.data.train_split
    configured_val = root / config.data.val_split

    if protocol in ("mstar", "fusar_ship"):
        return configured_train.name, configured_val.name
    if protocol == "sar_acd":
        return None, None
    raise ValueError(f"Unsupported few-shot protocol: {protocol}")


def _build_few_shot_protocol_datasets(config, *, transform, seed):
    root = Path(config.data.root)
    protocol = OmegaConf.select(config, "few_shot.protocol", default=None) or _infer_few_shot_protocol(root)
    train_split, val_split = _resolve_protocol_split_dirs(config, protocol)

    if protocol == "sar_acd":
        full_dataset = ImageFolder(root, transform=transform)
        return build_protocol_train_val_subsets(full_dataset, val_ratio=0.9, seed=seed)

    train_dataset, test_dataset = build_classification_datasets(root, train_split, val_split, transform=transform)
    protocol_train_dataset, _ = build_protocol_train_val_subsets(train_dataset, val_ratio=0.2, seed=seed)
    return protocol_train_dataset, test_dataset


def _build_finetune_train_transform_from_config(config):
    """Build the train transform for full fine-tuning."""

    crop_size = int(config.data.crop_size)
    num_output_channels = int(OmegaConf.select(config, "data.num_output_channels", default=1))
    transform_steps = [
        transforms.Grayscale(num_output_channels=num_output_channels),
        ResizeAndPad(crop_size),
        transforms.RandomHorizontalFlip(
            p=float(OmegaConf.select(config, "full_finetune.horizontal_flip_p", default=0.1))
        ),
        transforms.RandomVerticalFlip(
            p=float(OmegaConf.select(config, "full_finetune.vertical_flip_p", default=0.1))
        ),
        transforms.ToTensor(),
    ]

    if bool(OmegaConf.select(config, "data.apply_normalize", default=True)):
        normalize_mean = float(config.data.normalize_mean)
        normalize_std = float(config.data.normalize_std)
        mean = [normalize_mean] * num_output_channels
        std = [normalize_std] * num_output_channels
        transform_steps.append(transforms.Normalize(mean=mean, std=std))

    return transforms.Compose(transform_steps)


def _resolve_inner_vit(backbone):
    """Return the module that owns ViT blocks for native and wrapped backbones."""

    if hasattr(backbone, "blocks"):
        return backbone
    if hasattr(backbone, "model") and hasattr(backbone.model, "blocks"):
        return backbone.model
    return None


def _set_module_requires_grad(module, requires_grad):
    for parameter in module.parameters():
        parameter.requires_grad = requires_grad


def _apply_finetune_freezing(
    backbone,
    *,
    freeze_backbone=False,
    freeze_patch_embed=False,
    freeze_blocks=0,
):
    """Apply static fine-tuning freezes before optimizer construction."""

    if freeze_blocks < 0:
        raise ValueError("freeze_blocks must be non-negative")

    if freeze_backbone:
        _set_module_requires_grad(backbone, False)
        return

    inner = _resolve_inner_vit(backbone)
    if inner is None:
        if freeze_patch_embed or freeze_blocks:
            raise ValueError("Backbone does not expose ViT patch_embed/blocks for partial freezing")
        return

    if freeze_patch_embed and hasattr(inner, "patch_embed"):
        _set_module_requires_grad(inner.patch_embed, False)

    for block in list(inner.blocks)[:freeze_blocks]:
        _set_module_requires_grad(block, False)


def _l2_sp_penalty(backbone, reference_parameters):
    penalty = None
    for name, parameter in backbone.named_parameters():
        reference = reference_parameters.get(name)
        if reference is None or not parameter.requires_grad:
            continue
        value = (parameter - reference).pow(2).sum()
        penalty = value if penalty is None else penalty + value
    if penalty is None:
        return None
    return penalty


def run_few_shot_finetune(
    *,
    backbone,
    train_loader,
    test_loader,
    epochs,
    lr,
    weight_decay,
    warmup_epochs,
    warmup_start_lr,
    num_classes,
    device,
    label_smoothing=0.1,
    min_lr=1e-6,
    layer_decay=0.75,
    log_every_epochs=None,
    freeze_backbone=False,
    freeze_patch_embed=False,
    freeze_blocks=0,
    head_warmup_epochs=0,
    backbone_lr_scale=1.0,
    l2_sp_weight=0.0,
):
    """Fine-tune backbone + linear head for few-shot evaluation.

    Uses BEiT-style layer-wise lr decay, label smoothing, and cosine schedule
    with ``min_lr`` floor by default.
    """

    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if lr <= 0:
        raise ValueError("lr must be positive")
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    if len(train_loader) == 0:
        raise ValueError("train_loader must not be empty")
    if head_warmup_epochs < 0:
        raise ValueError("head_warmup_epochs must be non-negative")
    if backbone_lr_scale < 0:
        raise ValueError("backbone_lr_scale must be non-negative")
    if l2_sp_weight < 0:
        raise ValueError("l2_sp_weight must be non-negative")

    backbone = backbone.to(device)
    _apply_finetune_freezing(
        backbone,
        freeze_backbone=freeze_backbone,
        freeze_patch_embed=freeze_patch_embed,
        freeze_blocks=freeze_blocks,
    )
    head = nn.Linear(backbone.embed_dim, num_classes).to(device)
    nn.init.trunc_normal_(head.weight, std=2e-5)
    nn.init.zeros_(head.bias)

    if layer_decay < 1.0:
        param_groups = build_finetune_param_groups(backbone, head, weight_decay, layer_decay)
        for param_group in param_groups:
            if param_group.get("role") == "backbone":
                param_group["lr_scale"] *= backbone_lr_scale
        optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.999))
    else:
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": [parameter for parameter in backbone.parameters() if parameter.requires_grad],
                    "lr_scale": backbone_lr_scale,
                },
                {"params": list(head.parameters()), "lr_scale": 1.0},
            ],
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    l2_sp_reference = {}
    if l2_sp_weight > 0:
        l2_sp_reference = {
            name: parameter.detach().clone()
            for name, parameter in backbone.named_parameters()
            if parameter.requires_grad
        }

    warmup_backbone_parameters = [parameter for parameter in backbone.parameters() if parameter.requires_grad]
    if head_warmup_epochs > 0:
        for parameter in warmup_backbone_parameters:
            parameter.requires_grad = False

    steps_per_epoch = len(train_loader)
    max_steps = epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch
    global_step = 0
    use_amp = device.type == "cuda"

    for epoch in range(1, epochs + 1):
        if head_warmup_epochs > 0 and epoch == head_warmup_epochs + 1:
            for parameter in warmup_backbone_parameters:
                parameter.requires_grad = True
        warmup_active = head_warmup_epochs > 0 and epoch <= head_warmup_epochs
        if warmup_active or not any(parameter.requires_grad for parameter in backbone.parameters()):
            backbone.eval()
        else:
            backbone.train()
        head.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0
        for images, labels in train_loader:
            current_lr = cosine_schedule(
                global_step,
                max_steps,
                warmup_steps,
                peak=lr,
                end=min_lr,
                start=warmup_start_lr,
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr * param_group.get("lr_scale", 1.0)

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=use_amp):
                outputs = backbone.forward_features(images)
                logits = head(outputs["x_norm_clstoken"])
                loss = criterion(logits, labels)
                if l2_sp_weight > 0:
                    l2_sp = _l2_sp_penalty(backbone, l2_sp_reference)
                    if l2_sp is not None:
                        loss = loss + l2_sp_weight * l2_sp
            loss.backward()
            optimizer.step()

            batch_size = labels.shape[0]
            train_loss_sum += loss.item() * batch_size
            train_correct += logits.detach().float().argmax(dim=1).eq(labels).sum().item()
            train_total += batch_size
            global_step += 1

        if log_every_epochs and (epoch == 1 or epoch % log_every_epochs == 0 or epoch == epochs):
            train_loss = train_loss_sum / train_total
            train_top1 = train_correct / train_total
            print(
                f"[epoch {epoch:3d}/{epochs}]  lr={current_lr:.2e}"
                f"  train_loss={train_loss:.4f}  train_top1={train_top1:.4f}",
                flush=True,
            )

    backbone.eval()
    head.eval()

    def _forward(images):
        return head(backbone.forward_features(images)["x_norm_clstoken"])

    return _evaluate_predictions(_forward, test_loader, device=device, use_amp=use_amp)


def run_few_shot_evaluation(config):
    """Run the few-shot benchmark using backbone finetuning with layer decay."""

    print(OmegaConf.to_yaml(config, resolve=True).rstrip())
    transform = _build_eval_transform_from_config(config)
    device = get_runtime_device()
    batch_size = int(OmegaConf.select(config, "few_shot.batch_size", default=config.eval.batch_size))
    num_workers = int(config.eval.num_workers)

    base_backbone = load_eval_backbone_from_config(config.model, freeze=False)
    pristine_state = {k: v.detach().clone() for k, v in base_backbone.state_dict().items()}

    seed_results: dict[str, dict[str, float]] = {}
    top1_values: list[float] = []
    for seed_value in config.few_shot.seeds:
        seed = int(seed_value)
        set_seed(seed)
        protocol_train_dataset, test_dataset = _build_few_shot_protocol_datasets(config, transform=transform, seed=seed)
        train_subset = build_few_shot_subset(
            protocol_train_dataset, num_shots=int(config.few_shot.num_shots), seed=seed
        )
        train_loader = build_eval_loader(
            train_subset, batch_size=batch_size, num_workers=num_workers, shuffle=True, seed=seed
        )
        test_loader = build_eval_loader(test_dataset, batch_size=batch_size, num_workers=num_workers)

        base_backbone.load_state_dict(pristine_state)
        num_classes = (
            len(test_dataset.dataset.classes) if isinstance(test_dataset, Subset) else len(test_dataset.classes)
        )
        metrics = run_few_shot_finetune(
            backbone=base_backbone,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=int(config.few_shot.epochs),
            lr=float(config.few_shot.lr),
            weight_decay=float(config.few_shot.weight_decay),
            warmup_epochs=int(OmegaConf.select(config, "few_shot.warmup_epochs", default=5)),
            warmup_start_lr=float(OmegaConf.select(config, "few_shot.warmup_start_lr", default=0.0)),
            num_classes=num_classes,
            device=device,
            label_smoothing=float(OmegaConf.select(config, "few_shot.label_smoothing", default=0.1)),
            min_lr=float(OmegaConf.select(config, "few_shot.min_lr", default=1e-6)),
            layer_decay=float(OmegaConf.select(config, "few_shot.layer_decay", default=0.75)),
        )
        seed_results[str(seed)] = metrics
        top1_values.append(metrics["top1"])

    top1_tensor = torch.tensor(top1_values, dtype=torch.float32)
    summary = {
        "top1_mean": top1_tensor.mean().item(),
        "top1_std": top1_tensor.std(unbiased=False).item(),
    }
    metrics = {
        "num_shots": int(config.few_shot.num_shots),
        "per_seed": seed_results,
        "seeds": [int(seed) for seed in config.few_shot.seeds],
        "summary": summary,
    }
    _write_metrics(Path(config.eval.output_dir), metrics)
    return metrics


def run_full_finetune_evaluation(config):
    """Run full-train-set fine-tuning using the few-shot optimizer protocol."""

    print(OmegaConf.to_yaml(config, resolve=True).rstrip())
    train_transform = _build_finetune_train_transform_from_config(config)
    test_transform = _build_eval_transform_from_config(config)
    device = get_runtime_device()
    batch_size = int(OmegaConf.select(config, "full_finetune.batch_size", default=config.eval.batch_size))
    num_workers = int(config.eval.num_workers)

    base_backbone = load_eval_backbone_from_config(config.model, freeze=False)
    pristine_state = {k: v.detach().clone() for k, v in base_backbone.state_dict().items()}

    seed_results: dict[str, dict[str, float]] = {}
    top1_values: list[float] = []
    for seed_value in config.full_finetune.seeds:
        seed = int(seed_value)
        set_seed(seed)
        train_dataset, test_dataset = build_classification_datasets(
            config.data.root,
            config.data.train_split,
            config.data.val_split,
            transform=train_transform,
            val_transform=test_transform,
        )
        train_loader = build_eval_loader(
            train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, seed=seed
        )
        test_loader = build_eval_loader(test_dataset, batch_size=batch_size, num_workers=num_workers)

        base_backbone.load_state_dict(pristine_state)
        metrics = run_few_shot_finetune(
            backbone=base_backbone,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=int(config.full_finetune.epochs),
            lr=float(config.full_finetune.lr),
            weight_decay=float(config.full_finetune.weight_decay),
            warmup_epochs=int(OmegaConf.select(config, "full_finetune.warmup_epochs", default=5)),
            warmup_start_lr=float(OmegaConf.select(config, "full_finetune.warmup_start_lr", default=0.0)),
            num_classes=len(test_dataset.classes),
            device=device,
            label_smoothing=float(OmegaConf.select(config, "full_finetune.label_smoothing", default=0.1)),
            min_lr=float(OmegaConf.select(config, "full_finetune.min_lr", default=1e-6)),
            layer_decay=float(OmegaConf.select(config, "full_finetune.layer_decay", default=0.75)),
            log_every_epochs=int(OmegaConf.select(config, "full_finetune.log_every_epochs", default=1)),
            freeze_backbone=bool(OmegaConf.select(config, "full_finetune.freeze_backbone", default=False)),
            freeze_patch_embed=bool(OmegaConf.select(config, "full_finetune.freeze_patch_embed", default=False)),
            freeze_blocks=int(OmegaConf.select(config, "full_finetune.freeze_blocks", default=0)),
            head_warmup_epochs=int(OmegaConf.select(config, "full_finetune.head_warmup_epochs", default=0)),
            backbone_lr_scale=float(OmegaConf.select(config, "full_finetune.backbone_lr_scale", default=1.0)),
            l2_sp_weight=float(OmegaConf.select(config, "full_finetune.l2_sp_weight", default=0.0)),
        )
        seed_results[str(seed)] = metrics
        top1_values.append(metrics["top1"])

    top1_tensor = torch.tensor(top1_values, dtype=torch.float32)
    summary = {
        "top1_mean": top1_tensor.mean().item(),
        "top1_std": top1_tensor.std(unbiased=False).item(),
    }
    metrics = {
        "protocol": "full_finetune",
        "train_split": str(config.data.train_split),
        "val_split": str(config.data.val_split),
        "finetune_options": {
            "freeze_backbone": bool(OmegaConf.select(config, "full_finetune.freeze_backbone", default=False)),
            "freeze_patch_embed": bool(OmegaConf.select(config, "full_finetune.freeze_patch_embed", default=False)),
            "freeze_blocks": int(OmegaConf.select(config, "full_finetune.freeze_blocks", default=0)),
            "head_warmup_epochs": int(OmegaConf.select(config, "full_finetune.head_warmup_epochs", default=0)),
            "backbone_lr_scale": float(OmegaConf.select(config, "full_finetune.backbone_lr_scale", default=1.0)),
            "l2_sp_weight": float(OmegaConf.select(config, "full_finetune.l2_sp_weight", default=0.0)),
        },
        "per_seed": seed_results,
        "seeds": [int(seed) for seed in config.full_finetune.seeds],
        "summary": summary,
    }
    _write_metrics(Path(config.eval.output_dir), metrics)
    return metrics
