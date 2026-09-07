"""Evaluation runtime helpers for DINOSAR."""

import json
import random
from collections.abc import Mapping
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import InterpolationMode

from dinosar.common import cosine_schedule
from dinosar.distributed import get_runtime_device
from dinosar.model import load_backbone_from_config


class ResizeAndPad:
    """Resize keeping aspect ratio then pad to a square with zeros."""

    def __init__(self, size, interpolation=InterpolationMode.BICUBIC):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        w, h = img.size
        scale = self.size / max(w, h)
        new_w, new_h = round(w * scale), round(h * scale)
        img = transforms.functional.resize(img, [new_h, new_w], interpolation=self.interpolation, antialias=True)
        pad_left = (self.size - new_w) // 2
        pad_top = (self.size - new_h) // 2
        pad_right = self.size - new_w - pad_left
        pad_bottom = self.size - new_h - pad_top
        return transforms.functional.pad(img, [pad_left, pad_top, pad_right, pad_bottom], fill=0)


def build_eval_transform(
    *,
    crop_size,
    normalize_mean,
    normalize_std,
    apply_normalize=True,
    num_output_channels=1,
):
    """Build the deterministic grayscale eval transform.

    The image is resized so its longest side equals *crop_size* while
    preserving the aspect ratio, then padded with zeros to a
    ``(crop_size, crop_size)`` square.
    """

    transform_steps = [
        transforms.Grayscale(num_output_channels=num_output_channels),
        ResizeAndPad(crop_size),
        transforms.ToTensor(),
    ]
    if apply_normalize:
        mean = [normalize_mean] * num_output_channels if isinstance(normalize_mean, (float, int)) else normalize_mean
        std = [normalize_std] * num_output_channels if isinstance(normalize_std, (float, int)) else normalize_std
        transform_steps.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(transform_steps)


def _build_eval_transform_from_config(config):
    """Build the eval transform from a resolved OmegaConf config."""
    return build_eval_transform(
        crop_size=int(config.data.crop_size),
        normalize_mean=float(config.data.normalize_mean),
        normalize_std=float(config.data.normalize_std),
        apply_normalize=bool(OmegaConf.select(config, "data.apply_normalize", default=True)),
        num_output_channels=int(OmegaConf.select(config, "data.num_output_channels", default=1)),
    )


def build_classification_datasets(
    root,
    train_split,
    val_split,
    transform=None,
    val_transform=None,
):
    """Build ImageFolder datasets for train and validation splits."""

    root = Path(root)
    train_dataset = ImageFolder(root / train_split, transform=transform)
    val_dataset = ImageFolder(root / val_split, transform=transform if val_transform is None else val_transform)
    if train_dataset.classes != val_dataset.classes or train_dataset.class_to_idx != val_dataset.class_to_idx:
        raise ValueError(
            f"Train and val class vocabularies must match: train={train_dataset.classes!r}, val={val_dataset.classes!r}"
        )
    return train_dataset, val_dataset


def build_eval_loader(dataset, batch_size, num_workers, *, shuffle=False, seed=None):
    """Build a deterministic eval dataloader."""

    generator = None
    if shuffle and seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        drop_last=False,
        generator=generator,
    )


def _dataset_targets(dataset):
    if hasattr(dataset, "targets"):
        return [int(label) for label in dataset.targets]
    if isinstance(dataset, Subset):
        parent_targets = _dataset_targets(dataset.dataset)
        return [parent_targets[index] for index in dataset.indices]
    raise TypeError(f"Unsupported dataset type for target extraction: {type(dataset)!r}")


def build_few_shot_subset(dataset, *, num_shots, seed):
    """Select a deterministic per-class subset from an ImageFolder dataset."""

    if num_shots <= 0:
        raise ValueError("num_shots must be positive")

    by_class: dict[int, list[int]] = {}
    for index, label in enumerate(_dataset_targets(dataset)):
        by_class.setdefault(label, []).append(index)

    generator = torch.Generator()
    generator.manual_seed(seed)

    selected_indices: list[int] = []
    for label in sorted(by_class):
        indices = by_class[label]
        if len(indices) < num_shots:
            raise ValueError(f"class {label} has only {len(indices)} samples, need {num_shots}")
        order = torch.randperm(len(indices), generator=generator).tolist()
        selected_indices.extend(indices[i] for i in order[:num_shots])

    return Subset(dataset, sorted(selected_indices))


def build_protocol_train_val_subsets(dataset, *, val_ratio, seed):
    """Split an ImageFolder dataset into deterministic stratified train/val subsets."""

    if val_ratio <= 0 or val_ratio >= 1:
        raise ValueError("val_ratio must be in the range (0, 1)")

    by_class: dict[int, list[int]] = {}
    for index, label in enumerate(_dataset_targets(dataset)):
        by_class.setdefault(label, []).append(index)

    rng = random.Random(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []

    for label in sorted(by_class):
        indices = list(by_class[label])
        rng.shuffle(indices)
        num_val = round(len(indices) * val_ratio)
        if num_val <= 0:
            raise ValueError(f"class {label} has too few samples for val_ratio={val_ratio}")
        val_indices.extend(indices[:num_val])
        train_indices.extend(indices[num_val:])

    return Subset(dataset, sorted(train_indices)), Subset(dataset, sorted(val_indices))


def _resolve_model_config(model_config):
    """Resolve a model config to a plain dict."""

    if isinstance(model_config, DictConfig):
        return OmegaConf.to_container(model_config, resolve=True)
    return dict(model_config)


def load_eval_backbone(
    model_config,
    checkpoint_path,
    *,
    checkpoint_type="backbone",
    freeze=True,
):
    """Build a frozen DINOSAR backbone and load evaluation weights."""

    resolved = _resolve_model_config(model_config)
    for key in ("checkpoint_type", "checkpoint_path", "init_checkpoint"):
        resolved.pop(key, None)
    backbone = load_backbone_from_config(
        resolved,
        Path(checkpoint_path),
        checkpoint_type=checkpoint_type,
        initialize_weights=False,
        strict=True,
    )
    backbone.eval()
    backbone.requires_grad_(not freeze)
    return backbone


def load_eval_backbone_from_config(model_config, *, freeze=True):
    """Build a frozen backbone from a config section that includes checkpoint fields."""

    resolved = _resolve_model_config(model_config)
    checkpoint_path = resolved.get("checkpoint_path")
    if not checkpoint_path:
        raise ValueError("model.checkpoint_path must be set for eval backbone loading")
    checkpoint_type = resolved.get("checkpoint_type", "backbone")
    return load_eval_backbone(
        resolved,
        checkpoint_path,
        checkpoint_type=checkpoint_type,
        freeze=freeze,
    )


def extract_features(
    backbone,
    loader,
    *,
    device,
    feature_key="x_norm_clstoken",
    pool_patches=False,
):
    """Extract frozen features and labels from a dataloader.

    When pool_patches is True, extracts global-average-pooled patch features
    instead of using feature_key.
    """

    backbone = backbone.to(device)
    backbone.eval()
    features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    use_amp = device.type == "cuda"

    with torch.inference_mode():
        for batch in loader:
            images, batch_labels = batch
            images = images.to(device, non_blocking=True)
            with torch.autocast("cuda", enabled=use_amp):
                outputs = backbone.forward_features(images)
            if not isinstance(outputs, Mapping):
                raise TypeError("Expected a feature dictionary from forward_features")
            if pool_patches:
                features.append(outputs["x_norm_patchtokens"].float().mean(dim=1).cpu())
            else:
                features.append(outputs[feature_key].float().cpu())
            labels.append(batch_labels)

    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


@torch.inference_mode()
def _evaluate_predictions(forward_fn, loader, *, device, use_amp=False):
    """Compute cross-entropy loss and top-1 accuracy over a dataloader.

    The caller is responsible for setting model(s) to eval mode.
    *forward_fn* maps a single input batch to logits.
    """
    total_loss = 0.0
    total_samples = 0
    correct = 0
    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=use_amp):
            logits = forward_fn(inputs)
            loss = F.cross_entropy(logits, labels)
        batch_size = labels.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        correct += logits.float().argmax(dim=1).eq(labels).sum().item()
    if total_samples == 0:
        raise ValueError("loader must not be empty")
    return {"top1": correct / total_samples, "loss": total_loss / total_samples}


def run_knn_eval(
    *,
    train_features,
    train_labels,
    val_features,
    val_labels,
    ks,
    temperature,
    normalize_features,
    device=None,
    query_chunk_size=None,
    max_similarity_elements=100_000_000,
):
    """Run weighted k-NN on pre-extracted train and validation features."""

    if not ks:
        raise ValueError("ks must contain at least one value")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    train_features = train_features.float()
    val_features = val_features.float()
    if normalize_features:
        train_features = F.normalize(train_features, dim=1)
        val_features = F.normalize(val_features, dim=1)
    train_labels = train_labels.long()
    val_labels = val_labels.long()

    if device is not None:
        train_features = train_features.to(device)
        val_features = val_features.to(device)
        train_labels = train_labels.to(device)
        val_labels = val_labels.to(device)

    num_classes = int(torch.maximum(train_labels.max(), val_labels.max()).item()) + 1
    sorted_ks = sorted({int(k) for k in ks})
    max_k = min(sorted_ks[-1], train_features.shape[0])
    if query_chunk_size is None:
        if max_similarity_elements is None or max_similarity_elements <= 0:
            query_chunk_size = val_features.shape[0]
        else:
            query_chunk_size = max(1, min(val_features.shape[0], max_similarity_elements // train_features.shape[0]))

    correct_top1 = {k: 0 for k in sorted_ks}
    correct_top5 = {k: 0 for k in sorted_ks}
    total = 0

    for start in range(0, val_features.shape[0], query_chunk_size):
        end = min(start + query_chunk_size, val_features.shape[0])
        query_features = val_features[start:end]
        query_labels = val_labels[start:end]
        similarities = query_features @ train_features.T
        all_topk_similarities, all_topk_indices = similarities.topk(max_k, dim=1)

        for k in sorted_ks:
            effective_k = min(k, train_features.shape[0])
            topk_similarities = all_topk_similarities[:, :effective_k]
            topk_indices = all_topk_indices[:, :effective_k]
            topk_labels = train_labels[topk_indices]
            weights = torch.exp(topk_similarities / temperature)

            votes = torch.zeros((query_features.shape[0], num_classes), dtype=weights.dtype, device=weights.device)
            votes.scatter_add_(1, topk_labels, weights)

            top1_predictions = votes.argmax(dim=1)
            correct_top1[k] += top1_predictions.eq(query_labels).sum().item()
            top5_idx = votes.topk(min(5, num_classes), dim=1).indices
            correct_top5[k] += top5_idx.eq(query_labels.unsqueeze(1)).any(dim=1).sum().item()

        total += query_labels.numel()

    metrics: dict[str, float] = {}
    for k in sorted_ks:
        metrics[f"top1_k{k}"] = correct_top1[k] / total
        metrics[f"top5_k{k}"] = correct_top5[k] / total

    return metrics


class LinearProbeHead(nn.Module):
    """A single linear classifier trained on frozen features."""

    def __init__(self, in_dim, num_classes, *, use_batch_norm=False, batch_norm_eps=1e-6, batch_norm_affine=False):
        super().__init__()
        if use_batch_norm:
            self.classifier = nn.Sequential(
                nn.BatchNorm1d(in_dim, affine=batch_norm_affine, eps=batch_norm_eps),
                nn.Linear(in_dim, num_classes),
            )
        else:
            self.classifier = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.classifier(x)


def _build_feature_loader(features, labels, *, shuffle):
    batch_size = min(256, features.shape[0])
    generator = torch.Generator().manual_seed(0) if shuffle else None
    return DataLoader(
        TensorDataset(features, labels),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        pin_memory=False,
        generator=generator,
    )


def _build_optimizer(parameters, *, optimizer_name, lr, weight_decay, momentum=0.9):
    optimizer_name = optimizer_name.lower()
    if optimizer_name == "adamw":
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    if optimizer_name == "adam":
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    if optimizer_name == "sgd":
        return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay, momentum=momentum)
    raise ValueError(f"Unsupported optimizer_name: {optimizer_name}")


def _train_probe(
    *,
    probe,
    train_features,
    train_labels,
    val_features,
    val_labels,
    epochs,
    lr,
    weight_decay,
    device,
    normalize_features=True,
    optimizer_name="sgd",
    momentum=0.9,
    warmup_epochs=10,
):
    """Training loop for the linear probe on frozen features."""

    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if lr <= 0:
        raise ValueError("lr must be positive")

    train_features = train_features.float()
    val_features = val_features.float()
    if normalize_features:
        train_features = F.normalize(train_features, dim=1)
        val_features = F.normalize(val_features, dim=1)
    train_features = train_features.to(device)
    val_features = val_features.to(device)
    train_labels = train_labels.long().to(device)
    val_labels = val_labels.long().to(device)

    train_loader = _build_feature_loader(train_features, train_labels, shuffle=True)
    val_loader = _build_feature_loader(val_features, val_labels, shuffle=False)

    probe = probe.to(device)
    optimizer = _build_optimizer(
        probe.parameters(),
        optimizer_name=optimizer_name,
        lr=lr,
        weight_decay=weight_decay,
        momentum=momentum,
    )

    steps_per_epoch = len(train_loader)
    max_steps = epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    best_top1 = -1.0
    best_epoch = 0
    final_loss = 0.0
    final_top1 = 0.0
    final_train_top1 = 0.0
    global_step = 0

    for epoch in range(1, epochs + 1):
        probe.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0
        for batch_features, batch_labels in train_loader:
            current_lr = cosine_schedule(global_step, max_steps, warmup_steps, peak=lr, end=0.0, start=1e-6)
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr

            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = probe(batch_features)
            loss = F.cross_entropy(logits, batch_labels)
            loss.backward()
            optimizer.step()
            global_step += 1

            batch_size = batch_features.shape[0]
            train_loss_sum += loss.item() * batch_size
            train_correct += logits.detach().argmax(dim=1).eq(batch_labels).sum().item()
            train_total += batch_size

        train_top1 = train_correct / train_total if train_total > 0 else 0.0
        train_loss = train_loss_sum / train_total if train_total > 0 else 0.0

        probe.eval()
        val_metrics = _evaluate_predictions(probe, val_loader, device=device)
        final_loss = val_metrics["loss"]
        final_top1 = val_metrics["top1"]
        final_train_top1 = train_top1
        if final_top1 >= best_top1:
            best_top1 = final_top1
            best_epoch = epoch

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            print(
                f"[epoch {epoch:3d}/{epochs}]  lr={current_lr:.2e}"
                f"  train_loss={train_loss:.4f}  train_top1={train_top1:.4f}"
                f"  val_loss={final_loss:.4f}  val_top1={final_top1:.4f}"
                f"  best_top1={best_top1:.4f}@{best_epoch}"
            )

    return {
        "best_top1": best_top1,
        "best_epoch": float(best_epoch),
        "final_top1": final_top1,
        "final_loss": final_loss,
        "final_train_top1": final_train_top1,
    }


def run_linear_probe(
    *,
    train_features,
    train_labels,
    val_features,
    val_labels,
    epochs,
    lr,
    weight_decay,
    num_classes,
    device,
    normalize_features=True,
    optimizer_name="sgd",
    momentum=0.9,
    warmup_epochs=10,
    use_batch_norm=False,
):
    """Train a linear classifier on frozen features and evaluate it on validation features."""

    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    probe = LinearProbeHead(train_features.shape[1], num_classes, use_batch_norm=use_batch_norm)
    return _train_probe(
        probe=probe,
        train_features=train_features,
        train_labels=train_labels,
        val_features=val_features,
        val_labels=val_labels,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        device=device,
        normalize_features=normalize_features,
        optimizer_name=optimizer_name,
        momentum=momentum,
        warmup_epochs=warmup_epochs,
    )


def _write_metrics(output_dir, metrics):
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    return metrics_path


def _infer_few_shot_protocol(dataset_root):
    from dinosar.finetune import _infer_few_shot_protocol as _impl

    return _impl(dataset_root)


def run_few_shot_finetune(**kwargs):
    from dinosar.finetune import run_few_shot_finetune as _impl

    return _impl(**kwargs)


def _prepare_eval_features(config):
    """Build transform, datasets, loaders, backbone and extract features."""

    print(OmegaConf.to_yaml(config, resolve=True).rstrip())
    transform = _build_eval_transform_from_config(config)
    train_dataset, val_dataset = build_classification_datasets(
        config.data.root,
        config.data.train_split,
        config.data.val_split,
        transform=transform,
    )
    train_loader = build_eval_loader(
        train_dataset,
        batch_size=int(config.eval.batch_size),
        num_workers=int(config.eval.num_workers),
    )
    val_loader = build_eval_loader(
        val_dataset,
        batch_size=int(config.eval.batch_size),
        num_workers=int(config.eval.num_workers),
    )

    backbone = load_eval_backbone_from_config(config.model)
    device = get_runtime_device()
    train_features, train_labels = extract_features(
        backbone,
        train_loader,
        device=device,
        feature_key=str(config.eval.feature_key),
    )
    val_features, val_labels = extract_features(
        backbone,
        val_loader,
        device=device,
        feature_key=str(config.eval.feature_key),
    )
    return train_features, train_labels, val_features, val_labels, device


def run_knn_evaluation(config):
    """Run the k-NN evaluation task from a resolved config."""

    train_features, train_labels, val_features, val_labels, device = _prepare_eval_features(config)
    metrics = run_knn_eval(
        train_features=train_features,
        train_labels=train_labels,
        val_features=val_features,
        val_labels=val_labels,
        ks=[int(value) for value in config.knn.ks],
        temperature=float(config.knn.temperature),
        normalize_features=bool(config.eval.normalize_features),
        device=device,
        query_chunk_size=OmegaConf.select(config, "knn.query_chunk_size", default=None),
    )
    _write_metrics(Path(config.eval.output_dir), metrics)
    return metrics


def run_linear_probe_evaluation(config):
    """Run the linear-probe evaluation task from a resolved config."""

    train_features, train_labels, val_features, val_labels, device = _prepare_eval_features(config)
    num_classes = int(torch.maximum(train_labels.max(), val_labels.max()).item()) + 1
    metrics = run_linear_probe(
        train_features=train_features,
        train_labels=train_labels,
        val_features=val_features,
        val_labels=val_labels,
        epochs=int(config.linear.epochs),
        lr=float(config.linear.lr),
        weight_decay=float(config.linear.weight_decay),
        num_classes=num_classes,
        device=device,
        normalize_features=bool(config.eval.normalize_features),
        optimizer_name=str(OmegaConf.select(config, "linear.optimizer", default="sgd")),
        momentum=float(OmegaConf.select(config, "linear.momentum", default=0.9)),
        warmup_epochs=int(OmegaConf.select(config, "linear.warmup_epochs", default=10)),
        use_batch_norm=bool(OmegaConf.select(config, "linear.use_batch_norm", default=False)),
    )
    _write_metrics(Path(config.eval.output_dir), metrics)
    return metrics


def run_few_shot_evaluation(config):
    from dinosar.finetune import run_few_shot_evaluation as _impl

    return _impl(config)


def run_full_finetune_evaluation(config):
    from dinosar.finetune import run_full_finetune_evaluation as _impl

    return _impl(config)


_EVAL_TASKS = {
    "knn": run_knn_evaluation,
    "linear": run_linear_probe_evaluation,
    "few_shot": run_few_shot_evaluation,
    "full_finetune": run_full_finetune_evaluation,
}


def run_evaluation(config):
    """Dispatch to the evaluation task declared in ``config.eval.task``."""

    task = OmegaConf.select(config, "eval.task", default=None)
    if task is None:
        # Back-compat: infer from presence of a task section.
        for candidate in _EVAL_TASKS:
            if hasattr(config, candidate):
                task = candidate
                break
    if task not in _EVAL_TASKS:
        raise ValueError(f"Unknown eval.task: {task!r}. Valid: {sorted(_EVAL_TASKS)}")
    return _EVAL_TASKS[task](config)


if __name__ == "__main__":
    import argparse

    from dinosar.config import load_eval_config

    parser = argparse.ArgumentParser(description="DINOSAR Evaluation")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "overrides", nargs="*", help="OmegaConf dotlist overrides, e.g. model.checkpoint_path=weights.pth"
    )
    args = parser.parse_args()
    config = load_eval_config(args.config)
    if args.overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(args.overrides))
    run_evaluation(config)
