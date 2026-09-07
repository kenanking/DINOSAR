"""Training runtime for DINOSAR."""

import gc
import json
import time
import uuid
from contextlib import nullcontext
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import trackio
from omegaconf import OmegaConf
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel

from dinosar.common import get_peak_flops, log_message, resolve_schedule
from dinosar.config import next_output_dir, save_resolved_config
from dinosar.data import (
    ImageFolderDataset,
    MultiCropDataset,
    build_content_aware_multicrop_transform,
    build_ibot_collate_fn,
    build_multicrop_transform,
    build_train_dataloader,
)
from dinosar.distributed import (
    destroy_distributed_if_initialized,
    get_runtime_device,
    get_world_size,
    init_distributed_if_needed,
    is_main_process,
    set_seed,
    unwrap_model,
)
from dinosar.losses import DINOLoss, HiddenLayerDistillationLoss, KoLeoLoss, iBOTPatchLoss
from dinosar.model import DINO, StudentTeacherWrapper, load_backbone_from_config
from dinosar.optim import build_optimizer

# Not sent to Trackio (still in JSONL / stdout).
_TRAIN_METRICS_TRACKIO_EXCLUDE = frozenset(
    {
        "train/iter_time",
        "train/data_wait",
        "train/fwd_bwd_time",
        "train/optim_time",
        "train/avg_iter_time",
        "train/imgs_per_sec",
        "train/mfu",
        "train/elapsed",
        "train/eta_seconds",
    }
)

# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------


def prune_checkpoints(checkpoint_dir, keep_last_n):
    """Keep only the newest step checkpoints. ``keep_last_n < 0`` disables pruning."""

    if keep_last_n is None or int(keep_last_n) < 0:
        return
    checkpoint_dir = Path(checkpoint_dir)
    step_checkpoints = sorted(checkpoint_dir.glob("step_*.pth"))
    while len(step_checkpoints) > keep_last_n:
        step_checkpoints.pop(0).unlink(missing_ok=True)


def _atomic_save(obj, path):
    tmp_path = path.with_suffix(".pth.tmp")
    torch.save(obj, tmp_path)
    tmp_path.rename(path)


def save_training_checkpoint(
    checkpoint_dir,
    step,
    model,
    optimizer,
    extra_state=None,
):
    """Save a step checkpoint and refresh last.pth."""

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "global_step": step,
        "model_state_dict": unwrap_model(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if extra_state:
        checkpoint.update(extra_state)

    step_path = checkpoint_dir / f"step_{step:07d}.pth"
    _atomic_save(checkpoint, step_path)
    _atomic_save(checkpoint, checkpoint_dir / "last.pth")
    return step_path


def save_eval_backbone_checkpoint(checkpoint_dir, step, model):
    """Save a backbone-only eval export for the current teacher."""

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    step_path = checkpoint_dir / f"step_{step:07d}.pth"
    _atomic_save(unwrap_model(unwrap_model(model).teacher.backbone).state_dict(), step_path)
    return step_path


def load_training_checkpoint(
    checkpoint_path,
    model,
    optimizer=None,
    scaler=None,
    map_location="cpu",
):
    """Load a full training checkpoint and restore the available state."""

    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint format: {type(checkpoint)!r}")

    unwrap_model(model).load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    return {
        k: v
        for k, v in checkpoint.items()
        if k not in {"model_state_dict", "optimizer_state_dict", "scaler_state_dict"}
    }


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def build_train_loader(config, start_step=0, grad_accum_steps=1):
    """Build the pretraining dataloader from the resolved config."""

    base_dataset = ImageFolderDataset(config.data.data_path)
    aug_cfg = OmegaConf.select(config, "data.augmentation", default=None)
    aug_cfg = OmegaConf.to_container(aug_cfg, resolve=True) if aug_cfg is not None else None

    common_kwargs = dict(
        global_size=config.data.global_crops_size,
        local_size=config.data.local_crops_size,
        global_scale=tuple(config.data.global_crops_scale),
        local_scale=tuple(config.data.local_crops_scale),
        num_local_crops=config.data.num_local_crops,
        normalize_mean=float(config.data.normalize_mean),
        normalize_std=float(config.data.normalize_std),
        augmentation_config=aug_cfg,
    )

    cropping_strategy = OmegaConf.select(config, "data.cropping_strategy", default="random")
    if cropping_strategy == "random":
        multicrop_transform = build_multicrop_transform(**common_kwargs)
    elif cropping_strategy == "content_aware":
        ca_cfg = OmegaConf.select(config, "data.content_aware_crop", default=None)
        ca_cfg = OmegaConf.to_container(ca_cfg, resolve=True) if ca_cfg is not None else {}
        ca_cfg = {k: v for k, v in ca_cfg.items() if v is not None}
        multicrop_transform = build_content_aware_multicrop_transform(**common_kwargs, **ca_cfg)
    else:
        raise ValueError(f"Unknown cropping_strategy: {cropping_strategy!r}")
    train_dataset = MultiCropDataset(base_dataset, multicrop_transform)
    collate_fn = build_ibot_collate_fn(
        mask_ratio_min_max=tuple(config.ibot.mask_ratio_min_max),
        mask_sample_probability=float(config.ibot.mask_sample_probability),
        global_crop_size=int(config.data.global_crops_size),
        patch_size=int(config.model.patch_size),
    )
    return build_train_dataloader(
        train_dataset,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        prefetch_factor=config.data.prefetch_factor,
        pin_memory=config.data.pin_memory and torch.cuda.is_available(),
        collate_fn=collate_fn,
        seed=config.train.seed,
        start_step=start_step,
        grad_accum_steps=grad_accum_steps,
        drop_last=True,
    )


def _unpack_batch(batch, device, view_dtype=None):
    return {
        "views": [
            view.to(
                device=device,
                dtype=view_dtype if view_dtype is not None and view.is_floating_point() else view.dtype,
                non_blocking=True,
            )
            for view in batch["views"]
        ],
        "masks": [None if m is None else m.to(device, non_blocking=True) for m in batch["masks"]],
        "mask_indices_list": batch["mask_indices_list"].to(device, non_blocking=True),
        "masks_weight": batch["masks_weight"].to(device, non_blocking=True),
        "upperbound": batch["upperbound"],
    }


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def build_student_teacher_model(config):
    """Build the default eager student-teacher model from the resolved config."""

    model_config = OmegaConf.to_container(config.model, resolve=True)
    backbone = DINO.build_backbone_from_config(model_config, initialize_weights=True)
    target_blocks = OmegaConf.select(config, "hidden_distill.target_blocks", default=[])
    hidden_distill_blocks = tuple(int(block) for block in target_blocks) if target_blocks else None
    hidden_distill_weight = float(OmegaConf.select(config, "hidden_distill.loss_weight", default=0.0))
    use_predictor = bool(OmegaConf.select(config, "hidden_distill.use_predictor", default=False))
    return StudentTeacherWrapper.from_backbones(
        student=backbone,
        out_dim=config.dino.out_dim,
        bottleneck_dim=256,
        momentum=float(config.train.schedules.momentum.start),
        hidden_distill_blocks=hidden_distill_blocks if hidden_distill_weight > 0.0 and use_predictor else None,
        hidden_distill_use_predictor=hidden_distill_weight > 0.0 and use_predictor,
    )


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def build_loss_fn(config):
    """Build the DINO + KoLeo + iBOT loss module dict."""

    teacher_temp_sched = config.train.schedules.teacher_temp
    warmup_teacher_temp_steps = int(round(float(teacher_temp_sched.warmup_epochs) * int(config.train.iters_per_epoch)))
    return nn.ModuleDict(
        {
            "dino": DINOLoss(
                out_dim=int(config.dino.out_dim),
                warmup_teacher_temp=float(teacher_temp_sched.start),
                teacher_temp=float(teacher_temp_sched.peak),
                student_temp=float(config.dino.student_temp),
                num_global_crops=2,
                warmup_teacher_temp_steps=warmup_teacher_temp_steps,
            ),
            "koleo": KoLeoLoss(),
            "ibot": iBOTPatchLoss(
                patch_out_dim=int(config.ibot.head_n_prototypes),
                student_temp=float(config.ibot.student_temp),
            ),
            "hidden_distill": HiddenLayerDistillationLoss(
                loss_type=str(OmegaConf.select(config, "hidden_distill.loss_type", default="zscore_mse")),
                mask_policy=str(OmegaConf.select(config, "hidden_distill.mask_policy", default="visible")),
                block_weights=OmegaConf.select(config, "hidden_distill.block_weights", default=None),
            ),
        }
    )


def compute_losses(
    outputs,
    batch,
    loss_fn,
    teacher_temp,
    koleo_weight,
    ibot_weight,
    hidden_distill_weight,
    hidden_distill_blocks=None,
):
    """Compute DINO + KoLeo + iBOT + optional hidden distillation losses and return (total_loss, metrics)."""

    teacher_probs_global = loss_fn["dino"].build_teacher_probs(
        outputs["teacher_global"]["cls_after_head"],
        teacher_temp=teacher_temp,
    )
    student_local_cls = None if outputs["student_local"] is None else outputs["student_local"]["cls_after_head"]
    dino_loss_value, dino_metrics = loss_fn["dino"](
        student_output_global=outputs["student_global"]["cls_after_head"],
        teacher_probs_global=teacher_probs_global,
        student_output_local=student_local_cls,
    )

    n_global_crops = outputs["student_global"]["cls_pre_head"].shape[0]
    koleo_loss = sum(loss_fn["koleo"](x) for x in outputs["student_global"]["cls_pre_head"]) / n_global_crops

    teacher_masked_probs = loss_fn["ibot"].build_teacher_probs(
        outputs["teacher_global"]["masked_patch_after_head"],
        teacher_temp=teacher_temp,
    )
    ibot_loss = loss_fn["ibot"](
        outputs["student_global"]["masked_patch_after_head"],
        teacher_masked_probs,
        student_masks_flat=batch["masks"][0],
        n_masked_patches=int(batch["mask_indices_list"].shape[0]),
        masks_weight=batch["masks_weight"],
    )

    ref = outputs["student_global"]["cls_after_head"]
    hidden_distill_loss = torch.zeros((), device=ref.device)
    hidden_distill_metrics: dict[str, float] = {"hidden_distill_loss": 0.0}
    if "hidden_states" in outputs["student_global"]:
        if hidden_distill_weight > 0.0:
            hidden_distill_loss, hidden_distill_metrics = loss_fn["hidden_distill"](
                outputs["student_global"]["hidden_states"],
                outputs["teacher_global"]["hidden_states"],
                masks=batch["masks"][0],
                num_extra_tokens=int(outputs["student_global"].get("hidden_num_extra_tokens", 0)),
                target_blocks=hidden_distill_blocks,
            )
        else:
            # Delayed HD can leave optional predictor parameters unused unless they
            # are anchored in the graph before the distillation weight turns on.
            hidden_distill_loss = sum(
                tensor.float().sum() * 0.0
                for layer_views in outputs["student_global"]["hidden_states"]
                for tensor in layer_views
            )

    total_loss = (
        dino_loss_value
        + (koleo_weight * n_global_crops * koleo_loss)
        + (ibot_weight * ibot_loss)
        + ((hidden_distill_weight if hidden_distill_weight > 0.0 else 1.0) * hidden_distill_loss)
    )
    metrics = {
        **{k: float(v) for k, v in dino_metrics.items()},
        "koleo_loss": float(koleo_loss.item()),
        "ibot_loss": float(ibot_loss.item()),
        **hidden_distill_metrics,
    }
    return total_loss, metrics


def _grad_l2_norm(parameters):
    grads = [parameter.grad.detach() for parameter in parameters if parameter.grad is not None]
    if not grads:
        return 0.0
    total = sum(grad.float().pow(2).sum() for grad in grads)
    return float(torch.sqrt(total).item())


def _clip_student_gradients(model, max_norm):
    """Clip student backbone and heads independently, matching official semantics."""

    unwrapped = unwrap_model(model)
    modules = (
        unwrapped.student.backbone,
        unwrapped.student.head,
        unwrapped.student.ibot_head,
        unwrapped.hidden_distill_predictors,
    )
    for module in modules:
        params = [parameter for parameter in module.parameters() if parameter.grad is not None]
        if params:
            torch.nn.utils.clip_grad_norm_(params, max_norm=max_norm)


def _resolve_hidden_distill_weight(config, base_weight, global_step, iters_per_epoch):
    start_epoch = float(OmegaConf.select(config, "hidden_distill.start_epoch", default=0.0))
    warmup_epochs = float(OmegaConf.select(config, "hidden_distill.warmup_epochs", default=0.0))
    current_epoch = global_step / max(float(iters_per_epoch), 1.0)
    if current_epoch < start_epoch:
        return 0.0
    if warmup_epochs <= 0.0:
        return float(base_weight)
    warmup_progress = min((current_epoch - start_epoch) / warmup_epochs, 1.0)
    return float(base_weight) * max(warmup_progress, 0.0)


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Training setup
# ---------------------------------------------------------------------------


def _setup_model(config, device):
    """Build model, optional backbone init checkpoint, compile, and DDP wrap."""

    main_process = is_main_process()
    model = build_student_teacher_model(config)

    init_ckpt = getattr(config.model, "init_checkpoint", None)
    if init_ckpt:
        loaded_backbone = load_backbone_from_config(
            config.model,
            init_ckpt,
            checkpoint_type="backbone",
            initialize_weights=False,
            strict=False,
        )
        model.student.backbone.load_state_dict(loaded_backbone.state_dict(), strict=True)
        model.teacher.load_state_dict(model.student.state_dict())

    if main_process:
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        log_message(f"[init] model built on {device} | {n_params:.1f}M params")

    model = model.to(device)

    cc = getattr(config.train, "compile", None)
    if cc is not None and cc.enabled:
        log_message("[init] torch.compile enabled  mode=default  dynamic=False")
        model.student.backbone = torch.compile(model.student.backbone, mode="default", dynamic=False)
        model.teacher.backbone = torch.compile(model.teacher.backbone, mode="default", dynamic=False)

    if device.type == "cuda" and dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        idx = device.index or 0
        model = DistributedDataParallel(model, device_ids=[idx], output_device=idx)

    unwrapped = unwrap_model(model)
    return model, unwrapped


def _resolve_grad_accum_steps(config):
    """Return ``train.grad_accum_steps``, auto-derived from ``target_effective_batch`` when null."""

    explicit = getattr(config.train, "grad_accum_steps", None)
    if explicit is not None:
        return int(explicit)

    target = getattr(config.train, "target_effective_batch", None)
    if target is None:
        raise ValueError("set train.grad_accum_steps or train.target_effective_batch")

    micro_batch = int(config.data.batch_size) * get_world_size()
    target = int(target)
    if target < micro_batch or target % micro_batch != 0:
        raise ValueError(
            f"target_effective_batch={target} is not a positive multiple of batch_size × world_size = {micro_batch}"
        )
    derived = target // micro_batch
    config.train.grad_accum_steps = derived
    log_message(f"[init] grad_accum_steps auto-derived = {derived} (target={target} / micro={micro_batch})")
    return derived


def _setup_optimizer(config, model, device):
    """Build losses, AdamW, and AMP scaler."""

    loss_fn = build_loss_fn(config)
    grad_accum_steps = _resolve_grad_accum_steps(config)
    effective_batch = int(config.data.batch_size) * get_world_size() * grad_accum_steps
    peak_lr = float(config.train.schedules.lr.peak)
    end_lr = float(config.train.schedules.lr.end)
    initial_wd = float(config.train.schedules.weight_decay.start)
    log_message(
        f"[init] lr: peak={peak_lr:.2e}  end={end_lr:.2e}"
        f"  (bs={config.data.batch_size}×{get_world_size()}×{grad_accum_steps}={effective_batch})"
    )

    head_betas_cfg = getattr(config.train, "head_betas", None)
    head_betas = tuple(float(b) for b in head_betas_cfg) if head_betas_cfg is not None else None
    head_lr_mult_cfg = float(getattr(config.train, "head_lr_multiplier", 1.0))

    optimizer = build_optimizer(
        model,
        lr=peak_lr,
        weight_decay=initial_wd,
        betas=(float(config.train.adamw_beta1), float(config.train.adamw_beta2)),
        layerwise_decay=config.train.layerwise_decay,
        patch_embed_lr_mult=config.train.patch_embed_lr_mult,
        dino_head_wd_multiplier=config.train.dino_head_wd_multiplier,
        head_betas=head_betas,
        head_lr_multiplier=head_lr_mult_cfg,
    )
    amp_dtype = getattr(torch, str(config.train.amp_dtype))
    use_amp = bool(config.train.use_amp)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp and device.type == "cuda" and amp_dtype == torch.float16)
    return loss_fn, optimizer, scaler, peak_lr, end_lr, grad_accum_steps


def _setup_logging(config, output_dir, resume_state):
    """Resolve run name, init Trackio, and write resolved configs under ``output_dir``."""

    main_process = is_main_process()
    log_dir = output_dir / "logs"
    metrics_jsonl_path = log_dir / "metrics.jsonl"

    cfg_run_name = config.logging.run_name
    if cfg_run_name:
        trackio_run_name = str(cfg_run_name)
    elif resume_state.get("run_name"):
        trackio_run_name = resume_state["run_name"]
    else:
        if main_process:
            suffix = uuid.uuid4().hex[:5]
        else:
            suffix = ""
        if dist.is_available() and dist.is_initialized():
            payload = [suffix]
            dist.broadcast_object_list(payload, src=0)
            suffix = payload[0]
        trackio_run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{suffix}"

    trackio_run = None
    if main_process:
        log_dir.mkdir(parents=True, exist_ok=True)
        save_resolved_config(config, output_dir / "config.yaml")
        resolved_config = OmegaConf.to_container(config, resolve=True)
        resolved_config.setdefault("logging", {})["run_name"] = trackio_run_name
        (log_dir / "config.json").write_text(json.dumps(resolved_config, indent=2, sort_keys=True))
        if config.logging.enabled:
            trackio_run = trackio.init(
                project=config.logging.project_name,
                name=trackio_run_name,
                config=resolved_config,
                embed=False,
                resume="allow",
                auto_log_gpu=torch.cuda.is_available(),
            )
        log_message(f"[init] trackio run: {trackio_run_name}")

    return trackio_run, trackio_run_name, metrics_jsonl_path


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------


def run_training(config):
    """Run the configured training job and return the final trainer state."""

    trackio_run = None
    try:
        init_distributed_if_needed()
        main_process = is_main_process()
        if main_process:
            log_message(OmegaConf.to_yaml(config, resolve=True).rstrip())

        device = get_runtime_device()
        set_seed(config.train.seed)
        model, unwrapped = _setup_model(config, device)

        amp_dtype = getattr(torch, str(config.train.amp_dtype))
        use_amp = bool(config.train.use_amp)

        loss_fn, optimizer, scaler, peak_lr, end_lr, grad_accum_steps = _setup_optimizer(config, model, device)

        # Resume from checkpoint
        resume_state: dict[str, Any] = {}
        resume_path = getattr(config.train, "resume", None)
        if resume_path:
            log_message(f"[init] resuming from {resume_path}")
            resume_state = load_training_checkpoint(resume_path, model, optimizer, scaler, map_location="cpu")
            log_message(f"[init] resumed at global_step={resume_state.get('global_step', 0)}")

        output_dir = Path(config.train.output_dir)
        if not resume_path:
            output_dir = next_output_dir(output_dir)
        checkpoint_dir = output_dir / "ckpt"
        eval_dir = output_dir / "eval"

        trackio_run, trackio_run_name, metrics_jsonl_path = _setup_logging(config, output_dir, resume_state)

        # Unpack config into local variables
        iters_per_epoch = int(config.train.iters_per_epoch)
        max_steps = int(config.train.total_iters)
        clip_grad = config.train.clip_grad
        save_every = int(config.train.save_every_steps)
        log_every = int(config.train.log_every_steps)
        keep_last_n = int(config.train.keep_last_n_checkpoints)
        eval_save_every_cfg = getattr(config.train, "eval_save_every_steps", None)
        eval_save_every = int(eval_save_every_cfg) if eval_save_every_cfg is not None else None
        eval_keep_last_n = int(getattr(config.train, "eval_keep_last_n", -1))
        lr_sched = config.train.schedules.lr
        wd_sched = config.train.schedules.weight_decay
        momentum_sched = config.train.schedules.momentum
        teacher_temp_sched = config.train.schedules.teacher_temp
        freeze_last_layer_steps = int(round(float(lr_sched.get("freeze_last_layer_epochs", 0.0)) * iters_per_epoch))
        koleo_weight = float(config.dino.koleo_loss_weight)
        ibot_weight = float(config.ibot.loss_weight)
        hidden_distill_base_weight = float(OmegaConf.select(config, "hidden_distill.loss_weight", default=0.0))
        target_blocks = OmegaConf.select(config, "hidden_distill.target_blocks", default=[])
        hidden_distill_blocks = tuple(int(b) for b in target_blocks) if target_blocks else None
        forward_hidden_blocks = hidden_distill_blocks if hidden_distill_base_weight > 0 else None
        view_dtype = amp_dtype if use_amp and device.type == "cuda" else None

        # Throughput / MFU estimation
        # Student: 6N per token (fwd 2 + bwd_input 2 + bwd_weight 2) across all views.
        # Teacher: 2N per token (fwd only, no grad) across 2 global views.
        # Global and local crops have different resolutions → different patch counts.
        imgs_per_step = int(config.data.batch_size) * get_world_size() * grad_accum_steps
        student_params = sum(p.numel() for p in unwrapped.student.parameters())
        teacher_params = sum(p.numel() for p in unwrapped.teacher.parameters())
        global_patches = (int(config.data.global_crops_size) // int(config.model.patch_size)) ** 2
        local_patches = (int(config.data.local_crops_size) // int(config.model.patch_size)) ** 2
        num_local_crops = int(config.data.num_local_crops)
        student_tokens_per_img = 2 * global_patches + num_local_crops * local_patches
        student_flops = 6 * student_params * student_tokens_per_img
        teacher_flops = 2 * teacher_params * (2 * global_patches)
        flops_per_step = (student_flops + teacher_flops) * imgs_per_step
        gpu_peak_flops = get_peak_flops(device)

        if grad_accum_steps > 1:
            log_message(
                f"[init] gradient accumulation: {grad_accum_steps} micro-steps  effective_batch={imgs_per_step}"
            )

        # Loop state
        model.train()
        loss_fn.to(device)
        global_step = int(resume_state.get("global_step", 0))
        train_loader = build_train_loader(config, start_step=global_step, grad_accum_steps=grad_accum_steps)
        last_loss = 0.0
        last_lr = float(resume_state.get("last_lr", 0.0))
        last_wd = float(resume_state.get("last_weight_decay", float(wd_sched.start)))
        last_momentum = float(resume_state.get("last_momentum", 0.0))
        last_teacher_temp = float(resume_state.get("last_teacher_temp", 0.0))
        head_lr = float(resume_state.get("head_lr", 0.0))
        total_iter_time = float(resume_state.get("total_iter_time", 0.0))
        timed_steps = int(resume_state.get("timed_steps", 0))
        smooth_loss = float(resume_state.get("smooth_loss", 0.0))

        amp_str = f"amp={amp_dtype}" if use_amp else "amp=off"
        log_message(f"[train] starting from step {global_step}/{max_steps}  {amp_str}  device={device}")

        # ---------------------------------------------------------------
        # Training loop
        # ---------------------------------------------------------------
        iterator = iter(train_loader)
        first_step = True
        while global_step < max_steps:
            step_start = time.perf_counter()

            # Schedules
            last_lr = resolve_schedule(lr_sched, global_step, max_steps, iters_per_epoch)
            head_lr = 0.0 if global_step < freeze_last_layer_steps else last_lr
            last_teacher_temp = resolve_schedule(teacher_temp_sched, global_step, max_steps, iters_per_epoch)
            last_wd = resolve_schedule(wd_sched, global_step, max_steps, iters_per_epoch)
            last_momentum = resolve_schedule(momentum_sched, global_step, max_steps, iters_per_epoch)
            hidden_distill_weight = _resolve_hidden_distill_weight(
                config,
                hidden_distill_base_weight,
                global_step,
                iters_per_epoch,
            )

            for pg in optimizer.param_groups:
                group_lr = head_lr if pg.get("is_last_layer", False) else last_lr
                pg["lr"] = group_lr * float(pg.get("lr_multiplier", 1.0))
                pg["weight_decay"] = last_wd * float(pg.get("wd_multiplier", 1.0))

            # Forward / backward with gradient accumulation
            use_ddp = isinstance(model, DistributedDataParallel)
            optimizer.zero_grad(set_to_none=True)
            data_wait = 0.0
            for micro_step in range(grad_accum_steps):
                no_sync = use_ddp and micro_step < grad_accum_steps - 1
                with model.no_sync() if no_sync else nullcontext():
                    t_data = time.perf_counter()
                    try:
                        batch = next(iterator)
                    except StopIteration as exc:
                        raise RuntimeError("run_training() expects an infinite training loader.") from exc
                    batch_on_device = _unpack_batch(batch, device, view_dtype=view_dtype)
                    data_wait += time.perf_counter() - t_data
                    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                        outputs = model(
                            batch_on_device["views"],
                            masks=batch_on_device["masks"],
                            mask_indices_list=batch_on_device["mask_indices_list"],
                            hidden_distill_blocks=forward_hidden_blocks,
                        )
                        total_loss, loss_metrics = compute_losses(
                            outputs,
                            batch_on_device,
                            loss_fn,
                            last_teacher_temp,
                            koleo_weight,
                            ibot_weight,
                            hidden_distill_weight,
                            hidden_distill_blocks=hidden_distill_blocks,
                        )
                    scaler.scale(total_loss / grad_accum_steps).backward()

            # Optimizer step
            t_optim = time.perf_counter()
            scaler.unscale_(optimizer)
            grad_norm = None
            if clip_grad is not None:
                grad_norm_params = tuple(unwrapped.student.parameters()) + tuple(
                    unwrapped.hidden_distill_predictors.parameters()
                )
                grad_norm = torch.tensor(_grad_l2_norm(grad_norm_params), device=device)
                _clip_student_gradients(model, max_norm=clip_grad)
            scaler.step(optimizer)
            scaler.update()
            unwrapped.update_teacher(last_momentum)
            optim_time = time.perf_counter() - t_optim

            # Bookkeeping
            global_step += 1
            iter_time = time.perf_counter() - step_start
            total_iter_time += iter_time
            timed_steps += 1
            avg_iter = total_iter_time / timed_steps
            eta = avg_iter * max(max_steps - global_step, 0)
            last_loss = float(total_loss.item())
            smooth_loss = 0.9 * smooth_loss + 0.1 * last_loss
            debiased_smooth = smooth_loss / (1 - 0.9**timed_steps)
            cur_imgs_per_sec = imgs_per_step / iter_time
            cur_mfu = 100.0 * flops_per_step / (gpu_peak_flops * get_world_size() * iter_time)

            fwd_bwd_time = iter_time - data_wait - optim_time

            # Tensor-valued loss metrics materialize only on log steps (avoids per-step sync).
            is_log_step = main_process and global_step % log_every == 0
            step_metrics: dict[str, float] = {
                "train/loss": last_loss,
                "train/smooth_loss": debiased_smooth,
                "train/lr": last_lr,
                "train/weight_decay": last_wd,
                "train/momentum": last_momentum,
                "train/teacher_temp": last_teacher_temp,
                "train/hidden_distill_weight": hidden_distill_weight,
                "train/head_lr": head_lr,
                "train/iter_time": iter_time,
                "train/data_wait": data_wait,
                "train/fwd_bwd_time": fwd_bwd_time,
                "train/optim_time": optim_time,
                "train/avg_iter_time": avg_iter,
                "train/imgs_per_sec": cur_imgs_per_sec,
                "train/mfu": cur_mfu,
                "train/elapsed": total_iter_time,
                "train/eta_seconds": eta,
            }
            for k, v in loss_metrics.items():
                if torch.is_tensor(v):
                    if is_log_step:
                        step_metrics[f"train/{k}"] = float(v.item())
                else:
                    step_metrics[f"train/{k}"] = float(v)
            if grad_norm is not None and is_log_step:
                step_metrics["train/grad_norm"] = float(grad_norm.item())

            # Trackio: log_every_steps (same as print / JSONL).
            if is_log_step:
                if trackio_run is not None:
                    trackio_run.log(
                        {k: v for k, v in step_metrics.items() if k not in _TRAIN_METRICS_TRACKIO_EXCLUDE},
                        step=global_step,
                    )
                gn = step_metrics.get("train/grad_norm")
                hd_log = (
                    f"  hd={step_metrics['train/hidden_distill_loss']:.4f}  hd_w={hidden_distill_weight:.3f}"
                    if hidden_distill_base_weight > 0.0
                    else ""
                )
                log_message(
                    f"[step {global_step}/{max_steps}]"
                    f"  loss={debiased_smooth:.4f}"
                    f"  dino={step_metrics['train/dino_total_loss']:.4f}"
                    f"  koleo={step_metrics['train/koleo_loss']:.4f}"
                    f"  ibot={step_metrics['train/ibot_loss']:.4f}"
                    f"{hd_log}"
                    f"  lr={last_lr:.2e}  wd={last_wd:.2e}"
                    f"  time={iter_time:.2f}s  img/s={cur_imgs_per_sec:.0f}  mfu={cur_mfu:.1f}%"
                    f"  elapsed={timedelta(seconds=int(total_iter_time))}"
                    f"  eta={timedelta(seconds=int(eta))}" + (f"  grad_norm={gn:.2f}" if gn is not None else "")
                )
                with open(metrics_jsonl_path, "a") as f:
                    f.write(json.dumps({"step": global_step, **step_metrics}) + "\n")

            # Checkpointing
            is_last_step = global_step >= max_steps
            if main_process and (global_step % save_every == 0 or is_last_step):
                t = time.perf_counter()
                save_training_checkpoint(
                    checkpoint_dir,
                    global_step,
                    model,
                    optimizer,
                    extra_state={
                        "scaler_state_dict": scaler.state_dict(),
                        "run_name": trackio_run_name,
                        "last_lr": last_lr,
                        "last_weight_decay": last_wd,
                        "last_momentum": last_momentum,
                        "last_teacher_temp": last_teacher_temp,
                        "head_lr": head_lr,
                        "total_iter_time": total_iter_time,
                        "timed_steps": timed_steps,
                        "smooth_loss": smooth_loss,
                    },
                )
                prune_checkpoints(checkpoint_dir, keep_last_n=keep_last_n)
                log_message(f"[ckpt] saved step {global_step} to {checkpoint_dir}  ({time.perf_counter() - t:.1f}s)")

            if main_process and eval_save_every is not None and (global_step % eval_save_every == 0 or is_last_step):
                t = time.perf_counter()
                save_eval_backbone_checkpoint(eval_dir, global_step, model)
                prune_checkpoints(eval_dir, keep_last_n=eval_keep_last_n)
                log_message(f"[eval] saved step {global_step} to {eval_dir}  ({time.perf_counter() - t:.1f}s)")

            # GC management
            if first_step:
                gc.collect()
                gc.freeze()
                gc.disable()
                first_step = False
            elif global_step % 1000 == 0:
                gc.collect()

        gc.enable()
        state = {
            "global_step": global_step,
            "last_loss": last_loss,
            "last_lr": last_lr,
            "last_weight_decay": last_wd,
            "last_momentum": last_momentum,
            "last_teacher_temp": last_teacher_temp,
            "head_lr": head_lr,
            "elapsed": total_iter_time,
            "avg_iter_time": (total_iter_time / timed_steps) if timed_steps > 0 else 0.0,
            "eta_seconds": 0.0,
        }
        if main_process and timed_steps > 0:
            log_message(
                f"[done] training complete  steps={global_step}  loss={last_loss:.4f}  elapsed={timedelta(seconds=int(total_iter_time))}"
            )
    finally:
        if trackio_run is not None:
            trackio_run.finish()
        destroy_distributed_if_initialized()
    return state


if __name__ == "__main__":
    import argparse

    from dinosar.config import _DEFAULT_PRETRAIN_CONFIG, load_train_config

    parser = argparse.ArgumentParser(description="DINOSAR Training")
    parser.add_argument("--config", type=str, default=str(_DEFAULT_PRETRAIN_CONFIG))
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--local-rank", "--local_rank", type=int, default=None)
    parser.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides, e.g. train.schedules.lr.peak=1e-4")
    args = parser.parse_args()
    config = load_train_config(config_path=args.config, overrides=args.overrides)
    if args.resume:
        OmegaConf.update(config, "train.resume", args.resume)
    run_training(config)
