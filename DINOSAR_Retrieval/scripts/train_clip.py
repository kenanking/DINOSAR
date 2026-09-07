"""Unified-protocol CLIP training for one vision tower row.

AdamW with open_clip optimizer defaults, linear warmup + cosine decay, bf16
autocast, gradient clipping, gradient checkpointing, and per-epoch full
6-metric evaluation on the official 5K file. last.pth is rewritten every
epoch so a crashed run resumes without losing epochs; best.pth tracks the
best MeanRecall.
"""

import argparse
import json
import math
import random
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm

from openclip_retrieval.clip_model import build_clip_model, load_clip_checkpoint
from openclip_retrieval.config import ExperimentConfig, load_training_config
from openclip_retrieval.data import (
    build_eval_loader,
    build_tokenizer,
    build_train_loader,
    eval_dataset,
    train_dataset,
)
from openclip_retrieval.evaluation import evaluate_retrieval_model
from openclip_retrieval.losses import clip_loss
from openclip_retrieval.metrics import format_metrics


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="configs/train/<TOWER>.py")
    parser.add_argument("--resume", choices=["auto", "always", "never"], default="auto")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def lr_multiplier(step: int, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def build_optimizer(model: nn.Module, config: ExperimentConfig) -> optim.AdamW:
    def excluded(name: str, parameter: torch.Tensor) -> bool:
        return parameter.ndim < 2 or "bn" in name or "ln" in name or "bias" in name or "logit_scale" in name

    if config.optimizer.exclude_bias_and_norm_from_weight_decay:
        gain_or_bias = [p for n, p in model.named_parameters() if excluded(n, p) and p.requires_grad]
        rest = [p for n, p in model.named_parameters() if not excluded(n, p) and p.requires_grad]
        groups = [
            {"params": gain_or_bias, "weight_decay": 0.0},
            {"params": rest, "weight_decay": config.optimizer.weight_decay},
        ]
    else:
        groups = [
            {
                "params": [p for p in model.parameters() if p.requires_grad],
                "weight_decay": config.optimizer.weight_decay,
            }
        ]
    return optim.AdamW(
        groups,
        lr=config.optimizer.lr,
        betas=tuple(config.optimizer.betas),
        eps=config.optimizer.eps,
    )


def rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


class RunLogger:
    def __init__(self, out_dir: Path):
        self.log_path = out_dir / "log.txt"

    def log(self, message: str) -> None:
        line = f"{datetime.now().isoformat(timespec='seconds')} {message}"
        print(line, flush=True)
        with open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def save_checkpoint(path: Path, payload: dict) -> None:
    tmp_path = path.with_suffix(".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def main():
    args = parse_args()
    config = load_training_config(args.config)
    out_dir = config.outputs.root / config.outputs.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(out_dir)

    snapshot_path = out_dir / "config.json"
    if not snapshot_path.exists():
        with open(snapshot_path, "w", encoding="utf-8") as handle:
            json.dump(asdict(config), handle, indent=2, default=str)

    seed_everything(config.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = build_tokenizer(config.text)
    train_data = train_dataset(config, tokenizer)
    loader = build_train_loader(
        train_data,
        batch_size=config.training.batch_size,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        drop_last=config.training.drop_last_batch,
        prefetch_factor=config.data.prefetch_factor,
    )
    eval_data = eval_dataset(config, tokenizer)
    eval_loader = build_eval_loader(
        eval_data,
        batch_size=config.training.eval_batch_size,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        prefetch_factor=config.data.prefetch_factor,
    )

    model = build_clip_model(config)
    model.set_grad_checkpointing(config.training.grad_checkpointing)
    model.to(device)
    optimizer = build_optimizer(model, config)

    steps_per_epoch = len(loader)
    total_steps = steps_per_epoch * config.training.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: lr_multiplier(step, config.scheduler.warmup_steps, total_steps),
    )

    start_epoch = 0
    step = 0
    best_mean_recall = -1.0
    last_path = out_dir / "last.pth"
    best_path = out_dir / "best.pth"
    if args.resume in ("auto", "always") and last_path.exists():
        checkpoint = load_clip_checkpoint(model, last_path)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        restore_rng_state(checkpoint["rng_state"])
        start_epoch = checkpoint["epoch"] + 1
        step = checkpoint["step"]
        best_mean_recall = checkpoint["best_mean_recall"]
        logger.log(f"resumed from {last_path} at epoch {start_epoch} step {step}")

    metrics_path = out_dir / "metrics.jsonl"
    autocast_dtype = torch.bfloat16 if config.training.train_precision == "bf16" else torch.float32
    model.train()

    for epoch in range(start_epoch, config.training.epochs):
        epoch_loss = 0.0
        epoch_batches = 0
        progress = tqdm(loader, desc=f"epoch {epoch}", dynamic_ncols=True)
        for images, tokens, image_ids in progress:
            images = images.to(device, non_blocking=True)
            tokens = tokens.to(device, non_blocking=True)
            if config.training.false_negative_masking:
                image_ids = image_ids.to(device, non_blocking=True)
            else:
                image_ids = None
            optimizer.zero_grad(set_to_none=True)
            use_autocast = autocast_dtype != torch.float32
            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=use_autocast):
                image_features, text_features, logit_scale = model(images, tokens)
                loss = clip_loss(image_features, text_features, logit_scale, image_ids=image_ids)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip_norm)
            optimizer.step()
            scheduler.step()
            step += 1
            epoch_loss += loss.item()
            epoch_batches += 1
            progress.set_postfix(loss=f"{loss.item():.4f}")
            if step % config.training.log_every_steps == 0:
                logger.log(
                    f"epoch {epoch} step {step}/{total_steps} "
                    f"loss {loss.item():.4f} lr {scheduler.get_last_lr()[0]:.3e} "
                    f"logit_scale {model.logit_scale.item():.4f}"
                )

        metrics = evaluate_retrieval_model(model, eval_loader, device)
        mean_recall = metrics["MeanRecall"]
        record = {"epoch": epoch, "step": step, "train_loss": epoch_loss / max(1, epoch_batches), **metrics}
        with open(metrics_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        logger.log(f"epoch {epoch} eval {format_metrics(metrics)}")

        save_checkpoint(
            last_path,
            {
                "epoch": epoch,
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_mean_recall": max(best_mean_recall, mean_recall),
                "rng_state": rng_state(),
            },
        )
        if mean_recall > best_mean_recall:
            best_mean_recall = mean_recall
            save_checkpoint(
                best_path,
                {
                    "epoch": epoch,
                    "step": step,
                    "model": model.state_dict(),
                    "metrics": metrics,
                },
            )
            logger.log(f"epoch {epoch} new best MeanRecall {mean_recall:.2f} -> {best_path}")
        model.train()

    logger.log(f"done: {config.outputs.run_name} best MeanRecall {best_mean_recall:.2f}")


if __name__ == "__main__":
    main()
