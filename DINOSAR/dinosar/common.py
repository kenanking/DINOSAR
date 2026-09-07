"""Shared utilities for DINOSAR."""

import math
from collections.abc import Mapping

import torch

from dinosar.distributed import get_rank

# BF16 peak FLOPS by GPU model. More specific patterns first.
# Ref: https://github.com/pytorch/torchtitan/blob/main/torchtitan/tools/utils.py
_PEAK_FLOPS_TABLE = (
    # Blackwell
    (["b200"], 2.25e15),
    (["b100"], 1.8e15),
    # Hopper
    (["h200"], 989e12),
    (["h100", "nvl"], 835e12),
    (["h100", "pcie"], 756e12),
    (["h100"], 989e12),
    (["h800", "nvl"], 989e12),
    (["h800"], 756e12),
    # Ampere data center
    (["a100"], 312e12),
    (["a800"], 312e12),
    # Ada data center
    (["l40s"], 362e12),
    (["l4"], 121e12),
    # Consumer
    (["5090"], 209.5e12),
    (["4090"], 165.2e12),
    (["3090"], 71e12),
)


def log_message(message="", *, all_ranks=False, flush=True):
    """Print a message once from rank 0 unless all ranks are requested."""

    if all_ranks or get_rank() == 0:
        print(message, flush=flush)


def get_peak_flops(device):
    """BF16 peak FLOPS for the current GPU. Returns inf for unknown GPUs."""

    if device.type != "cuda":
        return float("inf")
    name = torch.cuda.get_device_name(device).lower()
    for patterns, flops in _PEAK_FLOPS_TABLE:
        if all(p in name for p in patterns):
            return flops
    return float("inf")


def cosine_schedule(step, total_iters, warmup_iters, peak, end=0.0, start=0.0):
    """Linear warmup from *start* to *peak*, then cosine decay to *end*.

    Use ``warmup_iters=0`` for cosine-only schedules.
    """

    if warmup_iters > 0 and step < warmup_iters:
        return start + (peak - start) * step / warmup_iters
    if total_iters <= warmup_iters:
        return peak
    if step >= total_iters:
        return end
    cosine_steps = total_iters - warmup_iters
    if cosine_steps <= 0:
        return end
    progress = min(max((step - warmup_iters) / cosine_steps, 0.0), 1.0)
    blend = 0.5 * (1.0 + math.cos(math.pi * progress))
    return end + (peak - end) * blend


def resolve_schedule(schedule, step, total_iters, iters_per_epoch):
    """Evaluate a schedules-v2 entry at ``step``.

    Expected keys: ``start``, ``peak``, ``end``, ``warmup_epochs``.
    Missing ``start`` defaults to 0.0; missing ``peak`` defaults to ``end``.
    """

    if not isinstance(schedule, Mapping):
        raise TypeError(f"Schedule entry must be a mapping, got {type(schedule).__name__}")
    end = float(schedule["end"])
    peak = float(schedule.get("peak", end))
    start = float(schedule.get("start", 0.0))
    warmup_iters = int(round(float(schedule.get("warmup_epochs", 0.0)) * iters_per_epoch))
    return cosine_schedule(step, total_iters, warmup_iters, peak, end=end, start=start)
