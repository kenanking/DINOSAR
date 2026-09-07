"""Distributed runtime and seeding helpers for DINOSAR."""

import os
import random

import numpy as np
import torch
import torch.distributed as dist


def _get_env_int(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return int(value)


def get_rank():
    """Return the current process rank."""

    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return max(_get_env_int("RANK", 0), 0)


def get_world_size():
    """Return the current world size."""

    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_local_rank():
    """Return the local rank when launched under torchrun."""

    return _get_env_int("LOCAL_RANK", 0)


def is_main_process():
    """Return True on rank 0."""

    return get_rank() == 0


def get_runtime_device():
    """Return the device selected for this process."""

    if not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda", get_local_rank())


def configure_cuda_defaults():
    """Set CUDA allocator and matmul precision defaults when a GPU is present."""

    if not torch.cuda.is_available():
        return
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    torch.set_float32_matmul_precision("high")


def init_distributed_if_needed():
    """Initialize torch.distributed when the current job is distributed."""

    configure_cuda_defaults()
    if not dist.is_available() or dist.is_initialized():
        return
    if _get_env_int("RANK", -1) == -1:
        return
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, init_method="env://")


def destroy_distributed_if_initialized():
    """Tear down torch.distributed when it was initialized."""

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _get_wrapped_module(model, attr):
    if isinstance(model, torch.nn.Module):
        module = model._modules.get(attr)
        if module is not None:
            return module

    model_dict = getattr(model, "__dict__", {})
    if attr in model_dict:
        return model_dict[attr]
    return None


def unwrap_model(model):
    """Return the underlying module for DDP and compiled-model wrappers."""

    while True:
        for attr in ("module", "_orig_mod"):
            wrapped_model = _get_wrapped_module(model, attr)
            if wrapped_model is not None and wrapped_model is not model:
                model = wrapped_model
                break
        else:
            return model


# ---------------------------------------------------------------------------
# Seeding helpers
# ---------------------------------------------------------------------------


def set_seed(seed):
    """Seed Python, NumPy, and PyTorch RNGs."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    """Seed NumPy and Python RNGs inside dataloader workers."""

    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
