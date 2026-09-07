import importlib
import shutil
from pathlib import Path

import pytest
import torch
import yaml
from omegaconf import OmegaConf
from PIL import Image

from dinosar.common import cosine_schedule
from dinosar.config import load_train_config
from dinosar.model import DINO, StudentTeacherWrapper
from dinosar.train import (
    _clip_student_gradients,
    _grad_l2_norm,
    _unpack_batch,
    build_loss_fn,
    build_optimizer,
    build_train_loader,
    compute_losses,
    load_training_checkpoint,
    prune_checkpoints,
    run_training,
    save_eval_backbone_checkpoint,
    save_training_checkpoint,
)


def _create_dummy_images(tmp_path, count=4):
    image_dir = tmp_path / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    names = []
    for i in range(count):
        img = Image.fromarray(torch.randint(0, 255, (64, 64), dtype=torch.uint8).numpy())
        name = f"{i:04d}.png"
        img.save(image_dir / name)
        names.append(name)
    (image_dir / "index.txt").write_text("\n".join(sorted(names)) + "\n")


def _minimal_config(tmp_path, *, epochs=1, iters_per_epoch=2, freeze_last_layer_epochs=0, save_every_steps=2):
    total_iters = epochs * iters_per_epoch
    return OmegaConf.create(
        {
            "data": {
                "data_path": str(tmp_path / "images"),
                "num_workers": 0,
                "prefetch_factor": None,
                "pin_memory": False,
                "batch_size": 2,
                "num_local_crops": 2,
                "global_crops_size": 64,
                "local_crops_size": 32,
                "global_crops_scale": [0.6, 1.0],
                "local_crops_scale": [0.1, 0.5],
                "normalize_mean": 0.5,
                "normalize_std": 0.5,
            },
            "model": {"arch": "vit_small", "patch_size": 16, "in_chans": 1},
            "dino": {
                "out_dim": 128,
                "student_temp": 0.1,
                "koleo_loss_weight": 0.1,
            },
            "ibot": {
                "loss_weight": 1.0,
                "mask_sample_probability": 0.5,
                "mask_ratio_min_max": [0.1, 0.5],
                "head_n_prototypes": 128,
                "student_temp": 0.1,
            },
            "hidden_distill": {
                "loss_weight": 0.0,
                "target_blocks": [0, 3, 7, 11],
            },
            "train": {
                "epochs": epochs,
                "iters_per_epoch": iters_per_epoch,
                "total_iters": total_iters,
                "save_every_steps": save_every_steps,
                "keep_last_n_checkpoints": 2,
                "eval_save_every_steps": None,
                "eval_keep_last_n": 5,
                "log_every_steps": 1,
                "seed": 42,
                "adamw_beta1": 0.9,
                "adamw_beta2": 0.99,
                "layerwise_decay": 0.95,
                "patch_embed_lr_mult": 0.1,
                "dino_head_wd_multiplier": 1.0,
                "head_lr_multiplier": 1.0,
                "head_betas": None,
                "grad_accum_steps": 1,
                "clip_grad": 1.0,
                "use_amp": False,
                "amp_dtype": "bfloat16",
                "output_dir": str(tmp_path / "output"),
                "resume": None,
                "schedules": {
                    "lr": {
                        "start": 0.0,
                        "peak": 5e-5,
                        "end": 1e-6,
                        "warmup_epochs": 0.5,
                        "freeze_last_layer_epochs": freeze_last_layer_epochs,
                    },
                    "weight_decay": {"start": 0.04, "peak": 0.04, "end": 0.4, "warmup_epochs": 0},
                    "momentum": {"start": 0.996, "peak": 0.996, "end": 0.9998, "warmup_epochs": 0},
                    "teacher_temp": {"start": 0.04, "peak": 0.07, "end": 0.07, "warmup_epochs": 1},
                },
                "compile": {"enabled": False},
            },
            "logging": {"project_name": "test", "run_name": "smoke", "enabled": False},
        }
    )


def test_run_training_smoke(tmp_path):
    _create_dummy_images(tmp_path)
    config = _minimal_config(tmp_path)
    state = run_training(config)
    assert state["global_step"] == 2
    assert state["last_lr"] > 0
    assert (tmp_path / "output" / "ckpt" / "last.pth").exists()


def test_run_training_saves_last_step_when_not_divisible(tmp_path):
    """max_steps=3 and save_every_steps=5 => the final step must still trigger a save."""
    _create_dummy_images(tmp_path)
    config = _minimal_config(tmp_path, epochs=1, iters_per_epoch=3, save_every_steps=5)
    state = run_training(config)
    assert state["global_step"] == 3
    ckpt_dir = tmp_path / "output" / "ckpt"
    assert (ckpt_dir / "last.pth").exists()
    assert (ckpt_dir / "step_0000003.pth").exists()


def test_run_training_with_hidden_distillation(tmp_path):
    _create_dummy_images(tmp_path)
    config = _minimal_config(tmp_path)
    config = OmegaConf.merge(
        config,
        OmegaConf.create({"hidden_distill": {"loss_weight": 0.5, "target_blocks": [0, 5, 11]}}),
    )
    state = run_training(config)
    assert state["global_step"] == 2


def test_delayed_hidden_distillation_anchors_predictor_gradients(tmp_path):
    config = _minimal_config(tmp_path)
    model = StudentTeacherWrapper.from_backbones(
        student=DINO.build_backbone(arch="vit_small", patch_size=16, in_chans=1),
        out_dim=128,
        hidden_distill_blocks=(0, 11),
        hidden_distill_use_predictor=True,
    )
    views = [torch.randn(2, 1, 64, 64), torch.randn(2, 1, 64, 64)]
    masks = torch.zeros(4, 16, dtype=torch.bool)
    masks[:, ::4] = True
    mask_indices_list = masks.flatten().nonzero().flatten()
    masks_weight = (1 / masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(masks)[masks]
    outputs = model(views, masks=[masks], mask_indices_list=mask_indices_list, hidden_distill_blocks=(0, 11))
    loss_fn = build_loss_fn(config)
    total_loss, metrics = compute_losses(
        outputs,
        {"masks": [masks], "mask_indices_list": mask_indices_list, "masks_weight": masks_weight},
        loss_fn,
        teacher_temp=0.04,
        koleo_weight=0.1,
        ibot_weight=1.0,
        hidden_distill_weight=0.0,
        hidden_distill_blocks=(0, 11),
    )

    total_loss.backward()

    predictor_grad = model.hidden_distill_predictors["0"].weight.grad
    assert metrics["hidden_distill_loss"] == 0.0
    assert predictor_grad is not None
    assert torch.count_nonzero(predictor_grad) == 0


def test_build_train_loader_passes_sigma_clip_augmentation_config(monkeypatch, tmp_path):
    config = _minimal_config(tmp_path, iters_per_epoch=1)
    config = OmegaConf.merge(
        config,
        OmegaConf.create(
            {
                "data": {
                    "augmentation": {
                        "gaussian_blur": {"enabled": True},
                        "sigma_clip": {"enabled": True, "p": 0.3, "sigma_min": 2.0, "sigma_max": 4.0},
                    }
                }
            }
        ),
    )

    recorded = {}

    monkeypatch.setattr("dinosar.train.ImageFolderDataset", lambda _: [torch.zeros(1, 64, 64) for _ in range(4)])

    def fake_build_multicrop_transform(**kwargs):
        recorded["augmentation_config"] = kwargs["augmentation_config"]
        return lambda image: [image, image, image, image]

    monkeypatch.setattr("dinosar.train.build_multicrop_transform", fake_build_multicrop_transform)

    def fake_build_train_dataloader(dataset, **kwargs):
        recorded["dataset_len"] = len(dataset)
        recorded["batch_size"] = kwargs["batch_size"]
        return "loader"

    monkeypatch.setattr("dinosar.train.build_train_dataloader", fake_build_train_dataloader)

    loader = build_train_loader(config)

    assert loader == "loader"
    assert recorded["dataset_len"] == 4
    assert recorded["batch_size"] == 2
    assert recorded["augmentation_config"]["sigma_clip"]["enabled"] is True
    assert recorded["augmentation_config"]["sigma_clip"]["p"] == 0.3


@pytest.mark.gpu
def test_run_training_with_compile_disabled(tmp_path):
    """Verify training completes with compile explicitly disabled (the HPC config)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    _create_dummy_images(tmp_path)
    config = _minimal_config(tmp_path)
    config = OmegaConf.merge(
        config,
        OmegaConf.create(
            {
                "train": {
                    "compile": {"enabled": False},
                    "use_amp": True,
                    "amp_dtype": "bfloat16",
                }
            }
        ),
    )
    state = run_training(config)
    assert state["global_step"] == 2
    assert state["last_lr"] > 0


@pytest.mark.gpu
def test_run_training_with_compile_backbone(tmp_path):
    """Verify backbone-level compile works on GPU with amp."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    _create_dummy_images(tmp_path)
    config = _minimal_config(tmp_path)
    config = OmegaConf.merge(
        config,
        OmegaConf.create(
            {
                "train": {
                    "compile": {"enabled": True},
                    "use_amp": True,
                    "amp_dtype": "bfloat16",
                }
            }
        ),
    )
    state = run_training(config)
    assert state["global_step"] == 2
    assert state["last_lr"] > 0


def test_optimizer_param_groups():
    model = StudentTeacherWrapper.from_backbones(
        student=DINO.build_backbone(arch="vit_small", patch_size=16, in_chans=1),
        teacher=DINO.build_backbone(arch="vit_small", patch_size=16, in_chans=1),
        out_dim=64,
    )
    optimizer = build_optimizer(
        model,
        lr=1e-4,
        weight_decay=0.04,
        layerwise_decay=0.95,
        patch_embed_lr_mult=0.1,
        dino_head_wd_multiplier=1.0,
    )
    assert len(optimizer.param_groups) > 2
    assert any(group["is_last_layer"] for group in optimizer.param_groups)
    assert any(group["lr_multiplier"] < 1.0 for group in optimizer.param_groups)
    assert any(group["wd_multiplier"] == 0.0 for group in optimizer.param_groups)


def test_optimizer_disables_hidden_distill_predictor_weight_decay():
    model = StudentTeacherWrapper.from_backbones(
        student=DINO.build_backbone(arch="vit_small", patch_size=16, in_chans=1),
        teacher=DINO.build_backbone(arch="vit_small", patch_size=16, in_chans=1),
        out_dim=64,
        hidden_distill_blocks=(7, 11),
        hidden_distill_use_predictor=True,
    )
    predictor_params = set(model.hidden_distill_predictors.parameters())
    optimizer = build_optimizer(
        model,
        lr=1e-4,
        weight_decay=0.04,
        layerwise_decay=0.95,
        patch_embed_lr_mult=0.1,
        dino_head_wd_multiplier=1.0,
    )

    predictor_groups = [
        group for group in optimizer.param_groups if any(parameter in predictor_params for parameter in group["params"])
    ]

    assert predictor_groups
    assert all(group["wd_multiplier"] == 0.0 for group in predictor_groups)


def test_freeze_last_layer(tmp_path):
    _create_dummy_images(tmp_path)
    config = _minimal_config(tmp_path, iters_per_epoch=1, freeze_last_layer_epochs=10)
    state = run_training(config)
    assert state["global_step"] == 1


def test_unpack_batch_dtype():
    unpacked = _unpack_batch(
        {
            "views": [torch.randn(2, 1, 64, 64), torch.randn(2, 1, 32, 32)],
            "masks": [torch.zeros(2, 16, dtype=torch.bool), None],
            "mask_indices_list": torch.tensor([0, 3, 4], dtype=torch.long),
            "masks_weight": torch.tensor([0.5, 0.5, 1.0]),
            "upperbound": 8,
        },
        device=torch.device("cpu"),
        view_dtype=torch.bfloat16,
    )
    assert [v.dtype for v in unpacked["views"]] == [torch.bfloat16, torch.bfloat16]
    assert unpacked["masks"][0].dtype == torch.bool


def test_unpack_batch_uses_non_blocking_transfers(monkeypatch):
    calls = []
    original_to = torch.Tensor.to

    def recording_to(self, *args, **kwargs):
        calls.append(kwargs.get("non_blocking"))
        return original_to(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", recording_to)

    _unpack_batch(
        {
            "views": [torch.randn(2, 1, 64, 64)],
            "masks": [torch.zeros(2, 16, dtype=torch.bool)],
            "mask_indices_list": torch.tensor([0, 1], dtype=torch.long),
            "masks_weight": torch.tensor([1.0, 1.0]),
            "upperbound": 2,
        },
        device=torch.device("cpu"),
        view_dtype=torch.bfloat16,
    )

    assert calls
    assert all(flag is True for flag in calls)


def test_prune_checkpoints(tmp_path):
    for step in [100, 200, 300, 400, 500]:
        (tmp_path / f"step_{step:07d}.pth").write_text("x")
    prune_checkpoints(tmp_path, keep_last_n=4)
    remaining = sorted(p.name for p in tmp_path.glob("step_*.pth"))
    assert remaining == ["step_0000200.pth", "step_0000300.pth", "step_0000400.pth", "step_0000500.pth"]


def test_prune_checkpoints_negative_keeps_all(tmp_path):
    for step in [100, 200, 300, 400, 500]:
        (tmp_path / f"step_{step:07d}.pth").write_text("x")
    prune_checkpoints(tmp_path, keep_last_n=-1)
    remaining = sorted(p.name for p in tmp_path.glob("step_*.pth"))
    assert len(remaining) == 5


def test_checkpoint_save_and_resume(tmp_path):
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    checkpoint_dir = tmp_path / "checkpoints"
    with torch.no_grad():
        model.weight.fill_(0.5)
        model.bias.fill_(0.25)
    save_training_checkpoint(
        checkpoint_dir=checkpoint_dir,
        step=12,
        model=model,
        optimizer=optimizer,
        extra_state={
            "scaler_state_dict": None,
            "last_lr": 1e-3,
            "last_weight_decay": 0.01,
            "last_momentum": 0.9,
            "last_teacher_temp": 0.04,
            "head_lr": 1e-4,
        },
    )
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    restored = load_training_checkpoint(
        checkpoint_path=checkpoint_dir / "last.pth",
        model=model,
        optimizer=optimizer,
    )
    assert restored["global_step"] == 12
    assert torch.allclose(model.weight, torch.full_like(model.weight, 0.5))
    assert torch.allclose(model.bias, torch.full_like(model.bias, 0.25))


def test_checkpoint_unwraps_model(tmp_path):
    class WrappedModel(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

    model = torch.nn.Linear(4, 2)
    wrapped = WrappedModel(model)
    optimizer = torch.optim.AdamW(wrapped.parameters(), lr=1e-3)
    with torch.no_grad():
        model.weight.fill_(0.5)
    save_training_checkpoint(checkpoint_dir=tmp_path, step=1, model=wrapped, optimizer=optimizer)
    checkpoint = torch.load(tmp_path / "last.pth", map_location="cpu")
    assert sorted(checkpoint["model_state_dict"]) == ["bias", "weight"]


def test_eval_checkpoint_unwraps_compiled_backbone(tmp_path):
    class CompiledWrapper(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self._orig_mod = module

    model = StudentTeacherWrapper.from_backbones(
        student=DINO.build_backbone(arch="vit_small", patch_size=16, in_chans=1),
        teacher=DINO.build_backbone(arch="vit_small", patch_size=16, in_chans=1),
        out_dim=128,
    )
    model.teacher.backbone = CompiledWrapper(model.teacher.backbone)

    checkpoint_path = save_eval_backbone_checkpoint(tmp_path, step=1, model=model)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    assert "cls_token" in checkpoint
    assert all(not key.startswith("_orig_mod.") for key in checkpoint)


def test_resume_after_dataset_mutation(tmp_path, image_folder):
    train_module = importlib.import_module("dinosar.train")
    output_dir = tmp_path / "run"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "data_path": str(image_folder),
                    "num_workers": 0,
                    "prefetch_factor": None,
                    "pin_memory": False,
                    "batch_size": 2,
                    "num_local_crops": 2,
                    "global_crops_size": 64,
                    "local_crops_size": 32,
                },
                "model": {"arch": "vit_small", "patch_size": 16, "in_chans": 1},
                "dino": {"out_dim": 128},
                "logging": {"enabled": False},
                "train": {
                    "epochs": 1,
                    "iters_per_epoch": 3,
                    "save_every_steps": 3,
                    "keep_last_n_checkpoints": 2,
                    "output_dir": str(output_dir),
                    "resume": str(output_dir / "ckpt" / "last.pth"),
                    "compile": {"enabled": False},
                    "schedules": {
                        "lr": {"start": 0.0, "peak": 5e-5, "end": 1e-6, "warmup_epochs": 0},
                        "weight_decay": {"start": 0.04, "peak": 0.04, "end": 0.4, "warmup_epochs": 0},
                        "momentum": {"start": 0.996, "peak": 0.996, "end": 0.9998, "warmup_epochs": 0},
                        "teacher_temp": {"start": 0.04, "peak": 0.07, "end": 0.07, "warmup_epochs": 0},
                    },
                },
            }
        )
    )
    config = load_train_config(config_path=str(config_path))
    model = train_module.build_student_teacher_model(config)
    loss_fn = train_module.build_loss_fn(config)
    optimizer = train_module.build_optimizer(
        model=model,
        lr=float(config.train.schedules.lr.peak),
        weight_decay=float(config.train.schedules.weight_decay.start),
        betas=(float(config.train.adamw_beta1), float(config.train.adamw_beta2)),
        layerwise_decay=config.train.layerwise_decay,
        patch_embed_lr_mult=config.train.patch_embed_lr_mult,
        dino_head_wd_multiplier=config.train.dino_head_wd_multiplier,
    )
    save_training_checkpoint(
        checkpoint_dir=Path(config.train.output_dir) / "ckpt",
        step=2,
        model=model,
        optimizer=optimizer,
        extra_state={
            "loss_state_dict": loss_fn.state_dict(),
            "scaler_state_dict": None,
            "last_lr": 1e-4,
            "last_weight_decay": 0.04,
            "last_momentum": 0.996,
            "last_teacher_temp": 0.07,
            "head_lr": 1e-4,
        },
    )
    added_dir = image_folder / "source_added"
    added_dir.mkdir()
    shutil.copy2(sorted(image_folder.glob("*.png"))[0], added_dir / "added.png")
    config = load_train_config(config_path=str(config_path))
    state = train_module.run_training(config)
    assert state["global_step"] == 3


def test_run_training_builds_loader_from_resumed_step(tmp_path, monkeypatch):
    _create_dummy_images(tmp_path)
    config = _minimal_config(tmp_path, iters_per_epoch=3, save_every_steps=3)
    train_module = importlib.import_module("dinosar.train")
    model = train_module.build_student_teacher_model(config)
    optimizer = train_module.build_optimizer(
        model=model,
        lr=float(config.train.schedules.lr.peak),
        weight_decay=float(config.train.schedules.weight_decay.start),
        betas=(float(config.train.adamw_beta1), float(config.train.adamw_beta2)),
        layerwise_decay=config.train.layerwise_decay,
        patch_embed_lr_mult=config.train.patch_embed_lr_mult,
        dino_head_wd_multiplier=config.train.dino_head_wd_multiplier,
    )
    checkpoint_dir = Path(config.train.output_dir) / "ckpt"
    save_training_checkpoint(
        checkpoint_dir=checkpoint_dir,
        step=2,
        model=model,
        optimizer=optimizer,
        extra_state={"scaler_state_dict": None},
    )
    config.train.resume = str(checkpoint_dir / "last.pth")

    observed = {}
    real_build_train_loader = train_module.build_train_loader

    def fake_build_train_loader(cfg, start_step=0, grad_accum_steps=1):
        observed["start_step"] = start_step
        observed["grad_accum_steps"] = grad_accum_steps
        return real_build_train_loader(cfg, start_step=start_step, grad_accum_steps=grad_accum_steps)

    monkeypatch.setattr(train_module, "build_train_loader", fake_build_train_loader)

    state = run_training(config)

    assert observed["start_step"] == 2
    assert observed["grad_accum_steps"] == 1
    assert state["global_step"] == 3


def test_lr_schedule():
    peak = 1.0
    end = 0.1
    max_steps = 1000
    warmup = 100

    # Warmup phase
    assert cosine_schedule(0, max_steps, warmup, peak, end=end) == 0.0
    assert abs(cosine_schedule(50, max_steps, warmup, peak, end=end) - 0.5) < 1e-9
    assert cosine_schedule(100, max_steps, warmup, peak, end=end) == peak
    # Official schedule starts cosine decay immediately after warmup.
    assert cosine_schedule(500, max_steps, warmup, peak, end=end) < peak
    assert cosine_schedule(500, max_steps, warmup, peak, end=end) > end
    assert abs(cosine_schedule(max_steps, max_steps, warmup, peak, end=end) - end) < 1e-9
    # No warmup
    assert cosine_schedule(0, 100, 0, peak, end=end) == peak
    assert cosine_schedule(100, 100, 0, peak, end=end) == end
    # Non-zero start with peak=end is the teacher-temperature warmup behavior.
    assert abs(cosine_schedule(1, 100, 2, 0.07, end=0.07, start=0.04) - 0.055) < 1e-12
    assert cosine_schedule(2, 100, 2, 0.07, end=0.07, start=0.04) == 0.07
    assert cosine_schedule(50, 100, 2, 0.07, end=0.07, start=0.04) == 0.07


def test_lr_schedule_pure_cosine_ema_style_monotone():
    base, final = 0.996, 0.9998
    assert abs(cosine_schedule(0, 1000, 0, base, end=final) - base) < 1e-9
    assert abs(cosine_schedule(1000, 1000, 0, base, end=final) - final) < 1e-9

    prev = cosine_schedule(0, 1000, 0, base, end=final)
    for step in range(1, 1001):
        val = cosine_schedule(step, 1000, 0, base, end=final)
        assert val >= prev - 1e-12
        prev = val


def test_lr_schedule_pure_cosine_weight_decay_endpoints():
    start_wd = 0.04
    end_wd = 0.4
    assert abs(cosine_schedule(0, 1000, 0, start_wd, end=end_wd) - start_wd) < 1e-9
    assert abs(cosine_schedule(1000, 1000, 0, start_wd, end=end_wd) - end_wd) < 1e-9
    assert start_wd < cosine_schedule(500, 1000, 0, start_wd, end=end_wd) < end_wd


def test_gradient_accumulation_smoke(tmp_path):
    _create_dummy_images(tmp_path, count=8)
    config = _minimal_config(tmp_path)
    config = OmegaConf.merge(
        config,
        OmegaConf.create({"train": {"grad_accum_steps": 2}, "data": {"batch_size": 1}}),
    )
    state = run_training(config)
    assert state["global_step"] == 2


def test_resume_passes_grad_accum_steps_to_loader(tmp_path, monkeypatch):
    """When grad_accum_steps > 1, build_train_loader must receive the value so the
    sampler skips the correct number of per-rank samples on resume."""
    _create_dummy_images(tmp_path, count=8)
    config = _minimal_config(tmp_path, epochs=2, iters_per_epoch=2, save_every_steps=2)
    config = OmegaConf.merge(
        config,
        OmegaConf.create({"train": {"grad_accum_steps": 2}, "data": {"batch_size": 1}}),
    )
    state = run_training(config)
    assert state["global_step"] == 4

    config.train.resume = str(tmp_path / "output" / "ckpt" / "last.pth")
    config.train.total_iters = 5

    train_module = importlib.import_module("dinosar.train")
    observed = {}
    real_build_train_loader = train_module.build_train_loader

    def spy_build_train_loader(cfg, start_step=0, grad_accum_steps=1):
        observed["start_step"] = start_step
        observed["grad_accum_steps"] = grad_accum_steps
        return real_build_train_loader(cfg, start_step=start_step, grad_accum_steps=grad_accum_steps)

    monkeypatch.setattr(train_module, "build_train_loader", spy_build_train_loader)

    state = run_training(config)
    assert observed["start_step"] == 4
    assert observed["grad_accum_steps"] == 2
    assert state["global_step"] == 5


def test_optimizer_head_betas_separate():
    """When head_betas is provided, head param groups get distinct beta values."""
    model = StudentTeacherWrapper.from_backbones(
        student=DINO.build_backbone(arch="vit_small", patch_size=16, in_chans=1),
        teacher=DINO.build_backbone(arch="vit_small", patch_size=16, in_chans=1),
        out_dim=64,
    )
    optimizer = build_optimizer(
        model,
        lr=1e-4,
        weight_decay=0.04,
        betas=(0.9, 0.99),
        layerwise_decay=0.95,
        patch_embed_lr_mult=0.1,
        dino_head_wd_multiplier=1.0,
        head_betas=(0.8, 0.95),
        head_lr_multiplier=0.5,
    )
    head_groups = [g for g in optimizer.param_groups if g.get("betas") == (0.8, 0.95)]
    backbone_groups = [g for g in optimizer.param_groups if g.get("betas", (0.9, 0.99)) == (0.9, 0.99)]
    assert len(head_groups) > 0, "Expected at least one head group with custom betas"
    assert len(backbone_groups) > 0, "Expected at least one backbone group with default betas"


def test_clip_student_gradients_clips_modules_independently():
    model = StudentTeacherWrapper.from_backbones(
        student=DINO.build_backbone(arch="vit_small", patch_size=16, in_chans=1),
        teacher=DINO.build_backbone(arch="vit_small", patch_size=16, in_chans=1),
        out_dim=64,
    )
    max_norm = 3.0

    with torch.no_grad():
        model.student.backbone.patch_embed.proj.weight.grad = 100 * torch.ones_like(
            model.student.backbone.patch_embed.proj.weight
        )
        model.student.head.last_layer.weight.grad = 0.01 * torch.ones_like(model.student.head.last_layer.weight)
        model.student.ibot_head.last_layer.weight.grad = 10 * torch.ones_like(model.student.ibot_head.last_layer.weight)

    _clip_student_gradients(model, max_norm=max_norm)
    bb_norm = _grad_l2_norm(model.student.backbone.parameters())
    dh_norm = _grad_l2_norm(model.student.head.parameters())
    ih_norm = _grad_l2_norm(model.student.ibot_head.parameters())

    assert bb_norm == pytest.approx(max_norm, rel=1e-5)
    assert ih_norm == pytest.approx(max_norm, rel=1e-5)
    assert 0.0 < dh_norm < max_norm
