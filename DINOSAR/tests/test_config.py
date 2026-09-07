import importlib.util
import math
from pathlib import Path


def _load_config_module():
    module_path = Path(__file__).resolve().parents[1] / "dinosar" / "config.py"
    spec = importlib.util.spec_from_file_location("dinosar_config", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_train_config_merges_base_and_variant(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text(
        "model:\n  arch: vit_small\ntrain:\n  epochs: 50\n  iters_per_epoch: 100\n  seed: 7\n"
        "data:\n  num_workers: 8\n  prefetch_factor: 4\n"
    )
    variant = tmp_path / "variant.yaml"
    variant.write_text("defaults:\n  - base\ntrain:\n  epochs: 80\n")

    cfg = _load_config_module().load_train_config(variant)

    assert cfg.train.epochs == 80
    assert cfg.train.iters_per_epoch == 100
    assert cfg.train.total_iters == 80 * 100
    assert cfg.train.seed == 7


def test_num_workers_zero_disables_prefetch(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "data:\n  num_workers: 0\n  prefetch_factor: 4\ntrain:\n  iters_per_epoch: 10\n  epochs: 1\n"
    )

    cfg = _load_config_module().load_train_config(config_file)

    assert cfg.data.num_workers == 0
    assert cfg.data.prefetch_factor is None


def test_eval_config_has_no_train_defaults(tmp_path):
    config_file = tmp_path / "eval.yaml"
    config_file.write_text(
        "model:\n  arch: vit_small\n  patch_size: 16\n  in_chans: 1\ndata:\n  root: ./data\neval:\n  batch_size: 64\n"
    )

    cfg = _load_config_module().load_eval_config(config_file)

    assert cfg.eval.batch_size == 64
    assert "train" not in cfg


def test_pretrain_config_composes_model_defaults():
    cfg = _load_config_module().load_train_config("dinosar/configs/pretrain/unisar7m/vitb16_reg.yaml")

    assert cfg.model.arch == "vit_base"
    assert cfg.model.patch_size == 16
    assert cfg.dino.koleo_loss_weight == 0.1
    assert cfg.ibot.loss_weight == 1.0
    assert cfg.hidden_distill.loss_weight == 0.0
    assert list(cfg.hidden_distill.target_blocks) == []


def test_iters_per_epoch_auto_resolves_from_dataset_size():
    cfg = _load_config_module().load_train_config("dinosar/configs/pretrain/unisar7m/vitb16_reg.yaml")

    expected = math.ceil(int(cfg.data.dataset_size) / int(cfg.train.target_effective_batch))
    assert cfg.train.iters_per_epoch == expected
    assert cfg.train.total_iters == expected * cfg.train.epochs


def test_schedules_v2_fields_present():
    cfg = _load_config_module().load_train_config("dinosar/configs/pretrain/unisar7m/vitb16_reg.yaml")

    schedules = cfg.train.schedules
    assert {"lr", "weight_decay", "momentum", "teacher_temp"} <= set(schedules.keys())
    assert float(schedules.lr.peak) > 0.0
    assert float(schedules.momentum.end) > float(schedules.momentum.start)


def test_overrides_apply_before_total_iters_resolution():
    """train.epochs override must propagate into train.total_iters."""

    cfg = _load_config_module().load_train_config(
        "dinosar/configs/pretrain/unisar7m/vits16_reg.yaml",
        overrides=["train.epochs=5"],
    )

    assert cfg.train.epochs == 5
    assert cfg.train.total_iters == 5 * cfg.train.iters_per_epoch


def test_released_60e_training_recipes():
    for name, arch, batch, steps in [
        ("vits16_reg", "vit_small", 1280, 330360),
        ("vitb16_reg", "vit_base", 1536, 275340),
    ]:
        cfg = _load_config_module().load_train_config(f"dinosar/configs/pretrain/unisar7m/{name}.yaml")
        assert cfg.model.arch == arch
        assert cfg.model.in_chans == 1
        assert cfg.model.num_register_tokens == 4
        assert cfg.data.dataset_size == 7047666
        assert cfg.data.cropping_strategy == "content_aware"
        assert cfg.train.epochs == 60
        assert cfg.train.target_effective_batch == batch
        assert cfg.train.total_iters == steps
        assert cfg.train.schedules.lr.peak == 6e-4
        assert cfg.train.schedules.lr.warmup_epochs == 3
        assert cfg.hidden_distill.loss_weight == 0
