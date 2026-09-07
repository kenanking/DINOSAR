"""Strict frozen-dataclass configuration schema and config-file loading.

Config files under ``configs/`` are plain Python modules exposing ``config()``
returning a fully explicit values dict. Every field must be present in the
values dict; unknown keys are rejected. The vision tower
resolves through a registry mapping its name to a config dataclass that
parses itself (``from_config_values``).
"""

import importlib.util
import sys
from dataclasses import dataclass, fields
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


class ConfigError(ValueError):
    pass


def _checked_values(cls, values):
    if not isinstance(values, dict):
        raise ConfigError(f"{cls.__name__} expects a dict of values, got {type(values).__name__}")
    expected = {field.name for field in fields(cls)}
    given = set(values)
    unknown = given - expected
    if unknown:
        raise ConfigError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    missing = expected - given
    if missing:
        raise ConfigError(f"missing {cls.__name__} keys: {sorted(missing)}")
    return values


def _coerce_field(cls, name, annotation, value):
    origin = getattr(annotation, "__origin__", None)
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{cls.__name__}.{name} expects a list/tuple, got {value!r}")
        return tuple(value)
    if annotation is Path:
        return Path(value)
    if annotation in (int, float, str, bool) and not isinstance(value, annotation):
        raise ConfigError(f"{cls.__name__}.{name} expects {annotation.__name__}, got {value!r}")
    return value


@dataclass(frozen=True)
class StrictConfig:
    @classmethod
    def from_config_values(cls, values):
        checked = _checked_values(cls, values)
        coerced = {
            field.name: _coerce_field(cls, field.name, field.type, checked[field.name])
            for field in fields(cls)
        }
        return cls(**coerced)


@dataclass(frozen=True)
class VisionTowerConfig(StrictConfig):
    tower: str
    checkpoint: Path
    checkpoint_prefix: str
    channels: int
    image_size: int
    num_tokens: int
    mean: tuple[float, ...]
    std: tuple[float, ...]
    pooling: str


@dataclass(frozen=True)
class Dinov3TowerConfig(VisionTowerConfig):
    patch_size: int
    embed_dim: int
    depth: int
    num_heads: int
    num_register_tokens: int
    layerscale_init: float
    mask_k_bias: bool
    untie_global_and_local_cls_norm: bool


VISION_CONFIG_REGISTRY: dict[str, type[VisionTowerConfig]] = {
    "DINOSAR": Dinov3TowerConfig,
}


@dataclass(frozen=True)
class TextTowerConfig(StrictConfig):
    checkpoint: Path
    tokenizer_model: str
    context_length: int
    vocab_size: int
    width: int
    heads: int
    layers: int
    output_dim: int
    quick_gelu: bool


@dataclass(frozen=True)
class TransformConfig(StrictConfig):
    train_crop_scale: tuple[float, float]
    train_crop_ratio: tuple[float, float]
    interpolation: str


@dataclass(frozen=True)
class DataConfig(StrictConfig):
    train_csv: Path
    train_image_root: Path
    eval_csv: Path
    eval_image_root: Path
    full_eval_csv: Path
    full_eval_image_root: Path
    num_workers: int
    prefetch_factor: int
    pin_memory: bool


@dataclass(frozen=True)
class OptimizerConfig(StrictConfig):
    lr: float
    weight_decay: float
    betas: tuple[float, float]
    eps: float
    exclude_bias_and_norm_from_weight_decay: bool


@dataclass(frozen=True)
class SchedulerConfig(StrictConfig):
    kind: str
    warmup_steps: int


@dataclass(frozen=True)
class TrainingConfig(StrictConfig):
    epochs: int
    batch_size: int
    eval_batch_size: int
    grad_clip_norm: float
    train_precision: str
    eval_precision: str
    grad_checkpointing: bool
    false_negative_masking: bool
    seed: int
    log_every_steps: int
    drop_last_batch: bool


@dataclass(frozen=True)
class OutputConfig(StrictConfig):
    root: Path
    run_name: str


@dataclass(frozen=True)
class ExperimentConfig(StrictConfig):
    name: str
    vision: VisionTowerConfig
    text: TextTowerConfig
    data: DataConfig
    transform: TransformConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    training: TrainingConfig
    outputs: OutputConfig


_NESTED_BUILDERS = {
    "text": TextTowerConfig,
    "transform": TransformConfig,
    "data": DataConfig,
    "optimizer": OptimizerConfig,
    "scheduler": SchedulerConfig,
    "training": TrainingConfig,
    "outputs": OutputConfig,
}


def _build_vision(values):
    tower = values.get("tower")
    if tower not in VISION_CONFIG_REGISTRY:
        raise ConfigError(f"unknown vision tower discriminator: {tower!r}")
    return VISION_CONFIG_REGISTRY[tower].from_config_values(values)


def build_training_config(values):
    checked = _checked_values(ExperimentConfig, values)
    parts = {name: builder.from_config_values(checked[name]) for name, builder in _NESTED_BUILDERS.items()}
    parts["vision"] = _build_vision(checked["vision"])
    return ExperimentConfig(name=checked["name"], **parts)


def load_config_values(path: Path) -> dict:
    path = Path(path)
    if path.suffix != ".py":
        raise ConfigError(f"config file must be a .py file: {path}")
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ConfigError(f"cannot import config file: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "config"):
        raise ConfigError(f"config file {path} does not define config()")
    values = module.config()
    if not isinstance(values, dict):
        raise ConfigError(f"config() in {path} must return a dict")
    return values


def load_training_config(path: Path) -> ExperimentConfig:
    return build_training_config(load_config_values(path))
