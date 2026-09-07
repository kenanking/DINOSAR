"""OmegaConf-backed configuration loading for DINOSAR."""

import math
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

_PACKAGE_ROOT = Path(__file__).resolve().parent
_DEFAULT_PRETRAIN_CONFIG = _PACKAGE_ROOT / "configs" / "pretrain" / "base.yaml"


def resolve_config_path(config_path):
    """Resolve a config path against package-aware fallback locations."""

    raw_path = Path(config_path).expanduser()
    candidates = [raw_path]

    if not raw_path.is_absolute():
        candidates.extend(
            [
                _PACKAGE_ROOT.parent / raw_path,
                _PACKAGE_ROOT / raw_path,
                _PACKAGE_ROOT / "configs" / raw_path,
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(f"Config file not found: {raw_path}")


def _load_yaml_tree(config_path):
    """Load a YAML file and resolve any local defaults chain."""

    raw_config = OmegaConf.load(config_path)
    defaults = raw_config.get("defaults") if isinstance(raw_config, DictConfig) else None
    merged = OmegaConf.create({})

    if defaults:
        defaults_list = OmegaConf.to_container(defaults, resolve=False)
        for entry in defaults_list:
            if entry in (None, "_self_"):
                continue
            if not isinstance(entry, str):
                raise TypeError(f"Unsupported defaults entry in {config_path}: {entry!r}")
            child_path = config_path.parent / entry
            if child_path.suffix == "":
                child_path = child_path.with_suffix(".yaml")
            merged = OmegaConf.merge(merged, _load_yaml_tree(child_path))

    if isinstance(raw_config, DictConfig) and "defaults" in raw_config:
        stripped = OmegaConf.to_container(raw_config, resolve=False)
        stripped.pop("defaults", None)
        raw_config = OmegaConf.create(stripped)

    return OmegaConf.merge(merged, raw_config)


def _resolve_epoch_units(config):
    """Fill in ``train.iters_per_epoch`` and ``train.total_iters`` from epoch-based config."""

    iters_per_epoch = OmegaConf.select(config, "train.iters_per_epoch")
    if iters_per_epoch is None:
        dataset_size = OmegaConf.select(config, "data.dataset_size")
        target_eff_batch = OmegaConf.select(config, "train.target_effective_batch")
        if dataset_size is None or target_eff_batch is None:
            raise ValueError(
                "train.iters_per_epoch is null and cannot be inferred: "
                "set data.dataset_size and train.target_effective_batch."
            )
        iters_per_epoch = math.ceil(int(dataset_size) / int(target_eff_batch))
        config.train.iters_per_epoch = iters_per_epoch
    config.train.total_iters = int(iters_per_epoch) * int(config.train.epochs)
    return config


def load_train_config(config_path=None, overrides=None):
    """Load the default pretrain config, merge an optional variant tree and CLI overrides.

    ``overrides`` is an optional list of OmegaConf dotlist strings (e.g.
    ``["train.epochs=5", "data.cropping_strategy=random"]``). They are merged
    *before* epoch-derived fields are resolved so that overriding ``train.epochs``
    or ``train.iters_per_epoch`` correctly updates ``train.total_iters``.
    """

    config = _load_yaml_tree(_DEFAULT_PRETRAIN_CONFIG)
    if config_path is not None:
        config = OmegaConf.merge(config, _load_yaml_tree(resolve_config_path(config_path)))
    if overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(list(overrides)))
    if config.data.num_workers == 0:
        config.data.prefetch_factor = None
    return _resolve_epoch_units(config)


def load_eval_config(config_path):
    """Load an eval config tree without injecting pretrain defaults."""

    return _load_yaml_tree(resolve_config_path(config_path))


def save_resolved_config(config, output_path):
    """Write the fully resolved config to disk."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(OmegaConf.to_yaml(config, resolve=True))


def next_output_dir(base):
    """Return *base* if it doesn't exist, otherwise append ``_1``, ``_2``, ... until free."""

    base = Path(base)
    if not base.exists():
        return base
    n = 1
    while base.with_name(f"{base.name}_{n}").exists():
        n += 1
    return base.with_name(f"{base.name}_{n}")
