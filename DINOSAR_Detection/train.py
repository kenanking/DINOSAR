#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
import ast
import os
from pathlib import Path

from utils.project_paths import DETECTRON2_CONFIGS, setup_dependency_paths
setup_dependency_paths()

import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import (
    AMPTrainer,
    SimpleTrainer,
    default_argument_parser,
    default_setup,
    default_writers,
    hooks,
    launch
)
from detectron2.engine.defaults import create_ddp_model
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.utils import comm
from detectron2.data import DatasetCatalog
from omegaconf import OmegaConf


FREEZE_BACKBONE = False


def _patch_model_zoo_config_lookup() -> None:
    import detectron2.model_zoo.model_zoo as mz_dep

    if getattr(mz_dep.get_config_file, "_local_lookup_patched", False):
        return

    search_roots = (
        # PROJECT_ROOT / "configs",
        DETECTRON2_CONFIGS,
    )
    original_get_config_file = mz_dep.get_config_file

    def local_get_config_file(config_rel: str) -> str:
        try:
            return original_get_config_file(config_rel)
        except RuntimeError:
            for root in search_roots:
                candidate = root / config_rel
                if candidate.exists():
                    return str(candidate)
            raise

    local_get_config_file._local_lookup_patched = True
    mz_dep.get_config_file = local_get_config_file


def _patch_dataset_register_guard():
    if getattr(DatasetCatalog.register, "_duplicate_guarded", False):
        return

    orig = DatasetCatalog.register

    def safe_register(name, func):
        if name in DatasetCatalog.list():
            return
        return orig(name, func)

    safe_register._duplicate_guarded = True
    DatasetCatalog.register = safe_register


def _quiet_setup(cfg, args):
    default_setup(cfg, args)


def _configure_deterministic_algorithms(cfg) -> None:
    if getattr(cfg.train, "deterministic", False):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        warn_only = getattr(cfg.train, "deterministic_warn_only", False)
        torch.use_deterministic_algorithms(True, warn_only=warn_only)


def _decode_override_value(value: str):
    value_lower = value.lower()
    if value_lower == "true":
        return True
    if value_lower == "false":
        return False
    if value_lower in {"none", "null"}:
        return None
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value


def _apply_overrides(cfg, overrides):
    for override in overrides:
        key, value = override.split("=", 1)
        OmegaConf.update(cfg, key, _decode_override_value(value), merge=True)
    return cfg


def _cleanup_output_checkpoints(output_dir: str | Path | None) -> None:
    if not output_dir:
        return

    output_path = Path(output_dir)
    if not output_path.exists():
        return

    for pattern in ("model*.pth", "last_checkpoint"):
        for path in output_path.glob(pattern):
            if path.is_file():
                path.unlink()


def do_test(cfg, model):
    if "evaluator" in cfg.dataloader:
        ret = inference_on_dataset(
            model,
            instantiate(cfg.dataloader.test),
            instantiate(cfg.dataloader.evaluator),
        )
        print_csv_format(ret)
        return ret


def do_train(args, cfg):
    """
    Args:
        cfg: an object with the following attributes:
            model: instantiate to a module
            dataloader.{train,test}: instantiate to dataloaders
            dataloader.evaluator: instantiate to evaluator for test set
            optimizer: instantiate to an optimizer
            lr_multiplier: instantiate to a fvcore scheduler
            train: other misc config defined in `configs/common/train.py`, including:
                output_dir (str)
                init_checkpoint (str)
                amp.enabled (bool)
                max_iter (int)
                eval_period, log_period (int)
                device (str)
                checkpointer (dict)
                ddp (dict)
    """
    model = instantiate(cfg.model)

    # freeze
    if args.freeze_backbone:
        print()
        print("[ ✓ ]Freeze backbone net parameters")
        print()
        for p in model.backbone.net.parameters():
            p.requires_grad = False

    model.to(cfg.train.device)

    cfg.optimizer.params.model = model
    optim = instantiate(cfg.optimizer)

    train_loader = instantiate(cfg.dataloader.train)

    model = create_ddp_model(model, **cfg.train.ddp)
    trainer = (AMPTrainer if cfg.train.amp.enabled else SimpleTrainer)(model, train_loader, optim)
    def run_eval():
        return do_test(cfg, model)

    checkpointer = DetectionCheckpointer(
        model,
        cfg.train.output_dir,
        trainer=trainer,
    )
    best_checkpointer_cfg = getattr(cfg.train, "best_checkpointer", None)
    trainer.register_hooks(
        [
            hooks.IterationTimer(),
            hooks.LRScheduler(scheduler=instantiate(cfg.lr_multiplier)),
            (
                hooks.PeriodicCheckpointer(checkpointer, **cfg.train.checkpointer)
                if comm.is_main_process()
                else None
            ),
            hooks.EvalHook(
                cfg.train.eval_period,
                run_eval,
                eval_after_train=getattr(cfg.train, "eval_after_train", True),
            ),
            (
                hooks.BestCheckpointer(
                    cfg.train.eval_period,
                    checkpointer,
                    best_checkpointer_cfg.metric,
                    mode=getattr(best_checkpointer_cfg, "mode", "max"),
                    file_prefix=getattr(best_checkpointer_cfg, "file_prefix", "model_best"),
                )
                if comm.is_main_process()
                and best_checkpointer_cfg is not None
                and getattr(best_checkpointer_cfg, "enabled", True)
                else None
            ),
            (
                hooks.PeriodicWriter(
                    default_writers(cfg.train.output_dir, cfg.train.max_iter),
                    period=cfg.train.log_period,
                )
                if comm.is_main_process()
                else None
            ),
        ]
    )

    checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=args.resume)
    if args.resume and checkpointer.has_checkpoint():
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration
        start_iter = trainer.iter + 1
    else:
        start_iter = 0

    trainer.train(start_iter, cfg.train.max_iter)

    if comm.is_main_process() and not getattr(cfg.train, "keep_checkpoint_files", True):
        _cleanup_output_checkpoints(cfg.train.output_dir)


def main(args):
    _patch_model_zoo_config_lookup()
    _patch_dataset_register_guard()
    cfg = LazyConfig.load(args.config_file)
    cfg = _apply_overrides(cfg, args.opts)
    _configure_deterministic_algorithms(cfg)
    _quiet_setup(cfg, args)

    if args.eval_only:
        model = instantiate(cfg.model)
        model.to(cfg.train.device)
        model = create_ddp_model(model)
        DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
        print(do_test(cfg, model))
    else:
        args.freeze_backbone = FREEZE_BACKBONE

        do_train(args, cfg)


def invoke_main() -> None:
    args = default_argument_parser().parse_args()

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )


if __name__ == "__main__":
    invoke_main()  # pragma: no cover
