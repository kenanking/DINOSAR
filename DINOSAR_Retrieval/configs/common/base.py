"""Shared DINOSAR retrieval training configuration."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = REPO_ROOT / "weights"

SARVLM_CSV = Path("/path/to/SARVLM/csv")
STAGED_IMAGE_ROOT = REPO_ROOT / "data"
EVAL_IMAGE_ROOT = STAGED_IMAGE_ROOT
TRAIN_IMAGE_ROOT = STAGED_IMAGE_ROOT / "images"
FULL_EVAL_IMAGE_ROOT = STAGED_IMAGE_ROOT / "images"

TEXT_TOWER_CHECKPOINT = WEIGHTS / "openclip_vitb32_laion2b.pt"


def base_data_section():
    return {
        "train_csv": SARVLM_CSV / "train.csv",
        "train_image_root": TRAIN_IMAGE_ROOT,
        "eval_csv": SARVLM_CSV / "val_uni_image_and_caption_5000_L40.csv",
        "eval_image_root": EVAL_IMAGE_ROOT,
        "full_eval_csv": SARVLM_CSV / "val_uni_image_and_caption.csv",
        "full_eval_image_root": FULL_EVAL_IMAGE_ROOT,
        "num_workers": 16,
        "prefetch_factor": 2,
        "pin_memory": True,
    }


def base_text_section():
    return {
        "checkpoint": TEXT_TOWER_CHECKPOINT,
        "tokenizer_model": "ViT-B-32",
        "context_length": 77,
        "vocab_size": 49408,
        "width": 512,
        "heads": 8,
        "layers": 12,
        "output_dim": 512,
        "quick_gelu": False,
    }


def base_training_config(vision):
    return {
        "name": vision["tower"],
        "vision": vision,
        "text": base_text_section(),
        "data": base_data_section(),
        "transform": {
            "train_crop_scale": (0.9, 1.0),
            "train_crop_ratio": (3.0 / 4.0, 4.0 / 3.0),
            "interpolation": "bicubic",
        },
        "optimizer": {
            "lr": 5.0e-5,
            "weight_decay": 0.01,
            "betas": (0.9, 0.98),
            "eps": 1.0e-6,
            "exclude_bias_and_norm_from_weight_decay": True,
        },
        "scheduler": {
            "kind": "cosine",
            "warmup_steps": 1000,
        },
        "training": {
            "epochs": 10,
            "batch_size": 256,
            "eval_batch_size": 64,
            "grad_clip_norm": 1.0,
            "train_precision": "bf16",
            "eval_precision": "fp32",
            "grad_checkpointing": True,
            "false_negative_masking": True,
            "seed": 42,
            "log_every_steps": 50,
            "drop_last_batch": True,
        },
        "outputs": {
            "root": REPO_ROOT / "outputs",
            "run_name": vision["tower"],
        },
    }
