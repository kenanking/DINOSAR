"""DINOSAR row: DINOv3 ViT-B/16 + 4 registers, single channel, RoPE."""

from configs.common.base import WEIGHTS, base_training_config

VISION = {
    "tower": "DINOSAR",
    "checkpoint": WEIGHTS / "dinosar_b16_unisar7m_60e.pth",
    "checkpoint_prefix": "",
    "channels": 1,
    "image_size": 256,
    "num_tokens": 256,
    "mean": (0.219,),
    "std": (0.220,),
    "pooling": "cls",
    "patch_size": 16,
    "embed_dim": 768,
    "depth": 12,
    "num_heads": 12,
    "num_register_tokens": 4,
    "layerscale_init": 1.0e-5,
    "mask_k_bias": True,
    "untie_global_and_local_cls_norm": True,
}


def config():
    return base_training_config(VISION)
