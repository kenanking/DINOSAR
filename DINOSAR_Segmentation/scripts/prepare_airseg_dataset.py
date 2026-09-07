#!/usr/bin/env python
"""Prepare AIR-PolSARSeg as a reproducible MMSeg dataset.

This project uses one canonical image product:

    RGB = [(HV + VH) / 2, HH, VV]

Each channel is converted to uint8 with log1p compression, per-image percentile
scaling, and intensity inversion. Three-channel models read the PNG as color;
single-channel models read the same PNG as grayscale in their MMSeg pipeline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


ALL_CLASS_PALETTE = {
    (0, 0, 255): 0,      # Industrial
    (0, 255, 0): 1,      # Natural
    (0, 255, 255): 2,    # Water
    (255, 0, 0): 3,      # Land_Use
    (255, 255, 0): 4,    # Housing
    (255, 255, 255): 5,  # Other
}

BLACK = (0, 0, 0)
WATER = (0, 255, 255)
POLARIZATIONS = ("HH", "HV", "VH", "VV")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("datasets/Raw_AIR-PolarSAR-Seg"))
    parser.add_argument("--out-root", type=Path, default=Path("datasets/AIR-PolSAR-Seg-mmseg"))
    parser.add_argument("--percentile-low", type=float, default=1.0)
    parser.add_argument("--percentile-high", type=float, default=99.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_band(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path), dtype=np.float32)


def robust_uint8(x: np.ndarray, low: float, high: float) -> np.ndarray:
    x = np.log1p(np.maximum(x, 0.0))
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return np.zeros_like(x, dtype=np.uint8)

    lo, hi = np.percentile(finite, [low, high])
    if hi <= lo:
        out = np.zeros_like(x, dtype=np.float32)
    else:
        out = (x - lo) / (hi - lo)
    out = np.clip(out, 0.0, 1.0) * 255.0
    return (255.0 - out).astype(np.uint8)


def make_pseudo_rgb(channels: dict[str, np.ndarray], low: float, high: float) -> np.ndarray:
    fused = [
        0.5 * (channels["HV"] + channels["VH"]),
        channels["HH"],
        channels["VV"],
    ]
    return np.stack([robust_uint8(ch, low, high) for ch in fused], axis=-1)


def convert_all_class_mask(mask_rgb: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    out = np.full(mask_rgb.shape[:2], 255, dtype=np.uint8)
    stats = {"ignored_black": 0, "unknown": 0}

    for rgb, idx in ALL_CLASS_PALETTE.items():
        match = np.all(mask_rgb == np.asarray(rgb, dtype=np.uint8), axis=-1)
        out[match] = idx

    black = np.all(mask_rgb == np.asarray(BLACK, dtype=np.uint8), axis=-1)
    stats["ignored_black"] = int(black.sum())

    unknown = (out == 255) & ~black
    stats["unknown"] = int(unknown.sum())
    if stats["unknown"]:
        colors = np.unique(mask_rgb[unknown].reshape(-1, 3), axis=0)[:20]
        raise ValueError(f"Unknown label colors found: {colors.tolist()}")

    return out, stats


def convert_water_mask(mask_rgb: np.ndarray) -> np.ndarray:
    out = np.zeros(mask_rgb.shape[:2], dtype=np.uint8)
    out[np.all(mask_rgb == np.asarray(WATER, dtype=np.uint8), axis=-1)] = 1
    out[np.all(mask_rgb == np.asarray(BLACK, dtype=np.uint8), axis=-1)] = 255
    return out


def sample_ids(split_dir: Path) -> list[str]:
    return sorted(path.name[: -len("_HH.tiff")] for path in split_dir.glob("*_HH.tiff"))


def convert_split(args: argparse.Namespace, split: str) -> dict[str, int]:
    raw_split = args.raw_root / split
    out_split = args.out_root / split
    image_dir = out_split / "images"
    anno_dir = out_split / "annotations"
    water_dir = args.out_root / f"{split}_water" / "annotations"
    image_dir.mkdir(parents=True, exist_ok=True)
    anno_dir.mkdir(parents=True, exist_ok=True)
    water_dir.mkdir(parents=True, exist_ok=True)

    totals = {"samples": 0, "ignored_black": 0, "unknown": 0}
    for sample_id in tqdm(sample_ids(raw_split), desc=split):
        totals["samples"] += 1
        out_img = image_dir / f"{sample_id}.png"
        out_mask = anno_dir / f"{sample_id}.png"
        out_water = water_dir / f"{sample_id}.png"
        if not args.overwrite and out_img.exists() and out_mask.exists() and out_water.exists():
            continue

        channels = {pol: read_band(raw_split / f"{sample_id}_{pol}.tiff") for pol in POLARIZATIONS}
        image = make_pseudo_rgb(channels, args.percentile_low, args.percentile_high)

        mask_rgb = np.asarray(Image.open(raw_split / f"{sample_id}_gt.png").convert("RGB"), dtype=np.uint8)
        all_mask, stats = convert_all_class_mask(mask_rgb)
        water_mask = convert_water_mask(mask_rgb)
        totals["ignored_black"] += stats["ignored_black"]
        totals["unknown"] += stats["unknown"]

        Image.fromarray(image, mode="RGB").save(out_img)
        Image.fromarray(all_mask, mode="L").save(out_mask)
        Image.fromarray(water_mask, mode="L").save(out_water)

    return totals


def main() -> None:
    args = parse_args()
    splits = {}
    for split in ("train_set", "test_set"):
        splits[split] = convert_split(args, split)

    meta = {
        "raw_root": str(args.raw_root),
        "image_mode": "pseudo_rgb_hvvh_hh_vv",
        "channels": ["0.5 * (HV + VH)", "HH", "VV"],
        "normalization": {
            "transform": "log1p",
            "scale": "per-image percentile",
            "percentile_low": args.percentile_low,
            "percentile_high": args.percentile_high,
            "invert": True,
        },
        "classes": ["Industrial", "Natural", "Water", "Land_Use", "Housing", "Other"],
        "palette": {str(k): v for k, v in ALL_CLASS_PALETTE.items()},
        "ignore_index": 255,
        "splits": splits,
    }
    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
