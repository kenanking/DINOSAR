"""CSV datasets, per-tower image transforms, and batching.

Transforms follow the open_clip csv-dataset defaults: train uses
RandomResizedCrop (bicubic) at the tower resolution, eval uses
Resize + CenterCrop; tower-specific channel count and normalization apply in
both. Images decode to the tower's channel count (native grayscale for
single-channel towers, replicated RGB otherwise).
"""

import csv
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Grayscale,
    InterpolationMode,
    Normalize,
    RandomResizedCrop,
    Resize,
)

from .config import TextTowerConfig, TransformConfig, VisionTowerConfig

logger = logging.getLogger(__name__)

_INTERPOLATIONS = {"bicubic": InterpolationMode.BICUBIC, "bilinear": InterpolationMode.BILINEAR}


def read_csv_pairs(path: Path) -> list[tuple[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or {"imgpath", "caption"} - set(reader.fieldnames):
            raise ValueError(f"{path} must have imgpath,caption columns")
        return [(row["imgpath"], row["caption"]) for row in reader]


def _check_interpolation(transform: TransformConfig) -> InterpolationMode:
    try:
        return _INTERPOLATIONS[transform.interpolation]
    except KeyError:
        raise ValueError(f"unsupported interpolation: {transform.interpolation}") from None


def _to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    if array.ndim == 2:
        array = array[:, :, None]
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def build_train_transform(vision: VisionTowerConfig, transform: TransformConfig) -> Compose:
    return Compose(
        [
            Grayscale(num_output_channels=vision.channels),
            RandomResizedCrop(
                vision.image_size,
                scale=transform.train_crop_scale,
                ratio=transform.train_crop_ratio,
                interpolation=_check_interpolation(transform),
            ),
            _to_tensor,
            Normalize(mean=list(vision.mean), std=list(vision.std)),
        ]
    )


def build_eval_transform(vision: VisionTowerConfig, transform: TransformConfig) -> Compose:
    return Compose(
        [
            Grayscale(num_output_channels=vision.channels),
            Resize(vision.image_size, interpolation=_check_interpolation(transform)),
            CenterCrop(vision.image_size),
            _to_tensor,
            Normalize(mean=list(vision.mean), std=list(vision.std)),
        ]
    )


class CsvImageTextDataset(Dataset):
    def __init__(
        self,
        rows,
        image_root,
        transform,
        tokenizer,
        *,
        channels: int,
        image_size: int,
        lenient: bool,
    ):
        self.rows = rows
        self.image_root = Path(image_root)
        self.transform = transform
        self.tokenize = tokenizer
        self.fallback_shape = (channels, image_size, image_size)
        self.lenient = lenient
        unique_paths = {rel_path for rel_path, _ in rows}
        self.path_ids = {rel_path: index for index, rel_path in enumerate(sorted(unique_paths))}
        self.image_ids = [self.path_ids[rel_path] for rel_path, _ in rows]

    def __len__(self):
        return len(self.rows)

    def _image_tensor(self, rel_path: str) -> torch.Tensor:
        try:
            with Image.open(self.image_root / rel_path) as image:
                return self.transform(image)
        except Exception:
            if not self.lenient:
                raise
            logger.warning("replacing unreadable image with zeros: %s", rel_path)
            return torch.zeros(self.fallback_shape)

    def __getitem__(self, index):
        rel_path, caption = self.rows[index]
        return self._image_tensor(rel_path), self.tokenize([caption])[0], self.image_ids[index]


def collate_batch(batch):
    images = torch.stack([item[0] for item in batch])
    tokens = torch.stack([item[1] for item in batch])
    image_ids = torch.tensor([item[2] for item in batch], dtype=torch.long)
    return images, tokens, image_ids


def build_tokenizer(config: TextTowerConfig):
    from open_clip import get_tokenizer

    return get_tokenizer(config.tokenizer_model)


def build_train_loader(
    dataset, *, batch_size, num_workers, pin_memory, drop_last, prefetch_factor
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_batch,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )


def build_eval_loader(dataset, *, batch_size, num_workers, pin_memory, prefetch_factor) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=collate_batch,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )


def train_dataset(config, tokenizer) -> CsvImageTextDataset:
    rows = read_csv_pairs(config.data.train_csv)
    return CsvImageTextDataset(
        rows,
        config.data.train_image_root,
        build_train_transform(config.vision, config.transform),
        tokenizer,
        channels=config.vision.channels,
        image_size=config.vision.image_size,
        lenient=True,
    )


def eval_dataset(config, tokenizer) -> CsvImageTextDataset:
    rows = read_csv_pairs(config.data.eval_csv)
    return CsvImageTextDataset(
        rows,
        config.data.eval_image_root,
        build_eval_transform(config.vision, config.transform),
        tokenizer,
        channels=config.vision.channels,
        image_size=config.vision.image_size,
        lenient=False,
    )
