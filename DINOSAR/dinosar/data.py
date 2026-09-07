"""Data loading, transforms, and masking for DINOSAR pretraining."""

import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision.transforms import InterpolationMode

from dinosar.common import log_message
from dinosar.distributed import get_rank, get_world_size, seed_worker

# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------


class GaussianBlur:
    """Gaussian blur augmentation with built-in probability."""

    def __init__(self, p=1.0, sigma=(0.1, 2.0)):
        self.p = p
        self.sigma = sigma

    def __call__(self, x):
        if random.random() >= self.p:
            return x
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        ksize = int(sigma * 4) | 1  # Odd kernel size
        return TF.gaussian_blur(x, kernel_size=ksize, sigma=sigma)


class RandomCrop:
    """Random crop with scale and ratio augmentation.

    Args:
        size: Output size (int or tuple)
        scale: Range of crop scale
        ratio: Range of aspect ratio
    """

    def __init__(
        self,
        size,
        scale=(0.08, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        interpolation=InterpolationMode.BILINEAR,
    ):
        self.size = (size, size) if isinstance(size, int) else size
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation

    def get_params(self, img_h, img_w):
        """Get random crop parameters."""
        area = img_h * img_w

        for _ in range(10):
            target_area = random.uniform(self.scale[0], self.scale[1]) * area
            aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])

            w = int(round((target_area * aspect_ratio) ** 0.5))
            h = int(round((target_area / aspect_ratio) ** 0.5))

            if 0 < w <= img_w and 0 < h <= img_h:
                i = random.randint(0, img_h - h)
                j = random.randint(0, img_w - w)
                return i, j, h, w

        # Fallback to center crop
        in_ratio = img_w / img_h
        if in_ratio < min(self.ratio):
            w = img_w
            h = int(round(w / min(self.ratio)))
        elif in_ratio > max(self.ratio):
            h = img_h
            w = int(round(h * max(self.ratio)))
        else:
            w = img_w
            h = img_h

        i = (img_h - h) // 2
        j = (img_w - w) // 2
        return i, j, h, w

    def __call__(self, x):
        c, h, w = x.shape
        i, j, th, tw = self.get_params(h, w)
        x = TF.crop(x, i, j, th, tw)
        x = TF.resize(x, self.size, interpolation=self.interpolation, antialias=True)
        return x


class RandomHorizontalFlip:
    """Random horizontal flip."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, x):
        if random.random() < self.p:
            return TF.hflip(x)
        return x


class SigmaClip:
    """Random n-sigma clipping for SAR dynamic range invariance.

    SAR images span a huge dynamic range (60–80 dB) and different
    preprocessing choices (linear stretch, log transform, n-sigma clipping)
    produce very different pixel distributions.  This augmentation randomly
    clips to [μ − nσ, μ + nσ] and rescales to [0, 1], teaching the model
    invariance to such preprocessing.
    """

    def __init__(self, p=0.3, sigma_min=2.0, sigma_max=4.0):
        self.p = p
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, x):
        if random.random() >= self.p:
            return x
        n_sigma = random.uniform(self.sigma_min, self.sigma_max)
        mu = x.mean()
        std = x.std()
        if std < 1e-8:
            return x
        lower = mu - n_sigma * std
        upper = mu + n_sigma * std
        x = x.clamp(lower, upper)
        x_min = x.min()
        x_range = x.max() - x_min
        if x_range < 1e-8:
            return x
        return (x - x_min) / x_range


class Normalize:
    """Normalization for single-channel baseline images."""

    def __init__(
        self,
        mean=0.5,
        std=0.5,
    ):
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)
        self._cached_key = None
        self._cached_mean = None
        self._cached_std = None

    def __call__(self, x):
        key = (x.device.type, x.device.index, x.dtype)
        if key != self._cached_key:
            self._cached_key = key
            self._cached_mean = self.mean.to(device=x.device, dtype=x.dtype)
            self._cached_std = self.std.to(device=x.device, dtype=x.dtype)
        return (x - self._cached_mean) / self._cached_std


class ToTensor:
    """Convert numpy array or PIL Image to tensor."""

    def __call__(self, x):
        if isinstance(x, np.ndarray):
            # Assuming HWC format, convert to CHW
            if x.ndim == 3 and x.shape[2] in [1, 3]:
                x = np.transpose(x, (2, 0, 1))
            elif x.ndim == 2:
                x = x[np.newaxis, ...]
            x = torch.from_numpy(x).float() / 255.0 if x.dtype == np.uint8 else torch.from_numpy(x).float()
        return x


class Compose:
    """Compose multiple transforms."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


class DINOAugmentation:
    """DINOv3-style grayscale augmentation pipeline with SAR-specific transforms."""

    def __init__(
        self,
        crop_size,
        crop_scale,
        gaussian_blur_probability,
        mean=0.219,
        std=0.220,
        augmentation_config=None,
    ):
        gaussian_blur_enabled = True
        transforms = [
            ToTensor(),
            RandomCrop(
                size=crop_size,
                scale=crop_scale,
                interpolation=InterpolationMode.BICUBIC,
            ),
            RandomHorizontalFlip(p=0.5),
        ]

        if augmentation_config is not None:
            aug = augmentation_config
            gaussian_blur = aug.get("gaussian_blur") or {}
            gaussian_blur_enabled = gaussian_blur.get("enabled", True)

            sigma_clip = aug.get("sigma_clip") or {}
            if sigma_clip.get("enabled", False):
                transforms.append(
                    SigmaClip(
                        p=float(sigma_clip.get("p", 0.3)),
                        sigma_min=float(sigma_clip.get("sigma_min", 2.0)),
                        sigma_max=float(sigma_clip.get("sigma_max", 4.0)),
                    )
                )

        if gaussian_blur_enabled:
            transforms.append(GaussianBlur(p=gaussian_blur_probability, sigma=(0.1, 2.0)))

        transforms.append(Normalize(mean=mean, std=std))
        self.transforms = Compose(transforms)

    def __call__(self, x):
        return self.transforms(x)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

INDEX_FILENAME = "index.txt"
INDEX_LOCK_FILENAME = f"{INDEX_FILENAME}.lock"


def _format_image_count(count):
    noun = "image" if count == 1 else "images"
    return f"{count:,} {noun}"


def _iter_image_paths(root, extensions, *, log=False, log_every=100_000):
    seen = 0
    matched = 0
    last_log = time.monotonic()
    for dirpath, _, filenames in os.walk(root, followlinks=False):
        rel_dir = os.path.relpath(dirpath, root)
        for name in filenames:
            seen += 1
            ext = os.path.splitext(name)[1].lower()
            if ext in extensions:
                matched += 1
                yield name if rel_dir == "." else os.path.join(rel_dir, name)
            if log and log_every and seen % log_every == 0:
                now = time.monotonic()
                rate = log_every / max(now - last_log, 1e-6)
                last_log = now
                log_message(
                    f"[data] image index scan: visited={seen:,} matched={matched:,} ({rate:,.0f} files/s)",
                    all_ranks=True,
                )


def _scan_image_paths(root, *, log=False, log_every=100_000):
    return sorted(_iter_image_paths(root, IMAGE_EXTENSIONS, log=log, log_every=log_every))


def _read_index_file(index_path):
    samples = []
    with open(index_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                samples.append(line)
    return samples


def _read_valid_index_file(index_path):
    if not index_path.is_file():
        return None
    samples = _read_index_file(index_path)
    return samples if samples else None


def _write_index_file(index_path, root, samples):
    index_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = index_path.with_name(f".{index_path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write("# generated by dinosar.data.build_image_index\n")
            fh.write(f"# root={root.resolve()}\n")
            fh.write(f"# count={len(samples)}\n")
            for sample in samples:
                fh.write(sample + "\n")
        os.replace(tmp, index_path)
    finally:
        if tmp.exists():
            tmp.unlink()


def build_image_index(root, *, index_path=None, log=True, log_every=100_000):
    """Build ``index.txt`` for an image folder root and return sorted entries."""

    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset root is not a directory: {root}")
    index_path = Path(index_path) if index_path is not None else root / INDEX_FILENAME

    if log:
        log_message(f"[data] building image index: root={root} output={index_path}", all_ranks=True)
    samples = _scan_image_paths(root, log=log, log_every=log_every)
    if not samples:
        raise ValueError(f"No matching image files found under: {root}")

    _write_index_file(index_path, root, samples)
    if log:
        log_message(f"[data] wrote image index: {index_path} ({_format_image_count(len(samples))})", all_ranks=True)
    return samples


def _load_or_build_index(root, index_path):
    import fcntl

    lock_path = root / INDEX_LOCK_FILENAME
    log_message(f"[data] image index missing or empty: {index_path}; acquiring build lock {lock_path}")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a", encoding="utf-8") as lock_fh:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
        try:
            samples = _read_valid_index_file(index_path)
            if samples is not None:
                log_message(f"[data] loaded image index: {index_path} ({_format_image_count(len(samples))})")
                return samples
            return build_image_index(root, index_path=index_path, log=True)
        finally:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)


class ImageFolderDataset(Dataset[torch.Tensor]):
    """Load single-channel images listed in ``<root>/index.txt``.

    Missing or empty index files are built once under ``<root>/index.txt.lock``.
    Existing index files are read directly, avoiding a multi-million-file
    directory scan during normal training startup.

    For ad-hoc/test datasets, set ``require_index=False`` to fall back to a
    live scan without writing an index.
    """

    def __init__(self, root, *, require_index=True):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {self.root}")
        if not self.root.is_dir():
            raise NotADirectoryError(f"Dataset path is not a directory: {self.root}")

        index_path = self.root / INDEX_FILENAME
        samples = _read_valid_index_file(index_path)
        if samples is not None:
            self.samples = samples
            log_message(f"[data] loaded image index: {index_path} ({_format_image_count(len(self.samples))})")
        elif require_index:
            self.samples = _load_or_build_index(self.root, index_path)
        else:
            self.samples = _scan_image_paths(self.root)
            log_message(f"[data] scanned image dataset: {self.root} ({_format_image_count(len(self.samples))})")

        if not self.samples:
            raise ValueError(f"No image entries found for: {self.root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path = self.root / self.samples[index]
        image = Image.open(image_path).convert("L")
        array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).unsqueeze(0)


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------


class InfiniteDistributedSampler(Sampler[int]):
    """Yield per-rank indices forever from repeated distributed rounds.

    The ``advance`` parameter skips that many per-rank samples, enabling
    deterministic resume after checkpointing without replaying data.
    """

    def __init__(
        self,
        dataset_size,
        world_size,
        rank,
        *,
        shuffle=False,
        seed=0,
        advance=0,
    ):
        if dataset_size <= 0:
            raise ValueError("dataset_size must be a positive integer")
        if world_size <= 0:
            raise ValueError("world_size must be a positive integer")
        if world_size > dataset_size:
            raise ValueError("world_size must not exceed dataset_size")
        if rank < 0 or rank >= world_size:
            raise ValueError("rank must be within [0, world_size)")
        if advance < 0:
            raise ValueError("advance must be non-negative")

        self.dataset_size = dataset_size
        self.world_size = world_size
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.advance = advance
        self._round_length = math.ceil(dataset_size / world_size)
        self._padded_size = self._round_length * world_size

    def __iter__(self):
        round_idx = self.advance // self._round_length
        offset = self.advance % self._round_length
        while True:
            if self.shuffle:
                generator = torch.Generator()
                generator.manual_seed(self.seed + round_idx)
                indices = torch.randperm(self.dataset_size, generator=generator).tolist()
            else:
                indices = list(range(self.dataset_size))

            if self._padded_size > self.dataset_size:
                indices = indices + indices[: self._padded_size - self.dataset_size]

            yield from indices[self.rank : self._padded_size : self.world_size][offset:]
            round_idx += 1
            offset = 0

    def __len__(self):
        return self._round_length


# ---------------------------------------------------------------------------
# Multi-Crop
# ---------------------------------------------------------------------------


class MultiCropTransform:
    """Apply DINO-style global and local crop transforms."""

    def __init__(
        self,
        global_transform_1,
        global_transform_2,
        local_transform,
        num_local_crops=8,
    ):
        self.global_transform_1 = global_transform_1
        self.global_transform_2 = global_transform_2
        self.local_transform = local_transform
        self.num_local_crops = num_local_crops

    def __call__(self, image):
        views = [self.global_transform_1(image), self.global_transform_2(image)]
        views.extend(self.local_transform(image) for _ in range(self.num_local_crops))
        return views


class MultiCropDataset(Dataset[list[torch.Tensor]]):
    """Wrap a base dataset and return multi-crop views per sample."""

    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image = self.dataset[index]
        return self.transform(image)


def multicrop_collate(batch):
    """Stack the same view across the batch into a list of tensors."""

    num_views = len(batch[0])
    return [torch.stack([sample[view_index] for sample in batch], dim=0) for view_index in range(num_views)]


def build_multicrop_transform(
    global_size,
    local_size,
    global_scale,
    local_scale,
    num_local_crops,
    normalize_mean=0.219,
    normalize_std=0.220,
    augmentation_config=None,
):
    """Build the global and local crop pipeline."""
    global_transform_1 = DINOAugmentation(
        crop_size=global_size,
        crop_scale=global_scale,
        gaussian_blur_probability=1.0,
        mean=normalize_mean,
        std=normalize_std,
        augmentation_config=augmentation_config,
    )
    global_transform_2 = DINOAugmentation(
        crop_size=global_size,
        crop_scale=global_scale,
        gaussian_blur_probability=0.1,
        mean=normalize_mean,
        std=normalize_std,
        augmentation_config=augmentation_config,
    )
    local_transform = DINOAugmentation(
        crop_size=local_size,
        crop_scale=local_scale,
        gaussian_blur_probability=0.5,
        mean=normalize_mean,
        std=normalize_std,
        augmentation_config=augmentation_config,
    )
    return MultiCropTransform(
        global_transform_1=global_transform_1,
        global_transform_2=global_transform_2,
        local_transform=local_transform,
        num_local_crops=num_local_crops,
    )


def build_post_crop_transforms(gaussian_blur_probability, mean, std, augmentation_config=None):
    """Build augmentation pipeline to apply AFTER spatial cropping.

    Same transforms as ``DINOAugmentation`` minus ``ToTensor`` and
    ``RandomCrop``.  Order: HFlip → SigmaClip → GaussianBlur → Normalize.
    """
    gaussian_blur_enabled = True
    transforms = [RandomHorizontalFlip(p=0.5)]

    if augmentation_config is not None:
        aug = augmentation_config
        gaussian_blur_cfg = aug.get("gaussian_blur") or {}
        gaussian_blur_enabled = gaussian_blur_cfg.get("enabled", True)

        sigma_clip = aug.get("sigma_clip") or {}
        if sigma_clip.get("enabled", False):
            transforms.append(
                SigmaClip(
                    p=float(sigma_clip.get("p", 0.3)),
                    sigma_min=float(sigma_clip.get("sigma_min", 2.0)),
                    sigma_max=float(sigma_clip.get("sigma_max", 4.0)),
                )
            )

    if gaussian_blur_enabled:
        transforms.append(GaussianBlur(p=gaussian_blur_probability, sigma=(0.1, 2.0)))

    transforms.append(Normalize(mean=mean, std=std))
    return Compose(transforms)


def build_content_aware_multicrop_transform(
    global_size,
    local_size,
    global_scale,
    local_scale,
    num_local_crops,
    normalize_mean=0.219,
    normalize_std=0.220,
    augmentation_config=None,
    coarse_scales=(3, 7),
    max_anchor_prototypes=4,
    num_candidates=256,
    normalize_amplitude=True,
    fixed_concentration=None,
    disable_content=False,
    disable_proximity=False,
    disable_coverage=False,
):
    """Build a content-aware global-local crop pipeline."""
    from dinosar.content_aware_crop import ContentAwareMultiCropTransform

    post_global_1 = build_post_crop_transforms(1.0, normalize_mean, normalize_std, augmentation_config)
    post_global_2 = build_post_crop_transforms(0.1, normalize_mean, normalize_std, augmentation_config)
    post_local = build_post_crop_transforms(0.5, normalize_mean, normalize_std, augmentation_config)

    return ContentAwareMultiCropTransform(
        global_crop_size=global_size,
        local_crop_size=local_size,
        global_crop_scale=tuple(global_scale),
        local_crop_scale=tuple(local_scale),
        num_local_crops=num_local_crops,
        coarse_scales=tuple(coarse_scales) if not isinstance(coarse_scales, tuple) else coarse_scales,
        max_anchor_prototypes=max_anchor_prototypes,
        num_candidates=num_candidates,
        normalize_amplitude=normalize_amplitude,
        fixed_concentration=fixed_concentration,
        disable_content=disable_content,
        disable_proximity=disable_proximity,
        disable_coverage=disable_coverage,
        post_crop_transforms_global_1=post_global_1,
        post_crop_transforms_global_2=post_global_2,
        post_crop_transforms_local=post_local,
    )


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------


def build_train_dataloader(
    dataset,
    batch_size,
    num_workers,
    prefetch_factor=None,
    collate_fn=None,
    pin_memory=True,
    drop_last=False,
    seed=0,
    start_step=0,
    grad_accum_steps=1,
):
    """Build the default infinite training DataLoader with rank-aware seeding.

    Each optimizer step consumes ``grad_accum_steps`` micro-batches of
    ``batch_size`` samples per rank, so on resume the sampler must skip
    ``start_step * grad_accum_steps * batch_size`` per-rank samples.
    """

    sampler = InfiniteDistributedSampler(
        dataset_size=len(dataset),
        shuffle=True,
        seed=seed,
        advance=start_step * grad_accum_steps * batch_size,
        rank=get_rank(),
        world_size=get_world_size(),
    )

    generator = torch.Generator()
    generator.manual_seed(seed ^ (get_rank() << 16) ^ start_step)

    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
        "collate_fn": collate_fn,
        "worker_init_fn": seed_worker,
        "generator": generator,
    }

    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = prefetch_factor

    return DataLoader(**loader_kwargs)


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------


class MaskingGenerator:
    """Generate random block masks for the iBOT masked-patch objective.

    Samples contiguous rectangular regions on a 2-D patch grid until the
    target number of masked patches is reached, then fills remaining slots
    uniformly at random.
    """

    def __init__(
        self,
        input_size,
        num_masking_patches=None,
        min_num_patches=4,
        max_num_patches=None,
        min_aspect=0.3,
        max_aspect=None,
    ):
        if not isinstance(input_size, tuple):
            input_size = (input_size, input_size)
        self.height, self.width = input_size
        self.num_patches = self.height * self.width
        self.num_masking_patches = num_masking_patches
        self.min_num_patches = min_num_patches
        self.max_num_patches = num_masking_patches if max_num_patches is None else max_num_patches

        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def get_shape(self):
        return self.height, self.width

    def _mask(self, mask, max_mask_patches):
        delta = 0
        for _ in range(10):
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            height = int(round(math.sqrt(target_area * aspect_ratio)))
            width = int(round(math.sqrt(target_area / aspect_ratio)))
            if width < self.width and height < self.height:
                top = random.randint(0, self.height - height)
                left = random.randint(0, self.width - width)

                region = mask[top : top + height, left : left + width]
                new_count = int(height * width - region.sum())
                if 0 < new_count <= max_mask_patches:
                    region[:] = True
                    delta = new_count

                if delta > 0:
                    break
        return delta

    def complete_mask_randomly(self, mask, num_masking_patches):
        flattened = mask.flatten()
        to_add = np.random.choice(np.where(~flattened)[0], size=num_masking_patches - flattened.sum(), replace=False)
        flattened[to_add] = True
        return flattened.reshape(mask.shape)

    def __call__(self, num_masking_patches=0):
        mask = np.zeros(shape=self.get_shape(), dtype=bool)
        mask_count = 0
        while mask_count < num_masking_patches:
            max_mask_patches = num_masking_patches - mask_count
            max_mask_patches = min(max_mask_patches, self.max_num_patches)

            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            mask_count += delta

        return self.complete_mask_randomly(mask, num_masking_patches)


def build_ibot_collate_fn(
    *,
    mask_ratio_min_max,
    mask_sample_probability,
    global_crop_size,
    patch_size,
):
    """Build the pretraining collate function that emits iBOT mask metadata."""

    n_tokens_per_view = (global_crop_size // patch_size) ** 2
    mask_generator = MaskingGenerator(
        input_size=(global_crop_size // patch_size, global_crop_size // patch_size),
        num_masking_patches=n_tokens_per_view,
    )

    def collate_fn(batch):
        views = multicrop_collate(batch)
        n_global_crops = 2
        global_batch_size = views[0].shape[0]
        batch_for_masks = n_global_crops * global_batch_size

        n_samples_masked = int(batch_for_masks * mask_sample_probability)
        probs = torch.linspace(*mask_ratio_min_max, n_samples_masked + 1)
        upperbound = 0
        masks_list: list[torch.Tensor] = []
        for index in range(n_samples_masked):
            prob_max = float(probs[index + 1].item())
            mask = torch.as_tensor(mask_generator(int(n_tokens_per_view * prob_max)), dtype=torch.bool)
            masks_list.append(mask)
            upperbound += int(n_tokens_per_view * prob_max)
        for _ in range(n_samples_masked, batch_for_masks):
            masks_list.append(torch.as_tensor(mask_generator(0), dtype=torch.bool))

        random.shuffle(masks_list)

        collated_masks = torch.stack(masks_list).flatten(1)
        mask_indices_list = collated_masks.flatten().nonzero().flatten()
        masks_weight = (
            (1 / collated_masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(collated_masks)[collated_masks]
        )

        return {
            "views": views,
            "masks": [collated_masks, None],
            "mask_indices_list": mask_indices_list,
            "masks_weight": masks_weight,
            "upperbound": upperbound,
        }

    return collate_fn
