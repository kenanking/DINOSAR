import random

import torch

from dinosar.data import (
    DINOAugmentation,
    GaussianBlur,
    InfiniteDistributedSampler,
    Normalize,
    SigmaClip,
    build_multicrop_transform,
)
from dinosar.distributed import get_rank


def collect_rank_stream(sampler, count):
    it = iter(sampler)
    return [next(it) for _ in range(count)]


def build_reference_rank_stream(dataset_size, world_size, rank, seed, advance, count):
    sampler = InfiniteDistributedSampler(
        dataset_size=dataset_size,
        world_size=world_size,
        rank=rank,
        shuffle=True,
        seed=seed,
    )
    it = iter(sampler)
    for _ in range(advance):
        next(it)
    return [next(it) for _ in range(count)]


def test_infinite_distributed_sampler_advance_without_shuffle():
    sampler = InfiniteDistributedSampler(
        dataset_size=12,
        world_size=2,
        rank=0,
        shuffle=False,
        seed=7,
        advance=4,
    )
    assert collect_rank_stream(sampler, 3) == [8, 10, 0]


def test_infinite_distributed_sampler_advance_matches_reference_shuffle():
    sampler = InfiniteDistributedSampler(
        dataset_size=10,
        world_size=2,
        rank=1,
        shuffle=True,
        seed=11,
        advance=6,
    )
    actual = collect_rank_stream(sampler, 5)
    expected = build_reference_rank_stream(
        dataset_size=10,
        world_size=2,
        rank=1,
        seed=11,
        advance=6,
        count=5,
    )
    assert actual == expected


def test_infinite_distributed_sampler_advance_zero_matches_baseline():
    baseline = InfiniteDistributedSampler(
        dataset_size=10,
        world_size=2,
        rank=0,
        shuffle=True,
        seed=5,
        advance=0,
    )
    resumed = InfiniteDistributedSampler(
        dataset_size=10,
        world_size=2,
        rank=0,
        shuffle=True,
        seed=5,
        advance=0,
    )
    assert collect_rank_stream(baseline, 6) == collect_rank_stream(resumed, 6)


def test_infinite_distributed_sampler_advance_can_cross_rounds():
    sampler = InfiniteDistributedSampler(
        dataset_size=12,
        world_size=2,
        rank=1,
        shuffle=False,
        seed=0,
        advance=8,
    )
    assert collect_rank_stream(sampler, 4) == [5, 7, 9, 11]


def test_infinite_distributed_sampler_eventually_emits_all_indices():
    sampler = InfiniteDistributedSampler(
        dataset_size=5,
        world_size=2,
        rank=0,
        shuffle=False,
        seed=0,
    )
    seen = set(collect_rank_stream(sampler, 6))
    assert seen == {0, 2, 4}


def test_infinite_distributed_sampler_padding_keeps_round_length_equal():
    rank0 = InfiniteDistributedSampler(dataset_size=5, world_size=2, rank=0, shuffle=True, seed=3)
    rank1 = InfiniteDistributedSampler(dataset_size=5, world_size=2, rank=1, shuffle=True, seed=3)
    assert len(collect_rank_stream(rank0, 3)) == len(collect_rank_stream(rank1, 3)) == 3


def test_build_train_dataloader_offsets_generator_seed_on_resume(monkeypatch):
    recorded = []

    class RecordingDataLoader:
        def __init__(self, *args, **kwargs):
            recorded.append(kwargs["generator"].initial_seed())

    monkeypatch.setattr("dinosar.data.DataLoader", RecordingDataLoader)

    dataset = list(range(8))
    from dinosar.data import build_train_dataloader

    build_train_dataloader(
        dataset,
        batch_size=2,
        num_workers=0,
        prefetch_factor=None,
        pin_memory=False,
        seed=42,
        start_step=0,
    )
    build_train_dataloader(
        dataset,
        batch_size=2,
        num_workers=0,
        prefetch_factor=None,
        pin_memory=False,
        seed=42,
        start_step=5,
    )

    assert recorded == [42 ^ (get_rank() << 16) ^ 0, 42 ^ (get_rank() << 16) ^ 5]


def test_build_train_dataloader_advance_accounts_for_grad_accum(monkeypatch):
    """The sampler advance must be start_step * grad_accum_steps * batch_size."""
    recorded_advances = []

    original_init = InfiniteDistributedSampler.__init__

    def spy_init(self, *args, **kwargs):
        recorded_advances.append(kwargs.get("advance", 0))
        return original_init(self, *args, **kwargs)

    monkeypatch.setattr(InfiniteDistributedSampler, "__init__", spy_init)

    dataset = list(range(16))
    from dinosar.data import build_train_dataloader

    build_train_dataloader(
        dataset,
        batch_size=2,
        num_workers=0,
        prefetch_factor=None,
        pin_memory=False,
        seed=0,
        start_step=5,
        grad_accum_steps=1,
    )
    build_train_dataloader(
        dataset,
        batch_size=2,
        num_workers=0,
        prefetch_factor=None,
        pin_memory=False,
        seed=0,
        start_step=5,
        grad_accum_steps=4,
    )

    assert recorded_advances == [5 * 1 * 2, 5 * 4 * 2]


def test_grad_accum_resume_sample_stream_continuity():
    """An uninterrupted stream of N samples must equal the resumed stream
    starting at the same offset, even with grad_accum_steps > 1."""
    dataset_size = 20
    batch_size = 2
    grad_accum_steps = 3
    steps_before_resume = 4
    samples_consumed = steps_before_resume * grad_accum_steps * batch_size

    baseline = InfiniteDistributedSampler(
        dataset_size=dataset_size,
        world_size=1,
        rank=0,
        shuffle=True,
        seed=7,
        advance=0,
    )
    baseline_it = iter(baseline)
    for _ in range(samples_consumed):
        next(baseline_it)
    baseline_after = [next(baseline_it) for _ in range(10)]

    resumed = InfiniteDistributedSampler(
        dataset_size=dataset_size,
        world_size=1,
        rank=0,
        shuffle=True,
        seed=7,
        advance=samples_consumed,
    )
    resumed_it = iter(resumed)
    resumed_after = [next(resumed_it) for _ in range(10)]

    assert baseline_after == resumed_after


# ---------------------------------------------------------------------------
# GaussianBlur tests
# ---------------------------------------------------------------------------

_SAMPLE = torch.rand(1, 64, 64)


class TestGaussianBlur:
    def test_preserves_shape(self):
        out = GaussianBlur(p=1.0)(_SAMPLE)
        assert out.shape == _SAMPLE.shape

    def test_p_zero_is_identity(self):
        assert torch.equal(GaussianBlur(p=0.0)(_SAMPLE), _SAMPLE)

    def test_default_p_always_applies(self):
        random.seed(0)
        out = GaussianBlur(p=1.0, sigma=(0.5, 0.5))(_SAMPLE)
        assert not torch.equal(out, _SAMPLE)


# ---------------------------------------------------------------------------
# SigmaClip tests
# ---------------------------------------------------------------------------


class TestSigmaClip:
    def test_preserves_shape(self):
        out = SigmaClip(p=1.0)(_SAMPLE)
        assert out.shape == _SAMPLE.shape

    def test_p_zero_is_identity(self):
        assert torch.equal(SigmaClip(p=0.0)(_SAMPLE), _SAMPLE)

    def test_output_in_unit_range(self):
        out = SigmaClip(p=1.0, sigma_min=2.0, sigma_max=2.0)(_SAMPLE)
        assert out.min() >= 0.0
        assert out.max() <= 1.0 + 1e-6

    def test_constant_input_unchanged(self):
        x = torch.full((1, 8, 8), 0.5)
        out = SigmaClip(p=1.0)(x)
        assert torch.equal(out, x)

    def test_clips_outliers(self):
        x = torch.zeros(1, 10, 10)
        x[0, 0, 0] = 100.0
        out = SigmaClip(p=1.0, sigma_min=2.0, sigma_max=2.0)(x)
        assert out.max() <= 1.0 + 1e-6
        assert out.min() >= 0.0


class TestNormalize:
    def test_caches_by_dtype_and_preserves_values(self):
        normalize = Normalize(mean=0.5, std=0.25)
        x32 = torch.full((1, 4, 4), 0.75, dtype=torch.float32)
        out32 = normalize(x32)
        assert torch.allclose(out32, torch.ones_like(out32))
        assert normalize._cached_key == ("cpu", None, torch.float32)

        x64 = torch.full((1, 2, 3), 0.75, dtype=torch.float64)
        out64 = normalize(x64)
        assert torch.allclose(out64, torch.ones_like(out64))
        assert out64.dtype == torch.float64
        assert normalize._cached_key == ("cpu", None, torch.float64)


class TestDINOAugmentationConfig:
    def test_none_config_gives_baseline_pipeline(self):
        aug = DINOAugmentation(crop_size=64, crop_scale=(0.5, 1.0), gaussian_blur_probability=0.5)
        # ToTensor + Crop + HFlip + Blur + Normalize = 5
        assert len(aug.transforms.transforms) == 5

    def test_gaussian_blur_can_be_disabled_via_flag(self):
        cfg = {"gaussian_blur": {"enabled": False}}
        aug = DINOAugmentation(
            crop_size=64, crop_scale=(0.5, 1.0), gaussian_blur_probability=0.5, augmentation_config=cfg
        )
        assert len(aug.transforms.transforms) == 4

    def test_sigma_clip_enabled_adds_transform(self):
        cfg = {"sigma_clip": {"enabled": True, "p": 0.3, "sigma_min": 2.0, "sigma_max": 4.0}}
        aug = DINOAugmentation(
            crop_size=64, crop_scale=(0.5, 1.0), gaussian_blur_probability=0.5, augmentation_config=cfg
        )
        # ToTensor + Crop + HFlip + SigmaClip + Blur + Normalize = 6
        assert len(aug.transforms.transforms) == 6
        assert isinstance(aug.transforms.transforms[3], SigmaClip)
        assert isinstance(aug.transforms.transforms[4], GaussianBlur)

    def test_multicrop_with_sigma_clip_runs(self):
        cfg = {"sigma_clip": {"enabled": True, "p": 1.0, "sigma_min": 2.0, "sigma_max": 2.0}}
        transform = build_multicrop_transform(
            global_size=64,
            local_size=32,
            global_scale=(0.5, 1.0),
            local_scale=(0.2, 0.5),
            num_local_crops=2,
            augmentation_config=cfg,
        )
        x = torch.rand(1, 128, 128)
        views = transform(x)
        assert len(views) == 4
        assert views[0].shape == (1, 64, 64)
        assert views[2].shape == (1, 32, 32)
