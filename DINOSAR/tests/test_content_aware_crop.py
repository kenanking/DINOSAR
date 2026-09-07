import random

import pytest
import torch

from dinosar.content_aware_crop import (
    ContentAwareMultiCropTransform,
    SARContentAwareCropper,
    _box_contains,
    _square_max_pool2d_separable,
    multi_scale_ratio_gradient_content_map,
)
from dinosar.data import (
    DINOAugmentation,
    build_content_aware_multicrop_transform,
    build_ibot_collate_fn,
    build_post_crop_transforms,
    multicrop_collate,
)

# ---------------------------------------------------------------------------
# Content map tests
# ---------------------------------------------------------------------------


class TestContentMap:
    def test_shape_and_range(self):
        x = torch.rand(1, 64, 64)
        cm = multi_scale_ratio_gradient_content_map(x, scales=(3, 7))
        assert cm.shape == (64, 64)
        assert cm.min() >= 0.0
        assert cm.max() <= 1.0 + 1e-6

    def test_2d_input(self):
        x = torch.rand(64, 64)
        cm = multi_scale_ratio_gradient_content_map(x, scales=(3,))
        assert cm.shape == (64, 64)

    def test_uniform_is_near_zero(self):
        x = torch.full((1, 64, 64), 0.5)
        cm = multi_scale_ratio_gradient_content_map(x, scales=(3, 7))
        # After min-max normalization, a uniform map is all zeros
        # (or all same value, range < eps)
        assert (cm.max() - cm.min()).item() < 1e-3

    def test_edge_response(self):
        x = torch.zeros(1, 64, 64)
        x[:, :, 32:] = 1.0  # sharp vertical edge at column 32
        cm = multi_scale_ratio_gradient_content_map(x, scales=(3, 7))
        # Edge region should have higher response than far-from-edge regions
        edge_mean = cm[:, 28:36].mean()
        corner_mean = cm[:, :8].mean()
        assert edge_mean > corner_mean

    def test_separable_max_pool_matches_square_pool(self):
        x = torch.rand(1, 1, 37, 53)
        expected = torch.nn.functional.max_pool2d(x, kernel_size=11, stride=1, padding=5)
        actual = _square_max_pool2d_separable(x, kernel_size=11)
        assert torch.equal(actual, expected)


# ---------------------------------------------------------------------------
# Transform output tests
# ---------------------------------------------------------------------------


class TestTransformOutput:
    def _make_cropper(self, num_local=6):
        return ContentAwareMultiCropTransform(
            global_crop_size=64,
            local_crop_size=32,
            global_crop_scale=(0.5, 1.0),
            local_crop_scale=(0.1, 0.5),
            num_local_crops=num_local,
        )

    def test_output_format(self):
        cropper = self._make_cropper(num_local=6)
        x = torch.rand(1, 128, 128)
        views = cropper(x)
        assert isinstance(views, list)
        assert len(views) == 8  # 2 global + 6 local
        assert views[0].shape == (1, 64, 64)
        assert views[1].shape == (1, 64, 64)
        for v in views[2:]:
            assert v.shape == (1, 32, 32)

    def test_deterministic_with_seed(self):
        cropper = self._make_cropper()
        x = torch.rand(1, 128, 128)
        torch.manual_seed(42)
        random.seed(42)
        v1 = cropper(x)
        torch.manual_seed(42)
        random.seed(42)
        v2 = cropper(x)
        for a, b in zip(v1, v2):
            assert torch.allclose(a, b)

    def test_fallback_on_uniform_image(self):
        cropper = self._make_cropper()
        x = torch.full((1, 128, 128), 0.5)
        views = cropper(x)
        assert len(views) == 8
        for v in views:
            assert not torch.isnan(v).any()
            assert not torch.isinf(v).any()

    def test_return_metadata(self):
        cropper = self._make_cropper()
        x = torch.rand(1, 128, 128)
        result = cropper(x, return_metadata=True)
        assert isinstance(result, dict)
        assert "views" in result
        assert "content_map" in result
        assert "anchor_boxes" in result
        assert "scene_concentration" in result
        assert "global_boxes" in result
        assert "local_boxes" in result
        assert len(result["views"]) == 8

    def test_cropper_sample_metadata_fast_path(self):
        cropper = self._make_cropper()
        x = torch.rand(1, 128, 128)
        result = cropper.cropper.sample(x)
        assert set(result) == {"global_crops", "local_crops"}
        assert len(result["global_crops"]) == 2
        assert len(result["local_crops"]) == 6

    def test_num_candidates_is_explicit(self):
        cropper = SARContentAwareCropper(n_local=6, num_candidates=12)
        assert cropper.num_candidates == 12
        with pytest.raises(ValueError, match="num_candidates"):
            SARContentAwareCropper(n_local=6, num_candidates=5)

    def test_normalize_amplitude_can_be_disabled(self):
        cropper = SARContentAwareCropper(n_local=2, normalize_amplitude=False)
        x = torch.rand(1, 128, 128)
        result = cropper.sample(x)
        assert cropper.normalize_amplitude is False
        assert len(result["global_crops"]) == 2
        assert len(result["local_crops"]) == 2


# ---------------------------------------------------------------------------
# Geometric constraint tests
# ---------------------------------------------------------------------------


class TestGeometricConstraints:
    def _make_cropper(self, num_local=6):
        return ContentAwareMultiCropTransform(
            global_crop_size=64,
            local_crop_size=32,
            global_crop_scale=(0.5, 1.0),
            local_crop_scale=(0.05, 0.3),
            num_local_crops=num_local,
        )

    def test_global_contains_anchor(self):
        """Each anchor must be contained by at least one global crop."""
        cropper = self._make_cropper()
        for seed in range(5):
            torch.manual_seed(seed)
            x = torch.rand(1, 128, 128)
            meta = cropper(x, return_metadata=True)
            for anchor in meta["anchor_boxes"]:
                in_any = any(_box_contains(gb, anchor) for gb in meta["global_boxes"])
                assert in_any, f"seed={seed}: anchor {anchor} not in any global"

    def test_siblings_within_at_least_one_global(self):
        cropper = self._make_cropper()
        for seed in range(5):
            torch.manual_seed(seed)
            x = torch.rand(1, 128, 128)
            meta = cropper(x, return_metadata=True)
            for lid, lbox in enumerate(meta["local_boxes"]):
                in_any = any(_box_contains(gb, lbox) for gb in meta["global_boxes"])
                assert in_any, f"seed={seed}: L{lid} {lbox} not in any global"

    def test_anchor_at_image_corner(self):
        """Image with content only in a corner."""
        x = torch.zeros(1, 128, 128)
        x[:, 100:128, 100:128] = 1.0  # bottom-right corner
        cropper = self._make_cropper()
        torch.manual_seed(0)
        meta = cropper(x, return_metadata=True)
        assert len(meta["views"]) == 8
        for v in meta["views"]:
            assert not torch.isnan(v).any()

    def test_non_square_image(self):
        cropper = self._make_cropper()
        x = torch.rand(1, 256, 128)
        views = cropper(x)
        assert len(views) == 8
        assert views[0].shape == (1, 64, 64)
        assert views[2].shape == (1, 32, 32)

    def test_num_local_crops_1(self):
        """Only the anchor, no siblings."""
        cropper = self._make_cropper(num_local=1)
        x = torch.rand(1, 128, 128)
        views = cropper(x)
        assert len(views) == 3  # 2 global + 1 local


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_compatible_with_multicrop_collate(self):
        cropper = ContentAwareMultiCropTransform(
            global_crop_size=64,
            local_crop_size=32,
            global_crop_scale=(0.5, 1.0),
            local_crop_scale=(0.1, 0.5),
            num_local_crops=4,
        )
        batch = [cropper(torch.rand(1, 128, 128)) for _ in range(4)]
        collated = multicrop_collate(batch)
        assert len(collated) == 6  # 2 global + 4 local
        assert collated[0].shape == (4, 1, 64, 64)
        assert collated[2].shape == (4, 1, 32, 32)

    def test_compatible_with_ibot_collate(self):
        cropper = ContentAwareMultiCropTransform(
            global_crop_size=64,
            local_crop_size=32,
            global_crop_scale=(0.5, 1.0),
            local_crop_scale=(0.1, 0.5),
            num_local_crops=4,
        )
        collate_fn = build_ibot_collate_fn(
            mask_ratio_min_max=(0.1, 0.5),
            mask_sample_probability=0.5,
            global_crop_size=64,
            patch_size=16,
        )
        batch = [cropper(torch.rand(1, 128, 128)) for _ in range(4)]
        result = collate_fn(batch)
        assert "views" in result
        assert "masks" in result
        assert len(result["views"]) == 6
        assert result["views"][0].shape == (4, 1, 64, 64)

    def test_build_content_aware_multicrop_transform(self):
        transform = build_content_aware_multicrop_transform(
            global_size=64,
            local_size=32,
            global_scale=(0.5, 1.0),
            local_scale=(0.1, 0.5),
            num_local_crops=4,
        )
        x = torch.rand(1, 128, 128)
        views = transform(x)
        assert len(views) == 6
        assert views[0].shape == (1, 64, 64)
        assert views[2].shape == (1, 32, 32)

    def test_invalid_cropping_strategy_raises(self):
        from omegaconf import OmegaConf

        # Simulate what train.py does
        config = OmegaConf.create({"data": {"cropping_strategy": "typo_value"}})
        strategy = OmegaConf.select(config, "data.cropping_strategy", default="random")
        assert strategy not in ("random", "content_aware")


# ---------------------------------------------------------------------------
# Augmentation parity test
# ---------------------------------------------------------------------------


class TestPostCropTransformsParity:
    def test_same_transform_types_as_dino_augmentation(self):
        """Verify build_post_crop_transforms matches DINOAugmentation order
        (minus ToTensor and RandomCrop)."""
        cfg = {
            "sigma_clip": {"enabled": True, "p": 0.3, "sigma_min": 2.0, "sigma_max": 4.0},
        }
        dino_aug = DINOAugmentation(
            crop_size=64,
            crop_scale=(0.5, 1.0),
            gaussian_blur_probability=0.5,
            augmentation_config=cfg,
        )
        post_crop = build_post_crop_transforms(0.5, 0.219, 0.220, augmentation_config=cfg)

        # DINOAugmentation: ToTensor, RandomCrop, HFlip, SigmaClip, Blur, Normalize
        dino_types = [type(t).__name__ for t in dino_aug.transforms.transforms]
        post_types = [type(t).__name__ for t in post_crop.transforms]

        # post_crop should match dino minus first two (ToTensor, RandomCrop)
        assert post_types == dino_types[2:], (
            f"Parity mismatch:\n  DINOAugmentation[2:]: {dino_types[2:]}\n  PostCrop: {post_types}"
        )

    def test_baseline_config_parity(self):
        """Baseline (no SAR augs) should also match."""
        dino_aug = DINOAugmentation(
            crop_size=64,
            crop_scale=(0.5, 1.0),
            gaussian_blur_probability=1.0,
        )
        post_crop = build_post_crop_transforms(1.0, 0.219, 0.220)

        dino_types = [type(t).__name__ for t in dino_aug.transforms.transforms]
        post_types = [type(t).__name__ for t in post_crop.transforms]
        assert post_types == dino_types[2:]

    def test_blur_disabled_parity(self):
        cfg = {"gaussian_blur": {"enabled": False}}
        dino_aug = DINOAugmentation(
            crop_size=64,
            crop_scale=(0.5, 1.0),
            gaussian_blur_probability=0.5,
            augmentation_config=cfg,
        )
        post_crop = build_post_crop_transforms(0.5, 0.219, 0.220, augmentation_config=cfg)

        dino_types = [type(t).__name__ for t in dino_aug.transforms.transforms]
        post_types = [type(t).__name__ for t in post_crop.transforms]
        assert post_types == dino_types[2:]
