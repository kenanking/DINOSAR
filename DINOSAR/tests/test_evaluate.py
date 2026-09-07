from importlib import resources

import numpy as np
import torch
from PIL import Image

from dinosar.config import load_eval_config
from dinosar.evaluate import (
    build_classification_datasets,
    build_eval_loader,
    build_eval_transform,
    build_few_shot_subset,
    build_protocol_train_val_subsets,
    extract_features,
    load_eval_backbone,
    run_few_shot_evaluation,
    run_full_finetune_evaluation,
    run_knn_eval,
    run_knn_evaluation,
    run_linear_probe,
    run_linear_probe_evaluation,
)
from dinosar.model import DINO


def _write_image(path, value):
    array = np.full((32, 32, 3), value, dtype=np.uint8)
    Image.fromarray(array).save(path)


def _write_tiny_imagefolder(root):
    for split in ("train", "val"):
        for class_name in ("class0", "class1"):
            class_dir = root / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            _write_image(class_dir / "image_0.png", 32 if class_name == "class0" else 224)
            _write_image(class_dir / "image_1.png", 64 if split == "train" else 192)
    return root


def _write_multisample_imagefolder(root, *, train_images_per_class=4, val_images_per_class=2):
    for split, count in (("train", train_images_per_class), ("val", val_images_per_class)):
        for class_index, class_name in enumerate(("class0", "class1")):
            class_dir = root / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            for image_index in range(count):
                value = 32 + (class_index * 128) + (image_index * 8)
                _write_image(class_dir / f"image_{image_index}.png", value)
    return root


def _write_tiny_mstar_benchmark(root, *, train_images_per_class=10, test_images_per_class=4):
    for split, count in (("TRAIN", train_images_per_class), ("TEST", test_images_per_class)):
        for class_index, class_name in enumerate(("class0", "class1")):
            class_dir = root / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            for image_index in range(count):
                value = 32 + (class_index * 128) + (image_index * 8)
                _write_image(class_dir / f"image_{image_index}.png", value)
    return root


def _tiny_config_preamble(
    *, data_root, checkpoint_path, output_dir, train_split="train", val_split="val", extra_eval_lines=None
):
    lines = [
        "model:",
        "  arch: vit_small",
        "  patch_size: 16",
        "  in_chans: 1",
        "  checkpoint_type: backbone",
        f"  checkpoint_path: {checkpoint_path}",
        "data:",
        f"  root: {data_root}",
        f"  train_split: {train_split}",
        f"  val_split: {val_split}",
        "  crop_size: 32",
        "  normalize_mean: 0.5",
        "  normalize_std: 0.5",
        "eval:",
        f"  output_dir: {output_dir}",
        "  batch_size: 2",
        "  num_workers: 0",
    ]
    if extra_eval_lines:
        lines.extend(extra_eval_lines)
    return lines


def _write_tiny_knn_config(config_path, *, data_root, checkpoint_path, output_dir):
    lines = _tiny_config_preamble(
        data_root=data_root,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        extra_eval_lines=["  normalize_features: true", "  feature_key: x_norm_clstoken"],
    )
    lines.extend(["knn:", "  ks:", "    - 1", "  temperature: 0.07"])
    config_path.write_text("\n".join(lines) + "\n")


def _write_tiny_linear_probe_config(config_path, *, data_root, checkpoint_path, output_dir):
    lines = _tiny_config_preamble(
        data_root=data_root,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        extra_eval_lines=["  normalize_features: true", "  feature_key: x_norm_clstoken"],
    )
    lines.extend(["linear:", "  epochs: 2", "  lr: 0.1", "  weight_decay: 0.0", "  use_batch_norm_head: true"])
    config_path.write_text("\n".join(lines) + "\n")


def _write_tiny_few_shot_config(config_path, *, data_root, checkpoint_path, output_dir):
    lines = _tiny_config_preamble(
        data_root=data_root,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        train_split="TRAIN",
        val_split="TEST",
    )
    lines.extend(
        [
            "few_shot:",
            "  num_shots: 2",
            "  seeds:",
            "    - 0",
            "  epochs: 2",
            "  lr: 0.1",
            "  weight_decay: 0.0",
            "  batch_size: 2",
            "  warmup_epochs: 0",
            "  warmup_start_lr: 0.0",
        ]
    )
    config_path.write_text("\n".join(lines) + "\n")


def _write_tiny_full_finetune_config(config_path, *, data_root, checkpoint_path, output_dir, extra_full_lines=None):
    lines = _tiny_config_preamble(
        data_root=data_root,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        extra_eval_lines=["  task: full_finetune"],
    )
    lines.extend(
        [
            "full_finetune:",
            "  seeds:",
            "    - 0",
            "  epochs: 2",
            "  lr: 0.1",
            "  weight_decay: 0.0",
            "  batch_size: 2",
            "  warmup_epochs: 0",
            "  warmup_start_lr: 0.0",
        ]
    )
    if extra_full_lines:
        lines.extend(extra_full_lines)
    config_path.write_text("\n".join(lines) + "\n")


def test_knn_correctness():
    train_features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    train_labels = torch.tensor([0, 1])
    val_features = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
    val_labels = torch.tensor([0, 1])
    metrics = run_knn_eval(
        train_features=train_features,
        train_labels=train_labels,
        val_features=val_features,
        val_labels=val_labels,
        ks=(1,),
        temperature=0.07,
        normalize_features=True,
    )
    assert metrics["top1_k1"] == 1.0


def test_knn_chunked_matches_unchunked():
    train_features = torch.tensor([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.2, 0.8]])
    train_labels = torch.tensor([0, 0, 1, 1])
    val_features = torch.tensor([[0.9, 0.1], [0.1, 0.9], [0.6, 0.4]])
    val_labels = torch.tensor([0, 1, 0])

    unchunked = run_knn_eval(
        train_features=train_features,
        train_labels=train_labels,
        val_features=val_features,
        val_labels=val_labels,
        ks=(1, 3),
        temperature=0.07,
        normalize_features=True,
        query_chunk_size=val_features.shape[0],
    )
    chunked = run_knn_eval(
        train_features=train_features,
        train_labels=train_labels,
        val_features=val_features,
        val_labels=val_labels,
        ks=(1, 3),
        temperature=0.07,
        normalize_features=True,
        query_chunk_size=1,
    )

    assert chunked == unchunked


def test_linear_probe_convergence():
    train_features = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    train_labels = torch.tensor([0, 1])
    val_features = torch.tensor([[1.5, 0.0], [0.0, 1.5]])
    val_labels = torch.tensor([0, 1])
    result = run_linear_probe(
        train_features=train_features,
        train_labels=train_labels,
        val_features=val_features,
        val_labels=val_labels,
        epochs=20,
        lr=0.1,
        weight_decay=0.0,
        num_classes=2,
        device=torch.device("cpu"),
    )
    assert result["best_top1"] >= 0.99


def test_build_few_shot_subset_is_deterministic_and_balanced(tmp_path):
    root = _write_multisample_imagefolder(tmp_path / "data")
    transform = build_eval_transform(crop_size=32, normalize_mean=0.5, normalize_std=0.5)
    train_dataset, _ = build_classification_datasets(root, "train", "val", transform=transform)

    subset_a = build_few_shot_subset(train_dataset, num_shots=2, seed=7)
    subset_b = build_few_shot_subset(train_dataset, num_shots=2, seed=7)

    assert subset_a.indices == subset_b.indices
    assert len(subset_a.indices) == 4

    labels = [train_dataset.targets[index] for index in subset_a.indices]
    assert labels.count(0) == 2
    assert labels.count(1) == 2


def test_build_protocol_train_val_subsets_is_deterministic_and_balanced(tmp_path):
    root = _write_multisample_imagefolder(tmp_path / "data", train_images_per_class=10, val_images_per_class=2)
    transform = build_eval_transform(crop_size=32, normalize_mean=0.5, normalize_std=0.5)
    train_dataset, _ = build_classification_datasets(root, "train", "val", transform=transform)

    train_subset_a, val_subset_a = build_protocol_train_val_subsets(train_dataset, val_ratio=0.2, seed=3)
    train_subset_b, val_subset_b = build_protocol_train_val_subsets(train_dataset, val_ratio=0.2, seed=3)

    assert train_subset_a.indices == train_subset_b.indices
    assert val_subset_a.indices == val_subset_b.indices
    assert len(train_subset_a.indices) == 16
    assert len(val_subset_a.indices) == 4

    train_labels = [train_dataset.targets[index] for index in train_subset_a.indices]
    val_labels = [train_dataset.targets[index] for index in val_subset_a.indices]
    assert train_labels.count(0) == 8
    assert train_labels.count(1) == 8
    assert val_labels.count(0) == 2
    assert val_labels.count(1) == 2


def test_build_few_shot_subset_accepts_subset_input(tmp_path):
    root = _write_multisample_imagefolder(tmp_path / "data", train_images_per_class=10, val_images_per_class=2)
    transform = build_eval_transform(crop_size=32, normalize_mean=0.5, normalize_std=0.5)
    train_dataset, _ = build_classification_datasets(root, "train", "val", transform=transform)
    protocol_train_subset, _ = build_protocol_train_val_subsets(train_dataset, val_ratio=0.2, seed=3)

    few_shot_subset = build_few_shot_subset(protocol_train_subset, num_shots=4, seed=5)

    assert len(few_shot_subset.indices) == 8


def test_extract_features(tmp_path):
    root = _write_tiny_imagefolder(tmp_path)
    transform = build_eval_transform(crop_size=32, normalize_mean=0.5, normalize_std=0.5)
    train_dataset, _ = build_classification_datasets(root, "train", "val", transform=transform)
    loader = build_eval_loader(train_dataset, batch_size=2, num_workers=0)
    backbone = DINO.build_backbone_from_config(
        {"arch": "vit_small", "patch_size": 16, "in_chans": 1}, initialize_weights=True
    )
    features, labels = extract_features(backbone, loader, device=torch.device("cpu"))
    assert features.ndim == 2
    assert labels.ndim == 1
    assert features.shape[0] == labels.shape[0] == len(train_dataset)


def test_extract_features_pool_patches(tmp_path):
    root = _write_multisample_imagefolder(tmp_path / "data")
    transform = build_eval_transform(crop_size=32, normalize_mean=0.5, normalize_std=0.5)
    train_dataset, _ = build_classification_datasets(root, "train", "val", transform=transform)
    loader = build_eval_loader(train_dataset, batch_size=2, num_workers=0)
    backbone = DINO.build_backbone_from_config(
        {"arch": "vit_small", "patch_size": 16, "in_chans": 1}, initialize_weights=True
    )

    features, labels = extract_features(backbone, loader, device=torch.device("cpu"), pool_patches=True)

    assert features.shape == (len(train_dataset), backbone.embed_dim)
    assert labels.shape == (len(train_dataset),)


def test_build_eval_transform_uses_resize_and_pad():
    transform = build_eval_transform(
        crop_size=32,
        normalize_mean=0.5,
        normalize_std=0.5,
    )

    names = [type(step).__name__ for step in transform.transforms]

    assert names == ["Grayscale", "ResizeAndPad", "ToTensor", "Normalize"]


def test_build_eval_transform_supports_disabling_normalize():
    transform = build_eval_transform(
        crop_size=32,
        normalize_mean=0.5,
        normalize_std=0.5,
        apply_normalize=False,
    )

    names = [type(step).__name__ for step in transform.transforms]

    assert names == ["Grayscale", "ResizeAndPad", "ToTensor"]


def test_build_eval_transform_supports_three_output_channels():
    transform = build_eval_transform(
        crop_size=32,
        normalize_mean=0.5,
        normalize_std=0.5,
        num_output_channels=3,
        apply_normalize=False,
    )

    image = Image.fromarray(np.full((32, 32, 3), 128, dtype=np.uint8))
    tensor = transform(image)

    assert tensor.shape == (3, 32, 32)


def test_load_eval_backbone_uses_model_config_register_tokens(tmp_path):
    config = {
        "arch": "vit_small",
        "patch_size": 16,
        "in_chans": 1,
        "num_register_tokens": 4,
    }
    backbone = DINO.build_backbone_from_config(config, initialize_weights=True)
    with torch.no_grad():
        backbone.cls_token.fill_(0.25)
        backbone.storage_tokens.fill_(0.5)

    checkpoint_path = tmp_path / "backbone.pth"
    torch.save(backbone.state_dict(), checkpoint_path)

    loaded = load_eval_backbone(config, checkpoint_path, checkpoint_type="backbone")

    assert loaded.storage_tokens is not None
    assert loaded.storage_tokens.shape == (1, 4, loaded.embed_dim)
    assert torch.allclose(loaded.cls_token, backbone.cls_token)
    assert torch.allclose(loaded.storage_tokens, backbone.storage_tokens)


def test_knn_entrypoint(tmp_path):
    data_root = _write_tiny_imagefolder(tmp_path / "data")
    output_dir = tmp_path / "outputs"
    config_path = tmp_path / "knn.yaml"
    config = load_eval_config(resources.files("dinosar").joinpath("configs", "eval", "base.yaml"))
    backbone = DINO.build_backbone_from_config(
        {
            "arch": config.model.arch,
            "patch_size": config.model.patch_size,
            "in_chans": config.model.in_chans,
        },
        initialize_weights=True,
    )
    with torch.no_grad():
        backbone.cls_token.fill_(0.25)
    checkpoint_path = tmp_path / "backbone.pth"
    torch.save(backbone.state_dict(), checkpoint_path)
    _write_tiny_knn_config(config_path, data_root=data_root, checkpoint_path=checkpoint_path, output_dir=output_dir)
    config = load_eval_config(config_path)
    run_knn_evaluation(config)
    assert (output_dir / "metrics.json").exists()


def test_linear_probe_entrypoint(tmp_path):
    data_root = _write_tiny_imagefolder(tmp_path / "data")
    output_dir = tmp_path / "outputs"
    config_path = tmp_path / "linear_probe.yaml"
    config = load_eval_config(resources.files("dinosar").joinpath("configs", "eval", "base.yaml"))
    backbone = DINO.build_backbone_from_config(
        {
            "arch": config.model.arch,
            "patch_size": config.model.patch_size,
            "in_chans": config.model.in_chans,
        },
        initialize_weights=True,
    )
    with torch.no_grad():
        backbone.cls_token.fill_(0.25)
    checkpoint_path = tmp_path / "backbone.pth"
    torch.save(backbone.state_dict(), checkpoint_path)
    _write_tiny_linear_probe_config(
        config_path, data_root=data_root, checkpoint_path=checkpoint_path, output_dir=output_dir
    )
    config = load_eval_config(config_path)
    run_linear_probe_evaluation(config)
    assert (output_dir / "metrics.json").exists()


def test_few_shot_entrypoint(tmp_path):
    data_root = _write_tiny_mstar_benchmark(tmp_path / "MSTAR")
    output_dir = tmp_path / "outputs"
    config_path = tmp_path / "few_shot.yaml"
    config = load_eval_config(resources.files("dinosar").joinpath("configs", "eval", "base.yaml"))
    backbone = DINO.build_backbone_from_config(
        {
            "arch": config.model.arch,
            "patch_size": config.model.patch_size,
            "in_chans": config.model.in_chans,
        },
        initialize_weights=True,
    )
    with torch.no_grad():
        backbone.cls_token.fill_(0.25)
    checkpoint_path = tmp_path / "backbone.pth"
    torch.save(backbone.state_dict(), checkpoint_path)
    _write_tiny_few_shot_config(
        config_path, data_root=data_root, checkpoint_path=checkpoint_path, output_dir=output_dir
    )
    config = load_eval_config(config_path)
    metrics = run_few_shot_evaluation(config)
    assert metrics["num_shots"] == 2
    assert "summary" in metrics
    assert (output_dir / "metrics.json").exists()


def test_full_finetune_entrypoint(tmp_path):
    data_root = _write_multisample_imagefolder(tmp_path / "data", train_images_per_class=4, val_images_per_class=2)
    output_dir = tmp_path / "outputs"
    config_path = tmp_path / "full_finetune.yaml"
    config = load_eval_config(resources.files("dinosar").joinpath("configs", "eval", "base.yaml"))
    backbone = DINO.build_backbone_from_config(
        {
            "arch": config.model.arch,
            "patch_size": config.model.patch_size,
            "in_chans": config.model.in_chans,
        },
        initialize_weights=True,
    )
    with torch.no_grad():
        backbone.cls_token.fill_(0.25)
    checkpoint_path = tmp_path / "backbone.pth"
    torch.save(backbone.state_dict(), checkpoint_path)
    _write_tiny_full_finetune_config(
        config_path, data_root=data_root, checkpoint_path=checkpoint_path, output_dir=output_dir
    )
    config = load_eval_config(config_path)
    metrics = run_full_finetune_evaluation(config)
    assert metrics["protocol"] == "full_finetune"
    assert "summary" in metrics
    assert (output_dir / "metrics.json").exists()


def test_full_finetune_entrypoint_supports_preservation_options(tmp_path):
    data_root = _write_multisample_imagefolder(tmp_path / "data", train_images_per_class=4, val_images_per_class=2)
    output_dir = tmp_path / "outputs"
    config_path = tmp_path / "full_finetune.yaml"
    config = load_eval_config(resources.files("dinosar").joinpath("configs", "eval", "base.yaml"))
    backbone = DINO.build_backbone_from_config(
        {
            "arch": config.model.arch,
            "patch_size": config.model.patch_size,
            "in_chans": config.model.in_chans,
        },
        initialize_weights=True,
    )
    checkpoint_path = tmp_path / "backbone.pth"
    torch.save(backbone.state_dict(), checkpoint_path)
    _write_tiny_full_finetune_config(
        config_path,
        data_root=data_root,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        extra_full_lines=[
            "  freeze_patch_embed: true",
            "  freeze_blocks: 1",
            "  head_warmup_epochs: 1",
            "  backbone_lr_scale: 0.5",
            "  l2_sp_weight: 1.0e-4",
            "  horizontal_flip_p: 0.0",
            "  vertical_flip_p: 0.0",
        ],
    )
    config = load_eval_config(config_path)
    metrics = run_full_finetune_evaluation(config)

    assert metrics["finetune_options"]["freeze_patch_embed"] is True
    assert metrics["finetune_options"]["freeze_blocks"] == 1
    assert metrics["finetune_options"]["head_warmup_epochs"] == 1
    assert metrics["finetune_options"]["backbone_lr_scale"] == 0.5
    assert metrics["finetune_options"]["l2_sp_weight"] == 1.0e-4
