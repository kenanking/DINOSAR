"""Tests for ImageFolderDataset index-file loading and generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from dinosar.data import INDEX_FILENAME, ImageFolderDataset, build_image_index


def _make_image(path: Path, value: int = 128) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((4, 4), value, dtype=np.uint8)).save(path)


@pytest.fixture
def sample_root(tmp_path: Path) -> Path:
    _make_image(tmp_path / "a.png")
    _make_image(tmp_path / "sub" / "b.png")
    _make_image(tmp_path / "sub" / "deep" / "c.png")
    (tmp_path / "ignore.txt").write_text("not an image")
    return tmp_path


def test_dataset_builds_index_by_default(sample_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ds = ImageFolderDataset(sample_root)

    assert len(ds) == 3
    assert (sample_root / INDEX_FILENAME).is_file()
    assert ds.samples == ["a.png", "sub/b.png", "sub/deep/c.png"]

    output = capsys.readouterr().out
    assert "[data] building image index:" in output
    assert "[data] wrote image index:" in output
    assert "3 images" in output


def test_dataset_falls_back_to_scan_when_not_required(sample_root: Path) -> None:
    ds = ImageFolderDataset(sample_root, require_index=False)
    assert len(ds) == 3
    assert ds[0].shape == (1, 4, 4)
    assert not (sample_root / INDEX_FILENAME).exists()


def test_dataset_reads_index_file(sample_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (sample_root / INDEX_FILENAME).write_text("# header\n\nsub/b.png\na.png\n")
    ds = ImageFolderDataset(sample_root)
    assert len(ds) == 2
    assert ds.samples == ["sub/b.png", "a.png"]

    output = capsys.readouterr().out
    assert "[data] loaded image index:" in output
    assert "2 images" in output


def test_dataset_rebuilds_empty_index_file(sample_root: Path) -> None:
    (sample_root / INDEX_FILENAME).write_text("# empty\n")

    ds = ImageFolderDataset(sample_root)

    assert len(ds) == 3
    assert ds.samples == ["a.png", "sub/b.png", "sub/deep/c.png"]


def test_build_image_index_writes_sorted_relative_paths(sample_root: Path) -> None:
    samples = build_image_index(sample_root, log=False)

    assert samples == ["a.png", "sub/b.png", "sub/deep/c.png"]

    index_path = sample_root / INDEX_FILENAME
    assert index_path.is_file()

    lines = [line for line in index_path.read_text().splitlines() if line and not line.startswith("#")]
    assert lines == ["a.png", "sub/b.png", "sub/deep/c.png"]

    ds = ImageFolderDataset(sample_root)
    assert len(ds) == 3
