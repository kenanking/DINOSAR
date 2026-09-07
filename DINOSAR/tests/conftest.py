import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def image_folder(tmp_path):
    root = tmp_path / "sar"
    root.mkdir()

    names = []
    for idx in range(3):
        array = np.linspace(0, 255, num=256 * 256, dtype=np.uint8).reshape(256, 256)
        array = np.roll(array, shift=idx * 17, axis=0)
        name = f"sample_{idx}.png"
        Image.fromarray(array).save(root / name)
        names.append(name)
    (root / "index.txt").write_text("\n".join(sorted(names)) + "\n")

    return root
