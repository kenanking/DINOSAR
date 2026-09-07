from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DETECTRON2_ROOT = PROJECT_ROOT / "dependency" / "detectron2"
DETECTRON2_CONFIGS = DETECTRON2_ROOT / "configs"
DINOV3_ROOT = PROJECT_ROOT / "dependency" / "dinov3"


def setup_dependency_paths() -> None:
    """Make vendored dependencies importable from scripts, configs, and tools."""
    for path in (PROJECT_ROOT, DETECTRON2_ROOT, DETECTRON2_CONFIGS, DINOV3_ROOT):
        if path.exists():
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)
