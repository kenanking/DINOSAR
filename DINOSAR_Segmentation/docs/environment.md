# Environment

Use a dedicated conda environment.

```bash
scripts/create_conda_env.sh
conda activate dinosar-seg
```

The base environment installs PyTorch, MMSegmentation, MMEngine, timm, and data-preparation dependencies. `scripts/create_conda_env.sh` exports `PYTHONNOUSERSITE=1` so the environment does not silently reuse `~/.local` packages.

The root `requirements.txt` installs PyTorch, MMSegmentation, MMEngine, timm, MMCV, and data-preparation dependencies.

Current verified stack:

```text
torch 2.11.0+cu128
torchvision 0.26.0+cu128
mmcv 2.1.0
mmengine 0.10.7
mmseg 1.2.2
timm 1.0.26
MMCV compiler GCC 11.4, CUDA 12.8
```

Expected import check after MMCV is installed:

```bash
python - <<'PY'
import torch, mmcv, mmengine, mmseg, timm
import projects.dinosar_mmseg
print(torch.__version__)
print(mmcv.__version__, mmengine.__version__, mmseg.__version__)
PY
```
