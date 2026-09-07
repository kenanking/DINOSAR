# DINOSAR Retrieval

Fine-tune a CLIP dual encoder on SARVLM, initializing its vision tower with
[DINOSAR ViT-B/16 pretrained for 60 epochs](https://huggingface.co/YTang/DINOSAR).
Run commands from `DINOSAR_Retrieval/` with Python 3.11 or newer.

## Setup

```bash
pip install -e .
uvx hf download YTang/DINOSAR dinosar_b16_unisar7m_60e.pth --local-dir weights
```

Initialize the text tower from OpenCLIP ViT-B-32 (LAION-2B):

```bash
python - <<'PYTEXT'
from pathlib import Path
import open_clip
import torch
model, _, _ = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='laion2b_s34b_b79k')
Path('weights').mkdir(exist_ok=True)
torch.save(model.state_dict(), 'weights/openclip_vitb32_laion2b.pt')
PYTEXT
```

Obtain the SARVLM image and CSV release. Set `SARVLM_CSV` and the image-root
constants in `configs/common/base.py` to local paths, preserving the directory
structure referenced by the CSVs. Required files are `train.csv` and
`val_uni_image_and_caption_5000_L40.csv`; `val_uni_image_and_caption.csv`
is used for optional full-set evaluation.

## Train and evaluate

`configs/train/DINOSAR.py` uses single-channel 256 × 256 images, four register
tokens, mean 0.219 and standard deviation 0.220. Both towers are fine-tuned
for 10 epochs with batch size 256, AdamW lr=5e-5, and false-negative masking.

```bash
python scripts/train_clip.py --config configs/train/DINOSAR.py
python scripts/evaluate.py --config configs/train/DINOSAR.py \
  --checkpoint outputs/DINOSAR/last.pth
```

Append `--resume auto` to the training command to resume. Evaluation uses the
5,000-pair SARVLM split and reports image-to-text and text-to-image R@1/5/10
and their mean. The evaluation checkpoint is the fine-tuned dual encoder;
the released DINOSAR file initializes only its vision backbone.
