# AIR-PolSARSeg Segmentation Protocol

This workspace fine-tunes DINOSAR pretrained backbones on AIR-PolSARSeg with a
matched MMSeg protocol. The released configuration uses the
generated PNG dataset, UPerNet decoder, 72-epoch schedule, validation every
epoch, and MMSeg mIoU evaluation.

## Protocol

- Dataset: `datasets/AIR-PolSAR-Seg-mmseg`
- Classes: `Industrial`, `Natural`, `Water`, `Land_Use`, `Housing`, `Other`
- Decoder: UPerNet with auxiliary FCN head
- Schedule: 72 epochs, 88 iterations per epoch, 6336 iterations total
- Train batch size: 4
- Validation batch size: 1
- Metric: class-wise IoU/Acc, `mIoU`, `mAcc`, and `aAcc`

## Configuration

Use `configs/dinosar/vitb_upernet_airseg_pseudo_gray_60e_72ep.py`.
It initializes ViT-B from the released UniSAR-7M 60-epoch checkpoint,
reads the prepared PNGs as grayscale, and fine-tunes for 72 epochs.
See [the README](../README.md) for weight download, training, and evaluation.
Logs and fine-tuned checkpoints are written under `experiments/work_dirs/`.
