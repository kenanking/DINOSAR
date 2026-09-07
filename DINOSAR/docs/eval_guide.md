# Classification Evaluation

Run from `DINOSAR/` after `uv sync`. The released checkpoints are teacher
backbone state dictionaries; no weight conversion is needed for classification.

## Download weights

```bash
uvx hf download YTang/DINOSAR \
  dinosar_s16_unisar7m_60e.pth dinosar_b16_unisar7m_60e.pth \
  --local-dir experiments/weights
```

Use the matching architecture and drop-path setting for each checkpoint:

| Model | Filename | `model.arch` | `model.drop_path_rate` |
| --- | --- | --- | ---: |
| ViT-S/16 | `dinosar_s16_unisar7m_60e.pth` | `vit_small` | 0.1 |
| ViT-B/16 | `dinosar_b16_unisar7m_60e.pth` | `vit_base` | 0.2 |

All evaluation configs default to ViT-S with four registers. The example below
selects ViT-B explicitly. Inputs are converted to grayscale, resized and padded
to 224 × 224, and normalized with mean 0.219 and standard deviation 0.220.

## Data and protocols

Prepare classification data following [datasets.md](datasets.md).
MSTAR SOC10, FUSAR-Ship, and ATRNet-STAR SOC-40 use
`train/<class>/image` and `test/<class>/image`. SAR-ACD uses class folders
at the root, named `SAR-ACD`; its few-shot protocol uses a stratified 10/90 split.

| Config under `dinosar/configs/eval/` | Protocol |
| --- | --- |
| `knn.yaml` | Frozen k-NN, k=1 and 5, temperature 0.07 |
| `linear_probe.yaml` | Frozen linear probe, 100 epochs, SGD lr=0.03 |
| `few_shot.yaml` | 10/20/40 examples per class, 50 epochs, seeds 0–4 |
| `full_finetune.yaml` | Full training split, 50 epochs, seed 0 |

```bash
uv run -m dinosar.evaluate --config dinosar/configs/eval/knn.yaml \
  model.arch=vit_base model.drop_path_rate=0.2 \
  model.checkpoint_path=experiments/weights/dinosar_b16_unisar7m_60e.pth \
  data.root=/path/to/MSTAR data.train_split=train data.val_split=test \
  eval.output_dir=experiments/eval/vitb60e/MSTAR/knn
```

For linear probing or full fine-tuning, change the config and output directory.
For few-shot evaluation on MSTAR, FUSAR-Ship, or SAR-ACD:

```bash
for shots in 10 20 40; do
  uv run -m dinosar.evaluate --config dinosar/configs/eval/few_shot.yaml \
    model.arch=vit_base model.drop_path_rate=0.2 \
    model.checkpoint_path=experiments/weights/dinosar_b16_unisar7m_60e.pth \
    data.root=/path/to/MSTAR few_shot.num_shots="$shots" \
    eval.output_dir="experiments/eval/vitb60e/MSTAR/${shots}shot"
done
```

Each run writes `metrics.json` to its output directory. Few-shot evaluation
aggregates all five seeds automatically. Pretraining lasts 60 epochs;
the evaluation schedules above are separate downstream protocols.
