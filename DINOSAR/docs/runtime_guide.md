# Training the Final 60-Epoch Models

Run the commands below from `DINOSAR/` after `uv sync` and
[preparing UniSAR-7M](datasets.md).

| Model | Config under `dinosar/configs/pretrain/unisar7m/` | Batch per GPU | Effective batch | Optimizer steps |
| --- | --- | ---: | ---: | ---: |
| ViT-S/16 | `vits16_reg.yaml` | 320 | 1,280 | 330,360 |
| ViT-B/16 | `vitb16_reg.yaml` | 192 | 1,536 | 275,340 |

Both recipes use all 7,047,666 images for 60 epochs, CAMC, and the DINO,
iBOT, and KoLeo objectives. Historical resolved configurations and training
logs are available in the [model repository](https://huggingface.co/YTang/DINOSAR/tree/main/training).

## Launch and resume

```bash
NUM_GPUS=4 uv run bash scripts/pretrain.sh \
  dinosar/configs/pretrain/unisar7m/vitb16_reg.yaml \
  data.data_path=/path/to/UniSAR-7M
```

For ViT-S, replace `vitb16_reg.yaml` with `vits16_reg.yaml`.
Gradient accumulation is inferred from the GPU count and per-GPU batch size.
On four GPUs it is 1 for ViT-S and 2 for ViT-B. If reducing the batch size
for smaller GPUs, keep the effective batch unchanged and choose values such
that `batch_size × GPU_count` divides it exactly.

```bash
NUM_GPUS=4 uv run bash scripts/pretrain.sh \
  dinosar/configs/pretrain/unisar7m/vitb16_reg.yaml \
  --resume experiments/pretrain/unisar7m_vitb16_60e/ckpt/last.pth \
  data.data_path=/path/to/UniSAR-7M
```

Outputs go to `train.output_dir`: `config.yaml` records the resolved settings,
`logs/metrics.jsonl` records metrics, `ckpt/last.pth` is resumable, and `eval/`
contains exported teacher backbones. Add `logging.enabled=false` to disable
Trackio; local training metrics remain available.

## Export and use

The final exports are `eval/step_0330360.pth` for ViT-S and
`eval/step_0275340.pth` for ViT-B. These backbone-only checkpoints can be
used directly. To export from a full training checkpoint instead:

```bash
uv run python scripts/export_backbone_weights.py \
  --input experiments/pretrain/unisar7m_vitb16_60e/ckpt/last.pth \
  --output experiments/weights/dinosar_b16_unisar7m_60e.pth
```

See [classification evaluation](eval_guide.md),
[DINOSAR_Detection](../../DINOSAR_Detection/README.md), [DINOSAR_Segmentation](../../DINOSAR_Segmentation/README.md),
and [DINOSAR_Retrieval](../../DINOSAR_Retrieval/README.md) for downstream use.
