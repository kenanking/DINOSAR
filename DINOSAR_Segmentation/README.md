# DINOSAR Segmentation

Standalone MMSegmentation workspace for AIR-PolSARSeg semantic segmentation.
It fine-tunes DINOSAR pretrained ViT backbones under a matched UPerNet
protocol. Run commands from `DINOSAR_Segmentation/`.

## Layout

- `configs/`: MMSeg experiment configs
- `projects/dinosar_mmseg/`: dataset, metric, optimizer, and backbone adapters
- `scripts/`: environment, dataset preparation, train, and test entrypoints
- `docs/`: environment and preprocessing notes
- `datasets/`: generated AIR-PolSARSeg PNG dataset (not shipped)
- `experiments/`: logs, checkpoints, and pretrained weights (not shipped)

## Environment

```bash
scripts/create_conda_env.sh
conda activate dinosar-seg
```

See `docs/environment.md` for the verified package stack.

## Dataset

Place the raw AIR-PolSARSeg release under `datasets/Raw_AIR-PolarSAR-Seg`,
then generate the canonical pseudo-color PNG dataset:

```bash
python scripts/prepare_airseg_dataset.py \
  --raw-root datasets/Raw_AIR-PolarSAR-Seg \
  --out-root datasets/AIR-PolSAR-Seg-mmseg \
  --overwrite
```

The image formula is `[(HV + VH) / 2, HH, VV]` with `log1p`, per-image
percentile scaling, and intensity inversion. Details are in
`docs/data_preprocessing.md`.

## Pretrained Weights

Download the [UniSAR-7M 60-epoch ViT-B backbone](https://huggingface.co/YTang/DINOSAR):

```bash
uvx hf download YTang/DINOSAR dinosar_b16_unisar7m_60e.pth \
  --local-dir experiments/weights/DINOSAR
```

The adapter imports `dinosar.model` from `../DINOSAR/`, or from `DINOSAR_ROOT`
if set. The downloaded backbone requires no conversion.

## Training

Fine-tune the single-channel backbone with UPerNet for 72 epochs:

```bash
bash scripts/run_train.sh configs/dinosar/vitb_upernet_airseg_pseudo_gray_60e_72ep.py 1
```

Test a checkpoint:

```bash
bash scripts/run_test.sh configs/dinosar/vitb_upernet_airseg_pseudo_gray_60e_72ep.py /path/to/finetuned_checkpoint.pth 1
```

See [the task protocol](docs/segmentation.md) for metrics and schedule details.
