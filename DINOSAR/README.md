# DINOSAR Pretraining and Classification

Train ViT-S/16 or ViT-B/16 on [UniSAR-7M](https://huggingface.co/datasets/YTang/UniSAR-7M)
for 60 epochs with content-aware multi-crop (CAMC). Both released backbones use
single-channel inputs and four register tokens.

## Training

Run from this directory:

```bash
uv sync
uv run bash scripts/pretrain.sh dinosar/configs/pretrain/unisar7m/vitb16_reg.yaml \
  data.data_path=/path/to/UniSAR-7M
```

Use `vits16_reg.yaml` for ViT-S. The launcher defaults to four GPUs;
set `NUM_GPUS` to change this. See the [training guide](docs/runtime_guide.md)
for effective batch sizes, checkpoint export, and resume commands.

## Released weights and evaluation

The [Hugging Face model repository](https://huggingface.co/YTang/DINOSAR)
contains the two final weights, their resolved training configurations, and logs.
Follow the [evaluation guide](docs/eval_guide.md) for downloading weights,
k-NN, linear probing, few-shot learning, and full fine-tuning.

- [Dataset preparation](docs/datasets.md)
- [Patch similarity and PCA visualization](notebooks/patch_similarity_pca.ipynb)
- [License](LICENSE)

## Visualization demo

```bash
uv sync
uv run jupyter lab notebooks/patch_similarity_pca.ipynb
```

Run all cells with the bundled examples. Choose ViT-S or ViT-B in Settings;
the notebook reuses local weights or downloads the selected release from
[Hugging Face](https://huggingface.co/YTang/DINOSAR), then caches it for reuse.
See [example images](dinosar/assets/README.md) for the included inputs.
