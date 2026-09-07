# DINOSAR Detection

Fine-tune the [DINOSAR 60-epoch backbones](https://huggingface.co/YTang/DINOSAR)
with Faster R-CNN and SimpleFPN on SARDet-100K. Run commands from `DINOSAR_Detection/`.

## Environment and data

```bash
mkdir -p dependency
git clone https://github.com/facebookresearch/detectron2.git dependency/detectron2
git -C dependency/detectron2 checkout fd27788
git clone https://github.com/facebookresearch/dinov3.git dependency/dinov3
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
source env.sh
```

Prepare [SARDet-100K](https://github.com/zcablii/SARDet_100K) in COCO format:
`Annotations/{train,test}.json` and `Images/{train,test}/`.
Set `DATASET_ROOT` to this directory. Annotation files must exist when the
configuration is loaded, including during evaluation.

## Pretrained weights

```bash
uvx hf download YTang/DINOSAR dinosar_b16_unisar7m_60e.pth --local-dir weights
python tools/add_weight_prefix.py weights/dinosar_b16_unisar7m_60e.pth
```

This creates `weights/dinosar_b16_unisar7m_60e_detectron2.pth`, adding the
`backbone.net.model.` prefix required by Detectron2. Use the corresponding
`dinosar_s16_unisar7m_60e.pth` file and conversion command for ViT-S.

## Train and evaluate

The default launcher uses the paper's primary ViT-B protocol: 800 × 800 input,
12 fine-tuning epochs, total batch size 16, AdamW learning rate 1e-4.

```bash
DATASET_ROOT=/path/to/SARDet100K_coco_annotation NUM_GPUS=1 bash train_sardet.sh
```

Retained configurations under `configs/detection/faster_rcnn/dinov3/`:

| Config | Backbone | Input | Fine-tuning epochs |
| --- | --- | ---: | ---: |
| `vitb/sardet/single_channel_imgsz_800.py` | ViT-B | 800 | 12 (primary) |
| `vitb/sardet/single_channel_3x.py` | ViT-B | 512 | 36 (additional paper protocol) |
| `vitb/sardet/single_channel.py` | ViT-B | 512 | 12 (shared recipe) |
| `vits/sardet/single_channel_nstorage_4.py` | ViT-S | 512 | 12 |

Select a config with `CONFIG_FILE=... bash train_sardet.sh`. Each initializes
from the corresponding UniSAR-7M 60-epoch backbone. Outputs are stored under
`output/<config-relative-path>/<pretrained-checkpoint-stem>/`.
Append `--resume` to resume a run. To evaluate a **fine-tuned detector**:

```bash
DATASET_ROOT=/path/to/SARDet100K_coco_annotation python train.py \
  --config-file configs/detection/faster_rcnn/dinov3/vitb/sardet/single_channel_imgsz_800.py \
  --eval-only train.init_checkpoint=/path/to/output/model_final.pth
```

See [LICENSE](LICENSE) and the upstream dependency licenses.
