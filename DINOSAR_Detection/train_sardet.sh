#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
source env.sh
export DATASET_ROOT="${DATASET_ROOT:-/path/to/SARDet100K_coco_annotation}"
exec python train.py --num-gpus "${NUM_GPUS:-1}" \
  --config-file "${CONFIG_FILE:-configs/detection/faster_rcnn/dinov3/vitb/sardet/single_channel_imgsz_800.py}" "$@"
