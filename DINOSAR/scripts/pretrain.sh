#!/bin/bash
# Launch a released 60-epoch recipe; trailing arguments are OmegaConf overrides.
set -euo pipefail
CONFIG="${1:-dinosar/configs/pretrain/unisar7m/vitb16_reg.yaml}"
if (( $# > 0 )); then shift; fi
exec torchrun --standalone --nproc_per_node="${NUM_GPUS:-4}" \
  --master_port="${MASTER_PORT:-29500}" -m dinosar.train --config "$CONFIG" "$@"
