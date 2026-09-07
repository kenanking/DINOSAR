#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONFIG=${1:?Usage: scripts/run_train.sh CONFIG [GPUS] [TRAIN_ARGS...]}
GPUS=${2:-1}
if [[ $# -ge 2 ]]; then
  shift 2
else
  shift 1
fi
EXTRA_ARGS=("$@")
PORT=${PORT:-29511}

export PYTHONNOUSERSITE=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

TRAIN_PY=$(python - <<'PY'
from pathlib import Path
import mmseg
path = Path(mmseg.__file__).resolve().parent / ".mim" / "tools" / "train.py"
print(path)
PY
)
if [[ ! -f "${TRAIN_PY}" ]]; then
  echo "Cannot find MMSeg train.py at ${TRAIN_PY}. Install requirements.txt first." >&2
  exit 1
fi

if [[ "${GPUS}" == "1" ]]; then
  python "${TRAIN_PY}" "${CONFIG}" "${EXTRA_ARGS[@]}"
else
  python -m torch.distributed.run \
    --nproc_per_node="${GPUS}" \
    --master_port="${PORT}" \
    "${TRAIN_PY}" \
    "${CONFIG}" \
    --launcher pytorch \
    "${EXTRA_ARGS[@]}"
fi
