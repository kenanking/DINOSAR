#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONFIG=${1:?Usage: scripts/run_test.sh CONFIG CHECKPOINT [GPUS]}
CHECKPOINT=${2:?Usage: scripts/run_test.sh CONFIG CHECKPOINT [GPUS]}
GPUS=${3:-1}
PORT=${PORT:-29512}

export PYTHONNOUSERSITE=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

TEST_PY=$(python - <<'PY'
from pathlib import Path
import mmseg
path = Path(mmseg.__file__).resolve().parent / ".mim" / "tools" / "test.py"
print(path)
PY
)
if [[ ! -f "${TEST_PY}" ]]; then
  echo "Cannot find MMSeg test.py at ${TEST_PY}. Install requirements.txt first." >&2
  exit 1
fi

if [[ "${GPUS}" == "1" ]]; then
  python "${TEST_PY}" "${CONFIG}" "${CHECKPOINT}"
else
  python -m torch.distributed.run \
    --nproc_per_node="${GPUS}" \
    --master_port="${PORT}" \
    "${TEST_PY}" \
    "${CONFIG}" \
    "${CHECKPOINT}" \
    --launcher pytorch
fi
