#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [[ -z "${CONDA_ROOT:-}" ]]; then
  if command -v conda >/dev/null 2>&1; then
    CONDA_ROOT=$(conda info --base)
  else
    CONDA_ROOT="${HOME}/miniconda3"
  fi
fi
ENV_NAME=${ENV_NAME:-dinosar-seg}
PYTHON_VERSION=${PYTHON_VERSION:-3.10}

source "${CONDA_ROOT}/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Conda env '${ENV_NAME}' already exists; installing/updating Python dependencies."
else
  conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}" pip "setuptools<82"
fi

conda activate "${ENV_NAME}"
export PYTHONNOUSERSITE=1
python -m pip install --upgrade pip
python -m pip install -r "${ROOT}/requirements.txt"

cat <<MSG

Environment '${ENV_NAME}' is ready.
MSG
