#!/usr/bin/env bash
set -euo pipefail

DEVICE="${1:-auto}"
HF_CACHE="${2:-/tmp/hf_vctk_cache}"
KEEP_CACHE="${KEEP_CACHE:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="$(pwd)"

echo "Device: ${DEVICE}"
echo "HF cache: ${HF_CACHE}"

export HF_DATASETS_CACHE="${HF_CACHE}"
export HF_HOME="${HF_CACHE}/hf_home"

echo "==> Training AR model (HF cached, non-streaming)"
"${PYTHON_BIN}" scripts/train.py \
  --config configs/experiment/vctk_stream.yaml \
  --model-type ar \
  --device "${DEVICE}"

echo "==> Training NAR model (HF cached, non-streaming)"
"${PYTHON_BIN}" scripts/train.py \
  --config configs/experiment/vctk_stream.yaml \
  --model-type nar \
  --device "${DEVICE}"

if [ "${KEEP_CACHE}" -eq 0 ]; then
  echo "==> Cleaning HF cache"
  rm -rf "${HF_CACHE}"
else
  echo "==> Keeping HF cache at ${HF_CACHE}"
fi

echo "Done."
