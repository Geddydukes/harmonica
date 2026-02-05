#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${1:-./data/VCTK-Corpus}"
CACHE_DIR="${2:-./cache/vctk}"
DEVICE="${3:-auto}"
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="$(pwd)"

echo "VCTK data dir: ${DATA_DIR}"
echo "Cache dir: ${CACHE_DIR}"
echo "Device: ${DEVICE}"

echo "==> Preprocessing VCTK"
"${PYTHON_BIN}" scripts/preprocess.py \
  --dataset vctk \
  --data-dir "${DATA_DIR}" \
  --output-dir "${CACHE_DIR}" \
  --device "${DEVICE}"

echo "==> Training AR model"
"${PYTHON_BIN}" scripts/train.py \
  --config configs/experiment/vctk_base.yaml \
  --model-type ar \
  --device "${DEVICE}"

echo "==> Training NAR model"
"${PYTHON_BIN}" scripts/train.py \
  --config configs/experiment/vctk_base.yaml \
  --model-type nar \
  --device "${DEVICE}"

echo "Done."
