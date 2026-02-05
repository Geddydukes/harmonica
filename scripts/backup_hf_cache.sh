#!/usr/bin/env bash
set -euo pipefail

SRC_CACHE="${1:-/tmp/hf_vctk_cache}"
DEST_CACHE="${2:-./hf_cache_backup}"
INTERVAL_MINUTES="${3:-10}"

echo "Backing up HF cache"
echo "Source: ${SRC_CACHE}"
echo "Dest:   ${DEST_CACHE}"
echo "Every:  ${INTERVAL_MINUTES} minutes"

mkdir -p "${DEST_CACHE}"

while true; do
  ts="$(date '+%Y-%m-%d %H:%M:%S')"
  echo "[${ts}] Syncing cache..."
  rsync -a --delete "${SRC_CACHE}/" "${DEST_CACHE}/"
  echo "[${ts}] Done. Sleeping..."
  sleep "$((INTERVAL_MINUTES * 60))"
done
