#!/usr/bin/env bash
# Chạy thử end-to-end trên 3 file đầu tiên trong thư mục raw.
# Usage: ./scripts/run_sample.sh <raw_dir> [workdir]
set -euo pipefail

RAW_DIR="${1:?Usage: run_sample.sh <raw_dir> [workdir]}"
WORKDIR="${2:-./work_sample}"

python -m pipeline run \
  --raw-dir "$RAW_DIR" \
  --workdir "$WORKDIR" \
  --stages all \
  --limit 3

echo
echo "Xem báo cáo: $WORKDIR/s8_package/report.html"
echo "Dataset:     $WORKDIR/s8_package/dataset/"
