#!/bin/sh

set -u

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "$SCRIPT_DIR"

GPU_ID="${GPU_ID:-1}"
KAPPA="${KAPPA:-0.8}"
DTS="${DTS:-0.01 0.005}"
SEEDS="${SEEDS:-0 1 12 123 1234}"

mkdir -p running_log
log_file="running_log/evaluate_mmd.file"

failed=0
for dt in $DTS; do
  echo "===== evaluate_mmd dt=${dt} start $(date '+%F %T') =====" | tee -a "$log_file"
  status_file="$(mktemp)"
  (
    CUDA_VISIBLE_DEVICES="$GPU_ID" python evaluate_mmd.py \
      --dt "$dt" \
      --kappa "$KAPPA" \
      --seeds $SEEDS 2>&1
    echo $? > "$status_file"
  ) | tee -a "$log_file"
  status=$(cat "$status_file")
  rm -f "$status_file"
  echo "===== evaluate_mmd dt=${dt} end $(date '+%F %T') status=${status} =====" | tee -a "$log_file"
  if [ "$status" -ne 0 ]; then
    failed=1
  fi
done

if [ "$failed" -ne 0 ]; then
  echo "One or more evaluate_mmd runs failed. Log: $log_file"
  exit 1
fi

echo "All evaluate_mmd runs finished. Log: $log_file"
