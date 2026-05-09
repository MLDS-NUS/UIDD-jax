#!/bin/sh
#
# Launch the nonlinear-case UIDD hyperparameter sensitivity sweep without
# modifying the original training/evaluation code.
#
# Defaults use 4 GPUs with 2 processes per GPU:
#   sh run_uidd_sensitivity.sh
#
# Foreground run:
#   sh run_uidd_sensitivity.sh --foreground
#
# Common overrides:
#   SEEDS="0 1 12" GPUS="0 1 2 3" sh run_uidd_sensitivity.sh
#

set -u

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "$SCRIPT_DIR"

SEEDS="${SEEDS:-0 1 12 123 1234}"
GPUS="${GPUS:-0 1 2 3}"
MAX_PER_GPU="${MAX_PER_GPU:-2}"
DT="${DT:-0.01}"
KAPPA="${KAPPA:-0.8}"
LOG_ROOT="${LOG_ROOT:-running_log/uidd_sensitivity}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/uidd_sensitivity}"
PYTHON_BIN="${PYTHON_BIN:-/home/aiqing/anaconda3/envs/UIDD-cuda12/bin/python}"

if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="python"
fi

CUDA_LIBS="$("$PYTHON_BIN" - <<'PY'
import glob
import os
import sys
paths = []
paths.extend(glob.glob(os.path.join(sys.prefix, "lib", "python*", "site-packages", "nvidia", "*", "lib")))
paths.extend(glob.glob(os.path.join(sys.prefix, "lib", "site-packages", "nvidia", "*", "lib")))
paths.append("/usr/local/cuda-12.8/targets/x86_64-linux/lib")
seen = []
for path in paths:
    if os.path.isdir(path) and path not in seen:
        seen.append(path)
print(":".join(seen))
PY
)"
if [ -n "$CUDA_LIBS" ]; then
  LD_LIBRARY_PATH="$CUDA_LIBS:${LD_LIBRARY_PATH:-}"
  export LD_LIBRARY_PATH
fi

mkdir -p "$LOG_ROOT"

run_sweep() {
  # shellcheck disable=SC2086
  "$PYTHON_BIN" -u uidd_sensitivity.py all \
    --seeds $SEEDS \
    --gpus $GPUS \
    --max-per-gpu "$MAX_PER_GPU" \
    --dt "$DT" \
    --kappa "$KAPPA" \
    --output-root "$OUTPUT_ROOT" \
    --log-root "$LOG_ROOT"
}

if [ "${1:-}" = "--foreground" ]; then
  echo "$PYTHON_BIN -u uidd_sensitivity.py all --seeds $SEEDS --gpus $GPUS --max-per-gpu $MAX_PER_GPU --dt $DT --kappa $KAPPA --output-root $OUTPUT_ROOT --log-root $LOG_ROOT"
  run_sweep
  exit $?
fi

scheduler_log="$LOG_ROOT/scheduler_stdout.log"
nohup "$PYTHON_BIN" -u uidd_sensitivity.py all \
  --seeds $SEEDS \
  --gpus $GPUS \
  --max-per-gpu "$MAX_PER_GPU" \
  --dt "$DT" \
  --kappa "$KAPPA" \
  --output-root "$OUTPUT_ROOT" \
  --log-root "$LOG_ROOT" \
  > "$scheduler_log" 2>&1 &
pid=$!

echo "UIDD sensitivity sweep launched in background (PID $pid)."
echo "GPUs: $GPUS; max processes per GPU: $MAX_PER_GPU"
echo "Seeds: $SEEDS; dt=$DT; kappa=$KAPPA"
echo "Scheduler log: $scheduler_log"
echo "Per-job logs: $LOG_ROOT"
echo "Outputs and final CSV/plots: $OUTPUT_ROOT"
