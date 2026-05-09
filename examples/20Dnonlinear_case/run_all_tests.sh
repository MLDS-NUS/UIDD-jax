#!/bin/sh

set -u

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "$SCRIPT_DIR"

DT="${DT:-0.01}"
KAPPA="${KAPPA:-0.8}"
SEEDS="${SEEDS:-0 1 12 123 1234}"
MODELS="${MODELS:-HD2 Onsager}"
EVAL_GPU="${EVAL_GPU:-6}"
NUM_TRAJS="${NUM_TRAJS:-1000}"
SOLVE_SUBSTEPS="${SOLVE_SUBSTEPS:-10}"
OUT_NAME="${OUT_NAME:-mmd_summary_20D_all_dt${DT}}"
OUTPUTS_ROOT="${OUTPUTS_ROOT:-outputs}"
LOG_FILE="${LOG_FILE:-running_log/run_all_tests.file}"

if [ -x "/home/aiqing/anaconda3/envs/UIDD-cuda12/bin/python" ]; then
  PYTHON_BIN="${PYTHON_BIN:-/home/aiqing/anaconda3/envs/UIDD-cuda12/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi

mkdir -p running_log "$OUTPUTS_ROOT"

missing=0
for seed in $SEEDS; do
  for model in $MODELS; do
    ckpt="${OUTPUTS_ROOT}/20Dnonlinear${seed}_${model}_dt${DT}/model.eqx"
    if [ ! -f "$ckpt" ]; then
      echo "[missing] $ckpt"
      missing=1
    fi
  done
done

if [ "$missing" -ne 0 ]; then
  echo "Some checkpoints are missing; aborting tests."
  exit 1
fi

{
  echo "===== 20D nonlinear all tests start $(date '+%F %T') ====="
  echo "DT=$DT KAPPA=$KAPPA"
  echo "SEEDS=$SEEDS"
  echo "MODELS=$MODELS"
  echo "NUM_TRAJS=$NUM_TRAJS SOLVE_SUBSTEPS=$SOLVE_SUBSTEPS"
  echo "EVAL_GPU=$EVAL_GPU"
  echo "OUT_NAME=$OUT_NAME"

  CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$PYTHON_BIN" evaluate_mmd.py \
    --dt "$DT" \
    --kappa "$KAPPA" \
    --seeds $SEEDS \
    --models $MODELS \
    --num-trajs "$NUM_TRAJS" \
    --solve-substeps "$SOLVE_SUBSTEPS" \
    --outputs-root "$OUTPUTS_ROOT" \
    --out-name "$OUT_NAME"
  eval_status=$?

  echo "===== evaluate_mmd end $(date '+%F %T') status=${eval_status} ====="
  if [ "$eval_status" -ne 0 ]; then
    exit "$eval_status"
  fi

  NPZ_PATH="${OUTPUTS_ROOT}/${OUT_NAME}.npz" "$PYTHON_BIN" - <<'PY'
import os
import numpy as np

path = os.environ["NPZ_PATH"]
z = np.load(path)
models = [key[:-5] for key in z.files if key.endswith("_mean") and not key.startswith("err_")]
models = [m for m in ["HD2", "Onsager"] if m in models] + sorted(
    m for m in models if m not in {"HD2", "Onsager"}
)

print("")
print(f"Summary file: {path}")
print(f"t_end={float(z['ts'][-1]):.6g}, n_time={len(z['ts'])}")
print(f"noise_floor_final={float(z['mmd_ref'][-1]):.6e}")
print(f"noise_floor_time_mean={float(z['mmd_ref'].mean()):.6e}")

print("")
print("MMD summary")
print(f"{'model':<10} {'final_mean':>14} {'final_std':>14} {'time_mean':>14} {'max_mean':>14}")
for model in models:
    mean = z[f"{model}_mean"]
    std = z[f"{model}_std"]
    print(
        f"{model:<10} {float(mean[-1]):14.6e} {float(std[-1]):14.6e} "
        f"{float(mean.mean()):14.6e} {float(mean.max()):14.6e}"
    )

print("")
print("Structural relative errors")
print(f"{'model':<10} {'drift':>14} {'gradV':>14} {'MgradV':>14} {'WgradV':>14}")
for model in models:
    vals = []
    for metric in ["drift", "gradV", "MgradV", "WgradV"]:
        key = f"err_{model}_{metric}_mean"
        vals.append("-" if key not in z else f"{float(z[key]):.6e}")
    print(f"{model:<10} {vals[0]:>14} {vals[1]:>14} {vals[2]:>14} {vals[3]:>14}")
PY

  echo "===== 20D nonlinear all tests end $(date '+%F %T') ====="
} 2>&1 | tee "$LOG_FILE"
