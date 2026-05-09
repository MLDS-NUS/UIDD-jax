#!/bin/sh
#
# Train OnsagerReg with L2 and Sobolev regularization (two branches, in
# parallel on the same GPU; up to 2 python processes at any time), then
# evaluate both via evaluate_mmd.py. Self-backgrounds via nohup so a plain
# `sh run_reg.sh` returns to the prompt immediately.
#
# Override defaults via env vars, e.g.
#   GPU_ID=3 SEEDS="1 12" sh run_reg.sh
#

set -u

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "$SCRIPT_DIR"

GPU_ID="${GPU_ID:-7}"
KAPPA="${KAPPA:-0.8}"
DT="${DT:-0.01}"
SEEDS="${SEEDS:-0 1 12 123 1234}"
L2_WEIGHT="${L2_WEIGHT:-1e-4}"
SOBOLEV_WEIGHT="${SOBOLEV_WEIGHT:-1e-3}"

# Weight value is encoded directly in the dir label so different runs do not
# collide. Avoid using underscores in L2_WEIGHT / SOBOLEV_WEIGHT (the dir-name
# parser splits on '_').
LABEL_L2="OnsagerRegL2${L2_WEIGHT}"
LABEL_SOB="OnsagerRegSobolev${SOBOLEV_WEIGHT}"

mkdir -p running_log

run_branch() {
  label="$1"
  l2w="$2"
  sobw="$3"
  log_file="running_log/${label}.file"
  {
    for seed in $SEEDS; do
      echo "===== ${label} dt=${DT} seed=${seed} start $(date '+%F %T') ====="
      CUDA_VISIBLE_DEVICES="$GPU_ID" python nonlinear_case.py \
        Model_name=OnsagerReg \
        model.seed="$seed" \
        data.var="$KAPPA" \
        dt="$DT" \
        model.potential.l2_weight="$l2w" \
        model.potential.sobolev_weight="$sobw" \
        hydra.run.dir="./outputs/nonlinear${seed}_${label}_dt${DT}"
      status=$?
      echo "===== ${label} dt=${DT} seed=${seed} end $(date '+%F %T') status=${status} ====="
    done
  } >> "$log_file" 2>&1
}

run_schedule() {
  echo "[$(date '+%F %T')] GPU=$GPU_ID  KAPPA=$KAPPA  DT=$DT"
  echo "[$(date '+%F %T')] SEEDS=[$SEEDS]"
  echo "[$(date '+%F %T')] L2_WEIGHT=$L2_WEIGHT  SOBOLEV_WEIGHT=$SOBOLEV_WEIGHT"
  echo "[$(date '+%F %T')] Labels: $LABEL_L2 / $LABEL_SOB"

  # Two reg branches in parallel on the same GPU: at most 2 python processes
  # alive simultaneously during training.
  run_branch "$LABEL_L2"  "$L2_WEIGHT"  "0"               &
  pid_l2=$!
  run_branch "$LABEL_SOB" "0"           "$SOBOLEV_WEIGHT" &
  pid_sob=$!
  echo "[$(date '+%F %T')] Training: $LABEL_L2 PID=$pid_l2, $LABEL_SOB PID=$pid_sob"

  l2_ok=0
  sob_ok=0
  wait "$pid_l2"  || l2_ok=1
  wait "$pid_sob" || sob_ok=1

  if [ "$l2_ok" -ne 0 ] || [ "$sob_ok" -ne 0 ]; then
    echo "[$(date '+%F %T')] WARNING: at least one branch returned a non-zero status."
    echo "[$(date '+%F %T')] L2 status=$l2_ok, Sobolev status=$sob_ok"
  fi
  echo "[$(date '+%F %T')] Both training branches finished. Starting evaluate_mmd.py ..."

  eval_log="running_log/OnsagerReg_l2${L2_WEIGHT}_sob${SOBOLEV_WEIGHT}_evaluate_mmd.file"
  {
    echo "===== evaluate_mmd dt=${DT} start $(date '+%F %T') ====="
    CUDA_VISIBLE_DEVICES="$GPU_ID" python evaluate_mmd.py \
      --dt "$DT" \
      --kappa "$KAPPA" \
      --seeds $SEEDS \
      --models "$LABEL_L2" "$LABEL_SOB" \
      --out-name "mmd_summary_OnsagerReg_l2${L2_WEIGHT}_sob${SOBOLEV_WEIGHT}_dt${DT}"
    eval_status=$?
    echo "===== evaluate_mmd dt=${DT} end $(date '+%F %T') status=${eval_status} ====="
  } >> "$eval_log" 2>&1

  echo "[$(date '+%F %T')] Pipeline finished. Eval log: $eval_log"
}

if [ "${1:-}" = "--run-schedule" ]; then
  run_schedule
else
  scheduler_log="running_log/run_reg_scheduler.file"
  nohup sh "$SCRIPT_DIR/run_reg.sh" --run-schedule > "$scheduler_log" 2>&1 &
  pid=$!
  echo "OnsagerReg pipeline launched in background (PID $pid)."
  echo "GPU: $GPU_ID  (two training branches in parallel; eval afterwards)"
  echo "Labels: $LABEL_L2, $LABEL_SOB"
  echo "Logs:"
  echo "  scheduler: $scheduler_log"
  echo "  L2 train:  running_log/${LABEL_L2}.file"
  echo "  Sob train: running_log/${LABEL_SOB}.file"
  echo "  eval:      running_log/OnsagerReg_l2${L2_WEIGHT}_sob${SOBOLEV_WEIGHT}_evaluate_mmd.file"
fi
