#!/bin/sh
#
# Train OnsagerReg with L2 and Sobolev regularization (two branches, one per
# GPU), then
# evaluate both via evaluate_mmd.py. Self-backgrounds via nohup so a plain
# `sh run_reg.sh` returns to the prompt immediately.
#
# Override defaults via env vars, e.g.
#   GPU_ID_L2=6 GPU_ID_SOB=7 SEEDS="1 12" sh run_reg.sh
#

set -u

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "$SCRIPT_DIR"

GPU_ID_L2="${GPU_ID_L2:-6}"
GPU_ID_SOB="${GPU_ID_SOB:-7}"
KAPPA="${KAPPA:-0.8}"
DT="${DT:-0.01}"
SEEDS="${SEEDS:-0 1 12 123 1234}"
L2_WEIGHT="${L2_WEIGHT:-1e-4}"
SOBOLEV_WEIGHT="${SOBOLEV_WEIGHT:-1e-3}"
BATCH_SIZE="${BATCH_SIZE:-50000}"
LEARNING_RATE="${LEARNING_RATE:-5e-4}"
NUM_EPOCHS="${NUM_EPOCHS:-5000}"
PRINT_EVERY="${PRINT_EVERY:-100}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-500}"
PYTHON_BIN="${PYTHON_BIN:-python}"

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
  gpu_id="$4"
  log_file="running_log/${label}.file"
  {
    for seed in $SEEDS; do
      echo "===== ${label} dt=${DT} seed=${seed} start $(date '+%F %T') ====="
      CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON_BIN" nonlinear_case.py \
        Model_name=OnsagerReg \
        model.seed="$seed" \
        data.var="$KAPPA" \
        dt="$DT" \
        train.batch_size="$BATCH_SIZE" \
        train.opt.learning_rate="$LEARNING_RATE" \
        train.num_epochs="$NUM_EPOCHS" \
        train.print_every="$PRINT_EVERY" \
        train.checkpoint_every="$CHECKPOINT_EVERY" \
        model.potential.l2_weight="$l2w" \
        model.potential.sobolev_weight="$sobw" \
        hydra.run.dir="./outputs/20Dnonlinear${seed}_${label}_dt${DT}"
      status=$?
      echo "===== ${label} dt=${DT} seed=${seed} end $(date '+%F %T') status=${status} ====="
    done
  } >> "$log_file" 2>&1
}

run_schedule() {
  echo "[$(date '+%F %T')] GPU_L2=$GPU_ID_L2  GPU_SOB=$GPU_ID_SOB  KAPPA=$KAPPA  DT=$DT"
  echo "[$(date '+%F %T')] SEEDS=[$SEEDS]"
  echo "[$(date '+%F %T')] L2_WEIGHT=$L2_WEIGHT  SOBOLEV_WEIGHT=$SOBOLEV_WEIGHT"
  echo "[$(date '+%F %T')] batch_size=$BATCH_SIZE  lr=$LEARNING_RATE  epochs=$NUM_EPOCHS"
  echo "[$(date '+%F %T')] Labels: $LABEL_L2 / $LABEL_SOB"

  # Two reg branches in parallel, one process per GPU.
  run_branch "$LABEL_L2"  "$L2_WEIGHT"  "0"               "$GPU_ID_L2" &
  pid_l2=$!
  run_branch "$LABEL_SOB" "0"           "$SOBOLEV_WEIGHT" "$GPU_ID_SOB" &
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
    CUDA_VISIBLE_DEVICES="$GPU_ID_L2" "$PYTHON_BIN" evaluate_mmd.py \
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
  echo "GPUs: L2 on $GPU_ID_L2, Sobolev on $GPU_ID_SOB; eval afterwards on $GPU_ID_L2"
  echo "Training params: batch_size=$BATCH_SIZE, lr=$LEARNING_RATE, epochs=$NUM_EPOCHS"
  echo "Labels: $LABEL_L2, $LABEL_SOB"
  echo "Logs:"
  echo "  scheduler: $scheduler_log"
  echo "  L2 train:  running_log/${LABEL_L2}.file"
  echo "  Sob train: running_log/${LABEL_SOB}.file"
  echo "  eval:      running_log/OnsagerReg_l2${L2_WEIGHT}_sob${SOBOLEV_WEIGHT}_evaluate_mmd.file"
fi
