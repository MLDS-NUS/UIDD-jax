#!/bin/sh

set -u

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "$SCRIPT_DIR"

GPU_ID1="${GPU_ID1:-2}"
GPU_ID2="${GPU_ID2:-2}"
KAPPA="${KAPPA:-0.8}"

DTS="${DTS:-0.01}"
SEEDS="${SEEDS:-0 1 12 123 1234}"
BATCH_SIZE="${BATCH_SIZE:-100000}"
LEARNING_RATE="${LEARNING_RATE:-5e-4}"
NUM_EPOCHS="${NUM_EPOCHS:-600}"
NUM_RUNS="${NUM_RUNS:-90000}"
NUM_RUNS_TEST="${NUM_RUNS_TEST:-10000}"
PRINT_EVERY="${PRINT_EVERY:-100}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-500}"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p running_log

run_model() {
  label="$1"
  model_name="$2"
  gpu_id="$3"
  dt="$4"
  seed="$5"
  log_file="$6"

  {
    echo "===== ${label} dt=${dt} seed=${seed} start $(date '+%F %T') ====="
    CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON_BIN" nonlinear_case.py \
      Model_name="$model_name" \
      model.seed="$seed" \
      data.var="$KAPPA" \
      dt="$dt" \
      train.batch_size="$BATCH_SIZE" \
      train.opt.learning_rate="$LEARNING_RATE" \
      train.num_epochs="$NUM_EPOCHS" \
      train.print_every="$PRINT_EVERY" \
      train.checkpoint_every="$CHECKPOINT_EVERY" \
      data.num_runs="$NUM_RUNS" \
      data.num_runs_test="$NUM_RUNS_TEST"
    model_status=$?
    echo "===== ${label} dt=${dt} seed=${seed} end $(date '+%F %T') status=${model_status} ====="
  } >> "$log_file" 2>&1

  return "$model_status"
}

run_evaluation() {
  failed=0

  for dt in $DTS; do
    {
      echo "===== evaluate_mmd dt=${dt} start $(date '+%F %T') ====="
      CUDA_VISIBLE_DEVICES="$GPU_ID1" "$PYTHON_BIN" evaluate_mmd.py \
        --dt "$dt" \
        --kappa "$KAPPA" \
        --seeds $SEEDS
      eval_status=$?
      echo "===== evaluate_mmd dt=${dt} end $(date '+%F %T') status=${eval_status} ====="
    } >> running_log/evaluate_mmd.file 2>&1

    if [ "$eval_status" -ne 0 ]; then
      failed=1
    fi
  done

  return "$failed"
}

run_schedule() {
  failed=0

  for dt in $DTS; do
    for seed in $SEEDS; do
      echo "===== batch dt=${dt} seed=${seed} start $(date '+%F %T') ====="

      run_model UIDD HD2 "$GPU_ID1" "$dt" "$seed" running_log/uidd.file &
      pid_uidd=$!

      run_model Onsager Onsager "$GPU_ID2" "$dt" "$seed" running_log/onsager.file &
      pid_onsager=$!

      for pid in "$pid_uidd" "$pid_onsager"; do
        if ! wait "$pid"; then
          failed=1
        fi
      done

      echo "===== batch dt=${dt} seed=${seed} end $(date '+%F %T') ====="
    done
  done

  if [ "$failed" -ne 0 ]; then
    echo "One or more runs failed. Check running_log/*.file for details."
    exit 1
  fi

  echo "All UIDD and Onsager jobs finished. Starting evaluate_mmd."
  if ! run_evaluation; then
    echo "One or more evaluate_mmd runs failed. Check running_log/evaluate_mmd.file for details."
    exit 1
  fi

  echo "All UIDD, Onsager, and evaluate_mmd jobs finished."
}

if [ "${1:-}" = "--run-schedule" ]; then
  run_schedule
else
  nohup sh "$SCRIPT_DIR/train.sh" --run-schedule > running_log/train_scheduler.file 2>&1 &
  echo "UIDD and Onsager scheduler is running in the background."
  echo "Using GPUs $GPU_ID1 and $GPU_ID2 with at most one training process per GPU."
  echo "For each dt and seed: UIDD runs on $GPU_ID1, Onsager runs on $GPU_ID2."
  echo "Training params: num_runs=$NUM_RUNS, num_runs_test=$NUM_RUNS_TEST, batch_size=$BATCH_SIZE, lr=$LEARNING_RATE, epochs=$NUM_EPOCHS."
  echo "Scheduler log: running_log/train_scheduler.file"
  echo "Evaluation log after training: running_log/evaluate_mmd.file"
fi
