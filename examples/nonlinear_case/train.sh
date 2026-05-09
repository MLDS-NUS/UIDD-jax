#!/bin/sh

set -u

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "$SCRIPT_DIR"

GPU_ID1=3
GPU_ID2=5
GPU_ID3=6
KAPPA=0.8

DTS="0.01 0.005"
SEEDS="0 1 12 123 1234"

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
    CUDA_VISIBLE_DEVICES="$gpu_id" python nonlinear_case.py \
      Model_name="$model_name" \
      model.seed="$seed" \
      data.var="$KAPPA" \
      dt="$dt"
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
      CUDA_VISIBLE_DEVICES="$GPU_ID1" python evaluate_mmd.py \
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

      run_model SDE SDE "$GPU_ID3" "$dt" "$seed" running_log/sde.file &
      pid_sde=$!

      for pid in "$pid_uidd" "$pid_onsager" "$pid_sde"; do
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

  echo "All UIDD, Onsager, and SDE jobs finished. Starting evaluate_mmd."
  if ! run_evaluation; then
    echo "One or more evaluate_mmd runs failed. Check running_log/evaluate_mmd.file for details."
    exit 1
  fi

  echo "All UIDD, Onsager, SDE, and evaluate_mmd jobs finished."
}

if [ "${1:-}" = "--run-schedule" ]; then
  run_schedule
else
  nohup sh "$SCRIPT_DIR/train.sh" --run-schedule > running_log/train_scheduler.file 2>&1 &
  echo "UIDD, Onsager, and SDE scheduler is running in the background."
  echo "For each dt and seed, the three models run together; the next dt/seed starts only after all three finish."
  echo "Scheduler log: running_log/train_scheduler.file"
  echo "Evaluation log after training: running_log/evaluate_mmd.file"
fi
