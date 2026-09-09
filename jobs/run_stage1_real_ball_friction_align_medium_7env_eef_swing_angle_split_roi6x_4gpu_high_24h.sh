#!/usr/bin/env bash
#SBATCH -A yejin
#SBATCH --job-name=ball-roll1d-roi6x
#SBATCH -p yejin
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=500G
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@1800
#SBATCH --no-requeue
#SBATCH --output=/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/logs/slurm/%x-%j.out
#SBATCH --error=/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/logs/slurm/%x-%j.err

set -euo pipefail

WALLCLOCK_REQUEST_DIR=/tmp/${USER}/bwm_wallclock_checkpoint_requests
WALLCLOCK_REQUEST_FILE=${WALLCLOCK_REQUEST_DIR}/${SLURM_JOB_ID}.request
mkdir -p "$WALLCLOCK_REQUEST_DIR"
rm -f "$WALLCLOCK_REQUEST_FILE" "${WALLCLOCK_REQUEST_FILE}.tmp"
export BWM_WALLCLOCK_CHECKPOINT_REQUEST_FILE="$WALLCLOCK_REQUEST_FILE"

request_wallclock_checkpoint() {
  printf 'job=%s signal_time=%s\n' "$SLURM_JOB_ID" "$(date --iso-8601=seconds)" \
    > "${WALLCLOCK_REQUEST_FILE}.tmp"
  mv -f "${WALLCLOCK_REQUEST_FILE}.tmp" "$WALLCLOCK_REQUEST_FILE"
  printf '[wallclock_checkpoint] Slurm 23:30 signal received; request=%s\n' \
    "$WALLCLOCK_REQUEST_FILE"
}
trap request_wallclock_checkpoint USR1

REPO=/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt
cd "$REPO"

BASE_CONFIG=configs/train/train_real_ball_friction_align_medium_7env_60f_eef_swing_angle_episode90_10_c32_random_oldmethod_roi6x_4gpu_3env6skill_stage1_5500.yaml
METADATA_DIR=data/real_ball_friction_align_medium_7env_60f_eef_state_action_episode90_10_20260828
ANGLE_STATS=configs/train/action_stats/real_ball_friction_align_medium_train_eef_swing_angle_stats.json
RUN_TAG=real_ball_friction_align_medium_7env_60f_eef_swing_angle_episode90_10_c32_random_oldmethod_roi6x_4gpu_3env6skill_stage1_5500
RUN_CONFIG=tmp/run_configs/${RUN_TAG}_${SLURM_JOB_ID}.yaml
OUT_DIR=./outputs/${RUN_TAG}_${SLURM_JOB_ID}
SOURCE_DATA=/afs/ir/users/c/y/cyzhou05/TTT-Physics/datasets_real/ball_friction_8_17_align_medium
SHARED_TMP_ROOT=/tmp/${USER}/bwm_shared_cache
LOCAL_MODEL=${SHARED_TMP_ROOT}/Wan2.2-TI2V-5B
LOCAL_CKPT_PATH=${SHARED_TMP_ROOT}/ckpt/BLM/step-12000.safetensors
LOCAL_DATA=${SHARED_TMP_ROOT}/datasets_real/ball_friction_align_medium_7env_1view_20260818
MODEL_LOCK=${SHARED_TMP_ROOT}/Wan2.2-TI2V-5B.lock
CKPT_LOCK=${SHARED_TMP_ROOT}/ckpt.lock
DATA_LOCK=${SHARED_TMP_ROOT}/real_ball_friction_align_medium_7env_1view_20260818.lock

test -s "${METADATA_DIR}/train.jsonl"
test -s "${METADATA_DIR}/test.jsonl"
test -s "${METADATA_DIR}/episode_split.jsonl"
test -s "${ANGLE_STATS}"

mkdir -p logs/slurm tmp/run_configs outputs "$OUT_DIR/input_manifest" "$SHARED_TMP_ROOT"
cp "$BASE_CONFIG" "$RUN_CONFIG"
cp "${METADATA_DIR}/manifest_summary.json" \
   "${METADATA_DIR}/episode_split.jsonl" \
   "${ANGLE_STATS}" \
   "$OUT_DIR/input_manifest/"

(
  flock 9
  if [ ! -f "${LOCAL_MODEL}/.copy_complete" ]; then
    mkdir -p "$LOCAL_MODEL"
    rsync -a models/Wan2.2-TI2V-5B/ "$LOCAL_MODEL/"
    touch "${LOCAL_MODEL}/.copy_complete"
  fi
) 9>"$MODEL_LOCK"

(
  flock 8
  if [ ! -f "${LOCAL_CKPT_PATH}.copy_complete" ] || [ ! -f "$LOCAL_CKPT_PATH" ]; then
    mkdir -p "$(dirname "$LOCAL_CKPT_PATH")"
    rsync -a ckpt/BLM/ "$(dirname "$LOCAL_CKPT_PATH")/"
    touch "${LOCAL_CKPT_PATH}.copy_complete"
  fi
) 8>"$CKPT_LOCK"

(
  flock 7
  mkdir -p "$LOCAL_DATA"
  rsync -a --delete "$SOURCE_DATA/" "$LOCAL_DATA/"
) 7>"$DATA_LOCK"

sed -i \
  -e "s|^  output_path: .*|  output_path: \"${OUT_DIR}\"|" \
  -e "s|model_paths: \"models/Wan2.2-TI2V-5B\"|model_paths: \"${LOCAL_MODEL}\"|" \
  -e "s|ckpt/BLM/step-12000.safetensors|${LOCAL_CKPT_PATH}|" \
  -e "s|${SOURCE_DATA}|${LOCAL_DATA}|" \
  "$RUN_CONFIG"

PORT=$(.venv/bin/python - <<'PY'
import socket
s = socket.socket()
s.bind(("", 0))
print(s.getsockname()[1])
s.close()
PY
)

echo "[start] job=${SLURM_JOB_ID} output=${OUT_DIR} gpus=4 priority=high condition=eef_swing_angle"
echo "[action] normalized unwrapped roll_x in channel 0, channels 1:14 exactly zero"
echo "[split] episode-level randomized 90/10; exact manifest copied to ${OUT_DIR}/input_manifest"
echo "[batch] per_gpu=3env*6distinct_skill=18 global=72"
echo "[schedule] model1-300; initial4 cycle301-1300; add3 cycle1301-2300; then C200/model200"
echo "[spatial loss] fixed polygon=(100,218);(640,175);(640,302);(100,335), ROI weight=6x normalized"

export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=16
export NCCL_DEBUG=WARN
export BWM_LOCAL_CHECKPOINT_ROOT=/tmp/${USER}/bwm_grouped_checkpoints/${SLURM_JOB_ID}

set +e
.venv/bin/python -m accelerate.commands.launch \
  --multi_gpu \
  --num_machines 1 \
  --num_processes 4 \
  --num_cpu_threads_per_process 16 \
  --main_process_port "$PORT" \
  --mixed_precision bf16 \
  scripts/train_stage1_grouped_context.py --config "$RUN_CONFIG" --find_unused_parameters &
TRAIN_PID=$!
while true; do
  wait "$TRAIN_PID"
  TRAIN_STATUS=$?
  if kill -0 "$TRAIN_PID" 2>/dev/null; then
    continue
  fi
  break
done
set -e
exit "$TRAIN_STATUS"
