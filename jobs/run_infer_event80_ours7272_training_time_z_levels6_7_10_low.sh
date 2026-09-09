#!/usr/bin/env bash
#SBATCH -A yejin
#SBATCH --job-name=ev80-ztrain-6710
#SBATCH -p yejin-lo
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=04:00:00
#SBATCH --output=/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/logs/slurm/%x-%j.out
#SBATCH --error=/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/logs/slurm/%x-%j.err

set -euo pipefail

REPO=/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt
cd "$REPO"

export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

CONFIG=configs/train/train_push_box_event_tap_segmented80_pushmotion41f_10chunk_curriculum_c32_random_stage1_15300.yaml
CKPT=outputs/curc32r65ib2-nc015_88823/step-7272.safetensors
TABLE=outputs/curc32r65ib2-nc015_88823/step-7272.context_table.json
INDICES=405,406,409

OUT_ROOT=/afs/ir/users/c/y/cyzhou05/TTT-Physics/tmp/pushbox_event80_mu075_level6_level7_level10_frames
RAW_DIR=${OUT_ROOT}/training_time_raw
SHARED_ROOT=/tmp/${USER}/bwm_shared_cache
LOCAL_MODEL=${SHARED_ROOT}/Wan2.2-TI2V-5B
LOCAL_CKPT=${SHARED_ROOT}/trained_ckpts/event80_ours_step7272.safetensors
LOCAL_TABLE=${SHARED_ROOT}/trained_ckpts/event80_ours_step7272.context_table.json

mkdir -p "$RAW_DIR" "$OUT_ROOT/level6" "$OUT_ROOT/level7" "$OUT_ROOT/level10" \
  "$SHARED_ROOT/trained_ckpts" logs/slurm

(
  flock 9
  if [ ! -f "$LOCAL_MODEL/.copy_complete" ]; then
    mkdir -p "$LOCAL_MODEL"
    rsync -a models/Wan2.2-TI2V-5B/ "$LOCAL_MODEL/"
    touch "$LOCAL_MODEL/.copy_complete"
  fi
) 9>"$SHARED_ROOT/Wan2.2-TI2V-5B.lock"

(
  flock 8
  if [ ! -s "$LOCAL_CKPT" ]; then
    rsync -a "$CKPT" "$LOCAL_CKPT"
  fi
  if [ ! -s "$LOCAL_TABLE" ]; then
    rsync -a "$TABLE" "$LOCAL_TABLE"
  fi
) 8>"$SHARED_ROOT/trained_ckpts/event80_ours_step7272.lock"

cat > "$OUT_ROOT/training_time_inference.txt" <<EOF
slurm_job_id=$SLURM_JOB_ID
method=ours_step7272_training_time_environment_latent
checkpoint=$CKPT
context_table=$TABLE
friction_mu=0.075
sample_indices=$INDICES
inner_loop=disabled
seed=20260708
EOF

# One process loads the model once and generates all three trajectories.
.venv/bin/python scripts/infer.py \
  --config "$CONFIG" \
  --model_paths "$LOCAL_MODEL" \
  --ckpt_path "$LOCAL_CKPT" \
  --physical_context_table_path "$LOCAL_TABLE" \
  --output_path "$RAW_DIR" \
  --sample_indices "$INDICES" \
  --num_inference_steps 25 \
  --cfg_scale 1 \
  --fps 20 \
  --quality 6 \
  --seed 20260708 \
  --skip_existing

FFMPEG=$(.venv/bin/python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')

extract_frames() {
  local level=$1
  local episode=$2
  local sample
  local ep
  local prediction
  sample=$(printf '%04d' "$episode")
  ep=$(printf '%06d' "$episode")
  prediction=$(find "$RAW_DIR" -maxdepth 1 -type f \
    -name "sample${sample}_episode${ep}_frames0065-0105.mp4" -print -quit)
  test -n "$prediction"
  "$FFMPEG" -v error -y -i "$prediction" \
    -vf "select='eq(n,5)+eq(n,10)+eq(n,15)+eq(n,20)+eq(n,40)'" \
    -fps_mode vfr -start_number 1 \
    "$OUT_ROOT/level${level}/Training_time_prediction_%d.png"
}

extract_frames 6 405
extract_frames 7 406
extract_frames 10 409

touch "$OUT_ROOT/training_time_complete"
echo "[done] raw=$RAW_DIR frames=$OUT_ROOT"
