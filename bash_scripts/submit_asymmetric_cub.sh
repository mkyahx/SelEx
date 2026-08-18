#!/bin/bash
#SBATCH --job-name=selex-asym-cub
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-2

set -euo pipefail
: "${MASK_ROOT:?Set MASK_ROOT to the CUB TokenCut mask directory.}"

SAVE_DIR="${SAVE_DIR:-$(pwd)/dev_outputs}"
mkdir -p "$SAVE_DIR"
SEED="${SLURM_ARRAY_TASK_ID:-0}"
if [[ -n "${SLURM_ARRAY_JOB_ID:-}" ]]; then
  EXP_NUM="${SLURM_ARRAY_JOB_ID}"
else
  EXP_NUM=$(($(find "$SAVE_DIR" -mindepth 1 -maxdepth 1 | wc -l) + 1))
fi
LOG_FILE="${SAVE_DIR%/}/logfile_${EXP_NUM}_seed_${SEED}.out"
echo "Running seed ${SEED}; logging to ${LOG_FILE}"

python methods/contrastive_training/asymmetric_mask_training.py \
  --dataset_name cub \
  --batch_size 128 \
  --grad_from_block 10 \
  --epochs 200 \
  --base_model vit_dino \
  --num_workers 4 \
  --use_ssb_splits True \
  --sup_con_weight 0.35 \
  --weight_decay 5e-5 \
  --contrast_unlabel_only False \
  --transform imagenet \
  --lr 0.1 \
  --seed "${SEED}" \
  --deterministic True \
  --eval_funcs v1 v2 \
  --unsupervised_smoothing 1.0 \
  --mask_root "$MASK_ROOT" \
  --max_foreground_tokens 128 \
  --report False \
  > "$LOG_FILE" 2>&1
