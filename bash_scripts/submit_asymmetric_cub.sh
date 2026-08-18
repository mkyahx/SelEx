#!/bin/bash
#SBATCH --job-name=selex-asym-cub
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-2

set -euo pipefail
: "${MASK_ROOT:?Set MASK_ROOT to the CUB TokenCut mask directory.}"

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
  --seed "${SLURM_ARRAY_TASK_ID}" \
  --deterministic True \
  --eval_funcs v1 v2 \
  --unsupervised_smoothing 1.0 \
  --mask_root "$MASK_ROOT" \
  --max_foreground_tokens 128 \
  --report False
