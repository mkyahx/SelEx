#!/bin/bash
#SBATCH --job-name=asymmetric_a2_cars_repro
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mkyahx@connect.hku.hk
#SBATCH --output=/userhome/cs/mkyahx/dev_outputs/SelEx/air_%j.out
#SBATCH --error=/userhome/cs/mkyahx/dev_outputs/SelEx/air_%j.err

set -e
set -x

mkdir -p /userhome/cs/mkyahx/dev_outputs/SelEx
source /userhome/cs/mkyahx/miniconda3/etc/profile.d/conda.sh
conda activate selex
cd /userhome/cs/mkyahx/SelEx/
nvidia-smi

export CUBLAS_WORKSPACE_CONFIG=:4096:8



PYTHON='/userhome/cs/mkyahx/miniconda3/envs/selex/bin/python'

hostname
nvidia-smi

export CUDA_VISIBLE_DEVICES=0

# Get unique log file,
SAVE_DIR=/userhome/cs/mkyahx/SelEx/dev_outputs/

EXP_NUM=$(ls ${SAVE_DIR} | wc -l)
EXP_NUM=$((${EXP_NUM}+1))
echo $EXP_NUM

${PYTHON} -m methods.contrastive_training.contrastive_training \
            --dataset_name 'aircraft' \
            --batch_size 128 \
            --grad_from_block 10 \
            --epochs 200 \
            --base_model vit_dino \
            --num_workers 4 \
            --use_ssb_splits True \
            --sup_con_weight 0.35 \
            --weight_decay 5e-5 \
            --contrast_unlabel_only False \
            --transform 'imagenet' \
            --lr 0.1 \
            --eval_funcs 'v1' 'v2' \
            --unsupervised_smoothing 0.5 \
 > ${SAVE_DIR}logfile_${EXP_NUM}.out

conda deactivate
echo "Finish"