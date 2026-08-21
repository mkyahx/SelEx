PYTHON='/userhome/cs/mkyahx/miniconda3/envs/selex/bin/python'

hostname
nvidia-smi

export CUDA_VISIBLE_DEVICES=0

# Get unique log file.
SAVE_DIR=/userhome/cs/mkyahx/SelEx/dev_outputs/
mkdir -p ${SAVE_DIR}

EXP_NUM=$(ls ${SAVE_DIR} | wc -l)
EXP_NUM=$((${EXP_NUM}+1))
echo $EXP_NUM

${PYTHON} -m methods.contrastive_training.contrastive_train_debug \
            --dataset_name 'cifar10' \
            --batch_size 128 \
            --grad_from_block 11 \
            --epochs 200 \
            --base_model vit_dino \
            --num_workers 4 \
            --use_ssb_splits 'True' \
            --sup_con_weight 0.35 \
            --weight_decay 5e-5 \
            --contrast_unlabel_only 'False' \
            --transform 'imagenet' \
            --lr 0.1 \
            --eval_funcs 'v1' 'v2' \
            --debug_log_interval 1 \
            --debug_kmeans_iter_limit 100 \
            --debug_detect_anomaly 'True' \
> ${SAVE_DIR}debug_logfile_${EXP_NUM}.out
