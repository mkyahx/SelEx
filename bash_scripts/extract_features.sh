PYTHON='/userhome/cs/mkyahx/miniconda3/envs/selex/bin/python'

hostname
nvidia-smi

export CUDA_VISIBLE_DEVICES=0

${PYTHON} -m methods.clustering.extract_features --dataset scars --use_best_model 'True' \
 --warmup_model_dir '/userhome/cs/mkyahx/SelEx/osr_novel_categories/metric_learn_gcd/log/(19.08.2026_|_12.691)/checkpoints/model.pt'