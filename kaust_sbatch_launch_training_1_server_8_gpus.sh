#!/bin/bash
#SBATCH --job-name=hej0b
#SBATCH --time=14-00:00:00
#SBATCH -N 1
#SBATCH --mem=64GB
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:8
#SBATCH --reservation=A100
time=$(date "+%m%d-%H%M")

nvidia-smi

python -m torch.distributed.launch \
  --nproc_per_node 8 \
  /ibex/user/hej0b/zhengyang/PseudoTrain/run_glue.py \
  --model_name_or_path bert-base-cased \
  --task_name mrpc \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 10000000000000000 \
  --save_strategy no \
  --cache_dir /ibex/user/hej0b/zhengyang/PseudoTrain/cache \
  --output_dir /ibex/user/hej0b/zhengyang/PseudoTrain/tmp/mrpc/ \
  --overwrite_output_dir
