#!/bin/bash
#SBATCH --job-name=zhengyang
#SBATCH -p batch*
#SBATCH --time=60
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:2
time=$(date "+%m%d-%H%M")

python -m torch.distributed.launch \
  --nproc_per_node 2 \
  /ibex/user/hej0b/zhengyang/PseudoTrain/run_glue.py \
  --model_name_or_path bert-base-cased \
  --task_name mrpc \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 10 \
  --cache_dir /ibex/user/hej0b/zhengyang/PseudoTrain/cache \
  --output_dir /ibex/user/hej0b/zhengyang/PseudoTrain/tmp/mrpc/ \
  --overwrite_output_dir > /ibex/user/hej0b/zhengyang/PseudoTrain/zhengyang_$time.log
