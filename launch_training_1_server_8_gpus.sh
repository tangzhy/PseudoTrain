export TASK_NAME=mrpc
export WANDB_DISABLED=true
export NUM_SERVERS=1
export SERVER_INDEX=0
export NUM_GPUS_PER_SERVER=8

HF_ENDPOINT=https://hf-mirror.com torchrun \
  --nnodes $NUM_SERVERS \
  --node_rank $SERVER_INDEX \
  --nproc_per_node $NUM_GPUS_PER_SERVER \
  run_glue.py \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 100 \
  --learning_rate 2e-5 \
  --num_train_epochs 10000000000000000000000 \
  --save_strategy no \
  --cache_dir ./cache \
  --output_dir ./output/$TASK_NAME/ \
  --report_to none \
  --overwrite_output_dir
