export TASK_NAME=mrpc
export WANDB_DISABLED=true
export NUM_SERVERS=2
export SERVER_INDEX=1
export NUM_GPUS_PER_SERVER=8

python -m torch.distributed.launch \
  --nnodes $NUM_SERVERS \
  --node_rank $SERVER_INDEX \
  --nproc_per_node $NUM_GPUS_PER_SERVER \
  run_glue.py \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 30 \
  --cache_dir ./cache \
  --output_dir ./tmp/$TASK_NAME/ \
  --overwrite_output_dir
