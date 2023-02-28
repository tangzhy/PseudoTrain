## pytorch & transformers 多机多卡训练调试

此代码用于调通 单机多卡 和 双机多卡 两种常见训练模式。
例子来源: https://github.com/huggingface/transformers/tree/v4.9.2/examples/pytorch/text-classification

### 环境依赖

We need `python >= 3.7`, for package installation, please refer to requirements.txt.

```
accelerate
datasets >= 1.8.0
sentencepiece != 0.1.92
protobuf
torch >= 1.3
transformers >= 4.9.0
```

### 调通 单机多卡 训练：

可在单机环境下, 执行以下命令
```
cd ~/PseudoTrain
sh launch_training_1_server_8_gpus.sh
```

launch脚本如下
```
export TASK_NAME=mrpc
export WANDB_DISABLED=true
export NUM_SERVERS=1
export SERVER_INDEX=0
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
  --num_train_epochs 10 \
  --cache_dir ./cache \
  --output_dir ./tmp/$TASK_NAME/ \
  --overwrite_output_dir
```
其中`NUM_GPUS_PER_SERVER`可根据实际单机gpu数量调整为相应数量。

训练完成后，应该见到如下类似日志, 数值可能变化没有关系
```
  epoch                   =       10.0
  eval_accuracy           =     0.8137
  eval_combined_score     =     0.8404
  eval_f1                 =     0.8671
  eval_loss               =     0.4761
  eval_runtime            = 0:00:00.10
  eval_samples            =        408
  eval_samples_per_second =   3713.457
  eval_steps_per_second   =     63.711
```

### 调通 双机多卡 训练:

需要同时ssh进两台机器,**两台机器应该具有相同的python环境，以及相同gpu卡数, 并且两台机器可以通过nvlink通信!**

第一台机器执行以下命令
```
cd ~/PseudoTrain
sh launch_training_2_servers_8_gpus_node0.sh
```

第二台机器执行以下命令
```
cd ~/PseudoTrain
sh launch_training_2_servers_8_gpus_node1.sh
```

同样, `NUM_GPUS_PER_SERVER`可根据实际每台机器的gpu数量调整为相应数量。

跑完后应该也会见到如单机训练类似的日志。




