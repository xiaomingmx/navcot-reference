#!/bin/bash

# 添加 checkpoint 路径变量
checkpoint_path=$2  # 作为第二个命令行参数传入

pretrained_path='/root/autodl-tmp/navcot/Data_prepar/Training/LLaMA2_weight'
pretrained_type=consolidated
tokenizer_path="/root/autodl-tmp/navcot/Data_prepar/Training/tokenizer"
data_config='/root/autodl-tmp/navcot/NavCoT/LLaMA2-Accessory/accessory/configs/data/finetune/sg/alpaca.yaml'
llama_config='/root/autodl-tmp/navcot/Data_prepar/Training/tokenizer/code_7B_params.json'

data_parallel=sdp
model_parallel=1

exp_name=/root/autodl-tmp/navcot/NavCoT/finetuned_model-r2r

echo "exp name: $exp_name"
mkdir -p "$exp_name"

# 切换到项目根目录
cd /root/autodl-tmp/navcot/NavCoT/LLaMA2-Accessory/accessory/

# 获取可用的GPU数量
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "检测到 $NUM_GPUS 个可用 GPU"

# 构建命令，添加 resume 参数（如果提供）
resume_arg=""
if [ -n "$checkpoint_path" ]; then
    resume_arg="--resume $checkpoint_path"
    echo "将从 checkpoint 恢复训练: $checkpoint_path"
fi

# 使用实际可用的GPU数量
CUDA_VISIBLE_DEVICES=$1 torchrun --master_port=1112 --nproc_per_node=$NUM_GPUS main_finetune.py \
--output_dir "$exp_name" --epochs 2 --warmup_epochs 1 \
--batch_size 2 --accum_iter 2 --num_workers 4 \
--max_words 400 \
--lr 0.001 --min_lr 0.000005 --clip_grad 2 --weight_decay 0.02 \
--data_parallel "$data_parallel" --model_parallel_size "$model_parallel" --checkpointing \
--llama_type llama_peft --llama_config $llama_config --tokenizer_path "$tokenizer_path" \
--no_visual \
--pretrained_path "$pretrained_path" --pretrained_type="$pretrained_type" \
--data_config $data_config \
--teacher_forcing \
--precision "tf32" \
$resume_arg \
# --precision "fp16" \
2>&1 | tee -a "$exp_name"/output.log

echo "exp name: $exp_name"