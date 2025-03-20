#!/bin/bash
#SBATCH --job-name=train_job
#SBATCH --partition=8gpu
#SBATCH --gres=gpu:8
#SBATCH --nodelist=gpu-8-003
#SBATCH --output=logs/train_output.log
#SBATCH --error=logs/train_error.log

cd ~/llm-fine-tune

module add compilers/cuda/12.4 compilers/gcc/10.2.0 libraries/nccl/2.21.5
source activate myenv

split_ratio=1e-4
is_sft=False
is_preprocessed=False
is_tuned="untuned"
strategy="deepspeed_stage_3_offload"
upload_user="meta-llama"
model_type="Llama-3.1-8B-Instruct"
left_padding=False
quantization_type="origin"
peft_type="origin"
data_type="structural"
dataset_name="open-Korean"
data_max_length=2048
target_max_length=2048
precision="bf16"
batch_size=16
eval_batch_size=16
accumulate_grad_batches=8
lr=3e-5
weight_decay=1e-1
warmup_ratio=5e-2
eta_min_ratio=1e-2
epoch=5
step=1e+3
workers_ratio=8
use_all_workers=False

python main.py mode=train \
    split_ratio=$split_ratio \
    is_sft=$is_sft \
    is_preprocessed=$is_preprocessed \
    is_tuned=$is_tuned \
    strategy=$strategy \
    upload_user=$upload_user \
    model_type=$model_type \
    left_padding=$left_padding \
    quantization_type=$quantization_type \
    peft_type=$peft_type \
    data_type=$data_type \
    dataset_name=$dataset_name \
    data_max_length=$data_max_length \
    target_max_length=$target_max_length \
    precision=$precision \
    batch_size=$batch_size \
    eval_batch_size=$eval_batch_size \
    accumulate_grad_batches=$accumulate_grad_batches \
    lr=$lr \
    weight_decay=$weight_decay \
    warmup_ratio=$warmup_ratio \
    eta_min_ratio=$eta_min_ratio \
    epoch=$epoch \
    step=$step \
    workers_ratio=$workers_ratio \
    use_all_workers=$use_all_workers
