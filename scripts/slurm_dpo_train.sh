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

split_ratio=1e-2
is_strict_split=False
is_sft=False
is_preprocessed=False
is_tuned="untuned"
strategy="deepspeed_stage_3_offload"
upload_user="Qwen"
model_type="Qwen3-8B"
revision="main"
left_padding=True
is_enable_thinking=False
quantization_type="origin"
peft_type="origin"
data_type="structural"
dataset_name="open-Korean"
data_max_length=1024
target_max_length=1024
precision="bf16"
batch_size=8
eval_batch_size=8
accumulate_grad_batches=2
dpo_beta=0.1
lr=5e-7
weight_decay=1e-1
warmup_ratio=5e-2
eta_min_ratio=1e-2
epoch=2
step=250
workers_ratio=8
use_all_workers=False
convert_at_end=False

python main.py --config-name=dpo.yaml mode=train \
    split_ratio=$split_ratio \
    is_strict_split=$is_strict_split \
    is_sft=$is_sft \
    is_preprocessed=$is_preprocessed \
    is_tuned=$is_tuned \
    strategy=$strategy \
    upload_user=$upload_user \
    model_type=$model_type \
    revision=$revision \
    left_padding=$left_padding \
    is_enable_thinking=$is_enable_thinking \
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
    dpo_beta=$dpo_beta \
    lr=$lr \
    weight_decay=$weight_decay \
    warmup_ratio=$warmup_ratio \
    eta_min_ratio=$eta_min_ratio \
    epoch=$epoch \
    step=$step \
    workers_ratio=$workers_ratio \
    use_all_workers=$use_all_workers \
    convert_at_end=$convert_at_end
