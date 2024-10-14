#!/bin/bash

split_ratio=1e-2
is_preprocessed=False
is_tuned="untuned"
strategy="deepspeed_stage_2_offload"
upload_user="meta-llama"
model_type="Meta-Llama-3.1-8B-Instruct"
left_padding=False
quantization_type="origin"
peft_type="origin"
data_max_length=1024
target_max_length=1024
precision="bf16"
batch_size=4
accumulate_grad_batches=4
lr=3e-7
weight_decay=1e-1
warmup_ratio=5e-2
eta_min_ratio=1e-3
epoch=4
step=1e+2

python main.py --config-name=dpo.yaml mode=train \
    split_ratio=$split_ratio \
    is_preprocessed=$is_preprocessed \
    is_tuned=$is_tuned \
    strategy=$strategy \
    upload_user=$upload_user \
    model_type=$model_type \
    left_padding=$left_padding \
    quantization_type=$quantization_type \
    peft_type=$peft_type \
    data_max_length=$data_max_length \
    target_max_length=$target_max_length \
    precision=$precision \
    batch_size=$batch_size \
    accumulate_grad_batches=$accumulate_grad_batches \
    lr=$lr \
    weight_decay=$weight_decay \
    warmup_ratio=$warmup_ratio \
    eta_min_ratio=$eta_min_ratio \
    epoch=$epoch \
    step=$step
