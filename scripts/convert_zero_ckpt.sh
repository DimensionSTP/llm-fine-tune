#!/bin/bash

path="src/postprocessing"
is_sft=False
is_preprocessed=False
is_tuned="untuned"
strategy="deepspeed_stage_3_offload"
upload_user="meta-llama"
model_type="Llama-3.1-8B-Instruct"
quantization_type="origin"
peft_type="origin"
dataset_name="open-Korean"
data_max_length=2048
target_max_length=2048
precision="bf16"
batch_size=16
accumulate_grad_batches=8

python $path/convert_zero_ckpt.py \
    is_sft=$is_sft \
    is_preprocessed=$is_preprocessed \
    is_tuned=$is_tuned \
    strategy=$strategy \
    upload_user=$upload_user \
    model_type=$model_type \
    quantization_type=$quantization_type \
    peft_type=$peft_type \
    dataset_name=$dataset_name \
    data_max_length=$data_max_length \
    target_max_length=$target_max_length \
    precision=$precision \
    batch_size=$batch_size \
    accumulate_grad_batches=$accumulate_grad_batches
