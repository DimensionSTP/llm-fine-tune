#!/bin/bash

path="src/postprocessing"
is_sft=False
is_preprocessed=False
is_tuned="untuned"
strategy="deepspeed_stage_3_offload"
upload_user="Qwen"
model_type="Qwen3-8B"
quantization_type="origin"
peft_type="origin"
data_max_length=2048
target_max_length=2048
precision="bf16"
batch_size=16
accumulate_grad_batches=1
dataset_name="open-Korean"
model_detail="Qwen3-8B"
upload_tag="open-Korean"

python $path/upload_all_to_hf_hub.py \
    is_sft=$is_sft \
    is_preprocessed=$is_preprocessed \
    is_tuned=$is_tuned \
    strategy=$strategy \
    upload_user=$upload_user \
    model_type=$model_type \
    quantization_type=$quantization_type \
    peft_type=$peft_type \
    data_max_length=$data_max_length \
    target_max_length=$target_max_length \
    precision=$precision \
    batch_size=$batch_size \
    accumulate_grad_batches=$accumulate_grad_batches \
    dataset_name=$dataset_name \
    model_detail=$model_detail \
    upload_tag=$upload_tag
