#!/bin/bash

path="src/postprocessing"
is_sft=False
is_preprocessed=False
is_tuned="untuned"
strategy="deepspeed_stage_3_offload"
upload_user="meta-llama"
model_type="Llama-3.1-8B-Instruct"
left_padding=False
quantization_type="origin"
peft_type="origin"
data_max_length=2048
target_max_length=2048
precision="bf16"
batch_size=16
accumulate_grad_batches=8
dataset_name="open-Korean"
model_detail="Llama-3.1-8B-Instruct"
upload_tag="open-Korean"

python $path/upload_all_to_hf_hub.py \
    is_sft=$is_sft \
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
    dataset_name=$dataset_name \
    model_detail=$model_detail \
    upload_tag=$upload_tag
