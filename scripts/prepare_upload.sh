#!/bin/bash

path="src/postprocessing"
is_preprocessed=False
is_tuned="untuned"
strategy="deepspeed_stage_3_offload"
upload_user="meta-llama"
model_type="Meta-Llama-3.1-8B-Instruct"
left_padding=False
quantization_type="origin"
peft_type="origin"
data_max_length=512
target_max_length=512
precision="bf16"
batch_size=24
lr=1e-4
accumulate_grad_batches=4
epoch=2
model_detail="Llama-3.1-8B-Instruct"

python $path/prepare_upload.py \
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
    lr=$lr \
    accumulate_grad_batches=$accumulate_grad_batches \
    epoch=$epoch \
    model_detail=$model_detail
