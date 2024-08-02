#!/bin/bash

path="src/postprocessing"
is_preprocessed=True
is_tuned="untuned"
strategy="deepspeed_stage_3_offload"
upload_user="meta-llama"
model_type="Meta-Llama-3.1-8B-Instruct"
quantization_type="origin"
peft_type="origin"
data_max_length=384
target_max_length=384
precision="bf16"
batch_size=32
epoch=10

python $path/prepare_upload.py \
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
    epoch=$epoch
