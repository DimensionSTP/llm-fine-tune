#!/bin/bash

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
workers_ratio=8
use_all_workers=False
steps="170000 180000"

for step in $steps
do
    python main.py --config-name=dpo.yaml mode=predict \
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
        workers_ratio=$workers_ratio \
        use_all_workers=$use_all_workers \
        step=$step
done

for step in $steps
do
    python merge_predictions.py \
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
        workers_ratio=$workers_ratio \
        use_all_workers=$use_all_workers \
        step=$step
done
