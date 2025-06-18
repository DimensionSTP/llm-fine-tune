#!/bin/bash

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
data_max_length=2048
target_max_length=2048
precision="bf16"
batch_size=16
eval_batch_size=16
accumulate_grad_batches=1
workers_ratio=8
use_all_workers=False
steps="1000 1250"

for step in $steps
do
    python main.py mode=predict \
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
        workers_ratio=$workers_ratio \
        use_all_workers=$use_all_workers \
        step=$step
done

for step in $steps
do
    python merge_predictions.py \
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
        workers_ratio=$workers_ratio \
        use_all_workers=$use_all_workers \
        step=$step
done
