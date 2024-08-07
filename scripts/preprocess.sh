#!/bin/bash

path="src/preprocessing"
upload_user="meta-llama"
model_type="Meta-Llama-3.1-8B-Instruct"

python $path/merge_tokenizer.py upload_user=$upload_user model_type=$model_type
python $path/merge_model.py upload_user=$upload_user model_type=$model_type
