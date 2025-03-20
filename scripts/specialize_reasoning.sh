#!/bin/bash

path="src/preprocessing"
upload_user="meta-llama"
model_type="Llama-3.1-8B-Instruct"

python $path/specialize_reasoning.py upload_user=$upload_user model_type=$model_type
