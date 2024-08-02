#!/bin/bash

path="src/preprocessing"

python $path/merge_tokenizer.py
python $path/merge_model.py
