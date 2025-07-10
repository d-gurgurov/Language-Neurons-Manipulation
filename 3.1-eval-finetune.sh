#!/bin/bash

pip install vllm datasets
pip install accelerate
pip install numpy==1.26.4

HF_TOKEN=*
huggingface-cli login --token $HF_TOKEN


python eval_finetune.py \
    --base_model "meta-llama/Meta-Llama-3.1-8B" \
    --finetuned_model "finetune/Llama-3.1-8B_1/best_model" \
    --target_lang mlt_Latn \
    --output_dir "finetune/maltese_eval"