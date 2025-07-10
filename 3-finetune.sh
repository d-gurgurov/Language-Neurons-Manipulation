#!/bin/bash

pip install vllm datasets 
pip install -U accelerate

HF_TOKEN=*
huggingface-cli login --token $HF_TOKEN

export VLLM_USE_V1=0

MODEL="meta-llama/Llama-3.1-8B"
MODEL_NAME=${MODEL#*/}
RATIO=1
ACTIVATION_MASK="activation_mask/llama-3.1-${RATIO}"

python src/finetune.py --target_lang mt --num_samples 10000 --epochs 3 \
                 --output_dir "finetune/${MODEL_NAME}_${RATIO}" \
                 --model $MODEL --activation_mask $ACTIVATION_MASK --batch_size 8
                 

