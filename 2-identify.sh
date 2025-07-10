#!/bin/bash

HF_TOKEN=*

huggingface-cli login --token $HF_TOKEN --add-to-git-credential

export CUDA_VISIBLE_DEVICES=0

rates=(0.01 0.02 0.03 0.04 0.05)

for i in "${!rates[@]}"; do
    RATE=${rates[$i]}
    SAVE_PATH="llama-3.1-$((i + 1))"
    echo "Running with top_rate=$RATE, save_path=$SAVE_PATH" # llama_3-1 llama-3.1
    python src/identify.py --top_rate $RATE --activations "llama_3-1 llama-3.1" --save_path "$SAVE_PATH"
done
