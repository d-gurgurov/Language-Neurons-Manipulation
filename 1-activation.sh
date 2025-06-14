#!/bin/bash

pip install vllm==0.2.7

HF_TOKEN=*

huggingface-cli login --token $HF_TOKEN --add-to-git-credential

languages=("bo" "mt" "it" "es" "de" "ja" "ar" "zh" "af" "nl" "fr" "pt" "ru" "ko" "hi" "tr" "pl" "sv" "da" "no")

model="meta-llama/Meta-Llama-3-8B"

export CUDA_VISIBLE_DEVICES=0

for lang in "${languages[@]}"
do
    echo "Running activation.py for language: $lang"
    python src/activation.py -m $model -l $lang
done
