#!/bin/bash

# do not run on V100 or so. on this kind of GPUs, VLLM backs up to using an alternative for flash attention and everything crashes
pip install vllm
pip install -U transformers

HF_TOKEN=*

export VLLM_USE_V1=0

# meta-llama/Llama-3.1-8B 
# meta-llama/Meta-Llama-3-8B 
# meta-llama/Llama-3.1-70B
# CohereLabs/aya-expanse-8b
# CohereLabs/aya-expanse-32b

huggingface-cli login --token $HF_TOKEN --add-to-git-credential

# List of languages to process "bo" "mt" "it" "es" "de" "ja" "ar" "zh" "af" "nl" "fr" "pt" "ru" "ko" "hi" "tr" "pl" "sv" "da" "no" "en"
languages=("en")

# Model name
model="CohereLabs/aya-expanse-8b" # meta-llama/Meta-Llama-3-8B  meta-llama/Llama-3.1-8B

# GPU assignment
export CUDA_VISIBLE_DEVICES=0

# Loop through each language
for lang in "${languages[@]}"
do
    echo "Running activation.py for language: $lang"
    python src/activation.py -m $model -l $lang -s "aya_8 aya-8"
done
