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
# mistralai/Mistral-Nemo-Base-2407

huggingface-cli login --token $HF_TOKEN --add-to-git-credential

languages=("bo" "mt" "it" "es" "de" "ja" "ar")

model="mistralai/Mistral-Nemo-Base-2407"

export CUDA_VISIBLE_DEVICES=0,1 # if needed

for lang in "${languages[@]}"
do
    echo "Running activation.py for language: $lang"
    python src/activation.py -m $model -l $lang -s "nemo nemo"
done
