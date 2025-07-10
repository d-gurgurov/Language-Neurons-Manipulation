#!/bin/bash

pip install vllm datasets
pip install -U transformers
pip install fasttext
pip install numpy==1.26.4

HF_TOKEN=*
huggingface-cli login --token $HF_TOKEN

export VLLM_USE_V1=0

MODEL="mistralai/Mistral-Nemo-Base-2407" # mistralai/Mistral-Nemo-Base-2407 meta-llama/Llama-3.1-8B
ACTIVATION_PATH="nemo" # llama-3.1
MODEL_NAME=${MODEL#*/}
METHOD="negative"
STRENGTH=-1.5

for RATIO in {1..5}; do
    echo "Running with RATIO=$RATIO"
    
    python src/fallbacks.py -m "$MODEL" \
                        --activation_mask "activation_mask/${ACTIVATION_PATH}-${RATIO}" \
                        --progressive_only \
                        --output_dir "fallbacks/${METHOD}${STRENGTH}/${MODEL_NAME}_${RATIO}" \
                        --deactivation_method $METHOD \
                        --deactivation_strength $STRENGTH
                        
done
