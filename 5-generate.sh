#!/bin/bash

pip install vllm==0.2.7 datasets

HF_TOKEN=*
huggingface-cli login --token $HF_TOKEN

LANGUAGES=("bo" "mt" "it" "es" "de" "ja" "ar" "zh" "af" "nl" "fr" "pt" "ru" "ko" "hi" "tr" "pl" "sv" "da" "no")

SCRIPT="batch_generation.py"

MODEL="meta-llama/Meta-Llama-3-8B"
ACTIVATION_MASK="activation_mask/llama-3"

echo "Starting systematic analysis..."
echo "Total combinations: $((${#LANGUAGES[@]} * ${#LANGUAGES[@]}))"

COUNTER=0
TOTAL=$((${#LANGUAGES[@]} * ${#LANGUAGES[@]}))

for deactivate_lang in "${LANGUAGES[@]}"; do
    for activate_lang in "${LANGUAGES[@]}"; do
        COUNTER=$((COUNTER + 1))
        
        echo "[$COUNTER/$TOTAL] Deactivating: $deactivate_lang, Activating: $activate_lang"

        python "$SCRIPT" \
            --model "$MODEL" \
            --activation_mask "$ACTIVATION_MASK" \
            --deactivate_lang "$deactivate_lang" \
            --activate_lang "$activate_lang" 
        
        sleep 1
    done
done