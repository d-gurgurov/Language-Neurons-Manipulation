#!/bin/bash

pip install vllm datasets
pip install numpy==1.26.4 # --> required by Nemo 

HF_TOKEN=*
huggingface-cli login --token $HF_TOKEN

SCRIPT="batch_generation.py"
MODEL="meta-llama/Llama-3.1-8B" # mistralai/Mistral-Nemo-Base-2407 meta-llama/Llama-3.1-8B
MODEL_NAME=${MODEL#*/}
ACTIVATIONS_PATH="llama_3-1 llama-3.1" # llama_3-1 llama-3.1 | nemo nemo
MASK_NAME="llama-3.1" # llama-3.1 | nemo
STRENGTH=0 # -2.5

export VLLM_USE_V1=0

for NEURON_RATIO in 1 2 3 4 5; do
    ACTIVATION_MASK="activation_mask/${MASK_NAME}-${NEURON_RATIO}"
    OUTPUT="generation/${MODEL_NAME}_${NEURON_RATIO}"

    echo "Running for NEURON_RATIO=$NEURON_RATIO"
    echo "Output path: $OUTPUT"
    
    # only activate the target
    python $SCRIPT --no_deactivation --batch_mode \
        --output "$OUTPUT" \
        --activations_path "$ACTIVATIONS_PATH" \
        --model "$MODEL" \
        --activation_mask "$ACTIVATION_MASK"

    # activate and deactivate
    python $SCRIPT --batch_mode \
        --output "$OUTPUT" \
        --activations_path "$ACTIVATIONS_PATH" \
        --model "$MODEL" \
        --activation_mask "$ACTIVATION_MASK" \
        --deactivation_strength $STRENGTH
done
