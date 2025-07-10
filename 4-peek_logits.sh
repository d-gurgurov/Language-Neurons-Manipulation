#!/bin/bash

pip install vllm datasets
pip install -U accelerate
pip install numpy==1.26.4 # --> required by Nemo
pip install fasttext 

HF_TOKEN=*
huggingface-cli login --token $HF_TOKEN

export VLLM_USE_V1=0

OUTPUT="peek/aya-expanse-8b/"
MODEL="CohereLabs/aya-expanse-8b"

python src/logit_lens.py \
            --languages bo mt it es de ja ar zh af nl fr pt ru ko hi tr pl sv da no en  \
            --output $OUTPUT \
            --model $MODEL


OUTPUT="peek/aya-expanse-32b/"
MODEL="CohereLabs/aya-expanse-32b"

python src/logit_lens.py \
            --languages bo mt it es de ja ar zh af nl fr pt ru ko hi tr pl sv da no en  \
            --output $OUTPUT \
            --model $MODEL

OUTPUT="peek/Llama-3.1-8B/"
MODEL="meta-llama/Llama-3.1-8B"

python src/logit_lens.py \
            --languages bo mt it es de ja ar zh af nl fr pt ru ko hi tr pl sv da no en  \
            --output $OUTPUT \
            --model $MODEL

OUTPUT="peek/Mistral-Nemo-Base-2407/"
MODEL="mistralai/Mistral-Nemo-Base-2407"

python src/logit_lens.py \
            --languages bo mt it es de ja ar zh af nl fr pt ru ko hi tr pl sv da no en  \
            --output $OUTPUT \
            --model $MODEL