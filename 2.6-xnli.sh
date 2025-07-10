#!/bin/bash

pip install vllm datasets
pip install numpy==1.26.4 # --> required by Nemo 

HF_TOKEN=*
huggingface-cli login --token $HF_TOKEN

export VLLM_USE_V1=0

python xnli.py -m "meta-llama/Llama-3.1-8B" \
                          --activations_path "llama_3-1 llama-3.1" \
                          --activation_mask "llama-3.1"