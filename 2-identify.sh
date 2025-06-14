#!/bin/bash

pip install vllm==0.2.7

HF_TOKEN=*

huggingface-cli login --token $HF_TOKEN --add-to-git-credential

export CUDA_VISIBLE_DEVICES=0

python src/identify.py
