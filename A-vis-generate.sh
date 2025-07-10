#!/bin/bash

pip install vllm datasets 
pip install fasttext


RATIO=5
MODEL="Mistral-Nemo-Base-2407" # Mistral-Nemo-Base-2407 Llama-3.1-8B
INPUT="generation/${MODEL}_${RATIO}/activate"
OUTPUT="generation/${MODEL}_${RATIO}/activate"

python vis_forcing.py --input_path $INPUT --output_path $OUTPUT

INPUT="generation/${MODEL}_${RATIO}/deactivate_activate"
OUTPUT="generation/${MODEL}_${RATIO}/deactivate_activate"

python vis_forcing.py --input_path $INPUT --output_path $OUTPUT