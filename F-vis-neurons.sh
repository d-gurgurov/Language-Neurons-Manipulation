#!/bin/bash

RATIO=1
MODEL_NAME="llama-3.1" # llama-3.1
INPUT="activation_mask/${MODEL_NAME}-${RATIO}"
OUTPUT="neurons/${MODEL_NAME}-${RATIO}"

python src/vis_neurons.py --input_path $INPUT --output_path $OUTPUT