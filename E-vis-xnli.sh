#!/bin/bash

pip install fasttext

MODEL="meta-llama/Llama-3.1-8B"
MODEL_NAME=${MODEL#*/}

for RATIO in {1..5}; do
  INPUT="xnli/${MODEL_NAME}_${RATIO}"
  OUTPUT="xnli/${MODEL_NAME}_${RATIO}"

  echo "Processing ratio $RATIO..."
  python vis_xnli.py --input_path "$INPUT" --output_path "$OUTPUT"
done

MODEL="mistralai/Mistral-Nemo-Base-2407"
MODEL_NAME=${MODEL#*/}

for RATIO in {1..5}; do
  INPUT="xnli/${MODEL_NAME}_${RATIO}"
  OUTPUT="xnli/${MODEL_NAME}_${RATIO}"

  echo "Processing ratio $RATIO..."
  python vis_xnli.py --input_path "$INPUT" --output_path "$OUTPUT"
done