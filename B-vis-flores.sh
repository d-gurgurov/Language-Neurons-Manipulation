#!/bin/bash

pip install fasttext

MODEL="meta-llama/Llama-3.1-8B"
MODEL_NAME=${MODEL#*/}

for RATIO in {1..5}; do
  INPUT="flores/${MODEL_NAME}_${RATIO}"
  OUTPUT="flores/${MODEL_NAME}_${RATIO}"

  echo "Processing ratio $RATIO..."
  python src/vis_flores.py --input_path "$INPUT" --output_path "$OUTPUT"
done

MODEL="mistralai/Mistral-Nemo-Base-2407"
MODEL_NAME=${MODEL#*/}

for RATIO in {1..5}; do
  INPUT="flores/${MODEL_NAME}_${RATIO}"
  OUTPUT="flores/${MODEL_NAME}_${RATIO}"

  echo "Processing ratio $RATIO..."
  python src/vis_flores.py --input_path "$INPUT" --output_path "$OUTPUT"
done