#!/bin/bash

pip install vllm==0.2.7 sacrebleu datasets

HF_TOKEN=*
huggingface-cli login --token $HF_TOKEN

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

MODEL="meta-llama/Meta-Llama-3-8B"
ACTIVATION_DIR="activation_mask/llama-3"

langs=(
  "afr_Latn"  # af
  "mlt_Latn"  # mt
  "bod_Tibt"  # bo
  "ita_Latn"  # it
  "spa_Latn"  # es
  "deu_Latn"  # de
  "jpn_Jpan"  # ja
  "arb_Arab"  # ar
  "zho_Hans"  # zh
  "nld_Latn"  # nl
  "fra_Latn"  # fr
  "por_Latn"  # pt
  "rus_Cyrl"  # ru
  "kor_Hang"  # ko
  "hin_Deva"  # hi
  "tur_Latn"  # tr
  "pol_Latn"  # pl
  "swe_Latn"  # sv
  "dan_Latn"  # da
  "nob_Latn"  # no
)

SOURCE_LANG="eng_Latn"

for target_lang in "${langs[@]}"; do
  for activate_lang in "${langs[@]}"; do
    echo "Running translation: $SOURCE_LANG -> $target_lang, activating $activate_lang neurons"
    python src/flores.py -m "$MODEL" -a "$ACTIVATION_DIR" \
      --source_lang "$SOURCE_LANG" --target_lang "$target_lang" --activate_lang "$activate_lang"
    
    sleep 5
  done
done
