from datasets import load_dataset
from transformers import AutoTokenizer
import torch
import os
import random
from itertools import islice

# Configuration 
languages = ["bo", "mt", "it", "es", "de", "ja", "ar", "zh", "af", "nl", "fr", "pt", "ru", "ko", "hi", "tr", "pl", "sv", "da", "no", "en"]
target_size_mb = 500  # Total size in MB for train + valid
val_ratio = 0.02
save_dir = "data_nemo" 
# meta-llama/Llama-3.1-8B 
# meta-llama/Meta-Llama-3-8B 
# meta-llama/Llama-3.1-70B
# CohereLabs/aya-expanse-8b
# CohereLabs/aya-expanse-8b
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-Nemo-Base-2407", use_fast=True)

# Convert MB to bytes
target_total_bytes = target_size_mb * 1024 * 1024

for lang in languages:
    print(f"Processing language: {lang}")
    ds_streamed = load_dataset("uonlp/CulturaX", lang, split="train", streaming=True, use_auth_token=True)

    train_ids, val_ids = [], []
    total_bytes = 0
    split_point = int((1 - val_ratio) * target_total_bytes)

    for item in ds_streamed:
        text = item["text"]
        ids = tokenizer.encode(text, add_special_tokens=False)
        byte_tensor = torch.LongTensor(ids).numpy().tobytes()
        size_bytes = len(byte_tensor)

        if total_bytes + size_bytes > target_total_bytes:
            break

        if total_bytes < split_point:
            train_ids.extend(ids)
        else:
            val_ids.extend(ids)

        total_bytes += size_bytes

    def save_tensor(tensor_ids, split_name):
        tensor = torch.LongTensor(tensor_ids)
        save_path = os.path.join(save_dir, f"id.{lang}.{split_name}.nemo")
        torch.save(tensor, save_path)
        print(f"Saved {split_name} ({len(tensor)} tokens, ~{tensor.numpy().nbytes / 1024 / 1024:.2f} MB) to {save_path}")

    save_tensor(train_ids, "train")
    save_tensor(val_ids, "valid")
