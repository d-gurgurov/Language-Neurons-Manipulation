import argparse
import json
import os
from types import MethodType
from typing import List, Dict, Tuple

import torch
import torch.nn.functional as F
from vllm import LLM, SamplingParams

def load_flores_data(source_lang: str, target_lang: str, split: str = "dev", data_path: str = None) -> List[Dict]:
    """Load FLORES-200 dataset for translation from HuggingFace"""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install the datasets library: pip install datasets")
    
    try:
        # Load the dataset from HuggingFace
        dataset = load_dataset("facebook/flores", name=f"{source_lang}-{target_lang}")
        
        # Convert to list of dictionaries
        data = []
        for item in dataset[split]:
            data.append(dict(item))
        
        return data
        
    except Exception as e:
        if "Config" in str(e) or "not found" in str(e).lower():
            raise ValueError(f"Language pair '{source_lang}-{target_lang}' not found in FLORES-200 dataset. Please use valid FLORES-200 language codes.")
        else:
            raise e

def format_translation_prompt(item: Dict, source_lang: str, target_lang: str, lang_name_mapping: Dict) -> str:
    """Format a FLORES-200 item into a translation prompt"""
    source_text = item['sentence_' + source_lang]
    
    # Get human-readable language names
    source_name = lang_name_mapping.get(source_lang, source_lang)
    target_name = lang_name_mapping.get(target_lang, target_lang)
    
    prompt = f"Translate this {source_name} sentence into {target_name}: {source_text}. Translation:"
    
    return prompt

def get_language_mapping():
    """
    Mapping from 2-letter ISO 639-1 codes to FLORES-200 codes used in the original script.
    """
    return {
        "bo": "bod_Tibt",  # Tibetan
        "mt": "mlt_Latn",  # Maltese
        "it": "ita_Latn",  # Italian
        "es": "spa_Latn",  # Spanish
        "de": "deu_Latn",  # German
        "ja": "jpn_Jpan",  # Japanese
        "ar": "arb_Arab",  # Modern Standard Arabic
        "zh": "zho_Hans",  # Chinese (Simplified)
        "af": "afr_Latn",  # Afrikaans
        "nl": "nld_Latn",  # Dutch
        "fr": "fra_Latn",  # French
        "pt": "por_Latn",  # Portuguese
        "ru": "rus_Cyrl",  # Russian
        "ko": "kor_Hang",  # Korean
        "hi": "hin_Deva",  # Hindi
        "tr": "tur_Latn",  # Turkish
        "pl": "pol_Latn",  # Polish
        "sv": "swe_Latn",  # Swedish
        "da": "dan_Latn",  # Danish
        "no": "nob_Latn"   # Norwegian Bokmål
    }

def get_language_names():
    """Human-readable language names for FLORES-200 codes"""
    return {
        "bod_Tibt": "Tibetan",
        "mlt_Latn": "Maltese", 
        "ita_Latn": "Italian",
        "spa_Latn": "Spanish",
        "deu_Latn": "German",
        "jpn_Jpan": "Japanese",
        "arb_Arab": "Arabic",
        "zho_Hans": "Chinese",
        "afr_Latn": "Afrikaans",
        "nld_Latn": "Dutch",
        "fra_Latn": "French",
        "por_Latn": "Portuguese",
        "rus_Cyrl": "Russian",
        "kor_Hang": "Korean",
        "hin_Deva": "Hindi",
        "tur_Latn": "Turkish",
        "pol_Latn": "Polish",
        "swe_Latn": "Swedish",
        "dan_Latn": "Danish",
        "nob_Latn": "Norwegian"
    }

def compute_bleu_score(reference: str, candidate: str) -> float:
    """Compute BLEU score between reference and candidate translations"""
    try:
        from sacrebleu import sentence_bleu
        score = sentence_bleu(candidate, [reference])
        return score.score / 100.0  # Convert to 0-1 range
    except ImportError:
        print("Warning: sacrebleu not installed. Install with: pip install sacrebleu")
        print("Using simple word overlap score instead.")
        return compute_word_overlap(reference, candidate)

def compute_word_overlap(reference: str, candidate: str) -> float:
    """Simple word overlap score as fallback when BLEU is not available"""
    ref_words = set(reference.lower().split())
    cand_words = set(candidate.lower().split())
    
    if not ref_words:
        return 1.0 if not cand_words else 0.0
    
    intersection = ref_words.intersection(cand_words)
    return len(intersection) / len(ref_words)

def compute_average_activations():
    """Compute average activation values for each language"""
    n, over_zero = [], []
    
    # Use the old codes for loading activation data (assuming that's how they're stored)
    old_lang_names = ["bo", "mt", "it", "es", "de", "ja", "ar", "zh", "af", "nl", "fr", "pt", "ru", "ko", "hi", "tr", "pl", "sv", "da", "no"]
    
    for lang in old_lang_names:
        data = torch.load(f'data/activation.{lang}.train.llama-3')
        n.append(data['n'])
        over_zero.append(data['over_zero'])

    n = torch.tensor(n)
    over_zero = torch.stack(over_zero, dim=-1)
    
    # Compute average activation probabilities per language
    activation_probs = over_zero / n  # layer x inter x lang_num
    
    # Create mappings for both old and new codes
    lang_mapping = get_language_mapping()
    old_to_idx = {lang: idx for idx, lang in enumerate(old_lang_names)}
    flores_to_idx = {lang_mapping[old_lang]: idx for old_lang, idx in old_to_idx.items()}
    
    return activation_probs, old_to_idx, flores_to_idx, old_lang_names, lang_mapping

def evaluate_translation(model, source_lang: str, target_lang: str, sampling_params, max_samples: int = None) -> Dict:
    """Evaluate model on FLORES-200 translation task"""
    lang_name_mapping = get_language_names()
    
    # Load FLORES-200 data from HuggingFace
    try:
        data = load_flores_data(source_lang, target_lang, split="devtest")
    except ValueError as e:
        print(f"Warning: {str(e)}")
        return {"bleu_score": 0.0, "total": 0, "error": "language_pair_not_found"}
    except Exception as e:
        print(f"Warning: Error loading FLORES-200 data for {source_lang}-{target_lang}: {str(e)}")
        return {"bleu_score": 0.0, "total": 0, "error": "loading_failed"}
    
    if max_samples:
        data = data[:max_samples]
    
    # Format prompts
    prompts = [format_translation_prompt(item, source_lang, target_lang, lang_name_mapping) for item in data]
    
    # Generate responses
    outputs = model.generate(prompts, sampling_params)
    responses = [output.outputs[0].text.strip() for output in outputs]
    
    # Evaluate translation quality using BLEU scores
    bleu_scores = []
    results = []
    
    for i, (item, response) in enumerate(zip(data, responses)):
        reference = item['sentence_' + target_lang]
        candidate = response
        
        # Compute BLEU score
        bleu_score = compute_bleu_score(reference, candidate)
        bleu_scores.append(bleu_score)
        
        results.append({
            "id": i,
            "source_text": item['sentence_' + source_lang],
            "reference_translation": reference,
            "model_translation": candidate,
            "bleu_score": bleu_score
        })
    
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
    
    return {
        "bleu_score": avg_bleu,
        "total": len(data),
        "individual_scores": bleu_scores,
        "results": results
    }

def convert_lang_code(lang_code: str, lang_mapping: Dict[str, str]) -> str:
    """Convert language code to appropriate format"""
    if '_' in lang_code:
        return lang_code
    
    if lang_code in lang_mapping:
        return lang_mapping[lang_code]
    
    return lang_code

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="meta-llama/Meta-Llama-3-8B")
parser.add_argument("-a", "--activation_mask", type=str, default="activation_mask/llama-3-8b")
parser.add_argument("--source_lang", type=str, default="eng_Latn", 
                   help="Source language for translation (FLORES-200 code, e.g., eng_Latn for English)")
parser.add_argument("--target_lang", type=str, default="spa_Latn", 
                   help="Target language for translation (FLORES-200 code, e.g., spa_Latn for Spanish)")
parser.add_argument("--activate_lang", type=str, default="spa_Latn", 
                   help="Language neurons to activate (FLORES-200 code, e.g., spa_Latn for Spanish)")
parser.add_argument("--max_samples", default=None, help="Max samples to test (None for all)")
args = parser.parse_args()

# Get language mappings
_, old_to_idx, flores_to_idx, old_lang_names, lang_mapping = compute_average_activations()

# Convert language codes if needed
source_lang_flores = convert_lang_code(args.source_lang, lang_mapping)
target_lang_flores = convert_lang_code(args.target_lang, lang_mapping)
activate_lang_flores = convert_lang_code(args.activate_lang, lang_mapping)

print(f"Translating from {source_lang_flores} to {target_lang_flores}")
print(f"Activating {activate_lang_flores} neurons")

# Initialize model
model = LLM(model=args.model, tensor_parallel_size=torch.cuda.device_count(), enforce_eager=True)

eos_token_id = model.get_tokenizer().eos_token_id
print(f"EOS token ID: {eos_token_id}")

sampling_params = SamplingParams(
    temperature=0, 
    repetition_penalty=1.1, 
    max_tokens=100,  # Longer responses for translation
    stop_token_ids=[eos_token_id] if eos_token_id is not None else [],
    stop=["\n\n", "Translate", "Source:", "Target:"],
    skip_special_tokens=True
)

is_llama = bool(args.model.lower().find("llama") >= 0)

print("="*50)
print("BASELINE EVALUATION (no activation)")
print("="*50)

# Baseline evaluation
baseline_results = evaluate_translation(model, source_lang_flores, target_lang_flores, sampling_params, args.max_samples)
print(f"Baseline BLEU Score: {baseline_results['bleu_score']:.3f}")

print("="*50)
print(f"ENHANCED EVALUATION (activating {activate_lang_flores} neurons)")
print("="*50)

# Load language-specific neuron masks and compute average activations
activation_masks = torch.load(args.activation_mask)
avg_activations, old_to_idx, flores_to_idx, old_lang_names, lang_mapping = compute_average_activations()

# Get indices for target language
# Check if we need to map from FLORES-200 back to old code for activation lookup
activate_old_code = None
for old_code, flores_code in lang_mapping.items():
    if flores_code == activate_lang_flores:
        activate_old_code = old_code
        break

if activate_old_code is None:
    # Try direct lookup in case it's provided as old code
    if activate_lang_flores in old_to_idx:
        activate_old_code = activate_lang_flores
    else:
        raise ValueError(f"Language {activate_lang_flores} not found in activation data. Available FLORES-200 codes: {list(lang_mapping.values())}")

activate_idx = old_to_idx[activate_old_code]
activate_mask = activation_masks[activate_idx]

print(f"Activating {activate_lang_flores} ({activate_old_code}) neurons (index {activate_idx})")

def factory(layer_idx, activate_indices, boost_values):
    def llama_forward(self, x):
        gate_up, _ = self.gate_up_proj(x)  # b, l, 2i
        i = gate_up.size(-1)
        activation = F.silu(gate_up[:, :, : i // 2])
        
        # Boost activate language neurons
        if len(activate_indices) > 0:
            # Get current activations for these neurons
            current_activations = activation[:, :, activate_indices]
            # Add boost values (broadcast across batch and sequence dimensions)
            boost_tensor = boost_values.unsqueeze(0).unsqueeze(0).expand_as(current_activations)
            # Ensure dtype matches
            boost_tensor = boost_tensor.to(activation.dtype)
            activation[:, :, activate_indices] = current_activations +  boost_tensor #  
        
        x = activation * gate_up[:, :, i // 2 :]
        x, _ = self.down_proj(x)
        return x

    def bloom_forward(self, x: torch.Tensor):
        x, _ = self.dense_h_to_4h(x)
        x = self.gelu_impl(x)
        
        # Boost activate language neurons
        if len(activate_indices) > 0:
            current_activations = x[:, :, activate_indices]
            boost_tensor = boost_values.unsqueeze(0).unsqueeze(0).expand_as(current_activations)
            # Ensure dtype matches
            boost_tensor = boost_tensor.to(x.dtype)
            x[:, :, activate_indices] = current_activations + boost_tensor # 
        
        x, _ = self.dense_4h_to_h(x)
        return x

    if is_llama:
        return llama_forward
    else:
        return bloom_forward

# Apply the activation intervention to each layer
for layer_idx in range(len(activate_mask)):
    activate_indices_cpu = activate_mask[layer_idx]  # Keep on CPU for indexing
    activate_indices = activate_indices_cpu.to('cuda')  # Move to CUDA for model operations
    
    # Compute boost values for this layer
    if len(activate_indices_cpu) > 0:
        boost_values = avg_activations[layer_idx, activate_indices_cpu, activate_idx].to('cuda')
    else:
        boost_values = torch.tensor([]).to('cuda')
    
    if is_llama:
        obj = model.llm_engine.driver_worker.model_runner.model.model.layers[layer_idx].mlp
    else:
        obj = model.llm_engine.driver_worker.model_runner.model.transformer.h[layer_idx].mlp
    
    obj.forward = MethodType(factory(layer_idx, activate_indices, boost_values), obj)

# Enhanced evaluation
enhanced_results = evaluate_translation(model, source_lang_flores, target_lang_flores, sampling_params, args.max_samples)
print(f"Enhanced BLEU Score: {enhanced_results['bleu_score']:.3f}")

# Compute improvement
improvement = enhanced_results['bleu_score'] - baseline_results['bleu_score']
print(f"Improvement: {improvement:+.3f} ({improvement*100:+.1f}%)")

# Save results
results = {
    "model": args.model,
    "source_language": source_lang_flores,
    "target_language": target_lang_flores,
    "activated_language": activate_lang_flores,
    "activated_language_old_code": activate_old_code,
    "max_samples": args.max_samples,
    "baseline": baseline_results,
    "enhanced": enhanced_results,
    "improvement": improvement
}

output_dir = f"results/flores_translation_activation"
os.makedirs(output_dir, exist_ok=True)
output_file = f"{output_dir}/{source_lang_flores}_to_{target_lang_flores}_activate_{activate_lang_flores}.json"

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print(f"\nDetailed results saved to: {output_file}")

print("\n" + "="*50)
print("SUMMARY")
print("="*50)
print(f"Source Language: {source_lang_flores}")
print(f"Target Language: {target_lang_flores}")
print(f"Activated Language: {activate_lang_flores}")
print(f"Baseline BLEU Score: {baseline_results['bleu_score']:.3f}")
print(f"Enhanced BLEU Score: {enhanced_results['bleu_score']:.3f}")
print(f"Improvement: {improvement:+.3f} ({improvement*100:+.1f}%)")

# Print available language mappings for reference
print("\n" + "="*50)
print("AVAILABLE LANGUAGE CODES")
print("="*50)
print("Old Code -> FLORES-200 Code")
print("-" * 30)
for old_code, flores_code in sorted(lang_mapping.items()):
    print(f"{old_code:>8} -> {flores_code}")