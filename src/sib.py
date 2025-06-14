import argparse
import json
import os
from types import MethodType
from typing import List, Dict

import torch
import torch.nn.functional as F
from vllm import LLM, SamplingParams

def load_sib200_data(lang: str, split: str = "test", data_path: str = None) -> List[Dict]:
    """Load SIB-200 dataset for a specific language from HuggingFace"""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install the datasets library: pip install datasets")
    
    try:
        # Load the dataset from HuggingFace
        dataset = load_dataset("Davlan/sib200", lang)
        
        # Convert to list of dictionaries
        data = []
        for item in dataset[split]:
            data.append(dict(item))
        
        return data
        
    except Exception as e:
        if "Config" in str(e) or "not found" in str(e).lower():
            raise ValueError(f"Language '{lang}' not found in SIB-200 dataset. Please use a valid FLORES-200 language code.")
        else:
            raise e

def format_sib200_prompt(item: Dict, include_categories: bool = False) -> str:
    """Format a SIB-200 item into a prompt for topic classification"""
    text = item['text']
    
    # Define the 7 topic categories
    categories = [
        "science/technology",  # 0
        "travel",             # 1  
        "politics",           # 2
        "sports",             # 3
        "health",             # 4
        "entertainment",      # 5
        "geography"           # 6
    ]
    
    if include_categories:
        categories_text = ", ".join(categories)
        prompt = f"""Text: {text}

Question: What is the main topic of this text?

Categories: {categories_text}

Answer: """
    else:
        prompt = f"""Text: {text}

Question: What is the topic of this text? Choose from: science/technology, travel, politics, sports, health, entertainment, or geography.

Answer: """
    
    return prompt

def get_language_mapping():
    """
    Mapping from 2-letter ISO 639-1 codes to FLORES-200 codes used in SIB-200.
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

def get_category_mapping():
    """Get mapping from category names to label indices"""
    return {
        "science/technology": 0,
        "science": 0,
        "technology": 0,
        "travel": 1,
        "politics": 2,
        "sports": 3,
        "health": 4,
        "entertainment": 5,
        "geography": 6
    }

def get_category_names_list():
    """Get list of category names in order"""
    return ["science/technology", "travel", "politics", "sports", "health", "entertainment", "geography"]

def get_string_to_int_mapping():
    """Get mapping from string labels to integer labels"""
    return {
        "science/technology": 0,
        "travel": 1,
        "politics": 2,
        "sports": 3,
        "health": 4,
        "entertainment": 5,
        "geography": 6
    }

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

def evaluate_sib200(model, lang: str, sampling_params, max_samples: int = None, split: str = "test") -> Dict:
    """Evaluate model on SIB-200 benchmark"""
    # Load SIB-200 data from HuggingFace
    try:
        data = load_sib200_data(lang, split)
    except ValueError as e:
        print(f"Warning: {str(e)}")
        return {"accuracy": 0.0, "total": 0, "correct": 0, "error": "language_not_found"}
    except Exception as e:
        print(f"Warning: Error loading SIB-200 data for {lang}: {str(e)}")
        return {"accuracy": 0.0, "total": 0, "correct": 0, "error": "loading_failed"}
    
    if max_samples:
        data = data[:max_samples]
    
    # Debug: Print first few items to understand data structure
    print(f"Dataset contains {len(data)} samples")
    if data:
        print("Sample data point:")
        print(data[0])
        print(f"Label type: {type(data[0]['category'])}, value: {data[0]['category']}")
    
    # Format prompts
    prompts = [format_sib200_prompt(item) for item in data]
    
    # Generate responses
    outputs = model.generate(prompts, sampling_params)
    responses = [output.outputs[0].text.strip() for output in outputs]
    
    # Evaluate accuracy
    correct = 0
    results = []
    category_mapping = get_category_mapping()
    category_names = get_category_names_list()
    string_to_int = get_string_to_int_mapping()
    
    for i, (item, response) in enumerate(zip(data, responses)):
        # Handle both string and integer labels
        raw_label = item['category']
        if isinstance(raw_label, str):
            correct_label = string_to_int.get(raw_label, -1)  # -1 if unknown category
            correct_category = raw_label
        else:
            correct_label = int(raw_label)  # 0-6
            correct_category = category_names[correct_label] if 0 <= correct_label < len(category_names) else "unknown"
        
        # Extract predicted category from response
        predicted_label = None
        response_clean = response.strip().lower()
        
        # Try to match category names in the response
        for category, label in category_mapping.items():
            if category in response_clean:
                predicted_label = label
                break
        
        # If no direct match, try to extract based on common patterns
        if predicted_label is None:
            if "science" in response_clean or "technology" in response_clean or "tech" in response_clean:
                predicted_label = 0
            elif "travel" in response_clean or "tourism" in response_clean:
                predicted_label = 1
            elif "politic" in response_clean:
                predicted_label = 2
            elif "sport" in response_clean:
                predicted_label = 3
            elif "health" in response_clean or "medical" in response_clean:
                predicted_label = 4
            elif "entertainment" in response_clean or "culture" in response_clean:
                predicted_label = 5
            elif "geography" in response_clean or "geographic" in response_clean:
                predicted_label = 6
        
        # Check if correct (skip if correct_label is unknown)
        is_correct = predicted_label == correct_label and correct_label != -1
        if is_correct:
            correct += 1
        
        results.append({
            "id": i,
            "correct_label": correct_label,
            "correct_category": correct_category,
            "predicted_label": predicted_label,
            "predicted_category": category_names[predicted_label] if predicted_label is not None and 0 <= predicted_label < len(category_names) else None,
            "response": response,
            "is_correct": is_correct,
            "text": item['text']
        })
    
    accuracy = correct / len(data) if data else 0.0
    
    return {
        "accuracy": accuracy,
        "total": len(data),
        "correct": correct,
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
parser.add_argument("--test_lang", type=str, default="afr_Latn", 
                   help="Language to test on (FLORES-200 code, e.g., spa_Latn for Spanish)")
parser.add_argument("--activate_lang", type=str, default="deu_Latn", 
                   help="Language neurons to activate (FLORES-200 code, e.g., spa_Latn for Spanish)")
parser.add_argument("--max_samples", type=int, default=None, help="Max samples to test (None for all)")
parser.add_argument("--split", type=str, default="test", choices=["train", "validation", "test"],
                   help="Dataset split to evaluate on")
args = parser.parse_args()

# Get language mappings
_, old_to_idx, flores_to_idx, old_lang_names, lang_mapping = compute_average_activations()

# Convert language codes if needed
test_lang_flores = convert_lang_code(args.test_lang, lang_mapping)
activate_lang_flores = convert_lang_code(args.activate_lang, lang_mapping)

print(f"Testing on {test_lang_flores}, activating {activate_lang_flores} neurons")

# Initialize model
model = LLM(model=args.model, tensor_parallel_size=torch.cuda.device_count(), enforce_eager=True)

eos_token_id = model.get_tokenizer().eos_token_id
print(f"EOS token ID: {eos_token_id}")

sampling_params = SamplingParams(
    temperature=0, 
    repetition_penalty=1.1, 
    max_tokens=20,  # Slightly longer for topic classification
    stop_token_ids=[eos_token_id] if eos_token_id is not None else [],
    stop=["\n", "Question:", "Text:", "Categories:"],
    skip_special_tokens=True
)

is_llama = bool(args.model.lower().find("llama") >= 0)

print("="*50)
print("BASELINE EVALUATION (no activation)")
print("="*50)

# Baseline evaluation
baseline_results = evaluate_sib200(model, test_lang_flores, sampling_params, args.max_samples, args.split)
print(f"Baseline Accuracy: {baseline_results['accuracy']:.3f} ({baseline_results['correct']}/{baseline_results['total']})")

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
            activation[:, :, activate_indices] = current_activations + boost_tensor
        
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
            x[:, :, activate_indices] = current_activations + boost_tensor
        
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
enhanced_results = evaluate_sib200(model, test_lang_flores, sampling_params, args.max_samples, args.split)
print(f"Enhanced Accuracy: {enhanced_results['accuracy']:.3f} ({enhanced_results['correct']}/{enhanced_results['total']})")

# Compute improvement
improvement = enhanced_results['accuracy'] - baseline_results['accuracy']
print(f"Improvement: {improvement:+.3f} ({improvement*100:+.1f}%)")

# Save results
results = {
    "model": args.model,
    "dataset": "SIB-200",
    "test_language": test_lang_flores,
    "activated_language": activate_lang_flores,
    "test_language_old_code": None,  # Will be filled if applicable
    "activated_language_old_code": activate_old_code,
    "max_samples": args.max_samples,
    "split": args.split,
    "baseline": baseline_results,
    "enhanced": enhanced_results,
    "improvement": improvement,
    "categories": ["science/technology", "travel", "politics", "sports", "health", "entertainment", "geography"]
}

# Add old code for test language if applicable
for old_code, flores_code in lang_mapping.items():
    if flores_code == test_lang_flores:
        results["test_language_old_code"] = old_code
        break

output_dir = f"results/sib200_activation"
os.makedirs(output_dir, exist_ok=True)
output_file = f"{output_dir}/{test_lang_flores}_activate_{activate_lang_flores}_{args.split}.json"

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print(f"\nDetailed results saved to: {output_file}")

print("\n" + "="*50)
print("SUMMARY")
print("="*50)
print(f"Dataset: SIB-200")
print(f"Split: {args.split}")
print(f"Test Language: {test_lang_flores}")
print(f"Activated Language: {activate_lang_flores}")
print(f"Baseline Accuracy: {baseline_results['accuracy']:.3f}")
print(f"Enhanced Accuracy: {enhanced_results['accuracy']:.3f}")
print(f"Improvement: {improvement:+.3f} ({improvement*100:+.1f}%)")

# Print available language mappings for reference
print("\n" + "="*50)
print("AVAILABLE LANGUAGE CODES")
print("="*50)
print("Old Code -> FLORES-200 Code")
print("-" * 30)
for old_code, flores_code in sorted(lang_mapping.items()):
    print(f"{old_code:>8} -> {flores_code}")

# Print category breakdown for detailed analysis
if baseline_results.get('results') and enhanced_results.get('results'):
    print("\n" + "="*50)
    print("CATEGORY-WISE PERFORMANCE")
    print("="*50)
    
    categories = ["science/technology", "travel", "politics", "sports", "health", "entertainment", "geography"]
    
    # Count correct predictions per category for baseline and enhanced
    baseline_category_correct = [0] * 7
    baseline_category_total = [0] * 7
    enhanced_category_correct = [0] * 7
    enhanced_category_total = [0] * 7
    
    for result in baseline_results['results']:
        label = result['correct_label']
        baseline_category_total[label] += 1
        if result['is_correct']:
            baseline_category_correct[label] += 1
    
    for result in enhanced_results['results']:
        label = result['correct_label']
        enhanced_category_total[label] += 1
        if result['is_correct']:
            enhanced_category_correct[label] += 1
    
    print(f"{'Category':<20} {'Baseline':<12} {'Enhanced':<12} {'Improvement':<12}")
    print("-" * 58)
    
    for i, category in enumerate(categories):
        if baseline_category_total[i] > 0:
            baseline_acc = baseline_category_correct[i] / baseline_category_total[i]
            enhanced_acc = enhanced_category_correct[i] / enhanced_category_total[i]
            improvement_cat = enhanced_acc - baseline_acc
            print(f"{category:<20} {baseline_acc:.3f}       {enhanced_acc:.3f}       {improvement_cat:+.3f}")
        else:
            print(f"{category:<20} {'N/A':<12} {'N/A':<12} {'N/A':<12}")