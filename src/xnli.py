import argparse
import json
import os
import time
from types import MethodType
from typing import List, Dict, Tuple
import re

import torch
import torch.nn.functional as F
from vllm import LLM, SamplingParams
from datasets import load_dataset

def load_xnli_data(language: str, split: str = "test") -> List[Dict]:
    """Load XNLI dataset for natural language inference from HuggingFace"""
    try:
        # Load the dataset from HuggingFace
        dataset = load_dataset("facebook/xnli", language)
        data = []
        for item in dataset[split]:
            data.append(dict(item))
        return data
    except Exception as e:
        print(f"Error loading XNLI data for language {language}: {e}")
        return []

def format_xnli_prompt(item: Dict, language: str, lang_name_mapping: Dict) -> str:
    """Format an XNLI item into a natural language inference prompt"""
    premise = item['premise']
    hypothesis = item['hypothesis']
    
    language_name = lang_name_mapping.get(language, language)
    
    prompt = f"""Given the following {language_name} premise and hypothesis, determine the relationship between them.

Premise: {premise}

Hypothesis: {hypothesis}

Options:
1. Entailment
2. Neutral
3. Contradiction

Answer:"""
    
    return prompt

def get_language_mapping():
    """
    Mapping from activation language codes to XNLI language codes.
    Only including languages that have activation data and are in XNLI.
    """
    # XNLI languages that overlap with our activation data
    return {
        "ar": "ar",      # Arabic
        "bg": "bg",      # Bulgarian (not in activation data)
        "de": "de",      # German
        "el": "el",      # Greek (not in activation data)
        "en": "en",      # English
        "es": "es",      # Spanish
        "fr": "fr",      # French
        "hi": "hi",      # Hindi
        "ru": "ru",      # Russian
        "sw": "sw",      # Swahili (not in activation data)
        "th": "th",      # Thai (not in activation data)
        "tr": "tr",      # Turkish
        "ur": "ur",      # Urdu (not in activation data)
        "vi": "vi",      # Vietnamese (not in activation data)
        "zh": "zh"       # Chinese
    }

def get_language_names():
    """Human-readable language names for XNLI codes"""
    return {
        "ar": "Arabic",
        "bg": "Bulgarian",
        "de": "German", 
        "el": "Greek",
        "en": "English",
        "es": "Spanish",
        "fr": "French",
        "hi": "Hindi",
        "ru": "Russian",
        "sw": "Swahili",
        "th": "Thai",
        "tr": "Turkish",
        "ur": "Urdu",
        "vi": "Vietnamese",
        "zh": "Chinese"
    }

def get_all_xnli_languages():
    """Get all XNLI languages"""
    return ["ar", "bg", "de", "el", "en", "es", "fr", "hi", "ru", "sw", "th", "tr", "ur", "vi", "zh"]

def get_activation_languages():
    """Get languages for which we have activation data"""
    return ["bo", "mt", "it", "es", "de", "ja", "ar", "zh", "af", "nl", "fr", "pt", "ru", "ko", "hi", "tr", "pl", "sv", "da", "no", "en"]

def extract_entailment_label(response: str) -> str:
    """Extract the entailment label from the model response"""
    response = response.strip()
    
    # First check for numeric responses (1, 2, 3)
    if response.isdigit():
        num = int(response)
        if num == 1:
            return "entailment"
        elif num == 2:
            return "neutral"
        elif num == 3:
            return "contradiction"
    
    # Check for numbers at the start of response
    if response and response[0].isdigit():
        num = int(response[0])
        if num == 1:
            return "entailment"
        elif num == 2:
            return "neutral"
        elif num == 3:
            return "contradiction"
    
    response_lower = response.lower()
    
    # Look for explicit labels
    if "entailment" in response_lower:
        return "entailment"
    elif "contradiction" in response_lower:
        return "contradiction"
    elif "neutral" in response_lower:
        return "neutral"
    
    # Look for word patterns
    patterns = [
        r'\bentailment\b',
        r'\bcontradiction\b', 
        r'\bneutral\b',
        r'\bentails\b',
        r'\bcontradicts\b'
    ]
    
    for i, pattern in enumerate(patterns[:3]):  # Only check the main three
        if re.search(pattern, response_lower):
            return ["entailment", "contradiction", "neutral"][i]
    
    # Look for yes/no patterns that might indicate entailment/contradiction
    if re.search(r'\byes\b|\btrue\b|\bcorrect\b', response_lower):
        return "entailment"
    elif re.search(r'\bno\b|\bfalse\b|\bincorrect\b', response_lower):
        return "contradiction"
    
    return None
    
def compute_accuracy(references: List[int], predictions: List[str]) -> float:
    """Compute accuracy for NLI classification"""
    correct = 0
    total = 0
    
    label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}
    
    for ref, pred in zip(references, predictions):
        total += 1
        ref_label = label_map.get(ref)
        if pred == ref_label:
            correct += 1
    
    return correct / total if total > 0 else 0.0

def compute_detailed_metrics(references: List[int], predictions: List[str]) -> Dict:
    """Compute detailed classification metrics including per-class accuracy"""
    label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}
    
    # Convert references to labels
    ref_labels = [label_map.get(ref) for ref in references]
    
    # Count correct predictions per class
    class_correct = {"entailment": 0, "neutral": 0, "contradiction": 0}
    class_total = {"entailment": 0, "neutral": 0, "contradiction": 0}
    
    total_correct = 0
    total_predictions = 0
    
    for ref_label, pred in zip(ref_labels, predictions):
        if ref_label is not None:
            class_total[ref_label] += 1
            total_predictions += 1
            
            if pred == ref_label:
                class_correct[ref_label] += 1
                total_correct += 1
    
    # Calculate per-class accuracies
    class_accuracies = {}
    for label in ["entailment", "neutral", "contradiction"]:
        if class_total[label] > 0:
            class_accuracies[label] = class_correct[label] / class_total[label]
        else:
            class_accuracies[label] = 0.0
    
    overall_accuracy = total_correct / total_predictions if total_predictions > 0 else 0.0
    
    return {
        "accuracy": overall_accuracy,
        "entailment_accuracy": class_accuracies["entailment"],
        "neutral_accuracy": class_accuracies["neutral"], 
        "contradiction_accuracy": class_accuracies["contradiction"],
        "class_counts": class_total,
        "class_correct": class_correct
    }

def compute_average_activations(activations_path):
    """Compute average activation values for each language"""
    n, over_zero, average_activations = [], [], []
    
    # Use the old codes for loading activation data (assuming that's how they're stored)
    old_lang_names = ["bo", "mt", "it", "es", "de", "ja", "ar", "zh", "af", "nl", "fr", "pt", "ru", "ko", "hi", "tr", "pl", "sv", "da", "no", "en"]
    
    for lang in old_lang_names:
        data = torch.load(f'data_{activations_path[0]}/activation.{lang}.train.{activations_path[1]}')
        n.append(data['n'])
        over_zero.append(data['over_zero'])
        average_activations.append(data['average_activations'])

    n = torch.tensor(n)
    over_zero = torch.stack(over_zero, dim=-1)
    average_activations = torch.stack(average_activations, dim=-1)  # layer x inter x lang_num
    
    # Compute average activation probabilities per language
    activation_probs = over_zero / n
    
    # Create mappings
    old_to_idx = {lang: idx for idx, lang in enumerate(old_lang_names)}
    
    return average_activations, old_to_idx, old_lang_names

def evaluate_xnli(model, language: str, sampling_params, max_samples: int = None) -> Dict:
    """Evaluate model on XNLI natural language inference task"""
    lang_name_mapping = get_language_names()

    data = load_xnli_data(language)[:1000]
    
    if max_samples is not None:
        data = data[:max_samples]
    
    # Format prompts
    prompts = [format_xnli_prompt(item, language, lang_name_mapping) for item in data]
    
    # Generate responses
    outputs = model.generate(prompts, sampling_params)
    responses = [output.outputs[0].text.strip() for output in outputs]
    
    # Extract predictions and references
    predictions = [extract_entailment_label(response) for response in responses]
    print(responses[:5])
    references = [item['label'] for item in data]
    print(references[:5])
    
    # Compute metrics
    metrics = compute_detailed_metrics(references, predictions)
    
    results = []
    label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}
    
    for i, (item, response, prediction) in enumerate(zip(data, responses, predictions)):
        ref_label = label_map.get(item['label'])
        correct = prediction == ref_label
        
        results.append({
            "id": i,
            "premise": item['premise'],
            "hypothesis": item['hypothesis'],
            "reference_label": ref_label,
            "model_response": response,
            "extracted_label": prediction,
            "correct": correct
        })
    
    return {
        "accuracy": metrics['accuracy'],
        "entailment_accuracy": metrics['entailment_accuracy'],
        "neutral_accuracy": metrics['neutral_accuracy'],
        "contradiction_accuracy": metrics['contradiction_accuracy'],
        "total": len(data),
        "correct_count": metrics['class_correct'],
        "class_counts": metrics['class_counts'],
        "results": results
    }

def convert_lang_code(lang_code: str, lang_mapping: Dict[str, str]) -> str:
    """Convert language code to appropriate format"""
    if lang_code in lang_mapping:
        return lang_mapping[lang_code]
    return lang_code

def factory(layer_idx, deactivate_indices=None, activate_indices=None, boost_values=None):
    def llama_forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        i = gate_up.size(-1)

        if gate_up.dim() == 3:
            silu_output = F.silu(gate_up[:, :, : i // 2])
            
            if deactivate_indices is not None and len(deactivate_indices) > 0:
                silu_output.index_fill_(2, deactivate_indices, 0)

            if activate_indices is not None and len(activate_indices) > 0 and boost_values is not None:
                boost_tensor = boost_values.to(silu_output.dtype).unsqueeze(0).unsqueeze(0)
                silu_output[:, :, activate_indices] += boost_tensor

            x = silu_output * gate_up[:, :, i // 2:]

        elif gate_up.dim() == 2:
            silu_output = F.silu(gate_up[:, : i // 2])

            if deactivate_indices is not None and len(deactivate_indices) > 0:
                silu_output.index_fill_(1, deactivate_indices, 0)

            if activate_indices is not None and len(activate_indices) > 0 and boost_values is not None:
                boost_tensor = boost_values.to(silu_output.dtype).unsqueeze(0)
                silu_output[:, activate_indices] += boost_tensor

            x = silu_output * gate_up[:, i // 2:]

        else:
            raise ValueError(f"Unexpected gate_up shape: {gate_up.shape}")

        x, _ = self.down_proj(x)
        return x

    return llama_forward

def reset_model_layers(model):
    """Reset all model layers to their original forward methods"""
    for layer_idx in range(len(model.llm_engine.model_executor.driver_worker.model_runner.model.model.layers)):
        layer = model.llm_engine.model_executor.driver_worker.model_runner.model.model.layers[layer_idx]
        if hasattr(layer.mlp, '_original_forward'):
            layer.mlp.forward = layer.mlp._original_forward

def apply_activation_intervention(model, activate_mask, avg_activations, activate_idx):
    """Apply activation intervention to model layers"""
    # Store original forward methods if not already stored
    for layer_idx in range(len(model.llm_engine.model_executor.driver_worker.model_runner.model.model.layers)):
        mlp = model.llm_engine.model_executor.driver_worker.model_runner.model.model.layers[layer_idx].mlp
        if not hasattr(mlp, '_original_forward'):
            mlp._original_forward = mlp.forward
    
    # Apply the activation intervention to each layer
    for layer_idx in range(len(activate_mask)):
        activate_indices_cpu = activate_mask[layer_idx]  # Keep on CPU for indexing
        activate_indices = activate_indices_cpu.to('cuda')  # Move to CUDA for model operations
        
        # Compute boost values for this layer
        if len(activate_indices_cpu) > 0:
            boost_values = avg_activations[layer_idx, activate_indices_cpu, activate_idx].to('cuda')
        else:
            boost_values = torch.tensor([]).to('cuda')
        
        obj = model.llm_engine.model_executor.driver_worker.model_runner.model.model.layers[layer_idx].mlp
        obj.forward = MethodType(factory(layer_idx, None, activate_indices, boost_values), obj)

def reset_activation_intervention(model):
    """Reset model to baseline (no activation intervention)"""
    for layer_idx in range(len(model.llm_engine.model_executor.driver_worker.model_runner.model.model.layers)):
        mlp = model.llm_engine.model_executor.driver_worker.model_runner.model.model.layers[layer_idx].mlp
        if hasattr(mlp, '_original_forward'):
            mlp.forward = mlp._original_forward

def run_full_evaluation_suite(args):
    """Run the complete evaluation suite with all combinations"""
    
    # Get all languages and ratios
    all_xnli_languages = get_all_xnli_languages()
    all_activation_languages = get_activation_languages()
    ratios = list(range(1, 6)) if args.ratios is None else args.ratios
    
    # Initialize model once
    print("Initializing model...")
    model = LLM(model=args.model, tensor_parallel_size=torch.cuda.device_count(), enforce_eager=True)
    
    eos_token_id = model.get_tokenizer().eos_token_id
    print(f"EOS token ID: {eos_token_id}")
    
    sampling_params = SamplingParams(
        temperature=0, 
        repetition_penalty=1.1, 
        max_tokens=32,  # Short responses for classification
        stop_token_ids=[eos_token_id] if eos_token_id is not None else [],
        stop=["\n\n", "Premise:", "Hypothesis:", "Answer:", "The relationship"],
        skip_special_tokens=True
    )
    
    # Track overall progress
    total_combinations = len(ratios) * len(all_xnli_languages) * len(all_activation_languages)
    current_combination = 0
    
    print(f"Starting evaluation of {total_combinations} combinations...")
    print(f"Ratios: {ratios}")
    print(f"XNLI languages: {len(all_xnli_languages)} - {all_xnli_languages}")
    print(f"Activation languages: {len(all_activation_languages)} - {all_activation_languages}")
    
    start_time = time.time()
    
    # Loop through all ratios
    for ratio in ratios:
        print(f"\n{'='*60}")
        print(f"PROCESSING RATIO {ratio}")
        print(f"{'='*60}")
        
        # Load activation masks and compute average activations for this ratio
        activation_dir = f"activation_mask/{args.activation_mask}-{ratio}"
        try:
            activation_masks = torch.load(activation_dir)
            activations_path = args.activations_path.split(" ")
            avg_activations, old_to_idx, old_lang_names = compute_average_activations(activations_path)
        except Exception as e:
            print(f"Error loading activation data for ratio {ratio}: {e}")
            continue
        
        # Create output directory for this ratio
        model_name = args.model.split('/')[-1]
        output_dir = f"{args.output_path}/{model_name}_{ratio}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Loop through all XNLI target languages
        for target_lang in all_xnli_languages:
            print(f"\n{'-'*40}")
            print(f"Target language: {target_lang}")
            print(f"{'-'*40}")
            
            # Run baseline evaluation once per target language
            print("Running baseline evaluation...")
            reset_activation_intervention(model)
            
            try:
                baseline_results = evaluate_xnli(model, target_lang, sampling_params, args.max_samples)
                baseline_acc = baseline_results['accuracy']
                baseline_ent = baseline_results['entailment_accuracy']
                baseline_neu = baseline_results['neutral_accuracy']
                baseline_con = baseline_results['contradiction_accuracy']
                print(f"Baseline - Overall: {baseline_acc:.3f}, Ent: {baseline_ent:.3f}, Neu: {baseline_neu:.3f}, Con: {baseline_con:.3f}")
            except Exception as e:
                print(f"Error in baseline evaluation for {target_lang}: {e}")
                continue
            
            # Loop through all activation languages
            for activate_lang in all_activation_languages:
                current_combination += 1
                elapsed_time = time.time() - start_time
                avg_time_per_combo = elapsed_time / current_combination if current_combination > 0 else 0
                eta = avg_time_per_combo * (total_combinations - current_combination)
                
                print(f"\nProgress: {current_combination}/{total_combinations} "
                      f"({100*current_combination/total_combinations:.1f}%) "
                      f"ETA: {eta/60:.1f} min")
                print(f"Activating {activate_lang} neurons for {target_lang} NLI")
                
                # Check if results already exist
                output_file = f"{output_dir}/{target_lang}_activate_{activate_lang}.json"
                if os.path.exists(output_file) and not args.overwrite:
                    print(f"Results already exist, skipping: {output_file}")
                    continue
                
                try:
                    # Get activation language index
                    if activate_lang in old_to_idx:
                        activate_idx = old_to_idx[activate_lang]
                    else:
                        print(f"Warning: Language {activate_lang} not found in activation data, skipping")
                        continue
                    
                    activate_mask = activation_masks[activate_idx]
                    
                    # Apply activation intervention
                    apply_activation_intervention(model, activate_mask, avg_activations, activate_idx)
                    
                    # Run enhanced evaluation
                    enhanced_results = evaluate_xnli(model, target_lang, sampling_params, args.max_samples)
                    enhanced_acc = enhanced_results['accuracy']
                    enhanced_ent = enhanced_results['entailment_accuracy']
                    enhanced_neu = enhanced_results['neutral_accuracy']
                    enhanced_con = enhanced_results['contradiction_accuracy']
                    
                    # Compute improvements
                    acc_improvement = enhanced_acc - baseline_acc
                    ent_improvement = enhanced_ent - baseline_ent
                    neu_improvement = enhanced_neu - baseline_neu
                    con_improvement = enhanced_con - baseline_con
                    
                    print(f"Enhanced - Overall: {enhanced_acc:.3f} (Δ{acc_improvement:+.3f})")
                    print(f"  Entailment: {enhanced_ent:.3f} (Δ{ent_improvement:+.3f})")
                    print(f"  Neutral: {enhanced_neu:.3f} (Δ{neu_improvement:+.3f})")
                    print(f"  Contradiction: {enhanced_con:.3f} (Δ{con_improvement:+.3f})")
                    
                    # Save results
                    results = {
                        "model": args.model,
                        "ratio": ratio,
                        "target_language": target_lang,
                        "activated_language": activate_lang,
                        "max_samples": args.max_samples,
                        "baseline": baseline_results,
                        "enhanced": enhanced_results,
                        "improvements": {
                            "accuracy": acc_improvement,
                            "entailment": ent_improvement,
                            "neutral": neu_improvement,
                            "contradiction": con_improvement
                        }
                    }
                    
                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump(results, f, indent=4, ensure_ascii=False)
                    
                    if args.verbose:
                        print(f"Results saved: {output_file}")
                        
                except Exception as e:
                    print(f"Error processing {activate_lang} activation: {e}")
                    continue
                
                # Small delay to prevent potential issues
                if args.delay > 0:
                    time.sleep(args.delay)
    
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Processed {current_combination} combinations")
    print(f"Average time per combination: {total_time/current_combination:.1f} seconds")

def main():
    parser = argparse.ArgumentParser(description="XNLI Natural Language Inference Evaluation with Neuron Activation")
    
    # Model and data arguments
    parser.add_argument("-m", "--model", type=str, default="meta-llama/Llama-3.1-8B",
                       help="Model name or path")
    parser.add_argument("--activations_path", type=str, default="llama_3-1 llama-3.1",
                       help="Activations path (space-separated)")
    
    # Evaluation control arguments
    parser.add_argument("--ratios", type=int, nargs='+', default=None,
                       help="Ratios to evaluate (default: 1-5)")
    parser.add_argument("--target_langs", type=str, nargs='+', default=None,
                       help="Specific target languages to evaluate (default: all XNLI languages)")
    parser.add_argument("--activate_langs", type=str, nargs='+', default=None,
                       help="Specific activation languages to evaluate (default: all activation languages)")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Max samples to test (None for all)")
    
    # Output and control arguments
    parser.add_argument("--output_path", type=str, default="xnli",
                       help="Output directory path")
    parser.add_argument("--overwrite", action="store_true",
                       help="Overwrite existing results")
    parser.add_argument("--delay", type=float, default=0,
                       help="Delay between evaluations (seconds)")
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose output")
    
    parser.add_argument("-a", "--activation_mask", type=str, default="activation_mask/llama-3-8b",
                       help="Activation mask path")
    parser.add_argument("--target_lang", type=str, default="es",
                       help="Target language (for single mode)")
    parser.add_argument("--activate_lang", type=str, default="es",
                       help="Language neurons to activate (for single mode)")
    
    global args
    args = parser.parse_args()
    
    # Set environment variable
    os.environ["VLLM_USE_V1"] = "0"
    
    if args.ratios is None:
        args.ratios = list(range(1, 6))
    
    run_full_evaluation_suite(args)

if __name__ == "__main__":
    main()