import argparse
import json
import os
import time
from types import MethodType
from typing import List, Dict, Tuple
import re
import string

import torch
import torch.nn.functional as F
from vllm import LLM, SamplingParams
from datasets import load_dataset

def load_xquad_data(language: str, split: str = "validation") -> List[Dict]:
    """Load XQUAD dataset for extractive QA from HuggingFace"""
    try:
        # Load the dataset from HuggingFace with the specific language config
        dataset = load_dataset("google/xquad", f"xquad.{language}")
        data = []
        for item in dataset[split]:
            data.append(dict(item))
        return data
    except Exception as e:
        print(f"Error loading XQUAD data for language {language}: {e}")
        return []

def format_xquad_prompt(item: Dict, language: str, lang_name_mapping: Dict) -> str:
    """Format an XQUAD item into an extractive QA prompt"""
    context = item['context']
    question = item['question']
    
    language_name = lang_name_mapping.get(language, language)
    
    prompt = f"""Answer the question based on the {language_name} context provided. Extract the exact answer from the context.

Context: {context}

Question: {question}

Answer:"""
    
    return prompt

def get_language_mapping():
    """
    Mapping from 2-letter ISO 639-1 codes to XQUAD language codes.
    Only including languages that have activation data.
    """
    return {
        "ar": "ar",      # Arabic
        "de": "de",      # German
        "zh": "zh",      # Chinese
        "en": "en",      # English
        "es": "es",      # Spanish
        "hi": "hi",      # Hindi
        "vi": "vi",      # Vietnamese
        "tr": "tr",      # Turkish
        "th": "th",      # Thai
        "ru": "ru",      # Russian
        "ro": "ro",      # Romanian
        "el": "el"       # Greek
    }

def get_language_names():
    """Human-readable language names for XQUAD codes"""
    return {
        "ar": "Arabic",
        "de": "German", 
        "zh": "Chinese",
        "en": "English",
        "es": "Spanish",
        "hi": "Hindi",
        "vi": "Vietnamese",
        "tr": "Turkish",
        "th": "Thai",
        "ru": "Russian",
        "ro": "Romanian",
        "el": "Greek"
    }

def get_all_xquad_languages():
    """Get all XQUAD languages"""
    return ["ar", "de", "zh", "en", "es", "hi", "vi", "tr", "th", "ru", "ro", "el"]

def get_activation_languages():
    """Get languages for which we have activation data"""
    return ["bo", "mt", "it", "es", "de", "ja", "ar", "zh", "af", "nl", "fr", "pt", "ru", "ko", "hi", "tr", "pl", "sv", "da", "no", "en"]

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.IGNORECASE)
        return re.sub(regex, ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_punc(lower(s)))

def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()

def compute_exact_match(prediction, ground_truth):
    """Compute exact match score"""
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def compute_f1_score(prediction, ground_truth):
    """Compute F1 score between prediction and ground truth"""
    prediction_tokens = get_tokens(prediction)
    ground_truth_tokens = get_tokens(ground_truth)
    
    if len(prediction_tokens) == 0 or len(ground_truth_tokens) == 0:
        return int(prediction_tokens == ground_truth_tokens)
    
    common_tokens = set(prediction_tokens) & set(ground_truth_tokens)
    
    if len(common_tokens) == 0:
        return 0
    
    precision = len(common_tokens) / len(prediction_tokens)
    recall = len(common_tokens) / len(ground_truth_tokens)
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def compute_squad_metrics(predictions: List[str], references: List[List[str]]) -> Dict:
    """Compute SQuAD-style metrics (EM and F1)"""
    exact_matches = []
    f1_scores = []
    
    for pred, ref_list in zip(predictions, references):
        # For each prediction, compute metrics against all possible answers
        max_em = 0
        max_f1 = 0
        
        for ref in ref_list:
            em = compute_exact_match(pred, ref)
            f1 = compute_f1_score(pred, ref)
            
            max_em = max(max_em, em)
            max_f1 = max(max_f1, f1)
        
        exact_matches.append(max_em)
        f1_scores.append(max_f1)
    
    return {
        "exact_match": sum(exact_matches) / len(exact_matches) if exact_matches else 0.0,
        "f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
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


def evaluate_xquad(model, language: str, sampling_params, max_samples: int = None) -> Dict:
    """Evaluate model on XQUAD extractive QA task"""
    lang_name_mapping = get_language_names()

    data = load_xquad_data(language)
    
    if not data:
        print(f"No data loaded for language {language}")
        return {"exact_match": 0.0, "f1_score": 0.0, "total": 0, "results": []}
    
    if max_samples is not None:
        data = data[:max_samples]
    
    # Debug: Print structure of first item
    if data:
        print(f"Sample data structure: {list(data[0].keys())}")
        print(f"Sample answers structure: {data[0].get('answers', 'No answers key')}")
    
    # Format prompts
    prompts = [format_xquad_prompt(item, language, lang_name_mapping) for item in data]
    
    # Generate responses
    outputs = model.generate(prompts, sampling_params)
    responses = [output.outputs[0].text.strip() for output in outputs]
    
    # Extract references (all possible answers for each question)
    references = []
    for item in data:
        # Check the structure of answers
        answers = item.get('answers', {})
        if isinstance(answers, dict):
            # Case 1: answers is a dict with 'text' key
            if 'text' in answers:
                answer_texts = answers['text'] if isinstance(answers['text'], list) else [answers['text']]
            else:
                # Case 2: answers dict has other structure
                answer_texts = [str(answers)]
        elif isinstance(answers, list):
            # Case 3: answers is a list of answer objects
            answer_texts = []
            for answer in answers:
                if isinstance(answer, dict) and 'text' in answer:
                    answer_texts.append(answer['text'])
                else:
                    answer_texts.append(str(answer))
        else:
            # Fallback: convert to string
            answer_texts = [str(answers)]
        
        references.append(answer_texts)
    
    # Compute metrics
    metrics = compute_squad_metrics(responses, references)
    
    results = []
    for i, (item, response) in enumerate(zip(data, responses)):
        # Get reference answers for this item
        answer_texts = references[i]
        item_metrics = compute_squad_metrics([response], [answer_texts])
        
        results.append({
            "id": item.get('id', i),
            "context": item['context'],
            "question": item['question'],
            "reference_answers": answer_texts,
            "model_answer": response,
            "exact_match": item_metrics['exact_match'],
            "f1_score": item_metrics['f1']
        })
    
    return {
        "exact_match": metrics['exact_match'],
        "f1_score": metrics['f1'],
        "total": len(data),
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
    all_xquad_languages = get_all_xquad_languages()
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
        max_tokens=64,  # Reasonable length for extractive answers
        stop_token_ids=[eos_token_id] if eos_token_id is not None else [],
        stop=["\n\n", "Question:", "Context:", "Answer:", "Q:", "A:"],
        skip_special_tokens=True
    )
    
    # Track overall progress
    total_combinations = len(ratios) * len(all_xquad_languages) * len(all_activation_languages)
    current_combination = 0
    
    print(f"Starting evaluation of {total_combinations} combinations...")
    print(f"Ratios: {ratios}")
    print(f"XQUAD languages: {len(all_xquad_languages)} - {all_xquad_languages}")
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
        
        # Loop through all XQUAD target languages
        for target_lang in all_xquad_languages:
            print(f"\n{'-'*40}")
            print(f"Target language: {target_lang}")
            print(f"{'-'*40}")
            
            # Run baseline evaluation once per target language
            print("Running baseline evaluation...")
            reset_activation_intervention(model)
            
            try:
                baseline_results = evaluate_xquad(model, target_lang, sampling_params, args.max_samples)
                baseline_em = baseline_results['exact_match']
                baseline_f1 = baseline_results['f1_score']
                print(f"Baseline EM: {baseline_em:.3f}, F1: {baseline_f1:.3f}")
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
                print(f"Activating {activate_lang} neurons for {target_lang} QA")
                
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
                    enhanced_results = evaluate_xquad(model, target_lang, sampling_params, args.max_samples)
                    enhanced_em = enhanced_results['exact_match']
                    enhanced_f1 = enhanced_results['f1_score']
                    
                    # Compute improvements
                    em_improvement = enhanced_em - baseline_em
                    f1_improvement = enhanced_f1 - baseline_f1
                    
                    print(f"Enhanced EM: {enhanced_em:.3f} (improvement: {em_improvement:+.3f})")
                    print(f"Enhanced F1: {enhanced_f1:.3f} (improvement: {f1_improvement:+.3f})")
                    
                    # Save results
                    results = {
                        "model": args.model,
                        "ratio": ratio,
                        "target_language": target_lang,
                        "activated_language": activate_lang,
                        "max_samples": args.max_samples,
                        "baseline": baseline_results,
                        "enhanced": enhanced_results,
                        "em_improvement": em_improvement,
                        "f1_improvement": f1_improvement
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
    parser = argparse.ArgumentParser(description="XQUAD Extractive QA Evaluation with Neuron Activation")
    
    # Model and data arguments
    parser.add_argument("-m", "--model", type=str, default="meta-llama/Llama-3.1-8B",
                       help="Model name or path")
    parser.add_argument("--activations_path", type=str, default="llama_3-1 llama-3.1",
                       help="Activations path (space-separated)")
    
    # Evaluation control arguments
    parser.add_argument("--ratios", type=int, nargs='+', default=None,
                       help="Ratios to evaluate (default: 1-5)")
    parser.add_argument("--target_langs", type=str, nargs='+', default=None,
                       help="Specific target languages to evaluate (default: all XQUAD languages)")
    parser.add_argument("--activate_langs", type=str, nargs='+', default=None,
                       help="Specific activation languages to evaluate (default: all activation languages)")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Max samples to test (None for all)")
    
    # Output and control arguments
    parser.add_argument("--output_path", type=str, default="xquad",
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