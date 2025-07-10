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

def load_belebele_data(language: str, split: str = "test") -> List[Dict]:
    """Load BELEBELE dataset for reading comprehension from HuggingFace"""
    try:
        dataset = load_dataset("facebook/belebele", language)
        data = []
        for item in dataset[split]:
            data.append(dict(item))
        return data
    except Exception as e:
        print(f"Error loading BELEBELE data for language {language}: {e}")
        return []

def format_belebele_prompt(item: Dict, language: str, lang_name_mapping: Dict) -> str:
    """Format a BELEBELE item into a multiple choice reading comprehension prompt"""
    passage = item['flores_passage']
    question = item['question']
    
    options = [
        f"A) {item['mc_answer1']}",
        f"B) {item['mc_answer2']}",
        f"C) {item['mc_answer3']}",
        f"D) {item['mc_answer4']}"
    ]
    
    language_name = lang_name_mapping.get(language, language)
    
    prompt = f"""Read the following {language_name} passage and answer the question.

Passage: {passage}

Question: {question}

{chr(10).join(options)}

Answer:"""
    
    return prompt

def get_language_mapping():
    """
    Mapping from 2-letter ISO 639-1 codes to FLORES-200 codes used in Belebele.
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
        "no": "nob_Latn",   # Norwegian Bokmål
        "en": "eng_Latn"
    }

def get_language_names():
    """Human-readable language names for BELEBELE codes"""
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
        "nob_Latn": "Norwegian",
        "eng_Latn": "English"
    }

def get_all_languages():
    """Get all supported languages from your language mapping"""
    lang_mapping = get_language_mapping()
    return list(lang_mapping.values())

def extract_answer_choice(response: str) -> str:
    """Extract the answer choice (A, B, C, D) from the model response"""
    response = response.strip().upper()
    
    patterns = [
        r'^([ABCD])\)',
        r'^([ABCD])\.',
        r'^([ABCD])\s',
        r'^\(([ABCD])\)',
        r'^([ABCD])$',
        r'ANSWER:\s*([ABCD])',
        r'THE ANSWER IS\s*([ABCD])',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            return match.group(1)
    
    # if no clear pattern, look for A, B, C, or D anywhere in the first few characters
    if len(response) > 0:
        first_char = response[0]
        if first_char in ['A', 'B', 'C', 'D']:
            return first_char
    
    return None

def compute_accuracy(references: List, predictions: List[str]) -> float:
    """Compute accuracy for multiple choice questions"""
    correct = 0
    total = 0
    
    num_to_choice = {1: 'A', 2: 'B', 3: 'C', 4: 'D'}
    
    for ref, pred in zip(references, predictions):
        total += 1
        if pred is not None:
            # Convert reference to letter if it's a number
            if isinstance(ref, str) and ref.isdigit():
                ref_letter = num_to_choice.get(int(ref))
            elif isinstance(ref, int):
                ref_letter = num_to_choice.get(ref)
            else:
                ref_letter = ref
            
            if pred == ref_letter:
                correct += 1
    
    return correct / total if total > 0 else 0.0

def compute_average_activations(activations_path):
    """Compute average activation values for each language"""
    n, over_zero = [], []
    
    # Use the old codes for loading activation data
    old_lang_names = ["bo", "mt", "it", "es", "de", "ja", "ar", "zh", "af", "nl", "fr", "pt", "ru", "ko", "hi", "tr", "pl", "sv", "da", "no", "en"]
    
    for lang in old_lang_names:
        data = torch.load(f'data_{activations_path[0]}/activation.{lang}.train.{activations_path[1]}')
        n.append(data['n'])
        over_zero.append(data['over_zero'])

    n = torch.tensor(n)
    over_zero = torch.stack(over_zero, dim=-1)
    
    # compute average activation probabilities per language
    activation_probs = over_zero / n
    
    # create mappings for both old and BELEBELE codes
    lang_mapping = get_language_mapping()
    old_to_idx = {lang: idx for idx, lang in enumerate(old_lang_names)}
    belebele_to_idx = {lang_mapping[old_lang]: idx for old_lang, idx in old_to_idx.items()}
    
    return activation_probs, old_to_idx, belebele_to_idx, old_lang_names, lang_mapping

def evaluate_belebele(model, language: str, sampling_params, max_samples: int = None) -> Dict:
    """Evaluate model on BELEBELE reading comprehension task"""
    lang_name_mapping = get_language_names()

    data = load_belebele_data(language)
    
    if max_samples is not None:
        data = data[:max_samples]
    
    prompts = [format_belebele_prompt(item, language, lang_name_mapping) for item in data]
    
    outputs = model.generate(prompts, sampling_params)
    responses = [output.outputs[0].text.strip() for output in outputs]
    
    predictions = [extract_answer_choice(response) for response in responses]
    num_to_choice = {1: 'A', 2: 'B', 3: 'C', 4: 'D'}
    references = []
    for item in data:
        ref = item['correct_answer_num']
        if isinstance(ref, str) and ref.isdigit():
            references.append(num_to_choice.get(int(ref)))
        elif isinstance(ref, int):
            references.append(num_to_choice.get(ref))
        else:
            references.append(ref)
    
    accuracy = compute_accuracy(references, predictions)
    
    results = []
    for i, (item, response, prediction, reference) in enumerate(zip(data, responses, predictions, references)):
        correct = prediction is not None and prediction == reference
        
        results.append({
            "id": i,
            "passage": item['flores_passage'],
            "question": item['question'],
            "options": [item['mc_answer1'], item['mc_answer2'], item['mc_answer3'], item['mc_answer4']],
            "correct_answer": reference,
            "model_response": response,
            "extracted_answer": prediction,
            "correct": correct
        })
    
    return {
        "accuracy": accuracy,
        "total": len(data),
        "correct_count": sum(1 for r in results if r['correct']),
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
    # store original forward methods
    for layer_idx in range(len(model.llm_engine.model_executor.driver_worker.model_runner.model.model.layers)):
        mlp = model.llm_engine.model_executor.driver_worker.model_runner.model.model.layers[layer_idx].mlp
        if not hasattr(mlp, '_original_forward'):
            mlp._original_forward = mlp.forward
    
    # apply the activation intervention to each layer
    for layer_idx in range(len(activate_mask)):
        activate_indices_cpu = activate_mask[layer_idx]  # Keep on CPU for indexing
        activate_indices = activate_indices_cpu.to('cuda')  # Move to CUDA for model operations
        
        # compute boost values for this layer
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
    
    all_languages = get_all_languages()
    ratios = list(range(1, 6)) if args.ratios is None else args.ratios
    
    print("Initializing model...")
    model = LLM(model=args.model, tensor_parallel_size=torch.cuda.device_count(), enforce_eager=True)
    
    eos_token_id = model.get_tokenizer().eos_token_id
    print(f"EOS token ID: {eos_token_id}")
    
    sampling_params = SamplingParams(
        temperature=0, 
        repetition_penalty=1.1, 
        max_tokens=32, 
        stop_token_ids=[eos_token_id] if eos_token_id is not None else [],
        stop=["\n\n", "Question:", "Passage:", "Answer:", "The answer"],
        skip_special_tokens=True
    )
    
    lang_mapping = get_language_mapping()
    
    total_combinations = len(ratios) * len(all_languages) * len(all_languages)
    current_combination = 0
    
    print(f"Starting evaluation of {total_combinations} combinations...")
    print(f"Ratios: {ratios}")
    print(f"Languages: {len(all_languages)}")
    print(f"Activation languages: {len(all_languages)}")
    
    start_time = time.time()
    
    for ratio in ratios:
        print(f"\n{'='*60}")
        print(f"PROCESSING RATIO {ratio}")
        print(f"{'='*60}")
        
        activation_dir = f"activation_mask/{args.activation_mask}-{ratio}"
        try:
            activation_masks = torch.load(activation_dir)
            activations_path = args.activations_path.split(" ")
            avg_activations, old_to_idx, belebele_to_idx, old_lang_names, lang_mapping = compute_average_activations(activations_path)
        except Exception as e:
            print(f"Error loading activation data for ratio {ratio}: {e}")
            continue
        
        model_name = args.model.split('/')[-1]
        output_dir = f"{args.output_path}/{model_name}_{ratio}"
        os.makedirs(output_dir, exist_ok=True)
        
        for target_lang in all_languages:
            print(f"\n{'-'*40}")
            print(f"Target language: {target_lang}")
            print(f"{'-'*40}")
            
            print("Running baseline evaluation...")
            reset_activation_intervention(model)
            
            try:
                baseline_results = evaluate_belebele(model, target_lang, sampling_params, args.max_samples)
                baseline_acc = baseline_results['accuracy']
                print(f"Baseline Accuracy: {baseline_acc:.3f}")
            except Exception as e:
                print(f"Error in baseline evaluation for {target_lang}: {e}")
                continue
            
            for activate_lang in all_languages:
                current_combination += 1
                elapsed_time = time.time() - start_time
                avg_time_per_combo = elapsed_time / current_combination if current_combination > 0 else 0
                eta = avg_time_per_combo * (total_combinations - current_combination)
                
                print(f"\nProgress: {current_combination}/{total_combinations} "
                      f"({100*current_combination/total_combinations:.1f}%) "
                      f"ETA: {eta/60:.1f} min")
                print(f"Activating {activate_lang} neurons for {target_lang} comprehension")
                
                output_file = f"{output_dir}/{target_lang}_activate_{activate_lang}.json"
                if os.path.exists(output_file) and not args.overwrite:
                    print(f"Results already exist, skipping: {output_file}")
                    continue
                
                try:
                    if activate_lang in belebele_to_idx:
                        activate_idx = belebele_to_idx[activate_lang]
                    else:
                        
                        activate_old_code = lang_mapping.get(activate_lang)
                        if activate_old_code and activate_old_code in old_to_idx:
                            activate_idx = old_to_idx[activate_old_code]
                        else:
                            print(f"Warning: Language {activate_lang} not found in activation data, skipping")
                            continue
                    
                    activate_mask = activation_masks[activate_idx]
                    
                    # Apply activation intervention
                    apply_activation_intervention(model, activate_mask, avg_activations, activate_idx)
                    
                    # Run enhanced evaluation
                    enhanced_results = evaluate_belebele(model, target_lang, sampling_params, args.max_samples)
                    enhanced_acc = enhanced_results['accuracy']
                    
                    improvement = enhanced_acc - baseline_acc
                    
                    print(f"Enhanced Accuracy: {enhanced_acc:.3f} "
                          f"(improvement: {improvement:+.3f}, {improvement*100:+.1f}%)")
                    
                    results = {
                        "model": args.model,
                        "ratio": ratio,
                        "target_language": target_lang,
                        "activated_language": activate_lang,
                        "max_samples": args.max_samples,
                        "baseline": baseline_results,
                        "enhanced": enhanced_results,
                        "improvement": improvement
                    }
                    
                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump(results, f, indent=4, ensure_ascii=False)
                    
                    if args.verbose:
                        print(f"Results saved: {output_file}")
                        
                except Exception as e:
                    print(f"Error processing {activate_lang} activation: {e}")
                    continue
                
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
    parser = argparse.ArgumentParser(description="BELEBELE Reading Comprehension Evaluation with Neuron Activation")
    
    # Model and data arguments
    parser.add_argument("-m", "--model", type=str, default="meta-llama/Llama-3.1-8B",
                       help="Model name or path")
    parser.add_argument("--activations_path", type=str, default="llama_3-1 llama-3.1",
                       help="Activations path (space-separated)")
    
    # Evaluation control arguments
    parser.add_argument("--ratios", type=int, nargs='+', default=None,
                       help="Ratios to evaluate (default: 1-5)")
    parser.add_argument("--target_langs", type=str, nargs='+', default=None,
                       help="Specific target languages to evaluate (default: all)")
    parser.add_argument("--activate_langs", type=str, nargs='+', default=None,
                       help="Specific activation languages to evaluate (default: all)")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Max samples to test (None for all)")
    
    # Output and control arguments
    parser.add_argument("--output_path", type=str, default="belebele",
                       help="Output directory path")
    parser.add_argument("--overwrite", action="store_true",
                       help="Overwrite existing results")
    parser.add_argument("--delay", type=float, default=0,
                       help="Delay between evaluations (seconds)")
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose output")
    
    parser.add_argument("-a", "--activation_mask", type=str, default="activation_mask/llama-3-8b",
                       help="Activation mask path (for single mode)")
    parser.add_argument("--target_lang", type=str, default="spa_Latn",
                       help="Target language (for single mode)")
    parser.add_argument("--activate_lang", type=str, default="spa_Latn",
                       help="Language neurons to activate (for single mode)")
    
    global args
    args = parser.parse_args()
    
    os.environ["VLLM_USE_V1"] = "0"
    
    if args.ratios is None:
        args.ratios = list(range(1, 6))
    
    run_full_evaluation_suite(args)

if __name__ == "__main__":
    main()