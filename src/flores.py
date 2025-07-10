import argparse
import json
import os
import time
from types import MethodType
from typing import List, Dict, Tuple

import torch
import torch.nn.functional as F
from vllm import LLM, SamplingParams
from datasets import load_dataset

def load_flores_data(source_lang: str, target_lang: str, split: str = "dev", data_path: str = None) -> List[Dict]:
    """Load FLORES-200 dataset for translation from HuggingFace"""
    # Load the dataset from HuggingFace
    dataset = load_dataset("facebook/flores", name=f"{source_lang}-{target_lang}")
    
    data = []
    for item in dataset[split]:
        data.append(dict(item))
    
    return data

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
        "no": "nob_Latn",   # Norwegian Bokmål
        "en": "eng_Latn"
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
        "nob_Latn": "Norwegian",
        "eng_Latn": "English"
    }

def get_all_languages():
    """Get all supported languages from the shell script"""
    return [
        "afr_Latn",  # af
        "mlt_Latn",  # mt
        "bod_Tibt",  # bo
        "ita_Latn",  # it
        "spa_Latn",  # es
        "deu_Latn",  # de
        "jpn_Jpan",  # ja
        "arb_Arab",  # ar
        "zho_Hans",  # zh
        "nld_Latn",  # nl
        "fra_Latn",  # fr
        "por_Latn",  # pt
        "rus_Cyrl",  # ru
        "kor_Hang",  # ko
        "hin_Deva",  # hi
        "tur_Latn",  # tr
        "pol_Latn",  # pl
        "swe_Latn",  # sv
        "dan_Latn",  # da
        "nob_Latn",  # no
        "eng_Latn"
    ]

def compute_bleu_score(reference: str, candidate: str) -> float:
    """Compute BLEU score between reference and candidate translations"""
    from sacrebleu import sentence_bleu
    score = sentence_bleu(candidate, [reference])
    return score.score / 100.0

def compute_average_activations(activations_path):
    """Compute average activation values for each language"""
    n, over_zero = [], []
    
    # Use the old codes for loading activation data (assuming that's how they're stored)
    old_lang_names = ["bo", "mt", "it", "es", "de", "ja", "ar", "zh", "af", "nl", "fr", "pt", "ru", "ko", "hi", "tr", "pl", "sv", "da", "no", "en"]
    
    for lang in old_lang_names:
        data = torch.load(f'data_{activations_path[0]}/activation.{lang}.train.{activations_path[1]}')
        n.append(data['n'])
        over_zero.append(data['over_zero'])

    n = torch.tensor(n)
    over_zero = torch.stack(over_zero, dim=-1)
    
    # Compute average activation probabilities per language
    activation_probs = over_zero / n
    
    # Create mappings for both old and new codes
    lang_mapping = get_language_mapping()
    old_to_idx = {lang: idx for idx, lang in enumerate(old_lang_names)}
    flores_to_idx = {lang_mapping[old_lang]: idx for old_lang, idx in old_to_idx.items()}
    
    return activation_probs, old_to_idx, flores_to_idx, old_lang_names, lang_mapping

def evaluate_translation(model, source_lang: str, target_lang: str, sampling_params, max_samples: int = None) -> Dict:
    """Evaluate model on FLORES-200 translation task"""
    lang_name_mapping = get_language_names()

    data = load_flores_data(source_lang, target_lang, split="devtest")
    
    # Format prompts
    prompts = [format_translation_prompt(item, source_lang, target_lang, lang_name_mapping) for item in data]
    
    # Generate responses
    outputs = model.generate(prompts, sampling_params)
    responses = [output.outputs[0].text.strip() for output in outputs]
    
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
    # This is a simplified reset - you might need to adjust based on your model architecture
    for layer_idx in range(len(model.llm_engine.model_executor.driver_worker.model_runner.model.model.layers)):
        layer = model.llm_engine.model_executor.driver_worker.model_runner.model.model.layers[layer_idx]
        # Reset to original forward method - this might need adjustment based on your setup
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
    all_languages = get_all_languages()
    ratios = list(range(1, 6)) if args.ratios is None else args.ratios
    source_lang = args.source_lang
    
    # Initialize model once
    print("Initializing model...")
    model = LLM(model=args.model, tensor_parallel_size=torch.cuda.device_count(), enforce_eager=True)
    
    eos_token_id = model.get_tokenizer().eos_token_id
    print(f"EOS token ID: {eos_token_id}")
    
    sampling_params = SamplingParams(
        temperature=0, 
        repetition_penalty=1.1, 
        max_tokens=128,
        stop_token_ids=[eos_token_id] if eos_token_id is not None else [],
        stop=["\n\n", "Translation", "Translate", "Source:", "Target:"],
        skip_special_tokens=True
    )
    
    # Get language mappings once
    lang_mapping = get_language_mapping()
    
    # Track overall progress
    total_combinations = len(ratios) * len(all_languages) * len(all_languages)
    current_combination = 0
    
    print(f"Starting evaluation of {total_combinations} combinations...")
    print(f"Ratios: {ratios}")
    print(f"Source language: {source_lang}")
    print(f"Target languages: {len(all_languages)}")
    print(f"Activation languages: {len(all_languages)}")
    
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
            avg_activations, old_to_idx, flores_to_idx, old_lang_names, lang_mapping = compute_average_activations(activations_path)
        except Exception as e:
            print(f"Error loading activation data for ratio {ratio}: {e}")
            continue
        
        # Create output directory for this ratio
        model_name = args.model.split('/')[-1]
        output_dir = f"{args.output_path}/{model_name}_{ratio}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Loop through all target languages
        for target_lang in all_languages:
            print(f"\n{'-'*40}")
            print(f"Target language: {target_lang}")
            print(f"{'-'*40}")
            
            # Run baseline evaluation once per target language
            print("Running baseline evaluation...")
            reset_activation_intervention(model)
            
            try:
                baseline_results = evaluate_translation(model, source_lang, target_lang, sampling_params, args.max_samples)
                baseline_bleu = baseline_results['bleu_score']
                print(f"Baseline BLEU: {baseline_bleu:.3f}")
            except Exception as e:
                print(f"Error in baseline evaluation for {source_lang}->{target_lang}: {e}")
                continue
            
            # Loop through all activation languages
            for activate_lang in all_languages:
                current_combination += 1
                elapsed_time = time.time() - start_time
                avg_time_per_combo = elapsed_time / current_combination if current_combination > 0 else 0
                eta = avg_time_per_combo * (total_combinations - current_combination)
                
                print(f"\nProgress: {current_combination}/{total_combinations} "
                      f"({100*current_combination/total_combinations:.1f}%) "
                      f"ETA: {eta/60:.1f} min")
                print(f"Activating {activate_lang} neurons for {source_lang}->{target_lang} translation")
                
                # Check if results already exist
                output_file = f"{output_dir}/{source_lang}_to_{target_lang}_activate_{activate_lang}.json"
                if os.path.exists(output_file) and not args.overwrite:
                    print(f"Results already exist, skipping: {output_file}")
                    continue
                
                try:
                    # Get activation language index
                    activate_old_code = None
                    for old_code, flores_code in lang_mapping.items():
                        if flores_code == activate_lang:
                            activate_old_code = old_code
                            break
                    
                    if activate_old_code is None:
                        if activate_lang in old_to_idx:
                            activate_old_code = activate_lang
                        else:
                            print(f"Warning: Language {activate_lang} not found in activation data, skipping")
                            continue
                    
                    activate_idx = old_to_idx[activate_old_code]
                    activate_mask = activation_masks[activate_idx]
                    
                    # Apply activation intervention
                    apply_activation_intervention(model, activate_mask, avg_activations, activate_idx)
                    
                    # Run enhanced evaluation
                    enhanced_results = evaluate_translation(model, source_lang, target_lang, sampling_params, args.max_samples)
                    enhanced_bleu = enhanced_results['bleu_score']
                    
                    # Compute improvement
                    improvement = enhanced_bleu - baseline_bleu
                    
                    print(f"Enhanced BLEU: {enhanced_bleu:.3f} "
                          f"(improvement: {improvement:+.3f}, {improvement*100:+.1f}%)")
                    
                    # Save results
                    results = {
                        "model": args.model,
                        "ratio": ratio,
                        "source_language": source_lang,
                        "target_language": target_lang,
                        "activated_language": activate_lang,
                        "activated_language_old_code": activate_old_code,
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
    parser = argparse.ArgumentParser(description="FLORES-200 Translation Evaluation with Neuron Activation")
    
    # Model and data arguments
    parser.add_argument("-m", "--model", type=str, default="meta-llama/Llama-3.1-8B",
                       help="Model name or path")
    parser.add_argument("--activations_path", type=str, default="llama_3-1 llama-3.1",
                       help="Activations path (space-separated)")
    parser.add_argument("--source_lang", type=str, default="eng_Latn", 
                       help="Source language for translation (FLORES-200 code)")
    
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
    parser.add_argument("--output_path", type=str, default="flores",
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
    
    # Set environment variable
    os.environ["VLLM_USE_V1"] = "0"
    
    if args.ratios is None:
        args.ratios = list(range(1, 6))
    
    run_full_evaluation_suite(args)

if __name__ == "__main__":
    main()