import argparse
import json
import os
from types import MethodType
from itertools import combinations
from collections import defaultdict

import torch
import torch.nn.functional as F
from vllm import LLM, SamplingParams
import fasttext
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def setup_language_detection():
    """Setup fasttext language detection model"""
    print("Loading language detection model...")
    try:
        model = fasttext.load_model('lid.176.bin')
    except:
        print("Downloading fasttext language detection model...")
        import urllib.request
        urllib.request.urlretrieve('https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin', 'lid.176.bin')
        model = fasttext.load_model('lid.176.bin')
    return model

def detect_language(text, model):
    """Detect language using fasttext, return language code and confidence"""
    if not text.strip():
        return "unknown", 0.0
    
    # Clean text for detection
    clean_text = text.replace('\n', ' ').strip()
    if not clean_text:
        return "unknown", 0.0
    
    predictions = model.predict(clean_text, k=1)
    lang_code = predictions[0][0].replace('__label__', '')
    confidence = predictions[1][0]
    return lang_code, confidence

def compute_average_activations():
    """Compute average activation values for each language"""
    lang_names = ["bo", "mt", "it", "es", "de", "ja", "ar", "zh", "af", "nl", "fr", "pt", "ru", "ko", "hi", "tr", "pl", "sv", "da", "no", "en"]
    
    # Create a mapping from language name to index
    lang_to_idx = {lang: idx for idx, lang in enumerate(lang_names)}
    
    return lang_to_idx, lang_names

def factory(idx, deactivate_indices=None, activate_indices=None, boost_values=None, deactivation_method="zero"):
    def llama_forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        i = gate_up.size(-1)

        # Determine dimensions (2D or 3D input)
        if gate_up.dim() == 3:
            silu_output = F.silu(gate_up[:, :, : i // 2])
            activation = silu_output.float()

            # Apply neuron deactivation with different methods
            if deactivate_indices is not None and len(deactivate_indices) > 0:
                if deactivation_method == "zero":
                    # Method 1: Set to zero (original)
                    silu_output.index_fill_(2, deactivate_indices, 0)
                elif deactivation_method == "negative":
                    # Method 2: Set to large negative values to suppress
                    silu_output.index_fill_(2, deactivate_indices, args.deactivation_strength)
                elif deactivation_method == "mean":
                    # Method 3: Set to mean activation of non-deactivated neurons
                    mean_val = silu_output.mean(dim=2, keepdim=True)
                    silu_output[:, :, deactivate_indices] = mean_val.expand(-1, -1, len(deactivate_indices))
                elif deactivation_method == "random":
                    # Method 4: Set to random small values
                    random_vals = torch.randn_like(silu_output[:, :, deactivate_indices]) * 0.1
                    silu_output[:, :, deactivate_indices] = random_vals
                elif deactivation_method == "mask":
                    # Method 5: Multiplicative masking (complete suppression)
                    silu_output[:, :, deactivate_indices] *= -0.1
                elif deactivation_method == "noise":
                    # Method 6: Add strong noise to disrupt patterns
                    noise = torch.randn_like(silu_output[:, :, deactivate_indices]) * 5.0
                    silu_output[:, :, deactivate_indices] += noise
                elif deactivation_method == "invert":
                    # Method 7: Invert the activations
                    silu_output[:, :, deactivate_indices] = -silu_output[:, :, deactivate_indices]

            # Optional: apply neuron boosting
            if activate_indices is not None and len(activate_indices) > 0 and boost_values is not None:
                boost_tensor = boost_values.to(silu_output.dtype).unsqueeze(0).unsqueeze(0)
                silu_output[:, :, activate_indices] += boost_tensor

            x = silu_output * gate_up[:, :, i // 2:]

        elif gate_up.dim() == 2:
            silu_output = F.silu(gate_up[:, : i // 2])
            activation = silu_output.float()

            # Apply neuron deactivation with different methods
            if deactivate_indices is not None and len(deactivate_indices) > 0:
                if deactivation_method == "zero":
                    silu_output.index_fill_(1, deactivate_indices, 0)
                elif deactivation_method == "negative":
                    silu_output.index_fill_(1, deactivate_indices, args.deactivation_strength)
                elif deactivation_method == "mean":
                    mean_val = silu_output.mean(dim=1, keepdim=True)
                    silu_output[:, deactivate_indices] = mean_val.expand(-1, len(deactivate_indices))
                elif deactivation_method == "random":
                    random_vals = torch.randn_like(silu_output[:, deactivate_indices]) * 0.1
                    silu_output[:, deactivate_indices] = random_vals
                elif deactivation_method == "mask":
                    silu_output[:, deactivate_indices] *= -0.1
                elif deactivation_method == "noise":
                    noise = torch.randn_like(silu_output[:, deactivate_indices]) * 5.0
                    silu_output[:, deactivate_indices] += noise
                elif deactivation_method == "invert":
                    silu_output[:, deactivate_indices] = -silu_output[:, deactivate_indices]

            if activate_indices is not None and len(activate_indices) > 0 and boost_values is not None:
                boost_tensor = boost_values.to(silu_output.dtype).unsqueeze(0)
                silu_output[:, activate_indices] += boost_tensor

            x = silu_output * gate_up[:, i // 2:]

        else:
            raise ValueError(f"Unexpected gate_up shape: {gate_up.shape}")

        x, _ = self.down_proj(x)
        return x

    return llama_forward

def generate_language_combinations(high_resource_langs, progressive_only=True):
    """Generate all combinations of languages to deactivate progressively"""
    combinations_list = []
    
    if progressive_only:
        # Progressive/cumulative: en, en+de, en+de+fr, en+de+fr+it, etc.
        for i in range(1, len(high_resource_langs) + 1):
            combinations_list.append(high_resource_langs[:i])
    else:
        # Original: all possible combinations
        # Start with individual languages
        for lang in high_resource_langs:
            combinations_list.append([lang])
        
        # Then pairs, triplets, etc.
        for r in range(2, len(high_resource_langs) + 1):
            for combo in combinations(high_resource_langs, r):
                combinations_list.append(list(combo))
    
    return combinations_list

def test_language_fallback(model, sampling_params, activation_masks, lang_to_idx, 
                          deactivate_langs, lang_detector, deactivation_method="zero"):
    """Test language fallback for a specific combination of deactivated languages"""
    
    print(f"Testing deactivation of: {deactivate_langs} (method: {deactivation_method})")
    
    # Combine deactivation masks for multiple languages
    combined_deactivate_mask = None
    for lang in deactivate_langs:
        if lang in lang_to_idx:
            lang_idx = lang_to_idx[lang]
            lang_mask = activation_masks[lang_idx]
            
            if combined_deactivate_mask is None:
                # Initialize combined mask - copy the structure
                combined_deactivate_mask = []
                for layer_idx in range(len(lang_mask)):
                    combined_deactivate_mask.append(lang_mask[layer_idx].clone())
            else:
                # Union of neurons to deactivate
                for layer_idx in range(len(lang_mask)):
                    combined_deactivate_mask[layer_idx] = torch.cat([
                        combined_deactivate_mask[layer_idx], 
                        lang_mask[layer_idx]
                    ]).unique()
    
    # Apply intervention to model
    if combined_deactivate_mask is not None:
        for layer_idx in range(len(combined_deactivate_mask)):
            deactivate_indices = combined_deactivate_mask[layer_idx].to('cuda')
            
            # Get the model layer
            obj = model.llm_engine.model_executor.driver_worker.model_runner.model.model.layers[layer_idx].mlp
            obj.forward = MethodType(factory(layer_idx, deactivate_indices=deactivate_indices, 
                                           deactivation_method=deactivation_method), obj)

    file_path = "mvicuna/en.txt"

    # Read lines into a list, stripping newline characters
    with open(file_path, "r", encoding="utf-8") as f:
        neutral_prompts = [line.strip() for line in f if line.strip()]

    
    results = []
    language_counts = defaultdict(int)
    
    print(f"  Testing {len(neutral_prompts)} prompts...")
    for i, prompt in enumerate(neutral_prompts):
        outputs = model.generate(["Q: "+prompt+"A: "], sampling_params)
        response = outputs[0].outputs[0].text.strip()
        
        # Detect language of response
        detected_lang, confidence = detect_language(response, lang_detector)
        language_counts[detected_lang] += 1
        
        # Print real-time results
        prompt_display = prompt if prompt else "[EMPTY]"
        if len(prompt_display) > 30:
            prompt_display = prompt_display[:27] + "..."
        response_display = response if response else "[NO RESPONSE]"
        if len(response_display) > 50:
            response_display = response_display[:47] + "..."
        
        print(f"    {i+1:2d}. '{prompt_display}' → {detected_lang} ({confidence:.2f}) | '{response_display}'")
        
        results.append({
            "prompt": prompt,
            "response": response,
            "detected_language": detected_lang,
            "confidence": confidence
        })
    
    # Find most common language
    most_common_lang = max(language_counts, key=language_counts.get) if language_counts else "unknown"
    
    # Print summary for this combination
    print(f"  → Summary: Most common = {most_common_lang}")
    print(f"  → Distribution: {dict(language_counts)}")
    print(f"  → Unique languages detected: {len(language_counts)}")
    print()
    
    return {
        "deactivated_languages": deactivate_langs,
        "most_common_output_language": most_common_lang,
        "language_distribution": dict(language_counts),
        "detailed_results": results
    }

def create_fallback_visualization(fallback_results, output_dir):
    """Create visualizations for language fallback hierarchy"""
    
    # Prepare data for visualization
    data = []
    all_output_languages = set()
    
    for result in fallback_results:
        deactivated = result["deactivated_languages"]
        output_lang = result["most_common_output_language"]
        lang_dist = result["language_distribution"]
        num_deactivated = len(deactivated)
        
        # Collect all languages that appear in any distribution
        all_output_languages.update(lang_dist.keys())
        
        data.append({
            "num_deactivated": num_deactivated,
            "deactivated_langs": ", ".join(sorted(deactivated)),
            "output_language": output_lang,
            "deactivated_set": frozenset(deactivated),
            "language_distribution": lang_dist
        })
    
    df = pd.DataFrame(data)
    
    # Create comprehensive heatmap showing all detected languages
    print(f"All detected output languages: {sorted(all_output_languages)}")
    
    # Create a detailed matrix with all languages and all combinations
    # PRESERVE THE ORIGINAL ORDER from fallback_results
    detailed_data = []
    combination_order = []  # Keep track of the original order
    
    for result in fallback_results:
        deactivated = result["deactivated_languages"]
        lang_dist = result["language_distribution"]
        num_deactivated = len(deactivated)
        deactivated_str = ", ".join(sorted(deactivated))  # Keep sorted for display consistency
        
        # Store the combination in the order it appears in results
        if deactivated_str not in combination_order:
            combination_order.append(deactivated_str)
        
        for lang in all_output_languages:
            count = lang_dist.get(lang, 0)
            detailed_data.append({
                "combination": deactivated_str,
                "num_deactivated": num_deactivated,
                "output_language": lang,
                "count": count
            })
    
    detailed_df = pd.DataFrame(detailed_data)
    
    # Create heatmap 1: By number of deactivated languages
    plt.figure(figsize=(15, 10))
    heatmap_data = detailed_df.pivot_table(
        index='num_deactivated', 
        columns='output_language', 
        values='count', 
        aggfunc='sum',
        fill_value=0
    )
    
    sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='YlOrRd', 
                cbar_kws={'label': 'Number of Prompts'})
    # plt.title('Language Fallback Hierarchy\n(Output Language by Number of Deactivated Languages)')
    plt.xlabel('Output Language')
    plt.ylabel('Number of Deactivated Languages')
    plt.tight_layout()
    # plt.savefig(f'{output_dir}/language_fallback_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create heatmap 2: By specific language combinations IN ORIGINAL ORDER
    plt.figure(figsize=(20, max(12, len(fallback_results))))
    combination_heatmap = detailed_df.pivot_table(
        index='combination',
        columns='output_language',
        values='count',
        fill_value=0
    )
    
    # Reorder the index to match the original order from fallback_results
    combination_heatmap = combination_heatmap.reindex(combination_order)
    
    sns.heatmap(combination_heatmap, annot=True, fmt='d', cmap='YlOrRd',
                cbar_kws={'label': 'Number of Prompts'})
    # plt.title('Detailed Language Fallback by Specific Combinations\n(In Progressive Order)')
    plt.xlabel('Output Language')
    plt.ylabel('Deactivated Language Combinations (Progressive Order)')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/detailed_combination_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create stacked bar chart showing language distribution for each combination
    # ALSO preserve the original order here
    plt.figure(figsize=(20, 10))
    
    # Use the original order from fallback_results
    combinations = [", ".join(sorted(r["deactivated_languages"])) for r in fallback_results]
    lang_matrix = np.zeros((len(combinations), len(all_output_languages)))
    lang_list = sorted(all_output_languages)
    
    for i, result in enumerate(fallback_results):
        lang_dist = result["language_distribution"]
        for j, lang in enumerate(lang_list):
            lang_matrix[i, j] = lang_dist.get(lang, 0)
    
    # Print detailed summary
    print(f"\nDetailed Language Fallback Analysis:")
    print("=" * 60)
    
    for i, result in enumerate(fallback_results):
        deactivated = result["deactivated_languages"]
        most_common = result["most_common_output_language"]
        lang_dist = result["language_distribution"]
        
        print(f"\n{i+1}. Deactivated: {deactivated}")
        print(f"   Most common output: {most_common}")
        print(f"   Full distribution: {lang_dist}")
        
        # Show percentage breakdown
        total = sum(lang_dist.values())
        if total > 0:
            percentages = {lang: (count/total)*100 for lang, count in lang_dist.items()}
            sorted_percentages = sorted(percentages.items(), key=lambda x: x[1], reverse=True)
            print(f"   Percentages: {[(lang, f'{pct:.1f}%') for lang, pct in sorted_percentages]}")
    
    # Save detailed results with all language information
    detailed_results = []
    for result in fallback_results:
        detailed_results.append({
            "deactivated_languages": result["deactivated_languages"],
            "most_common_output_language": result["most_common_output_language"],
            "complete_language_distribution": result["language_distribution"],
            "total_prompts": sum(result["language_distribution"].values()),
            "unique_languages_detected": len(result["language_distribution"]),
            "sample_outputs": [r["response"][:100] + "..." if len(r["response"]) > 100 else r["response"] 
                             for r in result["detailed_results"][:3]]
        })
    
    return detailed_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("-a", "--activation_mask", type=str, default="activation_mask/llama-3_5")
    parser.add_argument("--output_dir", type=str, default="results/language_fallback")
    parser.add_argument("--max_combinations", type=int, default=50, 
                       help="Maximum number of language combinations to test")
    parser.add_argument("--progressive_only", action='store_true',
                       help="Only test progressive combinations (en, en+de, en+de+fr, ...) instead of all combinations")
    parser.add_argument("--language_order", type=str, nargs='+', 
                       default=["en", "de", "fr", "it", "pt", "hi", "es"],
                       help="Order of languages for progressive testing (default: en de fr it pt hi es)")
    parser.add_argument("--deactivation_method", type=str, default="negative",
                       choices=["zero", "negative", "mean", "random", "mask", "noise", "invert"],
                       help="Method for deactivating neurons: zero, negative, mean, random, mask, noise, invert")
    parser.add_argument("--deactivation_strength", type=float, default=-1.0)      
    global args
    args = parser.parse_args()

    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    lang_detector = setup_language_detection()
    
    # Define high-resource languages to test
    high_resource_languages = ["en", "fr", "es", "it", "pt", "ru", "de", "ar", "zh", "ko", "ja"]
    # high_resource_languages = ["en", "pt", "ru", "de", "it", "es", "fr", "ar", "zh", "ko", "ja"]
    
    # let it run without english, then see what language pops up, and then deactivate that language and continue like this
    # Load model and setup
    model = LLM(model=args.model, tensor_parallel_size=torch.cuda.device_count(), enforce_eager=True)
    eos_token_id = model.get_tokenizer().eos_token_id
    
    sampling_params = SamplingParams(
        temperature=0, 
        repetition_penalty=1.1, 
        max_tokens=128,
        stop_token_ids=[eos_token_id] if eos_token_id is not None else [],
        stop=["\nQ:", "\nA:"],
        skip_special_tokens=True
    )
    
    # Load activation masks and compute averages
    activation_masks = torch.load(args.activation_mask)
    lang_to_idx, lang_names = compute_average_activations()
    
    # Generate combinations to test
    language_combinations = generate_language_combinations(high_resource_languages)
    
    # Limit combinations if specified
    if args.max_combinations > 0:
        language_combinations = language_combinations[:args.max_combinations]
    
    print(f"Testing {len(language_combinations)} language combinations...")
    
    # Test each combination
    fallback_results = []
    for i, combo in enumerate(language_combinations):
        print(f"\n{'='*60}")
        print(f"Progress: {i+1}/{len(language_combinations)}")
        print(f"{'='*60}")
        
        result = test_language_fallback(
            model, sampling_params, activation_masks, lang_to_idx, 
            combo, lang_detector, args.deactivation_method
        )
        fallback_results.append(result)
        
        # Save intermediate results
        if (i + 1) % 10 == 0:
            with open(f"{args.output_dir}/intermediate_results_{i+1}.json", "w") as f:
                json.dump(fallback_results, f, indent=2, ensure_ascii=False)
    
    # Create visualizations and analysis
    detailed_results = create_fallback_visualization(fallback_results, args.output_dir)
    
    # Save final results
    final_results = {
        "model": args.model,
        "high_resource_languages": high_resource_languages,
        "total_combinations_tested": len(language_combinations),
        "fallback_hierarchy": detailed_results,
        "raw_results": fallback_results
    }
    
    with open(f"{args.output_dir}/language_fallback_results.json", "w") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    print(f"\nLanguage fallback analysis complete!")
    print(f"Results saved to: {args.output_dir}")
    print(f"Total combinations tested: {len(language_combinations)}")

if __name__ == "__main__":
    main()