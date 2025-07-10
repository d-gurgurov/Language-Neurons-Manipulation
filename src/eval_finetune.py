import argparse
import json
import os
from typing import Dict, List, Optional
import re
from collections import defaultdict

import torch
from vllm import LLM, SamplingParams
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sacrebleu import BLEU
from tqdm import tqdm

def get_language_names():
    """Mapping of language codes to full language names"""
    return {
        'mt': 'Maltese',
        'es': 'Spanish', 
        'fr': 'French',
        'de': 'German',
        'it': 'Italian',
        'pt': 'Portuguese',
        'ru': 'Russian',
        'ar': 'Arabic',
        'zh': 'Chinese',
        'ja': 'Japanese',
        'ko': 'Korean',
        'hi': 'Hindi',
        'tr': 'Turkish',
        'pl': 'Polish',
        'sv': 'Swedish',
        'da': 'Danish',
        'no': 'Norwegian',
        'nl': 'Dutch',
        'af': 'Afrikaans',
        'bo': 'Tibetan',
        'en': 'English'
    }

def load_flores_data(source_lang: str, target_lang: str, split: str = "devtest", max_samples: Optional[int] = None):
    """Load FLORES-200 translation data"""
    try:
        dataset = load_dataset("facebook/flores", name=f"{source_lang}_Latn-{target_lang}", split=split)
        data = list(dataset)
        
        if max_samples is not None:
            data = data[:max_samples]
            
        return data
    except Exception as e:
        print(f"Error loading FLORES data for {source_lang}->{target_lang}: {e}")
        return []

def load_belebele_data(language: str, max_samples: Optional[int] = None):
    """Load BELEBELE reading comprehension data"""
    try:
        dataset = load_dataset("facebook/belebele", language, split="test")
        data = list(dataset)
        
        if max_samples is not None:
            data = data[:max_samples]
            
        return data
    except Exception as e:
        print(f"Error loading BELEBELE data for {language}: {e}")
        return []

def format_translation_prompt(item: Dict, source_lang: str, target_lang: str, lang_name_mapping: Dict) -> str:
    """Format translation prompt for FLORES evaluation"""
    source_name = lang_name_mapping.get(source_lang, source_lang)
    target_name = lang_name_mapping.get(target_lang, target_lang)
    source_text = item[f'sentence_{source_lang}_Latn']
    
    prompt = f"Translate this {source_name} sentence into {target_name}: {source_text}. Translation:"
    return prompt

def format_belebele_prompt(item: Dict, language: str, lang_name_mapping: Dict) -> str:
    """Format BELEBELE reading comprehension prompt"""
    language_name = lang_name_mapping.get(language, language)
    
    passage = item['flores_passage']
    question = item['question']
    
    # Format the multiple choice options
    options = [
        f"A) {item['mc_answer1']}",
        f"B) {item['mc_answer2']}", 
        f"C) {item['mc_answer3']}",
        f"D) {item['mc_answer4']}"
    ]
    
    prompt = f"""Read the following {language_name} passage and answer the question.

Passage: {passage}

Question: {question}

{chr(10).join(options)}

Answer:"""
    
    return prompt

def compute_bleu_score(reference: str, candidate: str) -> float:
    """Compute BLEU score for translation evaluation"""
    # Try the newer sacrebleu API first
    from sacrebleu import sentence_bleu
    score = sentence_bleu(candidate, [reference])
    return score.score

def extract_answer_choice(response: str) -> Optional[str]:
    """Extract A/B/C/D choice from model response"""
    response = response.strip().upper()
    
    # Look for patterns like "A)", "A.", "A", etc.
    patterns = [
        r'^([ABCD])\)',  # A), B), etc.
        r'^([ABCD])\.',  # A., B., etc.
        r'^([ABCD])\s',  # A , B , etc.
        r'^([ABCD])$',   # Just A, B, C, D
        r'\b([ABCD])\)',  # Anywhere with )
        r'\b([ABCD])\.',  # Anywhere with .
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            return match.group(1)
    
    return None

def compute_accuracy(references: List[str], predictions: List[Optional[str]]) -> float:
    """Compute accuracy for classification task"""
    if not references or not predictions:
        return 0.0
    
    correct = sum(1 for ref, pred in zip(references, predictions) if pred is not None and pred == ref)
    total = len(references)
    
    return correct / total if total > 0 else 0.0

def evaluate_translation(model, source_lang: str, target_lang: str, sampling_params, max_samples: Optional[int] = None) -> Dict:
    """Evaluate model on FLORES-200 translation task"""
    print(f"Evaluating translation: {source_lang} -> {target_lang}")
    
    lang_name_mapping = get_language_names()
    data = load_flores_data(source_lang, target_lang, split="devtest", max_samples=max_samples)
    
    if not data:
        return {"error": f"No data loaded for {source_lang}->{target_lang}"}
    
    # Format prompts
    prompts = [format_translation_prompt(item, source_lang, target_lang, lang_name_mapping) for item in data]
    
    # Generate responses
    print(f"Generating {len(prompts)} translations...")
    outputs = model.generate(prompts, sampling_params)
    responses = [output.outputs[0].text.strip() for output in outputs]
    
    bleu_scores = []
    results = []
    
    for i, (item, response) in enumerate(zip(data, responses)):
        reference = item[f'sentence_{target_lang}']
        candidate = response
        
        # Compute BLEU score
        bleu_score = compute_bleu_score(reference, candidate)
        bleu_scores.append(bleu_score)
        
        results.append({
            "id": i,
            "source_text": item[f'sentence_{source_lang}_Latn'],
            "reference_translation": reference,
            "model_translation": candidate,
            "bleu_score": bleu_score
        })
    
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
    
    return {
        "task": "translation",
        "source_lang": source_lang,
        "target_lang": target_lang,
        "bleu_score": avg_bleu,
        "total_samples": len(data),
        "individual_scores": bleu_scores,
        "results": results
    }

def evaluate_belebele(model, language: str, sampling_params, max_samples: Optional[int] = None) -> Dict:
    """Evaluate model on BELEBELE reading comprehension task"""
    print(f"Evaluating reading comprehension: {language}")
    
    lang_name_mapping = get_language_names()
    data = load_belebele_data(language, max_samples=max_samples)
    
    if not data:
        return {"error": f"No BELEBELE data loaded for {language}"}
    
    # Format prompts
    prompts = [format_belebele_prompt(item, language, lang_name_mapping) for item in data]
    
    # Generate responses
    print(f"Generating {len(prompts)} reading comprehension answers...")
    outputs = model.generate(prompts, sampling_params)
    responses = [output.outputs[0].text.strip() for output in outputs]
    
    # Extract predictions and compute accuracy
    predictions = [extract_answer_choice(response) for response in responses]
    
    # Convert numeric references to letters
    num_to_choice = {1: 'A', 2: 'B', 3: 'C', 4: 'D'}
    references = []
    for item in data:
        ref = item['correct_answer_num']
        if isinstance(ref, str) and ref.isdigit():
            references.append(num_to_choice.get(int(ref)))
        elif isinstance(ref, int):
            references.append(num_to_choice.get(ref))
        else:
            references.append(str(ref).upper() if ref else None)
    
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
        "task": "reading_comprehension",
        "language": language,
        "accuracy": accuracy,
        "total_samples": len(data),
        "correct_count": sum(1 for r in results if r['correct']),
        "results": results
    }

def create_comparison_plots(base_results: Dict, finetuned_results: Dict, output_dir: str):
    """Create comparison plots between base and fine-tuned models"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect results for plotting
    translation_data = []
    comprehension_data = []
    
    for task_name, task_results in base_results.items():
        if task_results.get("task") == "translation":
            translation_data.append({
                "Model": "Base",
                "Language Pair": f"{task_results['source_lang']}->{task_results['target_lang']}",
                "BLEU Score": task_results["bleu_score"]
            })
        elif task_results.get("task") == "reading_comprehension":
            comprehension_data.append({
                "Model": "Base", 
                "Language": task_results["language"],
                "Accuracy": task_results["accuracy"]
            })
    
    for task_name, task_results in finetuned_results.items():
        if task_results.get("task") == "translation":
            translation_data.append({
                "Model": "Fine-tuned",
                "Language Pair": f"{task_results['source_lang']}->{task_results['target_lang']}",
                "BLEU Score": task_results["bleu_score"]
            })
        elif task_results.get("task") == "reading_comprehension":
            comprehension_data.append({
                "Model": "Fine-tuned",
                "Language": task_results["language"], 
                "Accuracy": task_results["accuracy"]
            })
    
    # Plot translation results
    if translation_data:
        plt.figure(figsize=(12, 8))
        df_trans = pd.DataFrame(translation_data)
        sns.barplot(data=df_trans, x="Language Pair", y="BLEU Score", hue="Model")
        plt.title("Translation Performance Comparison (FLORES)")
        plt.xlabel("Language Pair")
        plt.ylabel("BLEU Score")
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "translation_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot reading comprehension results
    if comprehension_data:
        plt.figure(figsize=(10, 6))
        df_comp = pd.DataFrame(comprehension_data)
        sns.barplot(data=df_comp, x="Language", y="Accuracy", hue="Model")
        plt.title("Reading Comprehension Performance Comparison (BELEBELE)")
        plt.xlabel("Language")
        plt.ylabel("Accuracy")
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "comprehension_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()

def run_full_evaluation(model, target_lang: str, sampling_params, max_samples: Optional[int] = None) -> Dict:
    """Run full evaluation suite for a given target language"""
    results = {}
    
    # Translation tasks (English to target language)
    translation_result = evaluate_translation(
        model, "eng", target_lang, sampling_params, max_samples
    )
    results[f"translation_en_{target_lang}"] = translation_result
    
    # Reading comprehension in target language
    comprehension_result = evaluate_belebele(
        model, target_lang, sampling_params, max_samples
    )
    results[f"comprehension_{target_lang}"] = comprehension_result
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned vs base models on translation and comprehension tasks")
    parser.add_argument("--base_model", type=str, required=True,
                       help="Path to base model")
    parser.add_argument("--finetuned_model", type=str, required=True, 
                       help="Path to fine-tuned model (.pt weights file or model directory)")
    parser.add_argument("--target_lang", type=str, required=True,
                       help="Target language code (e.g., 'mt', 'es', 'fr')")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                       help="Output directory for results")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples per task (None for all)")
    parser.add_argument("--temperature", type=float, default=0.0,
                       help="Sampling temperature")
    parser.add_argument("--max_tokens", type=int, default=128,
                       help="Maximum tokens to generate")
    parser.add_argument("--repetition_penalty", type=float, default=1.1,
                       help="Repetition penalty")
    parser.add_argument("--tensor_parallel_size", type=int, default=None,
                       help="Tensor parallel size (default: auto-detect GPU count)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        max_tokens=args.max_tokens,
        skip_special_tokens=True
    )
    
    print(f"Evaluating models for target language: {args.target_lang}")
    print(f"Max samples per task: {args.max_samples}")
    
    # Auto-detect tensor parallel size if not specified
    tensor_parallel_size = args.tensor_parallel_size or torch.cuda.device_count()
    print(f"Using tensor parallel size: {tensor_parallel_size}")
    
    # Load and evaluate base model
    print("\n" + "="*60)
    print("EVALUATING BASE MODEL")
    print("="*60)
    
    base_model = LLM(
        model=args.base_model,
        tensor_parallel_size=tensor_parallel_size,
        enforce_eager=True
    )
    
    base_results = run_full_evaluation(
        base_model, args.target_lang, sampling_params, args.max_samples
    )
    
    # Clean up base model
    del base_model
    torch.cuda.empty_cache()
    
    # Load and evaluate fine-tuned model
    print("\n" + "="*60)
    print("EVALUATING FINE-TUNED MODEL")
    print("="*60)
    
    finetuned_model = LLM(
        model=args.finetuned_model,
        tensor_parallel_size=tensor_parallel_size,
        enforce_eager=True
    )
    
    finetuned_results = run_full_evaluation(
        finetuned_model, args.target_lang, sampling_params, args.max_samples
    )
    
    # Clean up fine-tuned model
    del finetuned_model
    torch.cuda.empty_cache()
    
    # Combine and save results
    evaluation_results = {
        "target_language": args.target_lang,
        "evaluation_config": {
            "max_samples": args.max_samples,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "repetition_penalty": args.repetition_penalty
        },
        "base_model_results": base_results,
        "finetuned_model_results": finetuned_results
    }
    
    # Save detailed results
    results_file = os.path.join(args.output_dir, f"evaluation_results_{args.target_lang}.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
    
    # Create comparison plots
    create_comparison_plots(base_results, finetuned_results, args.output_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    for task_name, base_result in base_results.items():
        finetuned_result = finetuned_results.get(task_name, {})
        
        if base_result.get("task") == "translation":
            base_bleu = base_result.get("bleu_score", 0)
            ft_bleu = finetuned_result.get("bleu_score", 0)
            improvement = ft_bleu - base_bleu
            
            print(f"\nTranslation ({base_result.get('source_lang', '')}->{base_result.get('target_lang', '')}):")
            print(f"  Base Model BLEU:       {base_bleu:.2f}")
            print(f"  Fine-tuned Model BLEU: {ft_bleu:.2f}")
            print(f"  Improvement:           {improvement:+.2f}")
            
        elif base_result.get("task") == "reading_comprehension":
            base_acc = base_result.get("accuracy", 0)
            ft_acc = finetuned_result.get("accuracy", 0)
            improvement = ft_acc - base_acc
            
            print(f"\nReading Comprehension ({base_result.get('language', '')}):")
            print(f"  Base Model Accuracy:       {base_acc:.3f}")
            print(f"  Fine-tuned Model Accuracy: {ft_acc:.3f}")
            print(f"  Improvement:               {improvement:+.3f}")
    
    print(f"\nDetailed results saved to: {results_file}")
    print(f"Comparison plots saved to: {args.output_dir}")

if __name__ == "__main__":
    main()