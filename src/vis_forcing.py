import json
import os
import glob
from collections import defaultdict, Counter
import fasttext
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import argparse

try:
    model = fasttext.load_model('lid.176.bin')
except:
    import urllib.request
    urllib.request.urlretrieve('https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin', 'lid.176.bin')
    model = fasttext.load_model('lid.176.bin')

def detect_language(text):
    """Detect language using fasttext, return language code"""
    if not text.strip():
        return "unknown"
    
    clean_text = text.replace('\n', ' ').strip()
    if not clean_text:
        return "unknown"
    
    predictions = model.predict(clean_text, k=1)
    lang_code = predictions[0][0].replace('__label__', '')
    return lang_code

def create_visualizations(language_success, detected_counts, all_detections):
    """Create only the activate vs deactivate heatmap visualization"""

    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")

    # Create a single-plot figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Prepare data
    lang_list = sorted(set(d["activate"] for d in all_detections if d["activate"]))
    deactivate_list = sorted(set(d["deactivate"] for d in all_detections if d["deactivate"]))

    heatmap_data = np.zeros((len(deactivate_list), len(lang_list)))

    for i, deact_lang in enumerate(deactivate_list):
        for j, act_lang in enumerate(lang_list):
            matches = [d for d in all_detections if d["deactivate"] == deact_lang and d["activate"] == act_lang]
            if matches:
                target_code = act_lang
                successes = sum(1 for m in matches if m["detected"] == target_code)
                heatmap_data[i, j] = (successes / len(matches)) * 100

    # Plot heatmap
    im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto')

    # Set ticks and labels
    ax.set_xticks(range(len(lang_list)))
    ax.set_yticks(range(len(deactivate_list)))
    ax.set_xticklabels(lang_list, rotation=45, ha='right', fontsize=16)
    ax.set_yticklabels(deactivate_list, fontsize=16)

    # Axis labels and title
    # ax.set_title('Success Rate: Deactivate vs Activate', fontsize=18, fontweight='bold')
    ax.set_xlabel('Activate Language', fontsize=20)
    ax.set_ylabel('Original Language', fontsize=20)

    if "deactivate_activate" in args.input_path:
        ax.set_ylabel('Deactivate Language', fontsize=20)
        
    
    # Add colorbar with larger label
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Success Rate (%)', fontsize=20)
    cbar.ax.tick_params(labelsize=12)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f'{args.output_path}/heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_results():
    """Analyze all intervention results"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="generation/Llama-3.1-8B_1/activate")
    parser.add_argument("--output_path", type=str, default="generation/Llama-3.1-8B_1/activate")

    global args
    args = parser.parse_args()

    results_dir = args.input_path
    json_files = glob.glob(os.path.join(results_dir, "*.json"))
    
    print(f"Found {len(json_files)} result files")
    
    # Data structures for analysis
    language_success = defaultdict(lambda: {"total": 0, "success": 0, "by_question": [0, 0, 0, 0, 0, 0]})
    all_detections = []
    activation_success = defaultdict(int)  # Count successful activations per target language
    
    # Language mapping for fasttext (just in case)
    lang_mapping = {
        "de": "de", "es": "es", "fr": "fr", "it": "it", "pt": "pt", "ru": "ru",
        "zh": "zh", "ja": "ja", "ko": "ko", "ar": "ar", "hi": "hi", "tr": "tr",
        "pl": "pl", "nl": "nl", "sv": "sv", "da": "da", "no": "no",
        "af": "af", "mt": "mt", "bo": "bo", "en": "en"
    }
    
    for file_path in json_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        deactivate_lang = data.get("deactivate_language")
        activate_lang = data.get("activate_language") 
        results = data.get("results", [])
        
        if not activate_lang or not results:
            continue
            
        target_lang_code = lang_mapping.get(activate_lang, activate_lang)
        
        # Analyze each question
        for i, result in enumerate(results):
            output_text = result.get("output", "")
            detected_lang = detect_language(output_text)
            
            all_detections.append({
                "deactivate": deactivate_lang,
                "activate": activate_lang,
                "question_idx": i,
                "detected": detected_lang,
                "output": output_text[:100] + "..." if len(output_text) > 100 else output_text
            })
            
            # Count success (detected language matches target language)
            language_success[activate_lang]["total"] += 1
            if detected_lang == target_lang_code:
                language_success[activate_lang]["success"] += 1
                if i < 6:  # Make sure we don't go out of bounds
                    language_success[activate_lang]["by_question"][i] += 1
                activation_success[activate_lang] += 1
    
    print(f"\n{'Language':<8} {'Success Rate':<12} {'Total Tests':<12} {'Q1':<4} {'Q2':<4} {'Q3':<4} {'Q4':<4} {'Q5':<4} {'Q6':<4}")
    print("-" * 70)
    
    overall_success = 0
    overall_total = 0
    
    for lang in sorted(language_success.keys()):
        stats = language_success[lang]
        success_rate = (stats["success"] / stats["total"]) * 100 if stats["total"] > 0 else 0
        
        print(f"{lang:<8} {success_rate:>7.1f}%     {stats['total']:>6}       "
              f"{stats['by_question'][0]:>2}  {stats['by_question'][1]:>2}  {stats['by_question'][2]:>2}  "
              f"{stats['by_question'][3]:>2}  {stats['by_question'][4]:>2}  {stats['by_question'][5]:>2}")
        
        overall_success += stats["success"]
        overall_total += stats["total"]
    
    overall_rate = (overall_success / overall_total) * 100 if overall_total > 0 else 0
    print("-" * 70)
    print(f"{'OVERALL':<8} {overall_rate:>7.1f}%     {overall_total:>6}")
    
    # Top performing languages
    print(f"\nTOP PERFORMING TARGET LANGUAGES:")
    sorted_langs = sorted(language_success.items(), 
                         key=lambda x: x[1]["success"]/x[1]["total"] if x[1]["total"] > 0 else 0, 
                         reverse=True)
    
    for i, (lang, stats) in enumerate(sorted_langs[:20]):
        rate = (stats["success"] / stats["total"]) * 100 if stats["total"] > 0 else 0
        print(f"{i+1}. {lang}: {rate:.1f}% ({stats['success']}/{stats['total']})")
    
    # Language detection distribution
    print(f"\nDETECTED LANGUAGES DISTRIBUTION:")
    detected_counts = Counter([d["detected"] for d in all_detections])
    for lang, count in detected_counts.most_common(10):
        print(f"{lang}: {count}")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    create_visualizations(language_success, detected_counts, all_detections)
    
    # Save detailed results
    output_file = f"{args.output_path}/analysis_results.json"
    analysis_results = {
        "overall_success_rate": overall_rate,
        "total_tests": overall_total,
        "language_statistics": dict(language_success),
        "detection_distribution": dict(detected_counts),
        "sample_detections": all_detections[:20]  # First 20 for inspection
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=2, ensure_ascii=False)
    
    print(f"\nDetailed analysis saved to: {output_file}")

if __name__ == "__main__":
    analyze_results()