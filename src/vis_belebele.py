import os
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def get_language_mapping():
    return {
        "bo": "bod_Tibt",
        "mt": "mlt_Latn",
        "it": "ita_Latn",
        "es": "spa_Latn",
        "de": "deu_Latn",
        "ja": "jpn_Jpan",
        "ar": "arb_Arab",
        "zh": "zho_Hans",
        "af": "afr_Latn",
        "nl": "nld_Latn",
        "fr": "fra_Latn",
        "pt": "por_Latn",
        "ru": "rus_Cyrl",
        "ko": "kor_Hang",
        "hi": "hin_Deva",
        "tr": "tur_Latn",
        "pl": "pol_Latn",
        "sv": "swe_Latn",
        "da": "dan_Latn",
        "no": "nob_Latn",
        "en": "eng_Latn"
    }

def reverse_mapping(lang_map):
    return {v: k for k, v in lang_map.items()}

def get_language_families():
    """Define language families for grouping"""
    return {
        'Romance': ['it', 'es', 'fr', 'pt'],
        'Germanic': ['de', 'af', 'nl', 'sv', 'da', 'no', 'en'],
        'Slavic': ['ru', 'pl'],
        'Sino-Tibetan': ['zh', 'bo'],
        'Other': ['mt', 'ja', 'ar', 'ko', 'hi', 'tr']
    }

def order_languages_by_family(languages, lang_families):
    """Order languages by family grouping"""
    ordered = []
    for family, family_langs in lang_families.items():
        for lang in family_langs:
            if lang in languages:
                ordered.append(lang)
    
    for lang in languages:
        if lang not in ordered:
            ordered.append(lang)
    
    return ordered

def visualize_results_as_confusion_matrix(folder):
    lang_map = get_language_mapping()
    rev_map = reverse_mapping(lang_map)
    lang_families = get_language_families()

    files = [f for f in os.listdir(folder) if f.endswith(".json")]

    test_langs = set()
    activated_langs = set()
    results_dict = {}

    for file in files:
        with open(os.path.join(folder, file), "r", encoding="utf-8") as f:
            data = json.load(f)
            test = data["target_language"]
            activate = data["activated_language"]
            baseline = data["baseline"]["accuracy"]
            enhanced = data["enhanced"]["accuracy"]
            improvement_raw = data["improvement"]
            if isinstance(improvement_raw, dict):
                improvement = improvement_raw["accuracy"]
            else:
                improvement = improvement_raw

            test_langs.add(test)
            activated_langs.add(activate)
            results_dict[(test, activate)] = improvement

    test_langs_old = [rev_map.get(lang, lang) for lang in test_langs]
    activated_langs_old = [rev_map.get(lang, lang) for lang in activated_langs]

    test_langs_ordered = order_languages_by_family(test_langs_old, lang_families)
    activated_langs_ordered = order_languages_by_family(activated_langs_old, lang_families)

    test_langs_flores_ordered = [lang_map.get(lang, lang) for lang in test_langs_ordered]
    activated_langs_flores_ordered = [lang_map.get(lang, lang) for lang in activated_langs_ordered]

    matrix = np.zeros((len(test_langs_flores_ordered), len(activated_langs_flores_ordered)))
    annot_matrix = np.empty((len(test_langs_flores_ordered), len(activated_langs_flores_ordered)), dtype=object)

    for i, test in enumerate(test_langs_flores_ordered):
        for j, activate in enumerate(activated_langs_flores_ordered):
            improvement = results_dict.get((test, activate))
            if improvement is not None:
                matrix[i, j] = improvement
                annot_matrix[i, j] = f"{improvement*100:+.1f}"
            else:
                matrix[i, j] = np.nan
                annot_matrix[i, j] = ""

    test_labels = test_langs_ordered
    activate_labels = activated_langs_ordered

    plt.figure(figsize=(16, 14))
    sns.heatmap(matrix, xticklabels=activate_labels, yticklabels=test_labels,
                cmap="coolwarm", annot=annot_matrix, fmt="", linewidths=0.5, square=True,
                center=0, cbar_kws={'label': 'BLEU Improvement (%)'})
    
    def get_family_positions(ordered_langs):
        positions = []
        current_pos = 0
        current_family = None
        
        for lang in ordered_langs:
            lang_family = None
            for family, family_langs in lang_families.items():
                if lang in family_langs:
                    lang_family = family
                    break
            
            if current_family is not None and lang_family != current_family:
                positions.append(current_pos)
            
            current_family = lang_family
            current_pos += 1
        
        return positions
    
    y_positions = get_family_positions(test_labels)
    x_positions = get_family_positions(activate_labels)
    
    ax = plt.gca()
    for pos in y_positions:
        ax.axhline(y=pos, color='black', linewidth=0.8, alpha=0.7)
    for pos in x_positions:
        ax.axvline(x=pos, color='black', linewidth=0.8, alpha=0.7)
    
    def add_family_labels(ordered_langs, axis='x'):
        current_pos = 0
        for family, family_langs in lang_families.items():
            family_count = sum(1 for lang in family_langs if lang in ordered_langs)
            if family_count > 0:
                # Calculate center position for family label
                center_pos = current_pos + (family_count - 1) / 2
                
                # Special adjustment for Sino-Tibetan
                if family == 'Sino-Tibetan' and axis == 'x':
                    center_pos += 0.5  # Move to the right
                
                if axis == 'x':
                    ax.text(center_pos, -0.5, family, ha='center', va='center', 
                           fontsize=11, fontweight='bold', rotation=0)
                else:  # axis == 'y'
                    ax.text(-1.5, center_pos, family, ha='center', va='center', 
                           fontsize=11, fontweight='bold', rotation=90)
                
                current_pos += family_count
    
    # Add family labels for both axes
    add_family_labels(activate_labels, 'x')
    # add_family_labels(test_labels, 'y')
    
    # Add thin border around the heatmap
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(0.7)
        spine.set_visible(True)
    
    # plt.title("BLEU Score Improvement by Test and Activated Language\n(Grouped by Language Family)")
    plt.xlabel("Activated Language", fontsize=20)
    plt.ylabel("Test Language", fontsize=20)
    plt.savefig(f"{args.output_path}/belebele_family_grouped.png", dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, default="input")
parser.add_argument("--output_path", type=str, default="output")

global args
args = parser.parse_args()

# Run visualization
visualize_results_as_confusion_matrix(args.input_path)