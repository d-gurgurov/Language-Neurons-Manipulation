import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, default="activation_mask/aya-8")
parser.add_argument("--output_path", type=str, default="plots")

global args
args = parser.parse_args()

# Set style for publication quality
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    "font.size": 12,
    "font.family": "serif",
    "axes.linewidth": 1.2,
    "grid.alpha": 0.3,
    "figure.dpi": 300
})

# Output directory
os.makedirs(args.output_path, exist_ok=True)

# Language codes used (same order as input)
languages = ["bo", "mt", "it", "es", "de", "ja", "ar", "zh", "af", "nl", "fr", "pt", "ru", "ko", "hi", "tr", "pl", "sv", "da", "no", "en"]

# Manually curated colors for maximum distinction and readability
distinct_colors = [
    '#1f77b4',  # blue
    '#ff7f0e',  # orange  
    '#2ca02c',  # green
    '#d62728',  # red
    '#9467bd',  # purple
    '#8c564b',  # brown
    '#e377c2',  # pink
    '#7f7f7f',  # gray
    '#bcbd22',  # olive
    '#17becf',  # cyan
    '#aec7e8',  # light blue
    '#ffbb78',  # light orange
    '#98df8a',  # light green
    '#ff9896',  # light red
    '#c5b0d5',  # light purple
    '#c49c94',  # light brown
    '#f7b6d3',  # light pink
    '#c7c7c7',  # light gray
    '#dbdb8d',  # light olive
    '#9edae5',  # light cyan
    '#8B4513'   # saddle brown
]

# Create language to color mapping
lang_colors = {}
for i, lang in enumerate(languages):
    lang_colors[lang] = distinct_colors[i % len(distinct_colors)]

def get_language_families():
    """Define language families for grouping"""
    return {
        'Romance': ['it', 'es', 'fr', 'pt'],
        'Germanic': ['de', 'af', 'nl', 'sv', 'da', 'no', 'en'],
        'Slavic': ['ru', 'pl'],
        'Sino-Tibetan': ['zh', 'bo'],
        'Other': ['mt', 'ja', 'ar', 'ko', 'hi', 'tr']
    }

def get_non_latin_scripts():
    """Languages with non-Latin scripts that should have asterisks"""
    return {'bo', 'ja', 'ar', 'zh', 'ko', 'hi', 'ru', 'pl'}

def format_language_label(lang):
    """Add asterisk for non-Latin script languages"""
    non_latin = get_non_latin_scripts()
    return f"{lang}*" if lang in non_latin else lang

def order_languages_by_family(languages, lang_families):
    """Order languages by family grouping"""
    ordered = []
    for family, family_langs in lang_families.items():
        for lang in family_langs:
            if lang in languages:
                ordered.append(lang)
    
    # Add any remaining languages not in families
    for lang in languages:
        if lang not in ordered:
            ordered.append(lang)
    
    return ordered

def get_family_positions(ordered_langs, lang_families):
    """Get positions where family boundaries should be drawn"""
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

def add_family_separators_and_labels(ax, ordered_langs, lang_families, axis='both'):
    """Add family separators and labels to the plot"""
    positions = get_family_positions(ordered_langs, lang_families)
    
    # Add separator lines
    if axis in ['both', 'x']:
        for pos in positions:
            ax.axvline(x=pos, color='black', linewidth=2, alpha=0.8)
    if axis in ['both', 'y']:
        for pos in positions:
            ax.axhline(y=pos, color='black', linewidth=2, alpha=0.8)
    
    # Add family labels
    current_pos = 0
    for family, family_langs in lang_families.items():
        family_count = sum(1 for lang in family_langs if lang in ordered_langs)
        if family_count > 0:
            center_pos = current_pos + (family_count - 1) / 2
            
            if axis in ['both', 'x']:
                ax.text(center_pos, -0.7, family, ha='center', va='center', 
                       fontsize=11, fontweight='bold', rotation=0)
            
            current_pos += family_count

# Load activation mask
print(f"Loading activation mask from: {args.input_path}")
final_indice = torch.load(f"{args.input_path}")

num_languages = len(final_indice)
num_layers = len(final_indice[0])

print(f"Loaded data for {num_languages} languages and {num_layers} layers")

# Build sets of (layer, neuron) pairs per language
lang_neuron_sets = []
for lang_index in range(num_languages):
    neuron_set = set()
    for layer, heads in enumerate(final_indice[lang_index]):
        for head in heads.tolist():
            neuron_set.add((layer, head))
    lang_neuron_sets.append(neuron_set)

print("Built neuron sets for all languages")

# Get language families and order languages
lang_families = get_language_families()
ordered_languages = order_languages_by_family(languages, lang_families)
ordered_indices = [languages.index(lang) for lang in ordered_languages]

print(f"Language order: {ordered_languages}")

# === Plot 1: Family-Grouped Overlap Heatmap ===
print("Creating family-grouped overlap heatmap...")

# Create overlap matrix with ordered languages
overlap_matrix = np.zeros((num_languages, num_languages), dtype=int)
for i, lang_i_idx in enumerate(ordered_indices):
    for j, lang_j_idx in enumerate(ordered_indices):
        intersection = len(lang_neuron_sets[lang_i_idx] & lang_neuron_sets[lang_j_idx])
        overlap_matrix[i, j] = intersection

# Create formatted labels with asterisks
formatted_labels = [format_language_label(lang) for lang in ordered_languages]

fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(
    overlap_matrix,
    xticklabels=formatted_labels,
    yticklabels=formatted_labels,
    cmap="Oranges",
    annot=True,
    fmt="d",
    cbar=True,
    linewidths=0.5,
    annot_kws={"size": 10},
    square=True,
    ax=ax
)

# Add family separators and labels
add_family_separators_and_labels(ax, ordered_languages, lang_families, 'both')

# Customize colorbar
cbar = ax.collections[0].colorbar
cbar.ax.set_ylabel("Neuron Overlap Count", fontsize=14, labelpad=15)
cbar.ax.tick_params(labelsize=12)

plt.xticks(rotation=45, ha="right", fontsize=12)
plt.yticks(rotation=0, fontsize=12)

# Add border around the heatmap
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1.5)
    spine.set_visible(True)

plt.tight_layout()
plt.savefig(f"{args.output_path}/language_overlap_family_grouped.png", dpi=300, bbox_inches='tight')
plt.close()

# === Plot 2: Cumulative Distribution Across All Languages ===
print("Creating cumulative distribution plot...")

# Calculate cumulative neuron counts across all languages
layer_counts_all = np.zeros(num_layers)
for lang_index in range(num_languages):
    for layer, heads in enumerate(final_indice[lang_index]):
        layer_counts_all[layer] += len(heads)

plt.figure(figsize=(12, 6))
bars = plt.bar(range(num_layers), layer_counts_all, color='steelblue', 
               edgecolor='black', linewidth=0.8, alpha=0.8)

# Add value labels on bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
             f'{int(height)}', ha='center', va='bottom', fontsize=10)

plt.xlabel("Layer Index", fontsize=14, fontweight='bold')
plt.ylabel("Neuron Count", fontsize=14, fontweight='bold')
plt.xticks(range(num_layers), fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='y', alpha=0.3, linestyle='--')

# Add thicker frame
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(1.2)
    spine.set_edgecolor('black')

plt.tight_layout()
plt.savefig(f"{args.output_path}/cumulative_neuron_distribution.png", dpi=300, bbox_inches='tight')
plt.close()

# === Plot 3: Comparative Line Plot for All Languages ===
print("Creating comparative line plot...")

# Prepare data for line plot
layer_counts_by_lang = np.zeros((num_languages, num_layers))
for lang_index in range(num_languages):
    for layer, heads in enumerate(final_indice[lang_index]):
        layer_counts_by_lang[lang_index, layer] = len(heads)

plt.figure(figsize=(14, 8))

# Plot lines for each language using custom colors
for lang_index, lang in enumerate(languages):
    linestyle = '--' if lang in get_non_latin_scripts() else '-'
    linewidth = 2.5 if lang in get_non_latin_scripts() else 2
    
    plt.plot(range(num_layers), layer_counts_by_lang[lang_index], 
            color=lang_colors[lang], linestyle=linestyle, linewidth=linewidth,
            label=format_language_label(lang), alpha=0.8)

plt.xlabel("Layer Index", fontsize=14, fontweight='bold')
plt.ylabel("Neuron Count", fontsize=14, fontweight='bold')
plt.xticks(range(num_layers), fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, alpha=0.3, linestyle='--')

# Create legend with frame
ax = plt.gca()
legend = ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', 
                   fontsize=10, ncol=1, frameon=True)
legend.set_title("", prop={'size': 12, 'weight': 'bold'})

# Add thicker frame
for spine in ax.spines.values():
    spine.set_linewidth(1.2)
    spine.set_edgecolor('black')

plt.tight_layout()
plt.savefig(f"{args.output_path}/comparative_line_plot.png", dpi=300, bbox_inches='tight')
plt.close()

# === Plot 4: Small Multiples - Individual Language Distributions ===
print("Creating small multiples plot...")

# Calculate grid dimensions
n_langs = len(languages)
cols = 7  # 7 columns for better layout
rows = (n_langs + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(20, 12))
if rows == 1:
    axes = axes.reshape(1, -1)

# Flatten axes for easier iteration
axes_flat = axes.flatten()

# Find global max for consistent y-axis scaling
global_max = 0
for lang_index in range(num_languages):
    neuron_counts = [len(heads) for heads in final_indice[lang_index]]
    global_max = max(global_max, max(neuron_counts))

# Create individual plots
for lang_index, lang in enumerate(languages):
    ax = axes_flat[lang_index]
    neuron_counts = [len(heads) for heads in final_indice[lang_index]]
    
    # Use custom color for each language
    lang_color = lang_colors[lang]
    
    # Create bar plot
    bars = ax.bar(range(num_layers), neuron_counts, color=lang_color, 
                  edgecolor='black', linewidth=0.5, alpha=0.8)
    
    # Formatting
    ax.set_title(format_language_label(lang), fontsize=12, fontweight='bold', pad=5)
    ax.set_ylim(0, global_max * 1.1)
    ax.set_xticks(range(0, num_layers, max(1, num_layers//5)))
    ax.tick_params(axis='both', which='major', labelsize=9)
    ax.grid(axis='y', alpha=0.3)
    
    # Add peak value annotation
    max_count = max(neuron_counts)
    max_layer = neuron_counts.index(max_count)
    ax.annotate(f'{max_count}', xy=(max_layer, max_count), 
                xytext=(max_layer, max_count + global_max*0.05),
                ha='center', va='bottom', fontsize=8, fontweight='bold')

# Add common axes labels
fig.text(0.5, 0.02, 'Layer Index', ha='center', va='center', fontsize=14, fontweight='bold')
fig.text(0.02, 0.5, 'Neuron Count', ha='center', va='center', 
         rotation=90, fontsize=14, fontweight='bold')

# Hide unused subplots
for i in range(n_langs, len(axes_flat)):
    axes_flat[i].set_visible(False)

# Add thicker frame to all subplots
for i in range(n_langs):
    for spine in axes_flat[i].spines.values():
        spine.set_linewidth(1.5)

plt.tight_layout()
plt.subplots_adjust(bottom=0.08, left=0.08, top=0.93)
plt.savefig(f"{args.output_path}/small_multiples_distribution.png", dpi=300, bbox_inches='tight')
plt.close()

print("All plots created successfully!")