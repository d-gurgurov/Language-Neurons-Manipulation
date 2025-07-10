import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import defaultdict
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import pandas as pd

# FastText language identification
try:
    import fasttext
    try:
        fasttext_model = fasttext.load_model('lid.176.bin')
    except:
        import urllib.request
        print("Downloading FastText language identification model...")
        urllib.request.urlretrieve('https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin', 'lid.176.bin')
        fasttext_model = fasttext.load_model('lid.176.bin')
    print("FastText language identification model loaded successfully")
except ImportError:
    print("Warning: FastText not available, falling back to rule-based language detection")
    fasttext_model = None

def get_test_questions():
    """Return test questions for each language"""
    return {
        "bo": ["Q: གནམ་གཤིས་ཇི་འདྲ་འདུག \nA:", "Q: ཁྱེད་རང་གི་མིང་ལ་ག་རེ་ཟེར་གྱི་འདུག \nA:", "Q: ད་རེས་ལོ་ཙམ་ག་ཚེས་རེད \nA:", "Q: ཁྱེད་རང་གི་དགའ་པོའི་ཁ་དོག་ག་རེ་རེད \nA:", "Q: གནམ་གཤིས་ཇི་འདྲ་འདུག \nA:", "Q: ཁྱེད་རང་གང་ནས་ཡིན \nA:"],
        "mt": ["Q: Kif int illum? \nA:", "Q: X'jisimek? \nA:", "Q: Liema sena aħna fiha? \nA:", "Q: Liema kulur tħobb l-aktar? \nA:", "Q: Kif hu t-temp? \nA:", "Q: Minn fejn int? \nA:"],
        "it": ["Q: Come stai oggi? \nA:", "Q: Come ti chiami? \nA:", "Q: Che anno è? \nA:", "Q: Qual è il tuo colore preferito? \nA:", "Q: Com'è il tempo? \nA:", "Q: Di dove sei? \nA:"],
        "es": ["Q: ¿Cómo estás hoy? \nA:", "Q: ¿Cómo te llamas? \nA:", "Q: ¿Qué año es? \nA:", "Q: ¿Cuál es tu color favorito? \nA:", "Q: ¿Cómo está el tiempo? \nA:", "Q: ¿De dónde eres? \nA:"],
        "de": ["Q: Wie geht es dir heute? \nA:", "Q: Wie heißt du? \nA:", "Q: Welches Jahr haben wir? \nA:", "Q: Was ist deine Lieblingsfarbe? \nA:", "Q: Wie ist das Wetter? \nA:", "Q: Woher kommst du? \nA:"],
        "ja": ["Q: 今日はどうですか？ \nA:", "Q: お名前は何ですか？ \nA:", "Q: 今年は何年ですか？ \nA:", "Q: 好きな色は何ですか？ \nA:", "Q: 天気はどうですか？ \nA:", "Q: どちらから来ましたか？ \nA:"],
        "ar": ["Q: كيف حالك اليوم؟ \nA:", "Q: ما اسمك؟ \nA:", "Q: ما هي السنة الحالية؟ \nA:", "Q: ما هو لونك المفضل؟ \nA:", "Q: كيف الطقس؟ \nA:", "Q: من أين أنت؟ \nA:"],
        "zh": ["Q: 你今天怎么样？ \nA:", "Q: 你叫什么名字？ \nA:", "Q: 现在是哪一年？ \nA:", "Q: 你最喜欢什么颜色？ \nA:", "Q: 天气怎么样？ \nA:", "Q: 你来自哪里？ \nA:"],
        "af": ["Q: Hoe gaan dit vandag? \nA:", "Q: Wat is jou naam? \nA:", "Q: Watter jaar is dit? \nA:", "Q: Wat is jou gunsteling kleur? \nA:", "Q: Hoe is die weer? \nA:", "Q: Waarvandaan kom jy? \nA:"],
        "nl": ["Q: Hoe gaat het vandaag? \nA:", "Q: Wat is je naam? \nA:", "Q: Welk jaar is het? \nA:", "Q: Wat is je favoriete kleur? \nA:", "Q: Hoe is het weer? \nA:", "Q: Waar kom je vandaan? \nA:"],
        "fr": ["Q: Comment ça va aujourd'hui? \nA:", "Q: Comment tu t'appelles? \nA:", "Q: Quelle année sommes-nous? \nA:", "Q: Quelle est ta couleur préférée? \nA:", "Q: Quel temps fait-il? \nA:", "Q: D'où viens-tu? \nA:"],
        "pt": ["Q: Como você está hoje? \nA:", "Q: Qual é o seu nome? \nA:", "Q: Que ano é este? \nA:", "Q: Qual é a sua cor favorita? \nA:", "Q: Como está o tempo? \nA:", "Q: De onde você é? \nA:"],
        "ru": ["Q: Как дела сегодня? \nA:", "Q: Как тебя зовут? \nA:", "Q: Какой сейчас год? \nA:", "Q: Какой твой любимый цвет? \nA:", "Q: Какая погода? \nA:", "Q: Откуда ты? \nA:"],
        "ko": ["Q: 오늘 어떻게 지내세요? \nA:", "Q: 이름이 뭐예요? \nA:", "Q: 올해가 몇 년도예요? \nA:", "Q: 좋아하는 색깔이 뭐예요? \nA:", "Q: 날씨가 어때요? \nA:", "Q: 어디서 왔어요? \nA:"],
        "hi": ["Q: आज आप कैसे हैं? \nA:", "Q: आपका नाम क्या है? \nA:", "Q: यह कौन सा साल है? \nA:", "Q: आपका पसंदीदा रंग क्या है? \nA:", "Q: मौसम कैसा है? \nA:", "Q: आप कहाँ से हैं? \nA:"],
        "tr": ["Q: Bugün nasılsın? \nA:", "Q: Adın ne? \nA:", "Q: Hangi yıldayız? \nA:", "Q: En sevdiğin renk nedir? \nA:", "Q: Hava nasıl? \nA:", "Q: Nerelisin? \nA:"],
        "pl": ["Q: Jak się masz dzisiaj? \nA:", "Q: Jak masz na imię? \nA:", "Q: Który to rok? \nA:", "Q: Jaki jest twój ulubiony kolor? \nA:", "Q: Jaka jest pogoda? \nA:", "Q: Skąd jesteś? \nA:"],
        "sv": ["Q: Hur mår du idag? \nA:", "Q: Vad heter du? \nA:", "Q: Vilket år är det? \nA:", "Q: Vilken är din favoritfärg? \nA:", "Q: Hur är vädret? \nA:", "Q: Var kommer du ifrån? \nA:"],
        "da": ["Q: Hvordan har du det i dag? \nA:", "Q: Hvad hedder du? \nA:", "Q: Hvilket år er det? \nA:", "Q: Hvad er din yndlingsfarve? \nA:", "Q: Hvordan er vejret? \nA:", "Q: Hvor kommer du fra? \nA:"],
        "no": ["Q: Hvordan har du det i dag? \nA:", "Q: Hva heter du? \nA:", "Q: Hvilket år er det? \nA:", "Q: Hva er favorittfargen din? \nA:", "Q: Hvordan er været? \nA:", "Q: Hvor kommer du fra? \nA:"],
        "en": ["Q: How are you today? \nA:", "Q: What is your name? \nA:", "Q: What year is it? \nA:", "Q: What is your favorite color? \nA:", "Q: What is the weather like? \nA:", "Q: Where are you from? \nA:"],
    }

def detect_language(text):
    """Detect language using FastText or simple fallback"""
    if fasttext_model is None:
        # Simple rule-based fallback
        if any(ord(c) > 127 for c in text):
            return "non-english"
        return "en"
    
    # Clean text for FastText
    text = text.replace('\n', ' ').strip()
    if not text:
        return "unknown"
    
    predictions = fasttext_model.predict(text, k=1)
    lang_code = predictions[0][0].replace('__label__', '')
    confidence = predictions[1][0]
    
    return lang_code # else "unknown" #  if confidence > 0.5

def get_layer_predictions(model, tokenizer, input_text, num_layers, device):
    """Get predictions from each layer using logit lens"""
    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    
    # Get hidden states from all layers
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        hidden_states = outputs.hidden_states  # (layer, batch, seq_len, hidden_dim)
    
    # Get the final layer norm and LM head
    final_layer_norm = model.model.norm if hasattr(model.model, 'norm') else model.model.final_layernorm
    lm_head = model.lm_head
    
    layer_predictions = []
    
    # For each layer, apply final layer norm and LM head to get logits
    for layer_idx in range(num_layers + 1):  # +1 for embedding layer
        if layer_idx == 0:
            # Skip embedding layer for now
            continue
            
        # Get hidden states from this layer
        layer_hidden = hidden_states[layer_idx]  # (batch, seq_len, hidden_dim)
        
        # Apply final layer norm
        normalized_hidden = final_layer_norm(layer_hidden)
        
        # Apply LM head to get logits
        logits = lm_head(normalized_hidden)  # (batch, seq_len, vocab_size)
        
        # Get prediction for the last token (where we'd generate next)
        last_token_logits = logits[0, -1, :]  # (vocab_size,)
        
        # Get top-k predictions
        top_k = 5
        top_probs, top_indices = torch.topk(F.softmax(last_token_logits, dim=-1), top_k)
        
        # Convert to tokens
        top_tokens = [tokenizer.decode(idx) for idx in top_indices]
        
        layer_predictions.append({
            'layer': layer_idx,
            'top_tokens': top_tokens,
            'top_probs': top_probs.detach().cpu().numpy(),
            'top_indices': top_indices.detach().cpu().numpy()
        })
    
    return layer_predictions

def analyze_language_evolution(layer_predictions, target_language):
    """Analyze how language predictions evolve across layers"""
    results = []
    
    for layer_data in layer_predictions:
        layer_idx = layer_data['layer']
        top_tokens = layer_data['top_tokens']
        top_probs = layer_data['top_probs']
        
        # Detect language for each top token
        token_languages = []
        for token in top_tokens:
            lang = detect_language(token)
            token_languages.append(lang)
        
        # Calculate language distribution
        lang_dist = defaultdict(float)
        for lang, prob in zip(token_languages, top_probs):
            lang_dist[lang] += prob
        
        # Check if target language is in top predictions
        target_lang_prob = lang_dist.get(target_language, 0.0)
        english_prob = lang_dist.get('en', 0.0)
        
        results.append({
            'layer': layer_idx,
            'target_lang_prob': target_lang_prob,
            'english_prob': english_prob,
            'lang_distribution': dict(lang_dist),
            'top_tokens': top_tokens,
            'top_probs': top_probs
        })
    
    return results

def create_language_evolution_plots(all_results, output_dir):
    """Create separate publication-quality plots for language evolution analysis"""
    
    # Set publication style
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'axes.linewidth': 1.2,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'figure.dpi': 300
    })
    
    # Prepare data for plotting
    languages = list(all_results.keys())
    num_layers = len(all_results[languages[0]][0])  # Assuming all have same number of layers
    
    # Define comprehensive color palette for up to 21 languages
    # Using a combination of qualitative color maps for maximum distinctiveness
    colors1 = plt.cm.tab10(np.linspace(0, 1, 10))  # 10 colors
    colors2 = plt.cm.Set3(np.linspace(0, 1, 12))    # 12 colors (avoiding too light ones)
    colors3 = plt.cm.Dark2(np.linspace(0, 1, 8))    # 8 colors
    
    # Combine and select most distinct colors
    all_colors = np.vstack([colors1, colors2[:8], colors3[:3]])  # Total: 21 colors
    
    # Manually curate for maximum distinction and readability
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
        '#8B4513'
    ]
    
    # Create language to color mapping
    lang_colors = {}
    for i, lang in enumerate(languages):
        lang_colors[lang] = distinct_colors[i % len(distinct_colors)]
    
    # Plot 1: Target Language Probability Evolution
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    for lang in languages:
        layers = []
        target_probs = []
        for question_results in all_results[lang]:
            for layer_result in question_results:
                layers.append(layer_result['layer'])
                target_probs.append(layer_result['target_lang_prob'])
        
        # Group by layer and calculate statistics
        layer_means = defaultdict(list)
        for layer, prob in zip(layers, target_probs):
            layer_means[layer].append(prob)
        
        layer_indices = sorted(layer_means.keys())
        mean_probs = [np.mean(layer_means[layer]) for layer in layer_indices]
        # std_probs = [np.std(layer_means[layer]) for layer in layer_indices]  # Commented out
        
        # Plot without error bars
        ax.plot(layer_indices, mean_probs, marker='o', label=f'{lang.upper()}', 
                linewidth=2.5, markersize=6, color=lang_colors[lang])
        # Commented out fill_between for std
        # ax.fill_between(layer_indices, 
        #                np.array(mean_probs) - np.array(std_probs),
        #                np.array(mean_probs) + np.array(std_probs),
        #                alpha=0.2, color=lang_colors[lang])
    
    ax.set_xlabel('Layer Index', fontweight='bold')
    ax.set_ylabel('Target Language Probability', fontweight='bold')
    # ax.set_title('Target Language Probability Evolution Across Layers', fontweight='bold', pad=20)  # Commented out
    
    # Create a legend with languages stacked vertically (single column)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0.5, num_layers + 0.5)
    ax.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'target_language_evolution.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Plot 2: English Probability Evolution
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    for lang in languages:
        layers = []
        english_probs = []
        for question_results in all_results[lang]:
            for layer_result in question_results:
                layers.append(layer_result['layer'])
                english_probs.append(layer_result['english_prob'])
        
        # Group by layer and calculate statistics
        layer_means = defaultdict(list)
        for layer, prob in zip(layers, english_probs):
            layer_means[layer].append(prob)
        
        layer_indices = sorted(layer_means.keys())
        mean_probs = [np.mean(layer_means[layer]) for layer in layer_indices]
        # std_probs = [np.std(layer_means[layer]) for layer in layer_indices]  # Commented out
        
        # Plot without error bars
        ax.plot(layer_indices, mean_probs, marker='s', label=f'{lang.upper()}', 
                linewidth=2.5, markersize=6, color=lang_colors[lang])
        # Commented out fill_between for std
        # ax.fill_between(layer_indices, 
        #                np.array(mean_probs) - np.array(std_probs),
        #                np.array(mean_probs) + np.array(std_probs),
        #                alpha=0.2, color=lang_colors[lang])
    
    ax.set_xlabel('Layer Index', fontweight='bold')
    ax.set_ylabel('English Probability', fontweight='bold')
    # ax.set_title('English Language Probability Evolution Across Layers', fontweight='bold', pad=20)  # Commented out
    
    # Create a legend with languages stacked vertically (single column)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0.5, num_layers + 0.5)
    ax.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'english_probability_evolution.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Plot 3: Language Transition Heatmap
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Create heatmap data
    heatmap_data = []
    for lang in languages:
        lang_layer_probs = []
        for layer_idx in range(1, num_layers + 1):
            layer_probs = []
            for question_results in all_results[lang]:
                for layer_result in question_results:
                    if layer_result['layer'] == layer_idx:
                        layer_probs.append(layer_result['target_lang_prob'])
            if layer_probs:
                lang_layer_probs.append(np.mean(layer_probs))
            else:
                lang_layer_probs.append(0.0)
        heatmap_data.append(lang_layer_probs)
    
    heatmap_data = np.array(heatmap_data)
    
    im = ax.imshow(heatmap_data, cmap='viridis', aspect='auto', interpolation='nearest')
    
    # Set ticks and labels - show all layers
    ax.set_xticks(range(num_layers))
    ax.set_xticklabels(range(1, num_layers + 1), rotation=45)
    ax.set_yticks(range(len(languages)))
    ax.set_yticklabels([lang.upper() for lang in languages])
    
    ax.set_xlabel('Layer Index', fontweight='bold')
    ax.set_ylabel('Language', fontweight='bold')
    # ax.set_title('Target Language Probability Heatmap Across Layers', fontweight='bold', pad=20)  # Commented out
    
    # Add colorbar with better formatting
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Target Language Probability', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'language_transition_heatmap.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Plot 4: Language Diversity (Entropy) Evolution
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Calculate entropy of language distributions per layer
    layer_entropies = defaultdict(list)
    for lang in languages:
        for question_results in all_results[lang]:
            for layer_result in question_results:
                layer_idx = layer_result['layer']
                lang_dist = layer_result['lang_distribution']
                
                # Calculate entropy
                probs = list(lang_dist.values())
                if probs:
                    entropy = -sum(p * np.log(p + 1e-10) for p in probs if p > 0)
                    layer_entropies[layer_idx].append(entropy)
    
    layer_indices = sorted(layer_entropies.keys())
    mean_entropies = [np.mean(layer_entropies[layer]) for layer in layer_indices]
    std_entropies = [np.std(layer_entropies[layer]) for layer in layer_indices]
    
    ax.plot(layer_indices, mean_entropies, marker='d', linewidth=3, 
            markersize=8, color='darkviolet', label='Mean Entropy')
    ax.fill_between(layer_indices, 
                   np.array(mean_entropies) - np.array(std_entropies),
                   np.array(mean_entropies) + np.array(std_entropies),
                   alpha=0.3, color='darkviolet', label='±1 Std Dev')
    
    ax.set_xlabel('Layer Index', fontweight='bold')
    ax.set_ylabel('Language Entropy (nats)', fontweight='bold')
    # ax.set_title('Language Prediction Diversity Across Layers', fontweight='bold', pad=20)  # Commented out
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0.5, num_layers + 0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'language_diversity_evolution.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Save color mapping for reference
    color_mapping = {lang: lang_colors[lang] for lang in languages}
    with open(os.path.join(output_dir, 'color_mapping.json'), 'w') as f:
        json.dump(color_mapping, f, indent=2)
    
    # Reset matplotlib parameters
    plt.rcParams.update(plt.rcParamsDefault)
    
    return lang_colors

def create_detailed_token_plot(results, language, output_dir, lang_colors):
    """Create improved detailed plot showing top tokens and their probabilities for each layer"""
    
    # Set publication style
    plt.rcParams.update({
        'font.size': 10,
        'font.family': 'serif',
        'axes.linewidth': 1.2,
        'figure.dpi': 300
    })
    
    num_questions = len(results)
    
    # Create separate plots for each question to avoid overcrowding
    for q_idx, question_results in enumerate(results):
        
        # Determine figure size based on number of layers
        num_layers = len(question_results)
        fig_width = max(16, num_layers * 0.8)  # Scale width with layers
        fig_height = 8
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(fig_width, fig_height))
        fig.suptitle(f'{language.upper()} - Question {q_idx + 1}: Top Token Evolution', 
                     fontsize=16, fontweight='bold', y=0.95)
        
        # Prepare data
        layers = [r['layer'] for r in question_results]
        max_tokens = 5  # Show top 5 tokens
        
        # Top subplot: Token text with probability coloring
        token_texts = []
        prob_values = []
        
        for layer_result in question_results:
            layer_tokens = layer_result['top_tokens'][:max_tokens]
            layer_probs = layer_result['top_probs'][:max_tokens]
            
            # Pad if necessary
            while len(layer_tokens) < max_tokens:
                layer_tokens.append("")
                layer_probs = np.append(layer_probs, 0.0)
            
            token_texts.append(layer_tokens)
            prob_values.append(layer_probs)
        
        prob_values = np.array(prob_values).T  # Shape: (max_tokens, num_layers)
        
        # Create probability heatmap
        im1 = ax1.imshow(prob_values, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
        
        # Set ticks and labels for probability plot
        layer_step = max(1, len(layers) // 15)  # Show at most 15 layer labels
        ax1.set_xticks(np.arange(0, len(layers), layer_step))
        ax1.set_xticklabels([str(layers[i]) for i in range(0, len(layers), layer_step)])
        ax1.set_yticks(range(max_tokens))
        ax1.set_yticklabels([f'Top {i+1}' for i in range(max_tokens)])
        ax1.set_xlabel('Layer Index', fontweight='bold')
        ax1.set_ylabel('Token Rank', fontweight='bold')
        ax1.set_title('Token Probability Heatmap', fontweight='bold', pad=15)
        
        # Add colorbar for probabilities
        cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
        cbar1.set_label('Probability', fontweight='bold')
        
        # Add token text annotations on the probability heatmap
        for i in range(max_tokens):
            for j, layer_result in enumerate(question_results):
                if i < len(layer_result['top_tokens']):
                    token = layer_result['top_tokens'][i]
                    prob = layer_result['top_probs'][i]
                    
                    # Clean and truncate token for display
                    display_token = token.replace('\n', '\\n').replace('\t', '\\t')
                    if len(display_token) > 8:
                        display_token = display_token[:8] + "..."
                    
                    # Choose text color based on probability
                    text_color = 'white' if prob > 0.5 else 'black'
                    
                    ax1.text(j, i, display_token, ha='center', va='center', 
                            fontsize=8, color=text_color, fontweight='bold')
        
        # Bottom subplot: Language detection visualization
        lang_evolution = []
        layer_indices = []
        
        for layer_result in question_results:
            layer_idx = layer_result['layer']
            lang_dist = layer_result['lang_distribution']
            
            # Get top 3 languages for this layer
            sorted_langs = sorted(lang_dist.items(), key=lambda x: x[1], reverse=True)[:3]
            
            lang_evolution.append({
                'layer': layer_idx,
                'top_langs': sorted_langs
            })
            layer_indices.append(layer_idx)
        
        # Create language evolution line plot
        target_lang_probs = [layer_result['target_lang_prob'] for layer_result in question_results]
        english_probs = [layer_result['english_prob'] for layer_result in question_results]
        
        # Use consistent colors from the main plots
        target_color = lang_colors.get(language, 'darkgreen')
        
        ax2.plot(layer_indices, target_lang_probs, marker='o', linewidth=3, 
                markersize=6, label=f'Target ({language.upper()})', color=target_color)
        ax2.plot(layer_indices, english_probs, marker='s', linewidth=3, 
                markersize=6, label='English', color='#8B0000')  # Dark red for English
        
        ax2.set_xlabel('Layer Index', fontweight='bold')
        ax2.set_ylabel('Language Probability', fontweight='bold')
        ax2.set_title('Language Probability Evolution', fontweight='bold', pad=15)
        ax2.legend(fontsize=12)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_ylim(-0.05, 1.05)
        
        # Highlight transition points
        max_diff_idx = np.argmax(np.abs(np.diff(target_lang_probs)))
        if max_diff_idx < len(layer_indices) - 1:
            ax2.axvline(x=layer_indices[max_diff_idx], color='orange', 
                       linestyle='--', alpha=0.7, linewidth=2, 
                       label='Max Change Point')
            ax2.legend(fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'detailed_tokens_{language}_q{q_idx+1}.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    # Create summary plot for all questions of this language
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Use distinct colors for different questions, but related to the language color
    base_color = lang_colors.get(language, '#1f77b4')
    question_colors = [base_color, 
                      plt.cm.Set2(0.2),  # Different shade
                      plt.cm.Set2(0.4)]  # Another shade
    
    for q_idx, question_results in enumerate(results):
        layer_indices = [r['layer'] for r in question_results]
        target_lang_probs = [r['target_lang_prob'] for r in question_results]
        
        ax.plot(layer_indices, target_lang_probs, marker='o', linewidth=2.5, 
               markersize=5, label=f'Question {q_idx+1}', 
               color=question_colors[q_idx % len(question_colors)], alpha=0.8)
    
    ax.set_xlabel('Layer Index', fontweight='bold')
    ax.set_ylabel('Target Language Probability', fontweight='bold')
    ax.set_title(f'{language.upper()} - Target Language Probability Across All Questions', 
                fontweight='bold', pad=20)
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'summary_{language}.png'), 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    

    # Set publication style
    plt.rcParams.update({
        'font.size': 10,
        'font.family': 'serif',
        'axes.linewidth': 1.2,
        'figure.dpi': 300
    })
    
    num_questions = len(results)
    
    # Create separate plots for each question to avoid overcrowding
    for q_idx, question_results in enumerate(results):
        
        # Determine figure size based on number of layers
        num_layers = len(question_results)
        fig_width = max(16, num_layers * 0.8)  # Scale width with layers
        fig_height = 8
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(fig_width, fig_height))
        fig.suptitle(f'{language.upper()} - Question {q_idx + 1}: Top Token Evolution', 
                     fontsize=16, fontweight='bold', y=0.95)
        
        # Prepare data
        layers = [r['layer'] for r in question_results]
        max_tokens = 5  # Show top 5 tokens
        
        # Top subplot: Token text with probability coloring
        token_texts = []
        prob_values = []
        
        for layer_result in question_results:
            layer_tokens = layer_result['top_tokens'][:max_tokens]
            layer_probs = layer_result['top_probs'][:max_tokens]
            
            # Pad if necessary
            while len(layer_tokens) < max_tokens:
                layer_tokens.append("")
                layer_probs = np.append(layer_probs, 0.0)
            
            token_texts.append(layer_tokens)
            prob_values.append(layer_probs)
        
        prob_values = np.array(prob_values).T  # Shape: (max_tokens, num_layers)
        
        # Create probability heatmap
        im1 = ax1.imshow(prob_values, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
        
        # Set ticks and labels for probability plot
        layer_step = max(1, len(layers) // 15)  # Show at most 15 layer labels
        ax1.set_xticks(np.arange(0, len(layers), layer_step))
        ax1.set_xticklabels([str(layers[i]) for i in range(0, len(layers), layer_step)])
        ax1.set_yticks(range(max_tokens))
        ax1.set_yticklabels([f'Top {i+1}' for i in range(max_tokens)])
        ax1.set_xlabel('Layer Index', fontweight='bold')
        ax1.set_ylabel('Token Rank', fontweight='bold')
        ax1.set_title('Token Probability Heatmap', fontweight='bold', pad=15)
        
        # Add colorbar for probabilities
        cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
        cbar1.set_label('Probability', fontweight='bold')
        
        # Add token text annotations on the probability heatmap
        for i in range(max_tokens):
            for j, layer_result in enumerate(question_results):
                if i < len(layer_result['top_tokens']):
                    token = layer_result['top_tokens'][i]
                    prob = layer_result['top_probs'][i]
                    
                    # Clean and truncate token for display
                    display_token = token.replace('\n', '\\n').replace('\t', '\\t')
                    if len(display_token) > 8:
                        display_token = display_token[:8] + "..."
                    
                    # Choose text color based on probability
                    text_color = 'white' if prob > 0.5 else 'black'
                    
                    ax1.text(j, i, display_token, ha='center', va='center', 
                            fontsize=8, color=text_color, fontweight='bold')
        
        # Bottom subplot: Language detection visualization
        lang_evolution = []
        layer_indices = []
        
        for layer_result in question_results:
            layer_idx = layer_result['layer']
            lang_dist = layer_result['lang_distribution']
            
            # Get top 3 languages for this layer
            sorted_langs = sorted(lang_dist.items(), key=lambda x: x[1], reverse=True)[:3]
            
            lang_evolution.append({
                'layer': layer_idx,
                'top_langs': sorted_langs
            })
            layer_indices.append(layer_idx)
        
        # Create language evolution line plot
        target_lang_probs = [layer_result['target_lang_prob'] for layer_result in question_results]
        english_probs = [layer_result['english_prob'] for layer_result in question_results]
        
        ax2.plot(layer_indices, target_lang_probs, marker='o', linewidth=3, 
                markersize=6, label=f'Target ({language.upper()})', color='darkgreen')
        ax2.plot(layer_indices, english_probs, marker='s', linewidth=3, 
                markersize=6, label='English', color='darkred')
        
        ax2.set_xlabel('Layer Index', fontweight='bold')
        ax2.set_ylabel('Language Probability', fontweight='bold')
        ax2.set_title('Language Probability Evolution', fontweight='bold', pad=15)
        ax2.legend(fontsize=12)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_ylim(-0.05, 1.05)
        
        # Highlight transition points
        max_diff_idx = np.argmax(np.abs(np.diff(target_lang_probs)))
        if max_diff_idx < len(layer_indices) - 1:
            ax2.axvline(x=layer_indices[max_diff_idx], color='orange', 
                       linestyle='--', alpha=0.7, linewidth=2, 
                       label='Max Change Point')
            ax2.legend(fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'detailed_tokens_{language}_q{q_idx+1}.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    # Create summary plot for all questions of this language
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    colors = ['darkgreen', 'darkblue', 'darkred']
    for q_idx, question_results in enumerate(results):
        layer_indices = [r['layer'] for r in question_results]
        target_lang_probs = [r['target_lang_prob'] for r in question_results]
        
        ax.plot(layer_indices, target_lang_probs, marker='o', linewidth=2.5, 
               markersize=5, label=f'Question {q_idx+1}', 
               color=colors[q_idx % len(colors)], alpha=0.8)
    
    ax.set_xlabel('Layer Index', fontweight='bold')
    ax.set_ylabel('Target Language Probability', fontweight='bold')
    ax.set_title(f'{language.upper()} - Target Language Probability Across All Questions', 
                fontweight='bold', pad=20)
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'summary_{language}.png'), 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Reset matplotlib parameters
    plt.rcParams.update(plt.rcParamsDefault)

def main():
    parser = argparse.ArgumentParser(description='Logit Lens Analysis for Language Evolution')
    parser.add_argument("-m", "--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct",
                        help="Model name or path")
    parser.add_argument("--output", type=str, default="logit_lens_results",
                        help="Output directory for results")
    parser.add_argument("--languages", nargs='+', 
                        default=["bo", "mt", "it", "es", "de", "ja", "ar", "zh", "af", "nl", "fr", "pt", "ru", "ko", "hi", "tr", "pl", "sv", "da", "no", "en"],
                        help="Languages to analyze")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use (auto, cuda, cpu)")
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load model and tokenizer
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device
    )
    
    # Get model info
    num_layers = len(model.model.layers)
    print(f"Model has {num_layers} layers")
    
    # Get test questions
    test_questions = get_test_questions()
    
    # Analyze each language
    all_results = {}
    
    for language in args.languages:
        print(f"\nAnalyzing language: {language}")
        
        if language not in test_questions:
            print(f"Warning: No test questions for language {language}")
            continue
        
        language_results = []
        questions = test_questions[language]
        
        for q_idx, question in enumerate(tqdm(questions, desc=f"Processing {language} questions")):
            print(f"  Question {q_idx + 1}: {question[:50]}...")
            
            # Get layer predictions
            layer_predictions = get_layer_predictions(model, tokenizer, question, num_layers, device)
            
            # Analyze language evolution
            evolution_results = analyze_language_evolution(layer_predictions, language)
            
            language_results.append(evolution_results)
        
        all_results[language] = language_results
    
    # Create overall language evolution plots (separate) and get color mapping
    lang_colors = create_language_evolution_plots(all_results, args.output)
    
    # Create detailed plots for each language with consistent colors
    # for language in all_results.keys():
    #     create_detailed_token_plot(all_results[language], language, args.output, lang_colors)
    
    # Save results to JSON
    # Convert numpy arrays to lists for JSON serialization
    json_results = {}
    for lang, lang_results in all_results.items():
        json_results[lang] = []
        for question_results in lang_results:
            json_question = []
            for layer_result in question_results:
                json_layer = {
                    'layer': layer_result['layer'],
                    'target_lang_prob': float(layer_result['target_lang_prob']),
                    'english_prob': float(layer_result['english_prob']),
                    'lang_distribution': {k: float(v) for k, v in layer_result['lang_distribution'].items()},
                    'top_tokens': layer_result['top_tokens'],
                    'top_probs': layer_result['top_probs'].tolist()
                }
                json_question.append(json_layer)
            json_results[lang].append(json_question)
    
    with open(os.path.join(args.output, 'logit_lens_results.json'), 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    
    print(f"\nAnalysis complete! Results saved to {args.output}")
    print(f"Generated publication-quality plots:")
    print(f"  - target_language_evolution.png: Target language probability across layers")
    print(f"  - english_probability_evolution.png: English probability across layers") 
    print(f"  - language_transition_heatmap.png: Heatmap of language transitions")
    print(f"  - language_diversity_evolution.png: Language entropy/diversity evolution")
    for language in args.languages:
        if language in all_results:
            print(f"  - detailed_tokens_{language}_q*.png: Per-question token analysis for {language}")
            print(f"  - summary_{language}.png: Summary plot for {language}")

if __name__ == "__main__":
    main()