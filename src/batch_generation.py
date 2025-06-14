import argparse
import json
import os
from types import MethodType

import torch
import torch.nn.functional as F
from vllm import LLM, SamplingParams

def get_test_questions():
    """Return test questions for each language"""
    return {
        "bo": [  # Tibetan
            "Q: གནམ་གཤིས་ཇི་འདྲ་འདུག \nA:",
            "Q: ཁྱེད་རང་གི་མིང་ལ་ག་རེ་ཟེར་གྱི་འདུག \nA:",
            "Q: ད་རེས་ལོ་ཙམ་ག་ཚེས་རེད \nA:"
        ],
        "mt": ["Q: Kif int illum? \nA:", "Q: X'jisimek? \nA:", "Q: Liema sena aħna fiha? \nA:"],
        "it": ["Q: Come stai oggi? \nA:", "Q: Come ti chiami? \nA:", "Q: Che anno è? \nA:"],
        "es": ["Q: ¿Cómo estás hoy? \nA:", "Q: ¿Cómo te llamas? \nA:", "Q: ¿Qué año es? \nA:"],
        "de": ["Q: Wie geht es dir heute? \nA:", "Q: Wie heißt du? \nA:", "Q: Welches Jahr haben wir? \nA:"],
        "ja": ["Q: 今日はどうですか？ \nA:", "Q: お名前は何ですか？ \nA:", "Q: 今年は何年ですか？ \nA:"],
        "ar": ["Q: كيف حالك اليوم؟ \nA:", "Q: ما اسمك؟ \nA:", "Q: ما هي السنة الحالية؟ \nA:"],
        "zh": ["Q: 你今天怎么样？ \nA:", "Q: 你叫什么名字？ \nA:", "Q: 现在是哪一年？ \nA:"],
        "af": ["Q: Hoe gaan dit vandag? \nA:", "Q: Wat is jou naam? \nA:", "Q: Watter jaar is dit? \nA:"],
        "nl": ["Q: Hoe gaat het vandaag? \nA:", "Q: Wat is je naam? \nA:", "Q: Welk jaar is het? \nA:"],
        "fr": ["Q: Comment ça va aujourd'hui? \nA:", "Q: Comment tu t'appelles? \nA:", "Q: Quelle année sommes-nous? \nA:"],
        "pt": ["Q: Como você está hoje? \nA:", "Q: Qual é o seu nome? \nA:", "Q: Que ano é este? \nA:"],
        "ru": ["Q: Как дела сегодня? \nA:", "Q: Как тебя зовут? \nA:", "Q: Какой сейчас год? \nA:"],
        "ko": ["Q: 오늘 어떻게 지내세요? \nA:", "Q: 이름이 뭐예요? \nA:", "Q: 올해가 몇 년도예요? \nA:"],
        "hi": ["Q: आज आप कैसे हैं? \nA:", "Q: आपका नाम क्या है? \nA:", "Q: यह कौन सा साल है? \nA:"],
        "tr": ["Q: Bugün nasılsın? \nA:", "Q: Adın ne? \nA:", "Q: Hangi yıldayız? \nA:"],
        "pl": ["Q: Jak się masz dzisiaj? \nA:", "Q: Jak masz na imię? \nA:", "Q: Który to rok? \nA:"],
        "sv": ["Q: Hur mår du idag? \nA:", "Q: Vad heter du? \nA:", "Q: Vilket år är det? \nA:"],
        "da": ["Q: Hvordan har du det i dag? \nA:", "Q: Hvad hedder du? \nA:", "Q: Hvilket år er det? \nA:"],
        "no": ["Q: Hvordan har du det i dag? \nA:", "Q: Hva heter du? \nA:", "Q: Hvilket år er det? \nA:"],
        "en": ["Q: How are you today? \nA:", "Q: What is your name? \nA:", "Q: What year is it? \nA:"],
    }

def compute_average_activations():
    """Compute average activation values for each language"""
    n, over_zero = [], []
    lang_names = ["bo", "mt", "it", "es", "de", "ja", "ar", "zh", "af", "nl", "fr", "pt", "ru", "ko", "hi", "tr", "pl", "sv", "da", "no"]
    
    for lang in lang_names:
        data = torch.load(f'data/activation.{lang}.train.llama-3')
        n.append(data['n'])
        over_zero.append(data['over_zero'])

    n = torch.tensor(n)
    over_zero = torch.stack(over_zero, dim=-1)
    
    # Compute average activation probabilities per language
    activation_probs = over_zero / n  # layer x inter x lang_num
    
    # Create a mapping from language name to index
    lang_to_idx = {lang: idx for idx, lang in enumerate(lang_names)}
    
    return activation_probs, lang_to_idx, lang_names

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="meta-llama/Meta-Llama-3-8B")
parser.add_argument("-a", "--activation_mask", type=str, default="activation_mask/llama-3")
parser.add_argument("--deactivate_lang", type=str, default="de", help="Language to deactivate (e.g., 'es' for Spanish, 'None' to skip deactivation)")
parser.add_argument("--activate_lang", type=str, default="ru", help="Language to activate (e.g., 'de' for German)")
parser.add_argument("--deactivate", action='store_true')
args = parser.parse_args()

# Get test questions
test_questions = get_test_questions()

model = LLM(model=args.model, tensor_parallel_size=torch.cuda.device_count(), enforce_eager=True)
# Get the correct EOS token ID from the tokenizer
eos_token_id = model.get_tokenizer().eos_token_id
print(f"EOS token ID: {eos_token_id}")

sampling_params = SamplingParams(temperature=0, repetition_penalty=1.1, max_tokens=512)
sampling_params = SamplingParams(
    temperature=0, 
    repetition_penalty=1.1, 
    max_tokens=512,
    stop_token_ids=[eos_token_id] if eos_token_id is not None else [],
    stop = ["\nQ:", "\nA:"],
    skip_special_tokens=True
)
is_llama = bool(args.model.lower().find("llama") >= 0)

# Load language-specific neuron masks
activation_masks = torch.load(args.activation_mask)

# Compute average activations for boosting
avg_activations, lang_to_idx, lang_names = compute_average_activations()

# Handle deactivation language (allow None)
if args.deactivate == False:
    deactivate_idx = None
    deactivate_mask = None
    print("Deactivation disabled")
else:
    if args.deactivate_lang not in lang_to_idx:
        raise ValueError(f"Deactivate language '{args.deactivate_lang}' not found in available languages: {list(lang_to_idx.keys())}")
    deactivate_idx = lang_to_idx[args.deactivate_lang]
    deactivate_mask = activation_masks[deactivate_idx]
    print(f"Deactivating {args.deactivate_lang} neurons (index {deactivate_idx})")

# Handle activation language
if args.activate_lang not in lang_to_idx:
    raise ValueError(f"Activate language '{args.activate_lang}' not found in available languages: {list(lang_to_idx.keys())}")
activate_idx = lang_to_idx[args.activate_lang]
activate_mask = activation_masks[activate_idx]
print(f"Activating {args.activate_lang} neurons (index {activate_idx})")

def factory(layer_idx, deactivate_indices, activate_indices, boost_values):
    def llama_forward(self, x):
        gate_up, _ = self.gate_up_proj(x)  # b, l, 2i
        i = gate_up.size(-1)
        activation = F.silu(gate_up[:, :, : i // 2])
        
        # Zero out deactivate language neurons (only if deactivation is enabled)
        if deactivate_indices is not None and len(deactivate_indices) > 0:
            activation.index_fill_(2, deactivate_indices, 0)
        
        # Boost activate language neurons
        if len(activate_indices) > 0:
            # Get current activations for these neurons
            current_activations = activation[:, :, activate_indices]
            # Add boost values (broadcast across batch and sequence dimensions)
            boost_tensor = boost_values.unsqueeze(0).unsqueeze(0).expand_as(current_activations)
            # Ensure dtype matches
            boost_tensor = boost_tensor.to(activation.dtype)
            activation[:, :, activate_indices] = current_activations +  boost_tensor # 
        
        x = activation * gate_up[:, :, i // 2 :]
        x, _ = self.down_proj(x)
        return x

    def bloom_forward(self, x: torch.Tensor):
        x, _ = self.dense_h_to_4h(x)
        x = self.gelu_impl(x)
        
        # Zero out deactivate language neurons (only if deactivation is enabled)
        if deactivate_indices is not None and len(deactivate_indices) > 0:
            x.index_fill_(2, deactivate_indices, 0)
        
        # Boost activate language neurons
        if len(activate_indices) > 0:
            current_activations = x[:, :, activate_indices]
            boost_tensor = boost_values.unsqueeze(0).unsqueeze(0).expand_as(current_activations)
            # Ensure dtype matches
            boost_tensor = boost_tensor.to(x.dtype)
            x[:, :, activate_indices] = boost_tensor
        
        x, _ = self.dense_4h_to_h(x)
        return x

    if is_llama:
        return llama_forward
    else:
        return bloom_forward

# Apply the intervention to each layer
for layer_idx in range(len(activate_mask)):  # Use activate_mask length since it's always present
    # Handle deactivation indices (may be None)
    if deactivate_mask is not None:
        deactivate_indices_cpu = deactivate_mask[layer_idx]
        deactivate_indices = deactivate_indices_cpu.to('cuda')
    else:
        deactivate_indices_cpu = None
        deactivate_indices = None
    
    # Handle activation indices (always present)
    activate_indices_cpu = activate_mask[layer_idx]
    activate_indices = activate_indices_cpu.to('cuda')
    
    # Compute boost values for this layer (average activation for target language neurons)
    if len(activate_indices_cpu) > 0:
        boost_values = avg_activations[layer_idx, activate_indices_cpu, activate_idx].to('cuda')
    else:
        boost_values = torch.tensor([]).to('cuda')
    
    if is_llama:
        obj = model.llm_engine.driver_worker.model_runner.model.model.layers[layer_idx].mlp
    else:
        obj = model.llm_engine.driver_worker.model_runner.model.transformer.h[layer_idx].mlp
    
    obj.forward = MethodType(factory(layer_idx, deactivate_indices, activate_indices, boost_values), obj)

# Test with all three questions
if args.deactivate_lang.lower() == "none": # 
    # Use English questions if no deactivation
    test_prompts = ["Q: How are you today? \nA:", "Q: What's your name? \nA:", "Q: What year is it? \nA:"]
else:
    if args.deactivate_lang not in test_questions:
        print(f"Warning: No questions found for {args.deactivate_lang}, using English")
        test_prompts = ["Q: How are you today? \nA:", "Q: What's your name? \nA:", "Q: What year is it? \nA:"]
    else:
        test_prompts = test_questions[args.deactivate_lang]

print("Testing all questions...")
print("="*50)

all_results = []
for i, test_prompt in enumerate(test_prompts):
    print(f"Question {i+1}: {test_prompt}")
    
    outputs = model.generate([test_prompt], sampling_params)
    response = outputs[0].outputs[0].text.strip()
    
    print(f"Response: {response}")
    print("-" * 30)
    
    # Store individual result
    result = {
        "question_idx": i,
        "input": test_prompt,
        "output": response
    }
    all_results.append(result)

# Save results
results = {
    "deactivate_language": args.deactivate_lang if args.deactivate_lang.lower() != "none" else None,
    "activate_language": args.activate_lang,
    "model": args.model,
    "results": all_results
}

output_dir = f"results/bilingual_intervention"
os.makedirs(output_dir, exist_ok=True)

# Handle filename when deactivation is disabled
if args.deactivate_lang.lower() == "none":
    output_file = f"{output_dir}/none_to_{args.activate_lang}.json"
else:
    output_file = f"{output_dir}/{args.deactivate_lang}_to_{args.activate_lang}.json"

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print(f"Results saved to: {output_file}")