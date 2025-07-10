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

def compute_average_activations():
    """Compute average activation values for each language"""
    n, over_zero = [], []
    lang_names = ["bo", "mt", "it", "es", "de", "ja", "ar", "zh", "af", "nl", "fr", "pt", "ru", "ko", "hi", "tr", "pl", "sv", "da", "no", "en"]
    
    for lang in lang_names:
        data = torch.load(f'data_{activations_path[0]}/activation.{lang}.train.{activations_path[1]}')
        n.append(data['n'])
        over_zero.append(data['over_zero'])

    n = torch.tensor(n)
    over_zero = torch.stack(over_zero, dim=-1)
    
    activation_probs = over_zero / n  # layer x inter x lang_num
    
    lang_to_idx = {lang: idx for idx, lang in enumerate(lang_names)}
    
    return activation_probs, lang_to_idx, lang_names

def factory(layer_idx, deactivate_indices=None, activate_indices=None, boost_values=None):
    def llama_forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        i = gate_up.size(-1)

        if gate_up.dim() == 3:
            silu_output = F.silu(gate_up[:, :, : i // 2])
            
            if deactivate_indices is not None and len(deactivate_indices) > 0:
                silu_output.index_fill_(2, deactivate_indices, args.deactivation_strength)

            if activate_indices is not None and len(activate_indices) > 0 and boost_values is not None:
                boost_tensor = boost_values.to(silu_output.dtype).unsqueeze(0).unsqueeze(0)
                silu_output[:, :, activate_indices] += boost_tensor
                # trying replacement
                silu_output[:, activate_indices] = boost_tensor

            x = silu_output * gate_up[:, :, i // 2:]

        elif gate_up.dim() == 2:
            silu_output = F.silu(gate_up[:, : i // 2])

            if deactivate_indices is not None and len(deactivate_indices) > 0:
                silu_output.index_fill_(1, deactivate_indices, args.deactivation_strength)

            if activate_indices is not None and len(activate_indices) > 0 and boost_values is not None:
                boost_tensor = boost_values.to(silu_output.dtype).unsqueeze(0)
                silu_output[:, activate_indices] += boost_tensor
                # trying replacement
                # silu_output[:, activate_indices] = boost_tensor

            x = silu_output * gate_up[:, i // 2:]

        else:
            raise ValueError(f"Unexpected gate_up shape: {gate_up.shape}")

        x, _ = self.down_proj(x)
        return x

    return llama_forward

def apply_intervention(model, activation_masks, avg_activations, lang_to_idx, deactivate_lang, activate_lang, is_llama):
    """Apply language intervention to the model"""
    
    # Handle deactivation
    if deactivate_lang is not None and deactivate_lang.lower() != "none":
        if deactivate_lang not in lang_to_idx:
            raise ValueError(f"Deactivate language '{deactivate_lang}' not found in available languages: {list(lang_to_idx.keys())}")
        deactivate_idx = lang_to_idx[deactivate_lang]
        deactivate_mask = activation_masks[deactivate_idx]
        print(f"Deactivating {deactivate_lang} neurons (index {deactivate_idx})")
    else:
        deactivate_mask = None
        print("Deactivation disabled")

    # Handle activation
    if activate_lang not in lang_to_idx:
        raise ValueError(f"Activate language '{activate_lang}' not found in available languages: {list(lang_to_idx.keys())}")
    activate_idx = lang_to_idx[activate_lang]
    activate_mask = activation_masks[activate_idx]
    print(f"Activating {activate_lang} neurons (index {activate_idx})")

    # Store original forward methods to restore later
    original_forwards = []
    
    # Apply the intervention to each layer
    for layer_idx in range(len(activate_mask)):
        if deactivate_mask is not None:
            deactivate_indices_cpu = deactivate_mask[layer_idx]
            deactivate_indices = deactivate_indices_cpu.to('cuda')
        else:
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
            obj = model.llm_engine.model_executor.driver_worker.model_runner.model.model.layers[layer_idx].mlp
        else:
            obj = model.llm_engine.model_executor.driver_worker.model_runner.model.model.layers[layer_idx].mlp
        
        # Store original forward method
        original_forwards.append(obj.forward)
        
        # Apply new forward method
        obj.forward = MethodType(factory(layer_idx, deactivate_indices, activate_indices, boost_values), obj)
    
    return original_forwards

def restore_original_forwards(model, original_forwards, is_llama):
    """Restore original forward methods"""
    for layer_idx, original_forward in enumerate(original_forwards):
        if is_llama:
            obj = model.llm_engine.model_executor.driver_worker.model_runner.model.model.layers[layer_idx].mlp
        else:
            obj = model.llm_engine.model_executor.driver_worker.model_runner.model.model.layers[layer_idx].mlp
        obj.forward = original_forward

def run_single_experiment(model, sampling_params, test_questions, deactivate_lang, activate_lang, activation_masks, avg_activations, lang_to_idx, is_llama, deactivation=True):
    """Run a single language intervention experiment"""
    
    print(f"\n{'='*60}")
    if not deactivation:
        print(f"Experiment: Activation only → {activate_lang}")
    else:
        print(f"Experiment: {deactivate_lang} → {activate_lang}")
    print(f"{'='*60}")
    
    if not deactivation:
        original_forwards = apply_intervention(model, activation_masks, avg_activations, lang_to_idx, "none", activate_lang, is_llama)
    else:
        original_forwards = apply_intervention(model, activation_masks, avg_activations, lang_to_idx, deactivate_lang, activate_lang, is_llama)

    test_prompts = test_questions[deactivate_lang]
    
    print("Testing all questions...")
    print("="*50)
    
    all_results = []
    for i, test_prompt in enumerate(test_prompts):
        print(f"Question {i+1}: {test_prompt}")
        
        outputs = model.generate([test_prompt], sampling_params)
        response = outputs[0].outputs[0].text.strip()
        
        print(f"Response: {response}")
        print("-" * 30)
        
        result = {
            "question_idx": i,
            "input": test_prompt,
            "output": response
        }
        all_results.append(result)
    
    restore_original_forwards(model, original_forwards, is_llama)
    
    results = {
        "deactivate_language": deactivate_lang if deactivate_lang and deactivate_lang.lower() != "none" else None,
        "activate_language": activate_lang,
        "model": args.model,
        "results": all_results
    }
    
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    if args.no_deactivation:
        output_dir = output_dir + "/activate"
        os.makedirs(output_dir, exist_ok=True)
        output_file = f"{output_dir}/{deactivate_lang}_to_{activate_lang}.json"
    else:
        output_dir = output_dir + "/deactivate_activate"
        os.makedirs(output_dir, exist_ok=True)
        output_file = f"{output_dir}/{deactivate_lang}_to_{activate_lang}.json"
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print(f"Results saved to: {output_file}")
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("-a", "--activation_mask", type=str, default="activation_mask/llama-3")
    parser.add_argument("--activations_path", type=str, default="data_llama-3")
    parser.add_argument("--output", type=str, default="results")
    parser.add_argument("--batch_mode", action='store_true', help="Run all language combinations")
    parser.add_argument("--deactivate_lang", type=str, default="de", help="Language to deactivate (single mode)")
    parser.add_argument("--activate_lang", type=str, default="ru", help="Language to activate (single mode)")
    parser.add_argument("--no_deactivation", action='store_true', help="Skip deactivation, only do activation (batch mode only)")
    parser.add_argument("--languages", nargs='+', default=["bo", "mt", "it", "es", "de", "ja", "ar", "zh", "af", "nl", "fr", "pt", "ru", "ko", "hi", "tr", "pl", "sv", "da", "no", "en"], help="Languages to test")
    parser.add_argument("--deactivation_strength", type=float, default=-1.0) 
    
    global args
    args = parser.parse_args()

    global activations_path
    activations_path = args.activations_path.split(" ")
    
    print("Loading model and data...")

    test_questions = get_test_questions() 
    model = LLM(model=args.model, tensor_parallel_size=torch.cuda.device_count(), enforce_eager=True)
    eos_token_id = model.get_tokenizer().eos_token_id
    print(f"EOS token ID: {eos_token_id}")
    
    sampling_params = SamplingParams(
        temperature=0, 
        repetition_penalty=1.1, 
        max_tokens=256,
        stop_token_ids=[eos_token_id] if eos_token_id is not None else [],
        stop = ["\nQ:", "\nA:"],
        skip_special_tokens=True
    )
    
    is_llama = bool(args.model.lower().find("llama") >= 0)
    
    # Load language-specific neuron masks
    activation_masks = torch.load(args.activation_mask)
    
    # Compute average activations for boosting
    avg_activations, lang_to_idx, lang_names = compute_average_activations()
    
    if args.batch_mode:
        languages = args.languages
        
        # Regular batch mode: all deactivation + activation combinations
        total_combinations = len(languages) * len(languages)
        counter = 0
        
        print(f"Starting systematic analysis...")
        print(f"Total combinations: {total_combinations}")
        
        all_experiment_results = []
        
        for deactivate_lang in languages:
            for activate_lang in languages:
                counter += 1
                print(f"\n[{counter}/{total_combinations}] Processing combination...")
                
                try:
                    if args.no_deactivation:
                        results = run_single_experiment(
                            model, sampling_params, test_questions, 
                            deactivate_lang, activate_lang, 
                            activation_masks, avg_activations, lang_to_idx, is_llama, deactivation=False
                        )
                        all_experiment_results.append(results)
                    else:
                        results = run_single_experiment(
                            model, sampling_params, test_questions, 
                            deactivate_lang, activate_lang, 
                            activation_masks, avg_activations, lang_to_idx, is_llama
                        )
                        all_experiment_results.append(results)
                    
                except Exception as e:
                    print(f"Error in experiment {deactivate_lang} → {activate_lang}: {e}")
                    continue

    else:
        results = run_single_experiment(
            model, sampling_params, test_questions, 
            args.deactivate_lang, args.activate_lang, 
            activation_masks, avg_activations, lang_to_idx, is_llama
        )
        print("Single experiment complete!")

if __name__ == "__main__":
    main()