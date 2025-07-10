import torch
import torch.nn.functional as F
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--top_rate", type=float, default=0.02, help="Top neuron rate to select")
parser.add_argument("--activations", type=str, default="", help="Activation paths")
parser.add_argument("--save_path", type=str, default="llama-3-1-2", help="Save path")
args = parser.parse_args()

activations_path = args.activations.split(" ")

n, over_zero = [], []
for lang in ["bo", "mt", "it", "es", "de", "ja", "ar", "zh", "af", "nl", "fr", "pt", "ru", "ko", "hi", "tr", "pl", "sv", "da", "no", "en"]:
    data = torch.load(f'data_{activations_path[0]}/activation.{lang}.train.{activations_path[1]}')
    n.append(data['n'])
    over_zero.append(data['over_zero'])

n = torch.tensor(n)
over_zero = torch.stack(over_zero, dim=-1)

num_layers, intermediate_size, lang_num = over_zero.size()
print(num_layers, intermediate_size, lang_num)

def activation():
    top_rate = args.top_rate
    filter_rate = 0.95
    activation_bar_ratio = 0.95
    activation_probs = over_zero / n # layer x inter x lang_num
    normed_activation_probs = activation_probs / activation_probs.sum(dim=-1, keepdim=True)
    normed_activation_probs[torch.isnan(normed_activation_probs)] = 0
    log_probs = torch.where(normed_activation_probs > 0, normed_activation_probs.log(), 0)
    entropy = -torch.sum(normed_activation_probs * log_probs, dim=-1)
    largest = False
    
    if torch.isnan(entropy).sum():
        print(torch.isnan(entropy).sum())
        raise ValueError
    
    flattened_probs = activation_probs.flatten()
    top_prob_value = flattened_probs.kthvalue(round(len(flattened_probs) * filter_rate)).values.item()
    print("filter rate =",top_prob_value)
    # dismiss the neruon if no language has an activation value over top 95%
    top_position = (activation_probs > top_prob_value).sum(dim=-1)
    entropy[top_position == 0] = -torch.inf if largest else torch.inf

    flattened_entropy = entropy.flatten()
    top_entropy_value = round(len(flattened_entropy) * top_rate)
    _, index = flattened_entropy.topk(top_entropy_value, largest=largest)
    row_index = index // entropy.size(1)
    col_index = index % entropy.size(1)
    selected_probs = activation_probs[row_index, col_index] # n x lang
    # for r, c in zip(row_index, col_index):
    #     print(r, c, activation_probs[r][c])

    # print(selected_probs.size(0), torch.bincount(selected_probs.argmax(dim=-1)))
    selected_probs = selected_probs.transpose(0, 1)
    activation_bar = flattened_probs.kthvalue(round(len(flattened_probs) * activation_bar_ratio)).values.item()
    print("second activation bar=", activation_bar)
    print((selected_probs > activation_bar).sum(dim=1).tolist())
    lang, indice = torch.where(selected_probs > activation_bar)
    num_after_activation_bar = len(indice)

    merged_index = torch.stack((row_index, col_index), dim=-1)
    final_indice = []
    for _, index in enumerate(indice.split(torch.bincount(lang).tolist())):
        lang_index = [tuple(row.tolist()) for row in merged_index[index]]
        lang_index.sort()
        layer_index = [[] for _ in range(num_layers)]
        for l, h in lang_index:
            layer_index[l].append(h)
        for l, h in enumerate(layer_index):
            layer_index[l] = torch.tensor(h).long()
        final_indice.append(layer_index)
    torch.save(final_indice, f"activation_mask/{args.save_path}")  

def activation():
    top_rate = args.top_rate
    filter_rate = 0.95
    activation_bar_ratio = 0.95
    activation_probs = over_zero / n  # [layer, inter, lang]
    normed_activation_probs = activation_probs / activation_probs.sum(dim=-1, keepdim=True)
    normed_activation_probs[torch.isnan(normed_activation_probs)] = 0

    log_probs = torch.where(normed_activation_probs > 0, normed_activation_probs.log(), 0)
    entropy = -torch.sum(normed_activation_probs * log_probs, dim=-1)
    largest = False

    if torch.isnan(entropy).sum():
        print(torch.isnan(entropy).sum())
        raise ValueError

    # --- Print original number of neurons ---
    total_neurons = activation_probs.size(0) * activation_probs.size(1)
    print("Total original neurons:", total_neurons)

    # --- First threshold: filter neurons that are ever above the filter_rate ---
    flattened_probs = activation_probs.flatten()
    top_prob_value = flattened_probs.kthvalue(round(len(flattened_probs) * filter_rate)).values.item()
    print("Filter rate threshold value =", top_prob_value)

    top_position = (activation_probs > top_prob_value).sum(dim=-1)
    entropy[top_position == 0] = -torch.inf if largest else torch.inf

    neurons_after_first_threshold = (top_position != 0).sum().item()
    print("Neurons after first threshold (filter rate):", neurons_after_first_threshold)

    # --- Second threshold: topk by entropy ---
    flattened_entropy = entropy.flatten()
    top_entropy_value = round(len(flattened_entropy) * top_rate)
    _, index = flattened_entropy.topk(top_entropy_value, largest=largest)
    row_index = index // entropy.size(1)
    col_index = index % entropy.size(1)
    selected_probs = activation_probs[row_index, col_index]  # [n, lang]

    selected_probs = selected_probs.transpose(0, 1)
    activation_bar = flattened_probs.kthvalue(round(len(flattened_probs) * activation_bar_ratio)).values.item()
    print("Second activation bar threshold value =", activation_bar)

    above_second_threshold = (selected_probs > activation_bar).sum().item()
    print("Neurons after second threshold (activation bar):", above_second_threshold)

    lang, indice = torch.where(selected_probs > activation_bar)
    num_after_activation_bar = len(indice)

    print((selected_probs > activation_bar).sum(dim=1).tolist())

    merged_index = torch.stack((row_index, col_index), dim=-1)
    final_indice = []
    for _, index in enumerate(indice.split(torch.bincount(lang).tolist())):
        lang_index = [tuple(row.tolist()) for row in merged_index[index]]
        lang_index.sort()
        layer_index = [[] for _ in range(num_layers)]
        for l, h in lang_index:
            layer_index[l].append(h)
        for l, h in enumerate(layer_index):
            layer_index[l] = torch.tensor(h).long()
        final_indice.append(layer_index)

    total_selected = sum(x.numel() for lang in final_indice for x in lang)
    print(f"Total selected neurons: {total_selected}")
    
    torch.save(final_indice, f"activation_mask/{args.save_path}")

activation()
