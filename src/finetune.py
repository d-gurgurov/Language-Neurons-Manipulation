import argparse
import json
import os
from types import MethodType
from itertools import combinations
from collections import defaultdict
import random
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

class LanguageSpecificDataset(Dataset):
    """Dataset for fine-tuning on target language text from CulturaX"""
    
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize with attention mask
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze()  # For causal LM, labels = input_ids
        }

class LanguageNeuronFineTuner:
    """Fine-tunes only language-specific neurons while freezing the rest of the model"""
    
    def __init__(self, model, tokenizer, activation_masks, target_lang, lang_to_idx):
        self.model = model
        self.tokenizer = tokenizer
        self.activation_masks = activation_masks
        self.target_lang = target_lang
        self.lang_to_idx = lang_to_idx
        
        # Get target language neuron indices
        if target_lang in lang_to_idx:
            self.target_neuron_indices = activation_masks[lang_to_idx[target_lang]]
        else:
            raise ValueError(f"Target language '{target_lang}' not found in activation masks")
        
        # Freeze all parameters initially
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Unfreeze only the target language neurons
        self.unfreeze_target_neurons()
        
        # Store original forward methods for restoration
        self.original_forwards = {}
        
    def unfreeze_target_neurons(self):
        """Unfreeze only the neurons associated with the target language"""
        print(f"Unfreezing neurons for language: {self.target_lang}")
        
        unfrozen_params = 0
        total_target_neurons = 0
        self.target_params = []  # Store references to target-specific parameters
        
        # Get model device and dtype
        model_device = next(self.model.parameters()).device
        model_dtype = next(self.model.parameters()).dtype
        
        for layer_idx, neuron_indices in enumerate(self.target_neuron_indices):
            if len(neuron_indices) == 0:
                continue
                
            # Convert neuron_indices to tensor if it isn't already, and ensure it's on CPU for indexing
            if not isinstance(neuron_indices, torch.Tensor):
                neuron_indices = torch.tensor(neuron_indices, dtype=torch.long)
            neuron_indices = neuron_indices.cpu()  # Keep indices on CPU for indexing
                
            # Get the MLP layer
            mlp_layer = self.model.model.layers[layer_idx].mlp
            
            # Check architecture type - some models have gate_up_proj, others have separate gate_proj and up_proj
            if hasattr(mlp_layer, 'gate_up_proj'):
                # Combined gate_up_proj architecture
                gate_up_weight = mlp_layer.gate_up_proj.weight
                
                # Extract target neuron weights (handle DTensor if present)
                if hasattr(gate_up_weight, 'to_local'):
                    # DTensor case - convert to local tensor first
                    gate_up_local = gate_up_weight.to_local()
                    gate_target_data = gate_up_local[neuron_indices, :].clone()
                    hidden_size = gate_up_local.shape[0] // 2
                    up_indices = neuron_indices + hidden_size
                    up_target_data = gate_up_local[up_indices, :].clone()
                else:
                    # Regular tensor case
                    gate_target_data = gate_up_weight[neuron_indices, :].clone()
                    hidden_size = gate_up_weight.shape[0] // 2
                    up_indices = neuron_indices + hidden_size
                    up_target_data = gate_up_weight[up_indices, :].clone()
                
                # Create trainable parameters
                gate_target_weights = nn.Parameter(gate_target_data.to(model_device, dtype=model_dtype))
                up_target_weights = nn.Parameter(up_target_data.to(model_device, dtype=model_dtype))
                
                # Store original indices and create trainable parameters
                mlp_layer._target_gate_indices = neuron_indices
                mlp_layer._target_up_indices = up_indices
                mlp_layer._target_gate_weights = gate_target_weights
                mlp_layer._target_up_weights = up_target_weights
                mlp_layer._architecture_type = "combined"
                
                # Add to target params list
                self.target_params.extend([gate_target_weights, up_target_weights])
                unfrozen_params += gate_target_weights.numel() + up_target_weights.numel()
                
            elif hasattr(mlp_layer, 'gate_proj') and hasattr(mlp_layer, 'up_proj'):
                # Separate gate_proj and up_proj architecture
                
                # Handle gate_proj weights
                gate_weight = mlp_layer.gate_proj.weight
                if hasattr(gate_weight, 'to_local'):
                    # DTensor case
                    gate_target_data = gate_weight.to_local()[neuron_indices, :].clone()
                else:
                    # Regular tensor case
                    gate_target_data = gate_weight[neuron_indices, :].clone()
                
                gate_target_weights = nn.Parameter(gate_target_data.to(model_device, dtype=model_dtype))
                mlp_layer._target_gate_indices = neuron_indices
                mlp_layer._target_gate_weights = gate_target_weights
                
                # Handle up_proj weights
                up_weight = mlp_layer.up_proj.weight
                if hasattr(up_weight, 'to_local'):
                    # DTensor case
                    up_target_data = up_weight.to_local()[neuron_indices, :].clone()
                else:
                    # Regular tensor case
                    up_target_data = up_weight[neuron_indices, :].clone()
                
                up_target_weights = nn.Parameter(up_target_data.to(model_device, dtype=model_dtype))
                mlp_layer._target_up_indices = neuron_indices
                mlp_layer._target_up_weights = up_target_weights
                mlp_layer._architecture_type = "separate"
                
                # Add to target params list
                self.target_params.extend([gate_target_weights, up_target_weights])
                unfrozen_params += gate_target_weights.numel() + up_target_weights.numel()
                
            else:
                raise ValueError(f"Unknown MLP architecture at layer {layer_idx}. "
                               f"Expected 'gate_up_proj' or ('gate_proj' and 'up_proj'), "
                               f"but found: {list(mlp_layer._modules.keys())}")
            
            # Handle down_proj weights
            down_weight = mlp_layer.down_proj.weight
            if hasattr(down_weight, 'to_local'):
                # DTensor case
                down_target_data = down_weight.to_local()[:, neuron_indices].clone()
            else:
                # Regular tensor case
                down_target_data = down_weight[:, neuron_indices].clone()
                
            down_target_weights = nn.Parameter(down_target_data.to(model_device, dtype=model_dtype))
            mlp_layer._target_down_indices = neuron_indices
            mlp_layer._target_down_weights = down_target_weights
            
            # Add to target params list
            self.target_params.append(down_target_weights)
            unfrozen_params += down_target_weights.numel()
            
            total_target_neurons += len(neuron_indices)
            
            # Replace the forward method with our custom one
            mlp_layer._original_forward = mlp_layer.forward
            mlp_layer.forward = self.create_custom_mlp_forward(mlp_layer)
        
        print(f"Created {unfrozen_params:,} trainable parameters across {total_target_neurons} target neurons")
        print(f"Target parameters device: {self.target_params[0].device if self.target_params else 'None'}")
        print(f"Target parameters dtype: {self.target_params[0].dtype if self.target_params else 'None'}")
        return unfrozen_params
        
    def create_custom_mlp_forward(self, mlp_layer):
        """Create a custom forward function that uses our target parameters in the computation"""
        def custom_forward(hidden_states):
            # Ensure hidden_states is on the correct device
            device = hidden_states.device
            dtype = hidden_states.dtype
            
            # Create modified weight matrices for this forward pass
            if mlp_layer._architecture_type == "combined":
                # Combined gate_up_proj architecture
                gate_up_weight = mlp_layer.gate_up_proj.weight
                
                # Handle DTensor case
                if hasattr(gate_up_weight, 'to_local'):
                    gate_up_local = gate_up_weight.to_local().clone()
                else:
                    gate_up_local = gate_up_weight.clone()
                
                # Update the target neuron weights in the cloned matrix
                gate_up_local[mlp_layer._target_gate_indices, :] = mlp_layer._target_gate_weights.to(device, dtype=dtype)
                gate_up_local[mlp_layer._target_up_indices, :] = mlp_layer._target_up_weights.to(device, dtype=dtype)
                
                # Perform gate_up projection with modified weights
                gate_up = F.linear(hidden_states, gate_up_local, 
                                 mlp_layer.gate_up_proj.bias.to_local() if hasattr(mlp_layer.gate_up_proj.bias, 'to_local') 
                                 else mlp_layer.gate_up_proj.bias)
                
                # Split into gate and up components
                gate, up = gate_up.chunk(2, dim=-1)
                intermediate_states = F.silu(gate) * up
                
            elif mlp_layer._architecture_type == "separate":
                # Separate gate_proj and up_proj architecture
                gate_weight = mlp_layer.gate_proj.weight
                up_weight = mlp_layer.up_proj.weight
                
                # Handle DTensor case for gate_proj
                if hasattr(gate_weight, 'to_local'):
                    gate_local = gate_weight.to_local().clone()
                else:
                    gate_local = gate_weight.clone()
                    
                # Handle DTensor case for up_proj
                if hasattr(up_weight, 'to_local'):
                    up_local = up_weight.to_local().clone()
                else:
                    up_local = up_weight.clone()
                
                # Update the target neuron weights in the cloned matrices
                gate_local[mlp_layer._target_gate_indices, :] = mlp_layer._target_gate_weights.to(device, dtype=dtype)
                up_local[mlp_layer._target_up_indices, :] = mlp_layer._target_up_weights.to(device, dtype=dtype)
                
                # Perform projections with modified weights
                gate = F.linear(hidden_states, gate_local, 
                              mlp_layer.gate_proj.bias.to_local() if hasattr(mlp_layer.gate_proj.bias, 'to_local')
                              else mlp_layer.gate_proj.bias)
                up = F.linear(hidden_states, up_local,
                            mlp_layer.up_proj.bias.to_local() if hasattr(mlp_layer.up_proj.bias, 'to_local')
                            else mlp_layer.up_proj.bias)
                intermediate_states = F.silu(gate) * up
            
            # Create modified down_proj weight matrix
            down_weight = mlp_layer.down_proj.weight
            if hasattr(down_weight, 'to_local'):
                down_local = down_weight.to_local().clone()
            else:
                down_local = down_weight.clone()
                
            down_local[:, mlp_layer._target_down_indices] = mlp_layer._target_down_weights.to(device, dtype=dtype)
            
            # Final down projection with modified weights
            output = F.linear(intermediate_states, down_local,
                            mlp_layer.down_proj.bias.to_local() if hasattr(mlp_layer.down_proj.bias, 'to_local')
                            else mlp_layer.down_proj.bias)
            
            return output
        
        return custom_forward
    
    def install_hooks(self):
        """Install custom forward methods for training"""
        for layer_idx, neuron_indices in enumerate(self.target_neuron_indices):
            if len(neuron_indices) == 0:
                continue
                
            mlp_layer = self.model.model.layers[layer_idx].mlp
            
            # Store original forward if not already stored
            if not hasattr(mlp_layer, '_original_forward'):
                mlp_layer._original_forward = mlp_layer.forward
            
            # Install custom forward
            mlp_layer.forward = self.create_custom_mlp_forward(mlp_layer)
    
    def remove_hooks(self):
        """Restore original forward methods"""
        for layer_idx in range(len(self.target_neuron_indices)):
            mlp_layer = self.model.model.layers[layer_idx].mlp
            if hasattr(mlp_layer, '_original_forward'):
                mlp_layer.forward = mlp_layer._original_forward

def integrate_trained_parameters(model, fine_tuner):
    """
    Integrate the trained target parameters back into the main model weights
    This is CRITICAL for save_pretrained to work correctly
    """
    print("Integrating trained parameters into model state...")
    
    with torch.no_grad():  # No gradients needed for this operation
        for layer_idx, neuron_indices in enumerate(fine_tuner.target_neuron_indices):
            if len(neuron_indices) == 0:
                continue
                
            mlp_layer = model.model.layers[layer_idx].mlp
            
            if not hasattr(mlp_layer, '_target_gate_weights'):
                continue  # Skip if no target weights were created
            
            # Get the device and dtype from the original model weights
            if hasattr(mlp_layer, 'gate_up_proj'):
                model_device = mlp_layer.gate_up_proj.weight.device
                model_dtype = mlp_layer.gate_up_proj.weight.dtype
            else:
                model_device = mlp_layer.gate_proj.weight.device
                model_dtype = mlp_layer.gate_proj.weight.dtype
            
            # Update the actual model weights with our trained parameters
            if mlp_layer._architecture_type == "combined":
                # Combined gate_up_proj architecture
                gate_up_weight = mlp_layer.gate_up_proj.weight
                
                if hasattr(gate_up_weight, 'to_local'):
                    # DTensor case
                    gate_up_local = gate_up_weight.to_local()
                    gate_up_local[mlp_layer._target_gate_indices] = mlp_layer._target_gate_weights.to(
                        device=gate_up_local.device, dtype=gate_up_local.dtype
                    )
                    gate_up_local[mlp_layer._target_up_indices] = mlp_layer._target_up_weights.to(
                        device=gate_up_local.device, dtype=gate_up_local.dtype
                    )
                else:
                    # Regular tensor case
                    gate_up_weight.data[mlp_layer._target_gate_indices] = mlp_layer._target_gate_weights.to(
                        device=model_device, dtype=model_dtype
                    )
                    gate_up_weight.data[mlp_layer._target_up_indices] = mlp_layer._target_up_weights.to(
                        device=model_device, dtype=model_dtype
                    )
                    
            elif mlp_layer._architecture_type == "separate":
                # Separate gate_proj and up_proj architecture
                gate_weight = mlp_layer.gate_proj.weight
                up_weight = mlp_layer.up_proj.weight
                
                if hasattr(gate_weight, 'to_local'):
                    # DTensor case
                    gate_weight.to_local()[mlp_layer._target_gate_indices] = mlp_layer._target_gate_weights.to(
                        device=gate_weight.device, dtype=gate_weight.dtype
                    )
                    up_weight.to_local()[mlp_layer._target_up_indices] = mlp_layer._target_up_weights.to(
                        device=up_weight.device, dtype=up_weight.dtype
                    )
                else:
                    # Regular tensor case
                    gate_weight.data[mlp_layer._target_gate_indices] = mlp_layer._target_gate_weights.to(
                        device=model_device, dtype=model_dtype
                    )
                    up_weight.data[mlp_layer._target_up_indices] = mlp_layer._target_up_weights.to(
                        device=model_device, dtype=model_dtype
                    )
            
            # Update down_proj weights
            down_weight = mlp_layer.down_proj.weight
            if hasattr(down_weight, 'to_local'):
                # DTensor case
                down_weight.to_local()[:, mlp_layer._target_down_indices] = mlp_layer._target_down_weights.to(
                    device=down_weight.device, dtype=down_weight.dtype
                )
            else:
                # Regular tensor case
                down_weight.data[:, mlp_layer._target_down_indices] = mlp_layer._target_down_weights.to(
                    device=model_device, dtype=model_dtype
                )
    
    print("Parameter integration complete!")


def clean_model_for_saving(model, fine_tuner):
    """
    Remove all custom attributes and parameters that shouldn't be saved
    """
    print("Cleaning model for saving...")
    
    for layer_idx, neuron_indices in enumerate(fine_tuner.target_neuron_indices):
        if len(neuron_indices) == 0:
            continue
            
        mlp_layer = model.model.layers[layer_idx].mlp
        
        # Remove custom attributes
        attrs_to_remove = [
            '_target_gate_indices', '_target_up_indices', '_target_down_indices',
            '_target_gate_weights', '_target_up_weights', '_target_down_weights',
            '_architecture_type', '_original_forward'
        ]
        
        for attr in attrs_to_remove:
            if hasattr(mlp_layer, attr):
                delattr(mlp_layer, attr)
    
    print("Model cleaning complete!")

def test_save_load_cycle(output_dir):
    """Test that the saved model can be loaded correctly"""
    try:
        print("\nTesting save/load cycle...")
        
        # Try loading the saved model
        test_model = AutoModelForCausalLM.from_pretrained(output_dir)
        test_tokenizer = AutoTokenizer.from_pretrained(output_dir)
        
        print("✅ Model loaded successfully!")
        print(f"Model type: {type(test_model)}")
        print(f"Model device: {next(test_model.parameters()).device}")
        print(f"Tokenizer vocab size: {len(test_tokenizer)}")
        
        # Quick inference test
        test_input = "Hello, how are you?"
        inputs = test_tokenizer(test_input, return_tensors="pt")
        
        with torch.no_grad():
            outputs = test_model.generate(**inputs, max_new_tokens=10)
        
        generated_text = test_tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Test generation: '{generated_text}'")
        print("✅ Save/load cycle successful!")
        
        return True
        
    except Exception as e:
        print(f"❌ Save/load test failed: {e}")
        return False

def load_culturax_data(language_code, num_samples=1000, max_length=512, validation_split=0.2):
    """Load text data from CulturaX dataset for the target language and split into train/validation"""
    print(f"Loading CulturaX data for language: {language_code}")
    
    try:
        # CulturaX dataset format - adjust the subset name as needed
        dataset_name = f"uonlp/CulturaX"
        
        # Load the dataset for the specific language
        dataset = load_dataset(dataset_name, language_code, split='train', streaming=True)
        
        texts = []
        for i, example in enumerate(dataset):
            if i >= num_samples:
                break
            
            # Extract text content (adjust field name based on CulturaX structure)
            text = example.get('text', '')
            if text and len(text.strip()) > 50:  # Filter out very short texts
                texts.append(text.strip()[:max_length])
        
        print(f"Loaded {len(texts)} text samples for {language_code}")
        
    except Exception as e:
        print(f"Error loading CulturaX data: {e}")
        print("Falling back to sample data generation...")
        
        # Fallback: generate sample data for testing
        texts = [
            f"This is sample text {i} in {language_code} for fine-tuning language-specific neurons."
            for i in range(num_samples)
        ]
    
    # Split into train and validation
    random.shuffle(texts)
    val_size = int(len(texts) * validation_split)
    train_texts = texts[val_size:]
    val_texts = texts[:val_size]
    
    print(f"Split into {len(train_texts)} training and {len(val_texts)} validation samples")
    
    return train_texts, val_texts

def validate_model(model, dataloader, device=None):
    """Validate the model on validation dataset"""
    if device is None:
        device = next(model.parameters()).device
        
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss

# Modified fine-tuning function with proper saving
def fine_tune_language_neurons(model, tokenizer, activation_masks, lang_to_idx, 
                                   target_lang, num_samples=1000, epochs=3, learning_rate=1e-4,
                                   batch_size=4, max_length=512, validation_split=0.2, 
                                   validate_every=1, output_dir="results/fine_tuning"):
    """Fine-tune language-specific neurons on target language data with validation"""
    
    print(f"Starting fine-tuning for language: {target_lang}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load target language data and split into train/validation
    train_texts, val_texts = load_culturax_data(target_lang, num_samples, max_length, validation_split)
    
    if len(train_texts) == 0:
        raise ValueError(f"No training data loaded for language: {target_lang}")
    
    # Create datasets and dataloaders
    train_dataset = LanguageSpecificDataset(train_texts, tokenizer, max_length)
    val_dataset = LanguageSpecificDataset(val_texts, tokenizer, max_length)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize fine-tuner
    fine_tuner = LanguageNeuronFineTuner(model, tokenizer, activation_masks, target_lang, lang_to_idx)
    fine_tuner.install_hooks()
    
    # Setup optimizer with only the target neuron parameters
    optimizer = optim.AdamW(fine_tuner.target_params, lr=learning_rate, weight_decay=0.01)
    
    # Debug: Count parameters by component
    print("\nParameter breakdown:")
    total_trainable = sum(p.numel() for p in fine_tuner.target_params)
    
    print(f"  Target neuron parameters: {total_trainable:,}")
    print(f"  Total model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Percentage of model being trained: {(total_trainable/sum(p.numel() for p in model.parameters()))*100:.4f}%")
    
    # Training loop with validation
    training_losses = []
    validation_losses = []
    best_val_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        print(f"\nEpoch {epoch + 1}/{epochs} - Training")
        
        for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}"):
            # Move batch to device - get device from model
            device = next(model.parameters()).device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(fine_tuner.target_params, max_norm=1.0)
            
            # Update only target neuron weights
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = epoch_loss / num_batches
        training_losses.append(avg_train_loss)
        
        # Validation phase
        if (epoch + 1) % validate_every == 0:
            print(f"Epoch {epoch + 1}/{epochs} - Validation")
            val_loss = validate_model(model, val_dataloader)
            validation_losses.append(val_loss)
            
            print(f"Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save best model with proper state integration
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                
                # CRITICAL: Proper model saving process
                print("Saving best model...")
                
                # 1. Remove custom forward methods
                fine_tuner.remove_hooks()
                
                # 2. Integrate trained parameters into model weights
                integrate_trained_parameters(model, fine_tuner)
                
                # 3. Clean model of custom attributes
                clean_model_for_saving(model, fine_tuner)
                
                # 4. Save the clean model
                best_model_dir = os.path.join(output_dir, "best_model")
                os.makedirs(best_model_dir, exist_ok=True)
                model.save_pretrained(best_model_dir)
                tokenizer.save_pretrained(best_model_dir)
                
                # 5. Recreate the fine-tuner for continued training
                fine_tuner = LanguageNeuronFineTuner(model, tokenizer, activation_masks, target_lang, lang_to_idx)
                fine_tuner.install_hooks()
                
                # 6. Recreate optimizer with new parameters
                optimizer = optim.AdamW(fine_tuner.target_params, lr=learning_rate, weight_decay=0.01)
                
                print(f"New best model saved to {best_model_dir} (Val Loss: {val_loss:.4f})")
        else:
            print(f"Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f}")
    
    # Final model save with proper cleanup
    print("Saving final model...")
    
    # 1. Remove custom forward methods
    fine_tuner.remove_hooks()
    
    # 2. Integrate final trained parameters
    integrate_trained_parameters(model, fine_tuner)
    
    # 3. Clean model of custom attributes
    clean_model_for_saving(model, fine_tuner)
    
    # 4. Save final model
    final_model_dir = os.path.join(output_dir, "final_model")
    os.makedirs(final_model_dir, exist_ok=True)
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    
    # Also save in the main output directory for backward compatibility
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    training_info = {
        "target_language": target_lang,
        "num_train_samples": len(train_texts),
        "num_val_samples": len(val_texts),
        "validation_split": validation_split,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "max_length": max_length,
        "training_losses": training_losses,
        "validation_losses": validation_losses,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "final_train_loss": training_losses[-1] if training_losses else None,
        "final_val_loss": validation_losses[-1] if validation_losses else None,
        "model_format": "huggingface_pytorch",
        "files_created": ["pytorch_model.bin", "config.json", "tokenizer.json", "tokenizer_config.json"]
    }
    
    with open(os.path.join(output_dir, "training_info.json"), "w") as f:
        json.dump(training_info, f, indent=2)
    
    print(f"\nFine-tuning completed!")
    print(f"Best model saved to: {os.path.join(output_dir, 'best_model')}")
    print(f"Final model saved to: {os.path.join(output_dir, 'final_model')}")
    print(f"Model also saved to: {output_dir}")
    
    # Test save/load cycle
    print("Testing save/load cycle...")
    test_save_load_cycle(output_dir)
    
    # Plot training and validation loss
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Training loss for all epochs
    plt.subplot(2, 1, 1)
    plt.plot(range(1, epochs + 1), training_losses, marker='o', label='Training Loss', color='blue')
    plt.title(f'Training Loss - Language-Specific Fine-tuning ({target_lang})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Training vs Validation loss (only for validation epochs)
    if validation_losses:
        plt.subplot(2, 1, 2)
        val_epochs = list(range(validate_every, epochs + 1, validate_every))
        train_loss_at_val = [training_losses[i-1] for i in val_epochs]
        
        plt.plot(val_epochs, train_loss_at_val, marker='o', label='Training Loss', color='blue')
        plt.plot(val_epochs, validation_losses, marker='s', label='Validation Loss', color='red')
        plt.title(f'Training vs Validation Loss ({target_lang})')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Mark best epoch
        if best_epoch in val_epochs:
            best_idx = val_epochs.index(best_epoch)
            plt.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7, 
                       label=f'Best Model (Epoch {best_epoch})')
            plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_validation_loss.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nFine-tuning completed!")
    print(f"Best model saved to: {os.path.join(output_dir, 'best_model')} (Epoch {best_epoch}, Val Loss: {best_val_loss:.4f})")
    print(f"Final model saved to: {os.path.join(output_dir, 'final_model')}")
    print(f"Model also saved to: {output_dir} (pytorch_model.bin)")
    print(f"Training info saved to: {output_dir}")
    print(f"\nTo load the model later, use:")
    print(f"  model = AutoModelForCausalLM.from_pretrained('{output_dir}')")
    print(f"  tokenizer = AutoTokenizer.from_pretrained('{output_dir}')")
    
    return model, training_losses, validation_losses

def save_checkpoint(model, optimizer, epoch, loss, path):
    """Save training checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, path)

def load_checkpoint(model, optimizer, path):
    """Load training checkpoint"""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss

def evaluate_fine_tuned_model(model, tokenizer, target_lang, test_prompts=None):
    """Evaluate the fine-tuned model on target language generation"""
    
    if test_prompts is None:
        # Default test prompts in various languages
        test_prompts = [
            "The weather today is",
            "I like to eat",
            "My favorite color is",
            "The capital city of France is",
            "Technology has changed our lives by"
        ]
    
    print(f"\nEvaluating fine-tuned model for {target_lang}:")
    print("=" * 60)
    
    model.eval()
    results = []
    
    with torch.no_grad():
        for i, prompt in enumerate(test_prompts):
            # Tokenize prompt and move to model device
            device = next(model.parameters()).device
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode response
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = generated_text[len(prompt):].strip()
            
            print(f"{i+1}. Prompt: '{prompt}'")
            print(f"   Response: '{response}'")
            print()
            
            results.append({
                "prompt": prompt,
                "response": response,
                "full_generation": generated_text
            })
    
    return results

def compute_average_activations():
    """Compute average activation values for each language"""
    lang_names = ["bo", "mt", "it", "es", "de", "ja", "ar", "zh", "af", "nl", "fr", "pt", "ru", "ko", "hi", "tr", "pl", "sv", "da", "no", "en"]
    
    # Create a mapping from language name to index
    lang_to_idx = {lang: idx for idx, lang in enumerate(lang_names)}
    
    return lang_to_idx, lang_names

def main():
    parser = argparse.ArgumentParser(description="Fine-tune language-specific neurons on target language corpus")
    parser.add_argument("-m", "--model", type=str, default="meta-llama/Meta-Llama-3-8B",
                       help="Model to fine-tune")
    parser.add_argument("-a", "--activation_mask", type=str, default="activation_mask/llama-3_5",
                       help="Path to activation masks")
    parser.add_argument("--target_lang", type=str, required=True,
                       help="Target language code (e.g., 'es', 'fr', 'de')")
    parser.add_argument("--output_dir", type=str, default="results/language_finetuning",
                       help="Output directory for fine-tuning results")
    parser.add_argument("--num_samples", type=int, default=1000,
                       help="Number of training samples from CulturaX")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate for fine-tuning")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Training batch size")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum sequence length")
    parser.add_argument("--evaluate_only", action='store_true',
                       help="Only evaluate existing fine-tuned model")
    parser.add_argument("--validation_split", type=float, default=0.05,
                       help="Fraction of data to use for validation (default: 0.2)")
    parser.add_argument("--validate_every", type=int, default=1,
                       help="Validate every N epochs (default: 1)")
    parser.add_argument("--use_fp16", action='store_true',
                       help="Use float16 precision instead of model's original precision")
    parser.add_argument("--use_best_model", action='store_true',
                       help="Use best validation model for final evaluation instead of last epoch model")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                       help="Path to checkpoint to resume training from")
    parser.add_argument("--force_gpu", action='store_true',
                       help="Force GPU usage, exit if no GPU available")
    parser.add_argument("--gpu_id", type=int, default=0,
                       help="Specific GPU ID to use (default: 0)")
    
    args = parser.parse_args()
    
    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Fine-tuning language-specific neurons for: {args.target_lang}")
    print(f"Model: {args.model}")
    print(f"Training samples: {args.num_samples}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.learning_rate}")
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Check device availability and set device
    if torch.cuda.is_available():
        if args.gpu_id >= torch.cuda.device_count():
            print(f"Warning: GPU {args.gpu_id} not available. Available GPUs: {torch.cuda.device_count()}")
            device = f"cuda:{min(args.gpu_id, torch.cuda.device_count()-1)}"
        else:
            device = f"cuda:{args.gpu_id}"
        print(f"Using GPU {device}: {torch.cuda.get_device_name(int(device.split(':')[1]))}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(int(device.split(':')[1])).total_memory / 1e9:.1f} GB")
    else:
        if args.force_gpu:
            raise RuntimeError("GPU requested but CUDA is not available!")
        device = "cpu"
        print("CUDA not available, using CPU")
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if args.use_fp16 else None,
        device_map={"": device}  # Force single device
    )
    
    # Ensure model is on the correct device
    model = model.to(device)
    
    # Clear GPU cache if using GPU
    if device.startswith('cuda'):
        torch.cuda.empty_cache()
    
    # Print device information
    print(f"Model device: {next(model.parameters()).device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"Current GPU: {torch.cuda.current_device()}")
        print(f"GPU name: {torch.cuda.get_device_name()}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load activation masks
    print("Loading activation masks...")
    activation_masks = torch.load(args.activation_mask)
    lang_to_idx, lang_names = compute_average_activations()
    
    if args.target_lang not in lang_to_idx:
        print(f"Available languages: {list(lang_to_idx.keys())}")
        raise ValueError(f"Target language '{args.target_lang}' not found in activation masks")
    
    if args.evaluate_only:
        # Only evaluate existing model
        print("Evaluation mode - loading existing fine-tuned model...")
        
        # Try to load from the main output directory first
        model_loaded = False
        
        # Try HuggingFace format first
        if os.path.exists(os.path.join(args.output_dir, "config.json")):
            try:
                model = AutoModelForCausalLM.from_pretrained(args.output_dir)
                tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
                print(f"Loaded HuggingFace format model from: {args.output_dir}")
                model_loaded = True
            except Exception as e:
                print(f"Failed to load HuggingFace format: {e}")
        
        # Try best model directory
        if not model_loaded and os.path.exists(os.path.join(args.output_dir, "best_model", "config.json")):
            try:
                best_model_dir = os.path.join(args.output_dir, "best_model")
                model = AutoModelForCausalLM.from_pretrained(best_model_dir)
                tokenizer = AutoTokenizer.from_pretrained(best_model_dir)
                print(f"Loaded best model from: {best_model_dir}")
                model_loaded = True
            except Exception as e:
                print(f"Failed to load best model: {e}")
        
        # Try final model directory
        if not model_loaded and os.path.exists(os.path.join(args.output_dir, "final_model", "config.json")):
            try:
                final_model_dir = os.path.join(args.output_dir, "final_model")
                model = AutoModelForCausalLM.from_pretrained(final_model_dir)
                tokenizer = AutoTokenizer.from_pretrained(final_model_dir)
                print(f"Loaded final model from: {final_model_dir}")
                model_loaded = True
            except Exception as e:
                print(f"Failed to load final model: {e}")
        
        # Fallback to .pt files
        if not model_loaded:
            if args.use_best_model:
                model_path = os.path.join(args.output_dir, "best_model.pt")
                model_type = "best validation"
            else:
                model_path = os.path.join(args.output_dir, "final_model.pt")
                model_type = "final"
                
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path))
                print(f"Loaded {model_type} model from: {model_path}")
                model_loaded = True
        
        if not model_loaded:
            print("No fine-tuned model found, using base model for evaluation")
        
        results = evaluate_fine_tuned_model(model, tokenizer, args.target_lang)
        
        # Save evaluation results
        with open(os.path.join(args.output_dir, "evaluation_results.json"), "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    else:
        # Fine-tune the model
        model, training_losses, validation_losses = fine_tune_language_neurons(
            model=model,
            tokenizer=tokenizer,
            activation_masks=activation_masks,
            lang_to_idx=lang_to_idx,
            target_lang=args.target_lang,
            num_samples=args.num_samples,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            max_length=args.max_length,
            validation_split=args.validation_split,
            validate_every=args.validate_every,
            output_dir=args.output_dir
        )
        
        # Evaluate the fine-tuned model
        print("Evaluating fine-tuned model...")
        
        # The model is already in its final state with integrated parameters
        results = evaluate_fine_tuned_model(model, tokenizer, args.target_lang)
        
        # Save evaluation results
        with open(os.path.join(args.output_dir, "evaluation_results.json"), "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nProcess completed! Results saved to: {args.output_dir}")
    print(f"\nModel files created:")
    print(f"  Main directory: {args.output_dir}/pytorch_model.bin")
    if not args.evaluate_only:
        print(f"  Best model: {args.output_dir}/best_model/pytorch_model.bin")
        print(f"  Final model: {args.output_dir}/final_model/pytorch_model.bin")

if __name__ == "__main__":
    main()