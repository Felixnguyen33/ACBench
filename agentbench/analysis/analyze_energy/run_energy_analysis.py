import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

def calculate_energy(logits: torch.Tensor, temperature: float = 1.0) -> np.ndarray:
    """Calculate free energy values from logits.
    
    Args:
        logits: Input logits tensor
        temperature: Temperature parameter for scaling logits (default: 1.0)
        
    Returns:
        Numpy array of energy values
    """
    if not isinstance(logits, torch.Tensor):
        logits = torch.tensor(logits, dtype=torch.float32)
    
    # Compute energy using vectorized operations
    energy = -temperature * torch.logsumexp(logits / temperature, dim=-1)
    return energy.numpy()

def plot_energy_distribution(energy_in: np.ndarray, energy_out: np.ndarray, 
                            bins: int = 100, save_path: str = None):
    """Plot histogram of energy distributions.
    
    Args:
        energy_in: Energy values for in-distribution samples
        energy_out: Energy values for out-of-distribution samples
        bins: Number of bins for the histograms (default: 50)
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(6, 4))
    plt.hist(energy_in, bins=bins, alpha=0.7, label='In-distribution', 
             density=True, color='royalblue')
    plt.hist(energy_out, bins=bins, alpha=0.7, label='Out-of-distribution',
             density=True, color='lightgray')
    
    plt.xlabel('Negative Energy', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Energy Distribution Comparison', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.2)
    
    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        except Exception as e:
            print(f"Error saving figure to {save_path}: {e}")   
    else:
        plt.show()
    plt.close()

def plot_combined_energy_distribution(orig_energy: np.ndarray, quant_energy: np.ndarray, 
                                    position: int, save_path: str = None):
    """Plot combined energy distributions for original and quantized models.
    
    Args:
        orig_energy: Energy values from original model
        quant_energy: Energy values from quantized model
        position: Token position
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(6, 4))
    plt.hist(orig_energy, bins=100, alpha=0.7, label='Original Model', 
             density=True, color='royalblue')
    plt.hist(quant_energy, bins=100, alpha=0.7, label='Quantized Model',
             density=True, color='lightcoral')
    
    plt.xlabel('Negative Energy', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title(f'Energy Distribution Comparison - Position {position}', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.2)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

def plot_logits_distribution(logits_in: torch.Tensor, logits_out: torch.Tensor, 
                           position: int, save_path: str = None):
    """Plot histogram of logits distributions.
    
    Args:
        logits_in: Logits values for in-distribution samples
        logits_out: Logits values for out-of-distribution samples
        position: Token position
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(6, 4))
    plt.hist(logits_in.flatten().numpy(), bins=100, alpha=0.7, 
             label='In-distribution', density=True, color='royalblue')
    plt.hist(logits_out.flatten().numpy(), bins=100, alpha=0.7,
             label='Out-of-distribution', density=True, color='lightgray')
    
    plt.xlabel('Logits', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title(f'Logits Distribution - Position {position}', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.2)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

def plot_combined_logits_distribution(orig_logits: torch.Tensor, quant_logits: torch.Tensor,
                                    position: int, save_path: str = None):
    """Plot combined logits distributions for original and quantized models.
    
    Args:
        orig_logits: Logits from original model
        quant_logits: Logits from quantized model
        position: Token position
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(6, 4))
    plt.hist(orig_logits.flatten().numpy(), bins=100, alpha=0.7,
             label='Original Model', density=True, color='royalblue')
    plt.hist(quant_logits.flatten().numpy(), bins=100, alpha=0.7,
             label='Quantized Model', density=True, color='lightcoral')
    
    plt.xlabel('Logits', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title(f'Logits Distribution Comparison - Position {position}', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.2)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

def setup_model_and_tokenizer(quantized=False):
    """Load Qwen model and tokenizer with optional quantization"""
    model_path = "/data2/share/phi/Phi-3.5-mini-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    if quantized:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            load_in_4bit=True,
            device_map="auto",
            bnb_4bit_compute_dtype=torch.float16
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not quantized:  # Only move to device if not quantized (quantized model handles this automatically)
        model = model.to(device)
    
    model.eval()
    return model, tokenizer, device

def get_wikitext_data():
    """Load and prepare WikiText-2 dataset"""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    # Filter out empty strings and very short sequences
    texts = [text for text in dataset["train"]["text"][:1000] if len(text.strip()) > 50]
    return texts

def process_batch(texts, model, tokenizer, device, seq_length=256, batch_size=4):
    """Process a batch of sequences and get logits"""
    all_logits = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        # Skip empty texts
        batch_texts = [text for text in batch_texts if text.strip()]
        if not batch_texts:
            continue
            
        inputs = tokenizer(batch_texts, return_tensors="pt", max_length=seq_length,
                          truncation=True, padding='max_length')
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        try:
            with torch.no_grad():
                outputs = model(**inputs)
                # Move logits to CPU to save GPU memory
                all_logits.extend([logits.cpu() for logits in outputs.logits])
        except Exception as e:
            print(f"Error processing batch: {e}")
            continue
            
    return all_logits

def main():
    # Create output directories
    os.makedirs("./energy_imgs", exist_ok=True)
    os.makedirs("./quant_energy_imgs", exist_ok=True)
    os.makedirs("./energy_data", exist_ok=True)
    os.makedirs("./combined_plots", exist_ok=True)
    # Add new directories for logits plots
    os.makedirs("./logits_imgs", exist_ok=True)
    os.makedirs("./quant_logits_imgs", exist_ok=True)
    os.makedirs("./combined_logits_imgs", exist_ok=True)
    
    energy_data = {}  # Store energy values for both models
    logits_data = {}  # Store logits for both models
    
    # Process for both original and quantized models
    for is_quantized in [False, True]:
        model_type = 'quantized' if is_quantized else 'original'
        energy_data[model_type] = {'in': {}, 'out': {}}
        logits_data[model_type] = {'in': {}, 'out': {}}
        
        # Setup
        model, tokenizer, device = setup_model_and_tokenizer(quantized=is_quantized)
        wiki_data = get_wikitext_data()
        
        seq_length = 256
        batch_size = 2
        
        print(f"Processing sequences for {model_type} model...")
        all_logits = process_batch(wiki_data, model, tokenizer, device, seq_length, batch_size)
        
        if not all_logits:
            print("No valid sequences processed!")
            continue
            
        print(f"Successfully processed {len(all_logits)} sequences")
        
        output_dir = "./quant_energy_imgs" if is_quantized else "./energy_imgs"
        
        for position in tqdm(range(seq_length), 
                           desc=f"Processing positions for {model_type} model"):
            in_dist_logits = []
            out_dist_logits = []
            
            for logits in all_logits:
                if position < logits.size(0):
                    in_dist_logits.append(logits[position])
                    
                    permuted_logits = logits[position].clone()
                    permuted_logits = permuted_logits[torch.randperm(permuted_logits.size(0))]
                    out_dist_logits.append(permuted_logits)
            
            if not in_dist_logits:
                continue
                
            in_dist_logits = torch.stack(in_dist_logits)
            out_dist_logits = torch.stack(out_dist_logits)
            
            energy_in = calculate_energy(in_dist_logits)
            energy_out = calculate_energy(out_dist_logits)
            
            # Save energy values
            energy_data[model_type]['in'][position] = energy_in
            energy_data[model_type]['out'][position] = energy_out
            
            # Save individual plots
            save_path = f"{output_dir}/position_{position:02d}.png"
            plot_energy_distribution(energy_in, energy_out, save_path=save_path)
            
            # Save numpy arrays
            np.save(f"./energy_data/{model_type}_energy_in_pos_{position:02d}.npy", energy_in)
            np.save(f"./energy_data/{model_type}_energy_out_pos_{position:02d}.npy", energy_out)
            
            # Store logits data
            logits_data[model_type]['in'][position] = in_dist_logits
            logits_data[model_type]['out'][position] = out_dist_logits
            
            # Save logits plots
            logits_output_dir = "./quant_logits_imgs" if is_quantized else "./logits_imgs"
            save_path = f"{logits_output_dir}/position_{position:02d}.png"
            plot_logits_distribution(in_dist_logits, out_dist_logits, position, save_path)
        
        del model
        torch.cuda.empty_cache()
    
    # Create combined plots for each position
    print("Creating combined plots...")
    for position in range(seq_length):
        if position in logits_data['original']['in'] and position in logits_data['quantized']['in']:
            # Combined logits plots
            save_path = f"./combined_logits_imgs/combined_position_{position:02d}.png"
            plot_combined_logits_distribution(
                logits_data['original']['in'][position],
                logits_data['quantized']['in'][position],
                position,
                save_path
            )
            
            if position in energy_data['original']['in'] and position in energy_data['quantized']['in']:
                save_path = f"./combined_plots/combined_position_{position:02d}.png"
                plot_combined_energy_distribution(
                    energy_data['original']['in'][position],
                    energy_data['quantized']['in'][position],
                    position,
                    save_path
                )

def plot_logits_difference(logits_diff, N, num_tokens):
    """Plot logits differences.
    
    Args:
        logits_diff: Array of logits differences
        N: Number of examples to plot
        num_tokens: Number of tokens to show
    """
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, num_tokens))
    
    # Calculate mean or max across the second dimension
    logits_summary = np.mean(logits_diff, axis=-1)  # or use np.max(logits_diff, axis=-1)
    
    for i in range(min(N, len(logits_diff))):
        plt.bar(range(num_tokens), logits_summary[i][:num_tokens], color=colors)
    
    plt.xlabel('Token Position')
    plt.ylabel('Mean Logits Difference')
    plt.title('Logits Differences Across Token Positions')
    plt.grid(alpha=0.2)
    plt.savefig('logits_difference.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main() 