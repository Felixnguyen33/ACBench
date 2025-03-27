import torch
import numpy as np
import matplotlib.pyplot as plt
import os

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
                            bins: int = 50, save_path: str = None):
    """Plot histogram of energy distributions.
    
    Args:
        energy_in: Energy values for in-distribution samples
        energy_out: Energy values for out-of-distribution samples
        bins: Number of bins for the histograms (default: 50)
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 6))
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
            os.makedirs(os.path.dirname(save_path), exist_ok=True) # Create directory if not exist
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        except Exception as e:
            print(f"Error saving figure to {save_path}: {e}")   
    else：
        plt.show()

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Generate Gaussian distributed logits
mean = 0
std = 1
num_samples = 1000
vocab_size = 100
logits_in = torch.normal(mean, std, size=(num_samples, vocab_size))

# Generate out-of-distribution logits: using a different mean and standard deviation
logits_out = torch.randn(num_samples, vocab_size) * 2 + 1 

# Calculate energy values
energy_in = calculate_energy(logits_in, temperature=1.0)
energy_out = calculate_energy(logits_out, temperature=1.0)

# Plot distributions
plot_energy_distribution(energy_in, energy_out, bins = 60, save_path="./energy_distribution.png")
