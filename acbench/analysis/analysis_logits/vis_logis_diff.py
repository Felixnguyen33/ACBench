import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau
from datasets import load_dataset

# COLOR_PALETTE = ['#2E8B57', '#FF8C00', '#DC143C', '#4169E1', '#8A2BE2', '#A0522D', '#696969']
COLOR_PALETTE = ['#FA7F6F', '#FFBE7A', '#82B0D2', '#8ECFC9', '#BEB8DC', '#E7DAD2', '#999999']

# Set academic paper style for matplotlib
plt.style.use('default')
plt.rcParams.update({
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.titlesize': 11
})

def get_model_outputs(model_path, apply_quant=True):
    """
    Load a model and get its output logits for WikiText-2 dataset.
    
    Args:
        model_path (str): Path to the model
        apply_quant (bool): Whether to apply 4-bit quantization
    
    Returns:
        numpy.ndarray: Model output logits
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model with quantization if specified
    if apply_quant:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True
        )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()

    # Load and process the WikiText-2 dataset
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    filtered_texts = [text for text in dataset['text'] if text.strip()]
    full_text = "\n\n".join(filtered_texts[:100])  # Limit to first 100 texts for testing
    
    # Tokenize the text and limit to model's max sequence length
    tokenized = tokenizer(full_text, return_tensors='pt')['input_ids'][0]
    max_length = min(2048, model.config.max_position_embeddings)  # Use smaller chunks
    
    # Split into sequences of max_length
    sequences = []
    for i in range(0, len(tokenized), max_length):
        if i + max_length <= len(tokenized):
            sequences.append(tokenized[i:i + max_length])
    sequences = torch.stack(sequences)
    
    if len(sequences) == 0:
        raise ValueError("No sequences to process after splitting")
    
    # Limit total sequences and process in smaller batches
    max_sequences = 20  # Limit total number of sequences
    batch_size = 2  # Use smaller batch size
    sequences = sequences[:min(len(sequences), max_sequences)]
    
    num_batches = len(sequences) // batch_size
    if num_batches == 0:
        num_batches = 1
        batch_size = len(sequences)
    
    batched_sequences = sequences[:num_batches * batch_size].view(num_batches, batch_size, -1)
    
    # Process batches
    all_logits = []
    for batch in batched_sequences:
        batch = batch.to(device)
        with torch.no_grad():
            outputs = model(batch)
        logits = outputs.logits.cpu().numpy()
        all_logits.append(logits)
        
    if not all_logits:
        raise ValueError("No logits were generated")
    
    return np.concatenate(all_logits, axis=0)

def get_model_outputs_both(model_path):
    """
    Load model once and get outputs for both quantized and non-quantized versions
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset first
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    filtered_texts = [text for text in dataset['text'] if text.strip()]
    full_text = "\n\n".join(filtered_texts[:500])  # Increased from 100 to 500
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Tokenize the text
    tokenized = tokenizer(full_text, return_tensors='pt')['input_ids'][0]
    max_length = 32  # Fixed length for 32 tokens as expected
    
    # Take exactly 32 tokens for multiple sequences
    sequences = []
    for i in range(0, len(tokenized) - max_length + 1, max_length):
        sequences.append(tokenized[i:i + max_length])
    sequences = torch.stack(sequences[:50])  # Take 50 sequences of 32 tokens each
    
    outputs_list = []
    
    # Process both quantized and non-quantized versions
    for apply_quant in [True, False]:
        if apply_quant:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                trust_remote_code=True
            )
        
        model.eval()
        
        # Process in batches
        batch_size = 10  # Increased batch size
        all_logits = []
        
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i + batch_size].to(device)
            with torch.no_grad():
                outputs = model(batch)
            logits = outputs.logits.cpu().numpy()
            all_logits.append(logits)
        
        del model  # Free up memory
        outputs_list.append(np.concatenate(all_logits, axis=0))
    
    return outputs_list[0], outputs_list[1]  # Returns (quant_logits, non_quant_logits)

def plot_logits_analysis(logits_before, logits_after, N, top_k_values=[3, 10, 20]):
    """
    Plot logits analysis charts as separate figures
    
    Args:
        logits_before: logits before quantization, shape [N, seq_len, vocab_size]
        logits_after: logits after quantization, shape [N, seq_len, vocab_size]
        N: number of sequences to analyze
        top_k_values: list of top-k values to analyze
    """
    # Flatten the sequence length and vocab size dimensions for analysis
    logits_before_flat = logits_before.reshape(logits_before.shape[0], -1)
    logits_after_flat = logits_after.reshape(logits_after.shape[0], -1)
    
    # 1. Logits difference heatmap
    plt.figure(figsize=(4, 3))
    logits_diff = logits_after_flat - logits_before_flat
    sample_points = min(100, logits_diff.shape[1])  # Take first 100 points or less
    im = plt.imshow(logits_diff[:N, :sample_points], cmap='seismic', aspect='auto')
    plt.title('Logits Difference Heatmap')
    plt.xlabel('Sample Points')
    plt.ylabel('Sequence Number')
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig('./logits_diff_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Kendall Tau correlation for each sequence
    plt.figure(figsize=(4, 3))
    kendall_scores = []
    for i in range(N):
        tau, _ = kendalltau(logits_before_flat[i], logits_after_flat[i])
        kendall_scores.append(tau)
    
    plt.plot(range(N), kendall_scores, 'b-', marker='o', markersize=3)
    plt.title('Kendall Tau Correlation')
    plt.xlabel('Sequence Number')
    plt.ylabel('Kendall Tau Value')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('./kendall_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Top-K correlation analysis
    plt.figure(figsize=(4, 3))
    for i, k in enumerate(top_k_values):
        topk_kendall = []
        for seq in range(N):
            top_k_indices = np.argsort(logits_before_flat[seq])[-k:]
            tau, _ = kendalltau(logits_before_flat[seq][top_k_indices], 
                              logits_after_flat[seq][top_k_indices])
            topk_kendall.append(tau)
        plt.plot(range(N), topk_kendall, label=f'Top-{k}', linewidth=1, 
                color=COLOR_PALETTE[i % len(COLOR_PALETTE)])
    
    plt.title('Top-K Kendall Correlation')
    plt.xlabel('Sequence Number')
    plt.ylabel('Kendall Tau Value')
    plt.legend(frameon=True, fancybox=True, framealpha=0.8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('./topk_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Box plot comparison
    plt.figure(figsize=(4, 3))
    box_data = []
    labels = []
    for i, k in enumerate(top_k_values):
        k_data = []
        for seq in range(N):
            top_k_indices = np.argsort(logits_before_flat[seq])[-k:]
            tau, _ = kendalltau(logits_before_flat[seq][top_k_indices], 
                              logits_after_flat[seq][top_k_indices])
            k_data.append(tau)
        box_data.append(k_data)
        labels.append(f'Top-{k}')
    
    plt.boxplot(box_data, labels=labels)
    plt.title('Top-K Correlation Distribution')
    plt.ylabel('Kendall Tau Value')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('./correlation_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()

def calculate_correlations(logits_after_quant, logits_before_quant, N, top_k):
    """Calculate various correlation metrics between quantized and non-quantized logits."""
    spearman_correlations = []
    kendall_taus = []
    top_k_spearman_correlations = {k: [] for k in top_k}
    top_k_kendall_taus = {k: [] for k in top_k}
    
    for i in range(N):
        logits_after_flat = logits_after_quant[i].flatten()
        logits_before_flat = logits_before_quant[i].flatten()
        
        # Calculate overall correlations
        spearman_corr, _ = spearmanr(logits_after_flat, logits_before_flat)
        kendall_tau, _ = kendalltau(logits_after_flat, logits_before_flat)
        spearman_correlations.append(spearman_corr)
        kendall_taus.append(kendall_tau)
        
        # Calculate top-k correlations
        for k in top_k:
            top_k_indices = np.argsort(logits_after_flat)[-k:]
            top_k_spearman_corr, _ = spearmanr(logits_after_flat[top_k_indices], 
                                              logits_before_flat[top_k_indices])
            top_k_kendall_tau, _ = kendalltau(logits_after_flat[top_k_indices], 
                                            logits_before_flat[top_k_indices])
            
            top_k_spearman_correlations[k].append(top_k_spearman_corr)
            top_k_kendall_taus[k].append(top_k_kendall_tau)
    
    return (spearman_correlations, kendall_taus, 
            top_k_spearman_correlations, top_k_kendall_taus)

def calculate_token_metrics(logits_before_quant, logits_after_quant, N, num_tokens, top_k_values=[3, 10, 20]):
    """Calculate token-wise correlation metrics for different top-k values."""
    token_metrics = []
    
    # For each token position
    for token_idx in range(num_tokens):
        token_data = {'token_index': token_idx}
        
        # Get all logits for this token position across all sequences
        before_logits = logits_before_quant[:, token_idx, :]  # Shape: [N, vocab_size]
        after_logits = logits_after_quant[:, token_idx, :]    # Shape: [N, vocab_size]
        
        # Calculate top-k correlations for each k
        for k in top_k_values:
            # For each sequence at this position
            k_kendall_scores = []
            k_spearman_scores = []
            
            for seq in range(N):
                # Get top-k indices based on before-quantization logits
                top_k_indices = np.argsort(before_logits[seq])[-k:]
                
                # Calculate correlations for top-k tokens
                kendall, _ = kendalltau(before_logits[seq][top_k_indices], 
                                      after_logits[seq][top_k_indices])
                spearman, _ = spearmanr(before_logits[seq][top_k_indices], 
                                      after_logits[seq][top_k_indices])
                
                k_kendall_scores.append(kendall)
                k_spearman_scores.append(spearman)
            
            # Average scores across sequences for this position
            token_data[f'kendall_top{k}'] = np.mean(k_kendall_scores)
            token_data[f'spearman_top{k}'] = np.mean(k_spearman_scores)
        
        token_metrics.append(token_data)
    
    return token_metrics

def plot_token_correlation_analysis(token_metrics, top_k_values=[3, 10, 20]):
    """Plot token-wise correlation analysis for different top-k values."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 2))
    df = pd.DataFrame(token_metrics)
    
    # Plot Kendall's Tau
    for i, k in enumerate(top_k_values):
        ax1.plot(df['token_index'], df[f'kendall_top{k}'], 
                label=f'Top-{k}', linewidth=1.3,
                color=COLOR_PALETTE[i % len(COLOR_PALETTE)],
                marker='o', markersize=3)
    
    ax1.set_xlabel('Token Position')
    ax1.set_ylabel('Kendall\'s Tau')
    # ax1.set_title('Token-wise Top-K Kendall Correlation')
    ax1.set_ylim(0.4, 0.9)
    ax1.legend(frameon=False, fancybox=True, framealpha=0.8, loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # Plot Spearman correlation
    for i, k in enumerate(top_k_values):
        ax2.plot(df['token_index'], df[f'spearman_top{k}'], 
                label=f'Top-{k}', linewidth=1.3,
                color=COLOR_PALETTE[i % len(COLOR_PALETTE)],
                marker='o', markersize=3)
    
    ax2.set_xlabel('Token Position')
    ax2.set_ylabel('Spearman Correlation')
    # ax2.set_title('Token-wise Top-K Spearman Correlation')
    ax2.set_ylim(0.5, 0.9)
    ax2.legend(frameon=False, fancybox=True, framealpha=0.8, loc='lower right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("./token_topk_correlations.png", dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Configuration
    model_path = "/data2/share/phi/Phi-3.5-mini-instruct"
    top_k = [3, 10, 20]
    
    # # Get model outputs using the new function
    # logits_after_quant, logits_before_quant = get_model_outputs_both(model_path)
    
    # # Calculate logits difference
    # logits_diff = logits_after_quant - logits_before_quant
    # N = logits_diff.shape[0]  # Number of sequences
    # num_tokens = 32  # Fixed number of tokens per sequence
    
    # # Generate analysis and plots
    # plot_logits_analysis(
    #     logits_before_quant,
    #     logits_after_quant,
    #     N=min(10, logits_before_quant.shape[0]),  # 选择前10个token进行分析
    #     top_k_values=[3, 10, 20]
    # )
    
    # correlations = calculate_correlations(logits_after_quant, logits_before_quant, N, top_k)
    # spearman_correlations, kendall_taus, top_k_spearman_correlations, top_k_kendall_taus = correlations
    
    # Calculate token-wise metrics with top-k analysis
    # token_metrics = calculate_token_metrics(logits_before_quant, logits_after_quant, 
    #                                      N, num_tokens, top_k_values=[3, 10, 20])
    
    token_metrics = pd.read_csv("./token_rankings.csv")
    # Plot token-wise correlation analysis
    plot_token_correlation_analysis(token_metrics, top_k_values=[3, 10, 20])
    
    # Save metrics to CSV
    pd.DataFrame(token_metrics).to_csv('./token_rankings.csv', index=False)

if __name__ == "__main__":
    main()