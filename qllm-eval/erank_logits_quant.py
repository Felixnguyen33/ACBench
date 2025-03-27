import argparse
import torch
import logging
import sys
from datetime import datetime
import os
import math
import matplotlib.pyplot as plt
import copy
import json 

from qllm_eval.quantization.quant_wrapper import quantize_model
from qllm_eval.utils import build_model_and_enc
from datasets import load_dataset

COLOR_PALETTE = ['#8ECFC9', '#FFBE7A', '#FA7F6F', '#82B0D2', '#BEB8DC', '#E7DAD2', '#999999']

candidate_model_dict = {
    "llama-2-7b-hf": "/data2/share/llama-2/Llama-2-7b-hf",
    "mistral-7b": "/data2/share/mistral/Mistral-7B-v0.1",
}

candidate_w_bits = [2, 3, 4, 8, 16]
candidate_kv_bits = [4, 8, 16]

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, choices=list(candidate_model_dict.keys()),
                   help="name of the model to use from available models")
parser.add_argument("--output_path", type=str, help="path to save the quantized model")
parser.add_argument("--use_flash_attn", action="store_true")
parser.add_argument("--w_group_size", type=int, default=128)
parser.add_argument("--w_bit", type=int, default=16)
parser.add_argument("--kv_group_size", type=int, default=128)
parser.add_argument("--kv_bit", type=int, default=16)
parser.add_argument("--top_k", type=int, default=10, help="number of top logits to show")
parser.add_argument("--log_file", type=str, default="logits.log", help="path to save logs")
parser.add_argument("--logits_path", type=str, default="./save_logits/", help="path to save logits")
parser.add_argument("--load", type=str, help="load results from JSON file instead of computing new ones")
args = parser.parse_args()

def setup_logging(log_file=None, log_level="INFO"):
    """Setup logging configuration"""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    # Set up basic configuration
    logging.basicConfig(
        level=getattr(logging, log_level),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Add file handler if log_file is specified
    if log_file:
        # Add timestamp to log filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_file.replace(".log", f"_{timestamp}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)

def analyze_logits(logits, tokenizer, input_ids, top_k=10):
    """Analyze logits for each position in detail"""
    batch_size, seq_len, vocab_size = logits.shape
    
    for b in range(batch_size):
        logging.info(f"\nAnalyzing sequence {b}:")
        for pos in range(seq_len):
            # Get the actual token at this position
            curr_token_id = input_ids[b, pos].item()
            curr_token = tokenizer.decode([curr_token_id])
            
            # Get logits for this position
            pos_logits = logits[b, pos]
            
            # Get top k predictions
            top_values, top_indices = torch.topk(pos_logits, k=top_k)
            top_tokens = [tokenizer.decode([idx.item()]) for idx in top_indices]
            
            # Get probability distribution using softmax
            probs = torch.softmax(pos_logits, dim=-1)
            top_probs = probs[top_indices]
            
            logging.info(f"\nPosition {pos}: Current token = '{curr_token}'")
            logging.info(f"Top {top_k} predictions:")
            for val, prob, tok in zip(top_values, top_probs, top_tokens):
                logging.info(f"  {tok}: logit = {val:.3f}, prob = {prob:.3%}")

def get_wikitext_prompt():
    """Load and return a random prompt from WikiText-2 test set"""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    
    # Filter out empty strings and very short sequences
    valid_texts = [text for text in dataset["text"] if len(text.strip()) > 50]
    
    # Select a random prompt
    prompt = torch.randint(0, len(valid_texts), (1,)).item()
    return valid_texts[prompt]

def compute_erank(logits):
    """Compute effective rank (erank) from logits"""
    with torch.no_grad():
        # Move computation to CPU and use float32 for better memory efficiency
        logits = logits.cpu().float()
        
        # Normalize the logits
        mean = logits.mean(dim=0)
        R = logits - mean
        norms = torch.norm(R, p=2, dim=1, keepdim=True)
        R = R/norms
        
        # Calculate covariance with batch processing
        batch_size = 1024  # Adjust this based on your available memory
        Z = torch.nn.functional.normalize(R, dim=1)
        
        # Compute correlation matrix in batches
        num_rows = Z.shape[0]
        A = torch.zeros((Z.shape[1], Z.shape[1]), dtype=Z.dtype)
        
        for i in range(0, num_rows, batch_size):
            end_idx = min(i + batch_size, num_rows)
            batch = Z[i:end_idx]
            A += torch.matmul(batch.T, batch)
        
        A = A / num_rows
        
        # Calculate erank using SVD
        try:
            # Try using regular SVD first
            eig_val = torch.svd(A / torch.trace(A))[1]
        except:
            # If regular SVD fails, use a more memory-efficient alternative
            eig_val = torch.linalg.svdvals(A / torch.trace(A))
        
        # Remove very small values to avoid NaN in log
        eig_val = eig_val[eig_val > 1e-10]
        entropy = -(eig_val * torch.log(eig_val)).nansum().item()
        erank = math.exp(entropy)
        
        # Clear some memory
        del R, Z, A
        torch.cuda.empty_cache()
        
    return erank

def save_logits():
    # Setup logging
    setup_logging(args.log_file, "INFO")
    logging.info("Starting logits saving")
    
    # Get model path from dictionary
    model_path = candidate_model_dict[args.model_name]
    logging.info(f"Using model: {args.model_name} from path: {model_path}")
    
    # Get WikiText prompt
    prompt = get_wikitext_prompt()
    
    # Create save directory if not exists
    os.makedirs(args.logits_path, exist_ok=True)
    
    # Load base model and tokenizer once
    base_model, enc = build_model_and_enc(model_path, args.use_flash_attn)
    input_ids = enc(prompt, return_tensors="pt")['input_ids'].to(next(base_model.parameters()).device)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save logits for w_bits and kv_bits only
    for w_bit in candidate_w_bits:
        args.w_bit = w_bit
        model = quantize_model(copy.deepcopy(base_model), args)
        
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits
            
        torch.save(logits, os.path.join(args.logits_path, f'w{w_bit}_{timestamp}.pt'))
    
    for kv_bit in candidate_kv_bits:
        args.kv_bit = kv_bit
        model = quantize_model(copy.deepcopy(base_model), args)
        
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits
            
        torch.save(logits, os.path.join(args.logits_path, f'kv{kv_bit}_{timestamp}.pt'))
        
    return timestamp

def plot_from_logits(timestamp, load_from_json=None):
    results = {
        'w_bits': {'bits': [], 'erank': []},
        'kv_bits': {'bits': [], 'erank': []}
    }
    
    if load_from_json:
        # Load results from JSON file
        with open(load_from_json, 'r') as f:
            results = json.load(f)
    else:
        # Load and process w_bits logits
        for w_bit in candidate_w_bits:
            logits = torch.load(os.path.join(args.logits_path, f'w{w_bit}_{timestamp}.pt'))
            erank_value = compute_erank(logits[0])
            results['w_bits']['bits'].append(w_bit)
            results['w_bits']['erank'].append(erank_value)
        
        # Load and process kv_bits logits
        for kv_bit in candidate_kv_bits:
            logits = torch.load(os.path.join(args.logits_path, f'kv{kv_bit}_{timestamp}.pt'))
            erank_value = compute_erank(logits[0])
            results['kv_bits']['bits'].append(kv_bit)
            results['kv_bits']['erank'].append(erank_value)

    # Create plots with two subplots instead of three
    fig, axes = plt.subplots(1, 2, figsize=(6, 2.5))
    plot_types = ['w_bits', 'kv_bits']
    titles = ['Weight Bits', 'KV Cache Bits']
    
    for idx, (plot_type, title) in enumerate(zip(plot_types, titles)):
        ax = axes[idx]
        
        bits = results[plot_type]['bits']
        erank = results[plot_type]['erank']
        
        if plot_type == 'w_bits':
            x_labels = [f'W{b}' for b in bits]
        else:  # kv_bits
            x_labels = [f'KV{b}' for b in bits]
        
        ax.plot(range(len(bits)), erank, marker='o', label='ERank', color=COLOR_PALETTE[len(COLOR_PALETTE)-3-idx])
        ax.set_xticks(range(len(bits)))
        ax.set_xticklabels(x_labels)
        ax.set_ylabel('ERank')
        ax.set_title(title)
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'quantization_analysis_{timestamp}_{args.model_fname}.png', dpi=250)
    plt.close()

    # with open(f'quantization_results_{timestamp}_{args.model_name}.json', 'w') as f:
    #     json.dump(results, f)

def main():
    if args.load:
        plot_from_logits(None, load_from_json=args.load)
    else:
        timestamp = save_logits()
        plot_from_logits(timestamp)

if __name__ == "__main__":
    main()
