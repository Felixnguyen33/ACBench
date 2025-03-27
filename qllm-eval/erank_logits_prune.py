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
    "qwen-2-4-wanda": "/mnt/sdd/dongpeijie/out_pruned_llm/qwen_7b/2-4/wanda/qwen-2.5-7b-chat-wanda-2-4-0.5",
    "qwen-un-wanda": "/mnt/sdd/dongpeijie/out_pruned_llm/qwen_7b/unstructured/wanda/qwen-2.5-7b-chat-wanda-un0.5",
}

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, choices=list(candidate_model_dict.keys()),
                   help="name of the model to use from available models")
parser.add_argument("--output_path", type=str, help="path to save results")
parser.add_argument("--use_flash_attn", action="store_true")
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
        # Move computation to CPU and use float64 for better numerical stability
        logits = logits.cpu().double()
        
        # Normalize the logits
        mean = logits.mean(dim=0)
        R = logits - mean
        norms = torch.norm(R, p=2, dim=1, keepdim=True)
        R = R/norms
        
        # Calculate covariance with smaller batch size
        batch_size = 128  # Reduced batch size
        Z = torch.nn.functional.normalize(R, dim=1)
        
        # Compute correlation matrix in batches
        num_rows = Z.shape[0]
        A = torch.zeros((Z.shape[1], Z.shape[1]), dtype=Z.dtype)
        
        for i in range(0, num_rows, batch_size):
            end_idx = min(i + batch_size, num_rows)
            batch = Z[i:end_idx]
            A += torch.matmul(batch.T, batch)
        
        A = A / num_rows

        # Use more stable SVD computation
        try:
            U, S, Vh = torch.linalg.svd(A / torch.trace(A), full_matrices=False)
            eig_val = S
        except RuntimeError:
            # If SVD fails, try with even smaller chunks
            logging.warning("Regular SVD failed, trying alternative method")
            eig_val = torch.linalg.eigvalsh(A / torch.trace(A))
            eig_val = torch.abs(eig_val)  # Ensure positive values
            
        # Remove very small values and sort
        eig_val = eig_val[eig_val > 1e-12]  # Increased threshold
        eig_val = torch.sort(eig_val, descending=True)[0]
        
        # Compute entropy and erank
        entropy = -(eig_val * torch.log(eig_val)).sum().item()
        erank = math.exp(entropy)
        
        # Clear memory
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
    
    # Load model and tokenizer
    model, enc = build_model_and_enc(model_path, args.use_flash_attn)
    input_ids = enc(prompt, return_tensors="pt")['input_ids'].to(next(model.parameters()).device)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits
        
    torch.save(logits, os.path.join(args.logits_path, f'{args.model_name}_{timestamp}.pt'))
        
    return timestamp

def plot_from_logits(timestamp, load_from_json=None):
    results = {'models': [], 'erank': []}
    
    if load_from_json:
        # Load results from JSON file
        with open(load_from_json, 'r') as f:
            results = json.load(f)
    else:
        # Load and process logits with weights_only=True
        logits = torch.load(os.path.join(args.logits_path, f'{args.model_name}_{timestamp}.pt'), weights_only=True)
        erank_value = compute_erank(logits[0])
        results['models'].append(args.model_name)
        results['erank'].append(erank_value)

    # Create single plot
    plt.figure(figsize=(6, 4))
    plt.plot(range(len(results['models'])), results['erank'], marker='o', color=COLOR_PALETTE[0])
    plt.xticks(range(len(results['models'])), results['models'], rotation=45)
    plt.ylabel('ERank')
    plt.title('ERank Analysis of Pruned Models')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'pruning_analysis_{timestamp}_{args.model_name}.png', dpi=250)
    plt.close()

def main():
    if args.load:
        plot_from_logits(None, load_from_json=args.load)
    else:
        timestamp = save_logits()
        plot_from_logits(timestamp)

if __name__ == "__main__":
    main()
