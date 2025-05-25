import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from datasets import load_dataset

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

def visualize_logits(model_path, quant_scheme, quant_method, 
                     smoothing_strength=0.8,
                     max_seq_length=2048,
                     num_calibration_samples=1024,
                     apply_quant=True):
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Load WikiText-2 dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    calibration_data = dataset['train']['text'][:64]  # Get first 64 samples
    
    # Combine samples with newline separator
    input_text = "\n".join([text for text in calibration_data if text.strip()])

    # Truncate to max sequence length if needed
    tokens = tokenizer.encode(input_text)
    if len(tokens) > max_seq_length:
        tokens = tokens[:max_seq_length]
        input_text = tokenizer.decode(tokens)

    # Load model with vLLM
    model = LLM(
        model=model_path,
        trust_remote_code=True,
        dtype="auto",
        gpu_memory_utilization=0.8,
        tensor_parallel_size=1
    )

    # Get model outputs using vLLM
    sampling_params = SamplingParams(temperature=0, max_tokens=1)
    outputs = model.generate(prompts=[input_text], sampling_params=sampling_params)
    logits = outputs[0].outputs[0].logits.detach().numpy()

    # Plot 5 subfigures in one plot
    plt.figure(figsize=(15, 25))
    cmap = plt.get_cmap('viridis')
    num_tokens = logits.shape[1] // 100
    for i in range(5):
        plt.subplot(5, 1, i+1)
        colors = cmap(np.linspace(0, 1, num_tokens))  # Generate colors for all tokens from the viridis colormap
        plt.bar(range(num_tokens), logits[i][:num_tokens], color=colors)
        plt.xlabel('Token Index')
        plt.ylabel('Logit Value')
        plt.title(f'Logits Visualization for Sequence Position {i}')
    
    # Save the logits to a numpy file
    if apply_quant:
        np.save("./logits_after_quant.npy", logits)
    else:
        np.save("./logits_before_quant.npy", logits)    

    # Plot the logits
    # plt.figure(figsize=(15, 5))
    # plt.imshow(logits, aspect='auto', cmap='viridis')
    # plt.colorbar(label='Logit Value')
    # plt.xlabel('Token Index')
    # plt.ylabel('Sequence Position')
    # plt.title('Logits Visualization for Quantized Model')
    
    if apply_quant:
        plt.savefig("./output_after_quant.png")
    else:
        plt.savefig("./output_before_quant.png")

def compare_logits(base_model_path, compressed_model_path, compression_name):
    # Get logits for both models
    visualize_logits(base_model_path, "", "", apply_quant=False)
    visualize_logits(compressed_model_path, "", "", apply_quant=True)
    
    # Load the saved logits
    base_logits = np.load("./logits_before_quant.npy")
    compressed_logits = np.load("./logits_after_quant.npy")
    
    # Create comparison plot
    plt.figure(figsize=(15, 25))
    num_tokens = base_logits.shape[1] // 100
    
    for i in range(5):
        plt.subplot(5, 1, i+1)
        plt.plot(range(num_tokens), base_logits[i][:num_tokens], 
                label='Uncompressed', alpha=0.7)
        plt.plot(range(num_tokens), compressed_logits[i][:num_tokens], 
                label=compression_name, alpha=0.7)
        plt.xlabel('Token Index')
        plt.ylabel('Logit Value')
        plt.title(f'Logits Comparison at Sequence Position {i}')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"./comparison_{compression_name.lower().replace(':', '_')}.png")
    plt.close()

# Paths
BASE_MODEL = "/data2/share/Qwen2.5/Qwen2.5-7B-Instruct"
WANDA_24_MODEL = "/path/to/out_pruned_llm/qwen_7b/2-4/wanda/qwen-2.5-7b-chat-wanda-2-4-0.5"
WANDA_UNSTRUCTURED_MODEL = "/path/to/out_pruned_llm/qwen_7b/unstructured/wanda/qwen-2.5-7b-chat-wanda-un0.5"
GPTQ_MODEL = "/data2/share/Qwen2.5/Qwen2.5-7B-Instruct-GPTQ-w4a16"

# Generate comparison plots
compare_logits(BASE_MODEL, WANDA_24_MODEL, "2:4-Wanda")
compare_logits(BASE_MODEL, WANDA_UNSTRUCTURED_MODEL, "Unstructured-Wanda")
compare_logits(BASE_MODEL, GPTQ_MODEL, "GPTQ-INT4")

# # Example usage
# visualize_logits(
#     model_path="/data2/share/llama3.1/llama-3.1-8B-Instruct",
#     quant_scheme="W8A8",
#     quant_method=["gptq"],
#     smoothing_strength=0.8,
#     max_seq_length=2048,
#     num_calibration_samples=128,
#     apply_quant=True,
# )


# # Example usage
# visualize_logits(
#     model_path="/data2/share/llama3.1/llama-3.1-8B-Instruct",
#     quant_scheme="W8A8",
#     quant_method=["gptq"],
#     smoothing_strength=0.8,
#     max_seq_length=2048,
#     num_calibration_samples=128,
#     apply_quant=False,
# )
