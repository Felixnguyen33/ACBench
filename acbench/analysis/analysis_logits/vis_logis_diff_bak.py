import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

def visualize_logits(model_path, quant_scheme, quant_method, 
                     smoothing_strength=0.8,
                     max_seq_length=2048,
                     num_calibration_samples=1024,
                     apply_quant=True):
    
    if apply_quant:
        # Configure the quantization algorithms using bitsandbytes
        from transformers import BitsAndBytesConfig

        # Define the quantization configuration
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )

        # Load the quantized model
        model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=quantization_config)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Generate some input text
    input_text = (
        "Consider the following system of linear equations with five variables: "
        "2x + 3y - z + 4w - 5v = 10, "
        "3x - 2y + 5z - w + 4v = 15, "
        "-x + 4y + 2z + 3w - 2v = 8, "
        "5x + y - 3z + 2w + 6v = 20, "
        "4x - 3y + 2z - 5w + 3v = 12. "
        "Find the values of x, y, z, w, and v that satisfy all five equations simultaneously. "
        "Additionally, analyze the stability and uniqueness of the solution by examining the determinant of the coefficient matrix and discussing the implications of the solution space."
    )
    inputs = tokenizer(input_text, return_tensors="pt")

    # Get the model's output
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract logits
    logits = outputs.logits[0].detach().numpy()
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

# Example usage
visualize_logits(
    model_path="/data2/share/llama3.1/llama-3.1-8B-Instruct",
    quant_scheme="W8A8",
    quant_method=["gptq"],
    smoothing_strength=0.8,
    max_seq_length=2048,
    num_calibration_samples=128,
    apply_quant=True,
)


# Example usage
visualize_logits(
    model_path="/data2/share/llama3.1/llama-3.1-8B-Instruct",
    quant_scheme="W8A8",
    quant_method=["gptq"],
    smoothing_strength=0.8,
    max_seq_length=2048,
    num_calibration_samples=128,
    apply_quant=False,
)