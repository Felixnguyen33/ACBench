#!/bin/bash

export CUDA_VISIBLE_DEVICES=1


# Base model path - adjust this to your model path
MODEL_PATH="/data2/share/llama-2/Llama-2-7b-hf/"  # or your specific model path

# Output path for saving quantized model (optional)
OUTPUT_PATH="./quantized_model"

# Path to reparameterization factors file (if using)
# REP_FILE="./rep_factors.pt"  # optional

# Run the main logits analysis script
python erank_logits.py \
    --output_path $OUTPUT_PATH \
    --logits_path ./save_logits/ \
    --use_flash_attn \
    --model_name llama-2-7b-hf 