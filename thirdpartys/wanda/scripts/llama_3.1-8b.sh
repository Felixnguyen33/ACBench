#!/bin/bash

# Set common variables
# model="decapoda-research/llama-7b-hf"
model="/data2/share/llama3.1/llama-3.1-8B-Instruct"
sparsity_ratio=0.5
cuda_device=3

# Set CUDA device visibility
export CUDA_VISIBLE_DEVICES=$cuda_device

# Define function to run python command
run_python_command () {
    python main.py \
    --model $model \
    --prune_method $1 \
    --sparsity_ratio $sparsity_ratio \
    --sparsity_type $2 \
    --save $3 \
    --save_model $4
}

# llama3.1 with wanda pruning method
# echo "Running with wanda pruning method"
# run_python_command "wanda" "unstructured" "out/llama_3.1_8b/unstructured/wanda/" "out/llama_3.1_8b/unstructured/wanda/llama-3.1-8b-instruct-wanda-un0.5"
# run_python_command "wanda" "2:4" "out/llama_3.1_8b/2-4/wanda/" "out/llama_3.1_8b/2-4/wanda/llama-3.1-8b-instruct-wanda-2-4-0.5"
# run_python_command "wanda" "4:8" "out/llama_3.1_8b/4-8/wanda/" "out/llama_3.1_8b/4-8/wanda/llama-3.1-8b-instruct-wanda-4-8-0.5"
# echo "Finished wanda pruning method"

# llama3.1 with sparsegpt pruning method
# echo "Running with sparsegpt pruning method"
run_python_command "sparsegpt" "unstructured" "out/llama_3.1_8b/unstructured/sparsegpt/" "out/llama_3.1_8b/unstructured/sparsegpt/llama-3.1-8b-instruct-sparsegpt-un0.5"
# run_python_command "sparsegpt" "2:4" "out/llama_3.1_8b/2-4/sparsegpt/" "out/llama_3.1_8b/2-4/sparsegpt/llama-3.1-8b-instruct-sparsegpt-2-4-0.5"
# run_python_command "sparsegpt" "4:8" "out/llama_3.1_8b/4-8/sparsegpt/" "out/llama_3.1_8b/4-8/sparsegpt/llama-3.1-8b-instruct-sparsegpt-4-8-0.5"
# echo "Finished sparsegpt pruning method"

# llama3.1 with magnitude pruning method
# echo "Running with magnitude pruning method"
# run_python_command "magnitude" "unstructured" "out/llama_3.1_8b/unstructured/magnitude/" "out/llama_3.1_8b/unstructured/magnitude/llama-3.1-8b-instruct-magnitude-un0.5"
# run_python_command "magnitude" "2:4" "out/llama_3.1_8b/2-4/magnitude/" "out/llama_3.1_8b/2-4/magnitude/llama-3.1-8b-instruct-magnitude-2-4-0.5"
# run_python_command "magnitude" "4:8" "out/llama_3.1_8b/4-8/magnitude/" "out/llama_3.1_8b/4-8/magnitude/llama-3.1-8b-instruct-magnitude-4-8-0.5"
# echo "Finished magnitude pruning method"
