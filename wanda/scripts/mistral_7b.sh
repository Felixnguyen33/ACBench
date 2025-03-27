#!/bin/bash

# Set common variables
# model="decapoda-research/llama-7b-hf"
model="/data2/share/mistral/Mistral-7B-Instruct-v0.3"
sparsity_ratio=0.5
cuda_device=6

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
echo "Running with wanda pruning method"
run_python_command "wanda" "unstructured" "out/mistral_7b/unstructured/wanda/" "out/mistral_7b/unstructured/wanda/mistral-7b-instruct-wanda-un0.5"
run_python_command "wanda" "2:4" "out/mistral_7b/2-4/wanda/" "out/mistral_7b/2-4/wanda/mistral-7b-instruct-wanda-2-4-0.5"
run_python_command "wanda" "4:8" "out/mistral_7b/4-8/wanda/" "out/mistral_7b/4-8/wanda/mistral-7b-instruct-wanda-4-8-0.5"
echo "Finished wanda pruning method"

# llama3.1 with sparsegpt pruning method
echo "Running with sparsegpt pruning method"
run_python_command "sparsegpt" "unstructured" "out/mistral_7b/unstructured/sparsegpt/" "out/mistral_7b/unstructured/sparsegpt/mistral-7b-instruct-sparsegpt-un0.5"
run_python_command "sparsegpt" "2:4" "out/mistral_7b/2-4/sparsegpt/" "out/mistral_7b/2-4/sparsegpt/mistral-7b-instruct-sparsegpt-2-4-0.5"
run_python_command "sparsegpt" "4:8" "out/mistral_7b/4-8/sparsegpt/" "out/mistral_7b/4-8/sparsegpt/mistral-7b-instruct-sparsegpt-4-8-0.5"
echo "Finished sparsegpt pruning method"

# llama3.1 with magnitude pruning method
echo "Running with magnitude pruning method"
run_python_command "magnitude" "unstructured" "out/mistral_7b/unstructured/magnitude/" "out/mistral_7b/unstructured/magnitude/mistral-7b-instruct-magnitude-un0.5"
run_python_command "magnitude" "2:4" "out/mistral_7b/2-4/magnitude/" "out/mistral_7b/2-4/magnitude/mistral-7b-instruct-magnitude-2-4-0.5"
run_python_command "magnitude" "4:8" "out/mistral_7b/4-8/magnitude/" "out/mistral_7b/4-8/magnitude/mistral-7b-instruct-magnitude-4-8-0.5"
echo "Finished magnitude pruning method"
