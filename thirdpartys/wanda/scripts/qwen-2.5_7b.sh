#!/bin/bash

# Set common variables
# model="/data2/share/internlm/internlm2_5-7b-chat"
model="/data2/share/Qwen2.5/Qwen2.5-7B-Instruct"
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

# internlm-7b with wanda pruning method
echo "Running with wanda pruning method"
run_python_command "wanda" "unstructured" "out/qwen_7b/unstructured/wanda/" "out/qwen_7b/unstructured/wanda/qwen-2.5-7b-chat-wanda-un0.5"
run_python_command "wanda" "2:4" "out/qwen_7b/2-4/wanda/" "out/qwen_7b/2-4/wanda/qwen-2.5-7b-chat-wanda-2-4-0.5"
# run_python_command "wanda" "4:8" "out/qwen_7b/4-8/wanda/"
echo "Finished wanda pruning method"

# internlm-7b with sparsegpt pruning method
echo "Running with sparsegpt pruning method"
run_python_command "sparsegpt" "unstructured" "out/qwen_7b/unstructured/sparsegpt/" "out/qwen_7b/unstructured/sparsegpt/qwen-2.5-7b-chat-sparsegpt-un0.5"
run_python_command "sparsegpt" "2:4" "out/qwen_7b/2-4/sparsegpt/" "out/qwen_7b/2-4/sparsegpt/qwen-2.5-7b-chat-sparsegpt-2-4-0.5"
# run_python_command "sparsegpt" "4:8" "out/qwen_7b/4-8/sparsegpt/"
echo "Finished sparsegpt pruning method"

# internlm-7b with magnitude pruning method
echo "Running with magnitude pruning method"
run_python_command "magnitude" "unstructured" "out/qwen_7b/unstructured/magnitude/" "out/qwen_7b/unstructured/magnitude/qwen-2.5-7b-chat-magnitude-un0.5"
run_python_command "magnitude" "2:4" "out/qwen_7b/2-4/magnitude/" "out/qwen_7b/2-4/magnitude/qwen-2.5-7b-chat-magnitude-2-4-0.5"
# run_python_command "magnitude" "4:8" "out/qwen_7b/4-8/magnitude/"
echo "Finished magnitude pruning method"