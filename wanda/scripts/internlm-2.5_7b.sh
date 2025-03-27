#!/bin/bash

# Set common variables
model="/data2/share/internlm/internlm2_5-7b-chat"
sparsity_ratio=0.5
cuda_device=4

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
run_python_command "wanda" "unstructured" "out/internlm_7b/unstructured/wanda/" "out/internlm_7b/unstructured/wanda/internlm-2.5-7b-chat-wanda-un0.5"
run_python_command "wanda" "2:4" "out/internlm_7b/2-4/wanda/" "out/internlm_7b/2-4/wanda/internlm-2.5-7b-chat-wanda-2-4-0.5"
# run_python_command "wanda" "4:8" "out/internlm_7b/4-8/wanda/"
echo "Finished wanda pruning method"

# internlm-7b with sparsegpt pruning method
echo "Running with sparsegpt pruning method"
run_python_command "sparsegpt" "unstructured" "out/internlm_7b/unstructured/sparsegpt/" "out/internlm_7b/unstructured/sparsegpt/internlm-2.5-7b-chat-sparsegpt-un0.5"
run_python_command "sparsegpt" "2:4" "out/internlm_7b/2-4/sparsegpt/" "out/internlm_7b/2-4/sparsegpt/internlm-2.5-7b-chat-sparsegpt-2-4-0.5"
# run_python_command "sparsegpt" "4:8" "out/internlm_7b/4-8/sparsegpt/"
echo "Finished sparsegpt pruning method"

# # internlm-7b with magnitude pruning method
# echo "Running with magnitude pruning method"
# run_python_command "magnitude" "unstructured" "out/internlm_7b/unstructured/magnitude/" "out/internlm_7b/unstructured/magnitude/internlm-2.5-7b-chat-magnitude-un0.5"
# run_python_command "magnitude" "2:4" "out/internlm_7b/2-4/magnitude/" "out/internlm_7b/2-4/magnitude/internlm-2.5-7b-chat-magnitude-2-4-0.5"
# # run_python_command "magnitude" "4:8" "out/internlm_7b/4-8/magnitude/"
# echo "Finished magnitude pruning method"