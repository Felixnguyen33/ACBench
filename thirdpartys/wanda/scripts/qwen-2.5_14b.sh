#!/bin/bash

# Set common variables
model="/data2/share/Qwen2.5-14B-Instruct"
sparsity_ratio=0.5

# Define function to run python command
run_python_command () {
    export CUDA_VISIBLE_DEVICES=$5
    python main.py \
    --model $model \
    --prune_method $1 \
    --sparsity_ratio $sparsity_ratio \
    --sparsity_type $2 \
    --save $3 \
    --save_model $4
}

# qwen-14b with wanda pruning method
echo "Running with wanda pruning method"
# run_python_command "wanda" "unstructured" "out/qwen_14b/unstructured/wanda/" "out/qwen_14b/unstructured/wanda/qwen-2.5-14b-instruct-wanda-un0.5" 6
run_python_command "wanda" "2:4" "out/qwen_14b/2-4/wanda/" "/path/to/out_pruned_llm/qwen_14b/2-4/wanda/qwen-2.5-14b-instruct-wanda-2-4-0.5" 1 &
# run_python_command "wanda" "4:8" "out/qwen_14b/4-8/wanda/" 6
echo "Finished wanda pruning method"

# qwen-14b with sparsegpt pruning method
echo "Running with sparsegpt pruning method"
run_python_command "sparsegpt" "unstructured" "out/qwen_14b/unstructured/sparsegpt/" "/path/to/out_pruned_llm/qwen_14b/unstructured/sparsegpt/qwen-2.5-14b-instruct-sparsegpt-un0.5" 2 &
run_python_command "sparsegpt" "2:4" "out/qwen_14b/2-4/sparsegpt/" "/path/to/out_pruned_llm/qwen_14b/2-4/sparsegpt/qwen-2.5-14b-instruct-sparsegpt-2-4-0.5" 3 &
# run_python_command "sparsegpt" "4:8" "out/qwen_14b/4-8/sparsegpt/" 7
echo "Finished sparsegpt pruning method"

# qwen-14b with magnitude pruning method
echo "Running with magnitude pruning method"
run_python_command "magnitude" "unstructured" "out/qwen_14b/unstructured/magnitude/" "/path/to/out_pruned_llm/qwen_14b/unstructured/magnitude/qwen-2.5-14b-instruct-magnitude-un0.5" 4 &
run_python_command "magnitude" "2:4" "out/qwen_14b/2-4/magnitude/" "/path/to/out_pruned_llm/qwen_14b/2-4/magnitude/qwen-2.5-14b-instruct-magnitude-2-4-0.5" 5 &
# run_python_command "magnitude" "4:8" "out/qwen_14b/4-8/magnitude/" 8
echo "Finished magnitude pruning method"