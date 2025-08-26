#!/bin/bash

DEVICE=${1:-"4,5"}  # Use first argument as device, default to "4,5"

# Set CUDA visible devices to use GPUs 4 and 5
export CUDA_VISIBLE_DEVICES=${DEVICE}

echo "Using GPUs: ${DEVICE}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

# Clear GPU memory before starting
python -c "
import torch
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        torch.cuda.empty_cache()
        print(f'Cleared GPU {i} memory')
        print(f'GPU {i} memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB')
"

# Memory optimization settings
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

MODEL="Qwen/Qwen2.5-14B"  # DeepSeek Distill 7B model from Hugging Face
TEMP=0.7
QUANT=none

tasks=(toolbench toolalpaca lumos alfworld webshop os)

MODEL_NAME=$(basename $MODEL)

# Create prediction directory if it doesn't exist
mkdir -p ./data/pred_traj/${MODEL_NAME}

for task in ${tasks[@]}; do
    echo "Processing task: ${task}"
    
    # Check if gold trajectory file exists
    if [ ! -f "./data/gold_traj/${task}/graph_eval.json" ]; then
        echo "Warning: Gold trajectory file not found for task ${task}, skipping..."
        continue
    fi
    
    # Create prediction directory for this task if it doesn't exist
    mkdir -p ./data/pred_traj/${MODEL_NAME}/${task}
    
    # Step 1: Generate workflow predictions
    echo "Generating workflow predictions for ${task}..."
    python acbench/node_eval.py \
        --task gen_workflow \
        --model_name "${MODEL}" \
        --gold_path ./data/gold_traj/${task}/graph_eval.json \
        --pred_path ./data/pred_traj/${MODEL_NAME}/${task}/graph_eval_two_shot.json \
        --task_type ${task} \
        --few_shot \
        --temperature ${TEMP} \
        --quantization ${QUANT} \
        --tensor_parallel_size 1 \
        --device cuda \
        --dtype fp16
    
    if [ $? -ne 0 ]; then
        echo "Error: Failed to generate predictions for task ${task}"
        exit 1
    fi
    
    # Step 2: Evaluate predictions and calculate P, R, F1 scores
    echo "Evaluating predictions for ${task}..."
    
    # Create eval_result directory if it doesn't exist
    mkdir -p ./data/eval_result/${MODEL_NAME}
    
    # Check if prediction file exists and is not empty
    if [ ! -f "./data/pred_traj/${MODEL_NAME}/${task}/graph_eval_two_shot.json" ]; then
        echo "Warning: Prediction file not found for task ${task}, skipping evaluation..."
        continue
    fi
    
    # Check if prediction file has content
    if [ ! -s "./data/pred_traj/${MODEL_NAME}/${task}/graph_eval_two_shot.json" ]; then
        echo "Warning: Prediction file is empty for task ${task}, skipping evaluation..."
        continue
    fi
    
    python acbench/node_eval.py \
        --task eval_workflow \
        --gold_path ./data/gold_traj/${task}/graph_eval.json \
        --pred_path ./data/pred_traj/${MODEL_NAME}/${task}/graph_eval_two_shot.json \
        --eval_model all-mpnet-base-v2 \
        --eval_output ./data/eval_result/${MODEL_NAME}/${MODEL_NAME}_${task}_node_eval_two_shot.json \
        --eval_type node
    
    if [ $? -ne 0 ]; then
        echo "Error: Failed to evaluate task ${task}"
        exit 1
    fi
    
    echo "Completed task: ${task}"
done