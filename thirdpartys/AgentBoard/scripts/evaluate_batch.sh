#!/bin/bash

# Create a cleanup function
cleanup() {
    # Kill all child processes
    pkill -P $$
    exit
}

# Set up trap
trap cleanup SIGINT SIGTERM

TASKS_LIST=("alfworld" "jericho" "pddl" "tool-query" "tool-operation" "scienceworld")

# MODEL_LIST=("Qwen2.5-7B-Instruct" "Qwen2.5-7B-Instruct-AWQ" "Qwen2.5-7B-Instruct-Mag-Un-0.5" \
# "Qwen2.5-7B-Instruct-Mag-2-4-0.5" "Qwen2.5-7B-Instruct-Wanda-Un-0.5" "Qwen2.5-7B-Instruct-Wanda-2-4-0.5" \
# "Qwen2.5-7B-Instruct-SparseGPT-Un-0.5" "Qwen2.5-7B-Instruct-SparseGPT-2-4-0.5" "InternLM2.5-7B-Instruct" \
# "InternLM2.5-7B-Instruct-Mag-Un-0.5" "InternLM2.5-7B-Instruct-Mag-2-4-0.5" "InternLM2.5-7B-Instruct-Wanda-Un-0.5" \
# "InternLM2.5-7B-Instruct-Wanda-2-4-0.5" "InternLM2.5-7B-Instruct-SparseGPT-Un-0.5" "InternLM2.5-7B-Instruct-SparseGPT-2-4-0.5" \
# "InternLM2.5-7B-Instruct-AWQ" "InternLM2.5-7B-Instruct-GPTQ-w4a16" "InternLM2.5-7B-Instruct-RTN-w4" \
# "Qwen2.5-7B-Instruct-RTN-w4" "Qwen2.5-7B-Instruct-GPTQ-w4a16")
# MODEL_LIST=("deepseek-qwen-1.5b" "deepseek-qwen-7b" "deepseek-llama-8b" "minicpm-4b" "megrez-3b" "qwen-3b-gptq-int4" "qwen-3b-gptq-int8" "qwen-3b-awq" "qwen-1.5b-gptq-int4" "qwen-1.5b-gptq-int8" "qwen-1.5b-awq" "gemma-2b" "phi-3.5")

# MODEL_LIST=("qwen-3b-gptq-int4" "qwen-3b-gptq-int8" "qwen-3b-awq" "qwen-1.5b-gptq-int4" "qwen-1.5b-gptq-int8" "qwen-1.5b-awq")

MODEL_LIST=("qwen-1.5b" "qwen-1.5b")

# Split MODEL_LIST into two parts
MODEL_LIST_1=("${MODEL_LIST[@]:0:1}")  # First 10 models
MODEL_LIST_2=("${MODEL_LIST[@]:1}")    # Remaining models

# Create logs directory if it doesn't exist
mkdir -p logs

# Run first half of models on GPUs 2,3
{
for model in "${MODEL_LIST_1[@]}"; do
    for task in "${TASKS_LIST[@]}"; do
        echo "Running evaluation for model: $model, task: $task on GPUs 4"
        CUDA_VISIBLE_DEVICES=4 python agentboard/eval_main.py \
            --cfg eval_configs/main_results_all_tasks.yaml \
            --tasks "$task" \
            --model "$model" \
            --log_path "results/$model" \
            --wandb \
            --project_name "evaluate-$model" \
            --baseline_dir data/baseline_results \
            2>&1 | tee "logs/${model}_${task}.log"
        echo "Completed evaluation for model: $model, task: $task"
    done
done
} &

# # Run second half of models on GPUs 4,5
# {
# for model in "${MODEL_LIST_2[@]}"; do
#     for task in "${TASKS_LIST[@]}"; do
#         echo "Running evaluation for model: $model, task: $task on GPUs 4,5"
#         CUDA_VISIBLE_DEVICES=6 python agentboard/eval_main.py \
#             --cfg eval_configs/main_results_all_tasks.yaml \
#             --tasks "$task" \
#             --model "$model" \
#             --log_path "results/$model" \
#             --wandb \
#             --project_name "evaluate-$model" \
#             --baseline_dir data/baseline_results \
#             2>&1 | tee "logs/${model}_${task}.log"
#         echo "Completed evaluation for model: $model, task: $task"
#     done
# done
# } &

# # Wait for both processes to complete
# wait