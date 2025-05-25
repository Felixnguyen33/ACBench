#!/bin/bash 


export CUDA_VISIBLE_DEVICES=6,7
export NUMEXPR_MAX_THREADS=40  # 或在代码中设置 os.environ["NUMEXPR_MAX_THREADS"] = "80"

# Qwen2.5-7B-Instruct-W8A8-smooth0.8
# Qwen2.5-7B-Instruct-GPTQ-w4a16
# # qwen-2.5-7b-chat-sparsegpt-un0.5
# # Qwen2.5-3B-Instruct-GPTQ-Int8
# # Qwen2.5-3B-Instruct-GPTQ-Int4
# # Qwen2.5-3B-Instruct-AWQ
# llama-3.1-8B-Instruct-W8A8-smooth0.8
# llama-3.1-8B-Instruct-W8A8-gptq
# llama-3.1-8B-Instruct-GPTQ-w4a16
# internlm2_5-20b-chat-4bit-awq
# internlm2_5-20b-chat

# Define small language models
declare -A SLM_MODELS=(
    ["qwen-2.5-7b-chat-sparsegpt-un0.5"]="/data2/share/wanda/out/qwen_7b/unstructured/sparsegpt/qwen-2.5-7b-chat-sparsegpt-un0.5"
    ["Qwen2.5-7B-Instruct-W8A8-smooth0.8"]="/data2/share/Qwen2.5/Qwen2.5-7B-Instruct-W8A8-smooth0.8"
    ["Qwen2.5-7B-Instruct-GPTQ-w4a16"]="/data2/share/Qwen2.5/Qwen2.5-7B-Instruct-GPTQ-w4a16"
    ["llama-3.1-8B-Instruct-W8A8-smooth0.8"]="/data2/share/llama3.1/llama-3.1-8B-Instruct-W8A8-smooth0.8"
    ["llama-3.1-8B-Instruct-W8A8-gptq"]="/data2/share/llama3.1/llama-3.1-8B-Instruct-W8A8-gptq"
    ["llama-3.1-8B-Instruct-GPTQ-w4a16"]="/data2/share/llama3.1/llama-3.1-8B-Instruct-GPTQ-w4a16"
    ["internlm2.5-20b-awq"]="/data2/share/internlm/internlm2_5-20b-chat-4bit-awq"
    ["internlm2.5-20b"]="/data2/share/internlm/internlm2_5-20b-chat"
    ["qwen-3b-gptq-int4"]="/data2/share/Qwen2.5/Qwen2.5-3B-Instruct-GPTQ-Int4"
    ["qwen-3b-gptq-int8"]="/data2/share/Qwen2.5/Qwen2.5-3B-Instruct-GPTQ-Int8"
    ["qwen-3b-awq"]="/data2/share/Qwen2.5/Qwen2.5-3B-Instruct-AWQ"
    ["qwen-1.5b-gptq-int4"]="/data2/share/Qwen2.5/Qwen2.5-1.5B-Instruct-GPTQ-Int4"
    ["qwen-1.5b-gptq-int8"]="/data2/share/Qwen2.5/Qwen2.5-1.5B-Instruct-GPTQ-Int8"
    ["qwen-1.5b-awq"]="/data2/share/Qwen2.5/Qwen2.5-1.5B-Instruct-AWQ"
    ["gemma-2b"]="/data2/share/gemma/gemma-2-2b-it"
    ["phi-3.5"]="/data2/share/phi/Phi-3.5-mini-instruct"
    ["qwen-14b"]="/data2/share/Qwen2.5/Qwen2.5-14B-Instruct"
    ["qwen-32b"]="/data2/share/Qwen2.5/Qwen2.5-32B-Instruct"
    ["qwen-14b-awq"]="/data2/share/Qwen2.5/Qwen2.5-14B-Instruct-AWQ"
    ["qwen-32b-awq"]="/data2/share/Qwen2.5/Qwen2.5-32B-Instruct-AWQ"
    ["qwen-14b-gptq-int4"]="/data2/share/Qwen2.5/Qwen2.5-14B-Instruct-GPTQ-Int4"
    ["qwen-32b-gptq-int4"]="/data2/share/Qwen2.5/Qwen2.5-32B-Instruct-GPTQ-Int4"
    ["internlm3-8b"]="/data2/share/internlm/internlm3-8b-instruct"
    ["internlm3-8b-awq"]="/data2/share/internlm/internlm3-8b-instruct-awq"
    ["internlm3-8b-gptq-int4"]="/data2/share/internlm/internlm3-8b-instruct-gptq-int4"
    ["deepseek-qwen-1.5b"]="/data2/share/deepseek/DeepSeek-R1-Distill-Qwen-1.5B"
    ["deepseek-qwen-7b"]="/data2/share/deepseek/DeepSeek-R1-Distill-Qwen-7B"
    ["deepseek-llama-8b"]="/data2/share/deepseek/DeepSeek-R1-Distill-Llama-8B"
    ["minicpm-4b"]="/data2/share/openbmb/MiniCPM3-4B"
    ["megrez-3b"]="/data2/share/megrez/Megrez-3B-Instruct"

)

function generate_log_name() {
    local model=$1
    local quant=$2
    local temp=$3
    local current_date=$(date +%Y%m%d)
    echo "EVALFLOW-${model}-${quant}-t${temp}-${current_date}-llmc"
}

function generate_slm_log_name() {
    local model=$1
    local temp=$2
    local current_date=$(date +%Y%m%d)
    echo "EVALFLOW-SLM-${model}-t${temp}-${current_date}"
}

function run_slm_EVALFLOW() {
    local model_key=$1
    local temp=$2
    local model_path=${SLM_MODELS[$model_key]}
    local log_name=$(generate_slm_log_name "$model_key" "$temp")
    local model_name=$(basename "$model_path")

    echo "Generating with model: $model_key, path: $model_path"
    echo "Log name: $log_name"

    tasks=(wikihow toolbench toolalpaca lumos alfworld webshop os)
    
    for task in ${tasks[@]}; do
        # Check if eval output file exists
        eval_output_file="./data/eval_result/${model_name}/${model_name}_${task}_graph_eval_two_shot.json"
        if [ ! -f "$eval_output_file" ]; then
            echo "Running evaluation for task $task as output not found"
            python agentbench/node_eval.py \
                --task eval_workflow \
                --model_name ${model_path} \
                --gold_path ./data/gold_traj/${task}/graph_eval.json \
                --pred_path ./data/pred_traj/${model_name}/${task}/${model_name}/graph_eval_two_shot.json \
                --eval_model all-mpnet-base-v2 \
                --eval_output ${eval_output_file} \
                --eval_type node \
                --task_type ${task} \
                --few_shot \
                --temperature ${temp} \
                --quantization "base"
        else
            echo "Skipping task $task as evaluation output already exists"
        fi
    done
}

# Check if temperature is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 {temperature}"
    exit 1
fi

TEMP=$1

# Create logs directory if it doesn't exist
mkdir -p logs

# Simplified loop without GPU cycling
for model_key in "${!SLM_MODELS[@]}"; do
    echo "Running with model: $model_key on GPU 0"
    run_slm_EVALFLOW "$model_key" "$TEMP"
done
