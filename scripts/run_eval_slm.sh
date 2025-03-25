#!/bin/bash 

# LOG_NAME_FORMAT: {TYPE}-{MODEL}-{PRUNE}-t{TEMPERATURE}-{DATE}
# TYPE: GENFLOW; EVAL
# MODEL: internlm2.5; qwen2.5
# PRUNE: magnitude-2-4; sparsegpt-2-4; wanda-2-4; magnitude-un; sparsegpt-un; wanda-un
# TEMPERATURE: from args
# DATE: current date

# DEMO
# bash ./scripts/eval.sh {model} {temp} {prune} {device}
# bash ./scripts/eval.sh /data2/share/peijiedong/AgentBench/thirdparty/wanda/out/internlm_7b/2-4/magnitude/internlm-2.5-7b-chat-magnitude-2-4-0.5 0 magnitude-2-4 1

# MODEL+PRUNE=internlm2.5+2-4/unstructured
# ROOT_DIR=/data2/share/peijiedong/AgentBench/thirdparty/wanda/out/internlm_7b
# 2-4:
# magnitude: internlm-2.5-7b-chat-magnitude-2-4-0.5
# sparsegpt: internlm-2.5-7b-chat-sparsegpt-2-4-0.5  
# wanda: internlm-2.5-7b-chat-wanda-2-4-0.5
# unstructured:
# magnitude: internlm-2.5-7b-chat-magnitude-un0.5
# sparsegpt: internlm-2.5-7b-chat-sparsegpt-un0.5
# wanda: internlm-2.5-7b-chat-wanda-un0.5

# MODEL+PRUNE=qwen2.5+2-4/unstructured
# ROOT_DIR=/data2/share/peijiedong/AgentBench/thirdparty/wanda/out/qwen_7b
# 2-4:
# magnitude: qwen-2.5-7b-chat-magnitude-2-4-0.5
# sparsegpt: qwen-2.5-7b-chat-sparsegpt-2-4-0.5
# wanda: qwen-2.5-7b-chat-wanda-2-4-0.5
# unstructured:
# magnitude: qwen-2.5-7b-chat-magnitude-un0.5
# sparsegpt: qwen-2.5-7b-chat-sparsegpt-un0.5
# wanda: qwen-2.5-7b-chat-wanda-un0.5

# AVA_CUDA_DEVICES=(0 1 2 3 4)
# declare -A MODEL_TO_DEVICE=(
#     ["internlm2.5"]="6"
#     ["qwen2.5"]="6"
# )
# MODEL_TYPES=("internlm2.5" "qwen2.5")
# PRUNE_TYPES=("magnitude-2-4" "sparsegpt-2-4" "wanda-2-4" "magnitude-un" "sparsegpt-un" "wanda-un")


# Define paths for small language models
declare -A SLM_MODELS=(
    ["deepseek-qwen-1.5b"]="/data2/share/deepseek/DeepSeek-R1-Distill-Qwen-1.5B"
    ["deepseek-qwen-7b"]="/data2/share/deepseek/DeepSeek-R1-Distill-Qwen-7B"
    ["deepseek-llama-8b"]="/data2/share/deepseek/DeepSeek-R1-Distill-Llama-8B"
    ["minicpm-4b"]="/data2/share/openbmb/MiniCPM3-4B"
    ["megrez-3b"]="/data2/share/megrez/Megrez-3B-Instruct"
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
    ["internlm2.5-20b"]="/data2/share/internlm/internlm2_5-20b-chat"
    ["internlm2.5-20b-awq"]="/data2/share/internlm/internlm2_5-20b-chat-4bit-awq"
)

export CUDA_VISIBLE_DEVICES=4

function generate_log_name() {
    local model=$1
    local prune=$2
    local temp=$3
    local current_date=$(date +%Y%m%d)
    echo "EVAL-${model}-${prune}-t${temp}-${current_date}"
}

# function run_eval() {
#     local model=$1
#     local temp=$2  
#     local prune=$3
#     local device=${MODEL_TO_DEVICE[$model]}
#     local log_name=$(generate_log_name "$model" "$prune" "$temp")

#     echo "Generating with log name: $log_name"

#     case "$model" in
#         "internlm2.5")
#             case "$prune" in
#                 "magnitude-2-4")
#                     bash ./scripts/eval.sh /data2/share/peijiedong/AgentBench/thirdparty/wanda/out/internlm_7b/2-4/magnitude/internlm-2.5-7b-chat-magnitude-2-4-0.5 "$temp" "$prune" "$device" > "logs/${log_name}.log" 2>&1
#                     ;;
#                 "sparsegpt-2-4")
#                     bash ./scripts/eval.sh /data2/share/peijiedong/AgentBench/thirdparty/wanda/out/internlm_7b/2-4/sparsegpt/internlm-2.5-7b-chat-sparsegpt-2-4-0.5 "$temp" "$prune" "$device" > "logs/${log_name}.log" 2>&1
#                     ;;
#                 "wanda-2-4")
#                     bash ./scripts/eval.sh /data2/share/peijiedong/AgentBench/thirdparty/wanda/out/internlm_7b/2-4/wanda/internlm-2.5-7b-chat-wanda-2-4-0.5 "$temp" "$prune" "$device" > "logs/${log_name}.log" 2>&1
#                     ;;
#                 "magnitude-un")
#                     bash ./scripts/eval.sh /data2/share/peijiedong/AgentBench/thirdparty/wanda/out/internlm_7b/unstructured/magnitude/internlm-2.5-7b-chat-magnitude-un0.5 "$temp" "$prune" "$device" > "logs/${log_name}.log" 2>&1
#                     ;;
#                 "sparsegpt-un")
#                     bash ./scripts/eval.sh /data2/share/peijiedong/AgentBench/thirdparty/wanda/out/internlm_7b/unstructured/sparsegpt/internlm-2.5-7b-chat-sparsegpt-un0.5 "$temp" "$prune" "$device" > "logs/${log_name}.log" 2>&1
#                     ;;
#                 "wanda-un")
#                     bash ./scripts/eval.sh /data2/share/peijiedong/AgentBench/thirdparty/wanda/out/internlm_7b/unstructured/wanda/internlm-2.5-7b-chat-wanda-un0.5 "$temp" "$prune" "$device" > "logs/${log_name}.log" 2>&1
#                     ;;
#             esac
#             ;;
#         "qwen2.5")
#             case "$prune" in
#                 "magnitude-2-4")
#                     bash ./scripts/eval.sh /data2/share/peijiedong/AgentBench/thirdparty/wanda/out/qwen_7b/2-4/magnitude/qwen-2.5-7b-chat-magnitude-2-4-0.5 "$temp" "$prune" "$device" > "logs/${log_name}.log" 2>&1
#                     ;;
#                 "sparsegpt-2-4")
#                     bash ./scripts/eval.sh /data2/share/peijiedong/AgentBench/thirdparty/wanda/out/qwen_7b/2-4/sparsegpt/qwen-2.5-7b-chat-sparsegpt-2-4-0.5 "$temp" "$prune" "$device" > "logs/${log_name}.log" 2>&1
#                     ;;
#                 "wanda-2-4")
#                     bash ./scripts/eval.sh /data2/share/peijiedong/AgentBench/thirdparty/wanda/out/qwen_7b/2-4/wanda/qwen-2.5-7b-chat-wanda-2-4-0.5 "$temp" "$prune" "$device" > "logs/${log_name}.log" 2>&1
#                     ;;
#                 "magnitude-un")
#                     bash ./scripts/eval.sh /data2/share/peijiedong/AgentBench/thirdparty/wanda/out/qwen_7b/unstructured/magnitude/qwen-2.5-7b-chat-magnitude-un0.5 "$temp" "$prune" "$device" > "logs/${log_name}.log" 2>&1
#                     ;;
#                 "sparsegpt-un")
#                     bash ./scripts/eval.sh /data2/share/peijiedong/AgentBench/thirdparty/wanda/out/qwen_7b/unstructured/sparsegpt/qwen-2.5-7b-chat-sparsegpt-un0.5 "$temp" "$prune" "$device" > "logs/${log_name}.log" 2>&1
#                     ;;
#                 "wanda-un")
#                     bash ./scripts/eval.sh /data2/share/peijiedong/AgentBench/thirdparty/wanda/out/qwen_7b/unstructured/wanda/qwen-2.5-7b-chat-wanda-un0.5 "$temp" "$prune" "$device" > "logs/${log_name}.log" 2>&1
#                     ;;
#             esac
#             ;;
#     esac
# }

function generate_slm_log_name() {
    local model=$1
    local temp=$2
    local current_date=$(date +%Y%m%d)
    echo "EVAL-SLM-${model}-t${temp}-${current_date}"
}

function run_slm_eval() {
    local model_key=$1
    local temp=$2
    local model_path=${SLM_MODELS[$model_key]}
    local log_name=$(generate_slm_log_name "$model_key" "$temp")

    echo "Evaluating with model: $model_key, path: $model_path"
    echo "Log name: $log_name"

    bash ./scripts/eval.sh "$model_path" "$temp" "base" "1" > "logs/${log_name}.log" 2>&1
}

# Check if temperature is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 {temperature}"
    exit 1
fi

TEMP=$1

# Create logs directory if it doesn't exist
mkdir -p logs

# # Traverse all combinations
# for model in "${MODEL_TYPES[@]}"; do
#     for prune in "${PRUNE_TYPES[@]}"; do
#         device=${MODEL_TO_DEVICE[$model]}
#         echo "Running with model: $model, prune: $prune, temp: $TEMP, device: $device"
#         run_eval "$model" "$TEMP" "$prune" "$device"
#     done
# done

# Run evaluation for all SLM models
for model_key in "${!SLM_MODELS[@]}"; do
    echo "Running evaluation with model: $model_key on GPU 1"
    run_slm_eval "$model_key" "$TEMP"
done