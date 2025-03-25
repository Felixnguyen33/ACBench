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

AVA_CUDA_DEVICES=(0 1 2 3 4)
declare -A MODEL_TO_DEVICE=(
    ["internlm2.5"]="6"
    ["qwen2.5"]="6"
)
MODEL_TYPES=("internlm2.5" "qwen2.5")
PRUNE_TYPES=("magnitude-2-4" "sparsegpt-2-4" "wanda-2-4" "magnitude-un" "sparsegpt-un" "wanda-un")

function generate_log_name() {
    local model=$1
    local prune=$2
    local temp=$3
    local current_date=$(date +%Y%m%d)
    echo "EVAL-${model}-${prune}-t${temp}-${current_date}"
}

function run_eval() {
    local model=$1
    local temp=$2  
    local prune=$3
    local device=${MODEL_TO_DEVICE[$model]}
    local log_name=$(generate_log_name "$model" "$prune" "$temp")

    echo "Generating with log name: $log_name"

    case "$model" in
        "internlm2.5")
            case "$prune" in
                "magnitude-2-4")
                    bash ./scripts/eval.sh /data2/share/peijiedong/AgentBench/thirdparty/wanda/out/internlm_7b/2-4/magnitude/internlm-2.5-7b-chat-magnitude-2-4-0.5 "$temp" "$prune" "$device" > "logs/${log_name}.log" 2>&1
                    ;;
                "sparsegpt-2-4")
                    bash ./scripts/eval.sh /data2/share/peijiedong/AgentBench/thirdparty/wanda/out/internlm_7b/2-4/sparsegpt/internlm-2.5-7b-chat-sparsegpt-2-4-0.5 "$temp" "$prune" "$device" > "logs/${log_name}.log" 2>&1
                    ;;
                "wanda-2-4")
                    bash ./scripts/eval.sh /data2/share/peijiedong/AgentBench/thirdparty/wanda/out/internlm_7b/2-4/wanda/internlm-2.5-7b-chat-wanda-2-4-0.5 "$temp" "$prune" "$device" > "logs/${log_name}.log" 2>&1
                    ;;
                "magnitude-un")
                    bash ./scripts/eval.sh /data2/share/peijiedong/AgentBench/thirdparty/wanda/out/internlm_7b/unstructured/magnitude/internlm-2.5-7b-chat-magnitude-un0.5 "$temp" "$prune" "$device" > "logs/${log_name}.log" 2>&1
                    ;;
                "sparsegpt-un")
                    bash ./scripts/eval.sh /data2/share/peijiedong/AgentBench/thirdparty/wanda/out/internlm_7b/unstructured/sparsegpt/internlm-2.5-7b-chat-sparsegpt-un0.5 "$temp" "$prune" "$device" > "logs/${log_name}.log" 2>&1
                    ;;
                "wanda-un")
                    bash ./scripts/eval.sh /data2/share/peijiedong/AgentBench/thirdparty/wanda/out/internlm_7b/unstructured/wanda/internlm-2.5-7b-chat-wanda-un0.5 "$temp" "$prune" "$device" > "logs/${log_name}.log" 2>&1
                    ;;
            esac
            ;;
        "qwen2.5")
            case "$prune" in
                "magnitude-2-4")
                    bash ./scripts/eval.sh /data2/share/peijiedong/AgentBench/thirdparty/wanda/out/qwen_7b/2-4/magnitude/qwen-2.5-7b-chat-magnitude-2-4-0.5 "$temp" "$prune" "$device" > "logs/${log_name}.log" 2>&1
                    ;;
                "sparsegpt-2-4")
                    bash ./scripts/eval.sh /data2/share/peijiedong/AgentBench/thirdparty/wanda/out/qwen_7b/2-4/sparsegpt/qwen-2.5-7b-chat-sparsegpt-2-4-0.5 "$temp" "$prune" "$device" > "logs/${log_name}.log" 2>&1
                    ;;
                "wanda-2-4")
                    bash ./scripts/eval.sh /data2/share/peijiedong/AgentBench/thirdparty/wanda/out/qwen_7b/2-4/wanda/qwen-2.5-7b-chat-wanda-2-4-0.5 "$temp" "$prune" "$device" > "logs/${log_name}.log" 2>&1
                    ;;
                "magnitude-un")
                    bash ./scripts/eval.sh /data2/share/peijiedong/AgentBench/thirdparty/wanda/out/qwen_7b/unstructured/magnitude/qwen-2.5-7b-chat-magnitude-un0.5 "$temp" "$prune" "$device" > "logs/${log_name}.log" 2>&1
                    ;;
                "sparsegpt-un")
                    bash ./scripts/eval.sh /data2/share/peijiedong/AgentBench/thirdparty/wanda/out/qwen_7b/unstructured/sparsegpt/qwen-2.5-7b-chat-sparsegpt-un0.5 "$temp" "$prune" "$device" > "logs/${log_name}.log" 2>&1
                    ;;
                "wanda-un")
                    bash ./scripts/eval.sh /data2/share/peijiedong/AgentBench/thirdparty/wanda/out/qwen_7b/unstructured/wanda/qwen-2.5-7b-chat-wanda-un0.5 "$temp" "$prune" "$device" > "logs/${log_name}.log" 2>&1
                    ;;
            esac
            ;;
    esac
}

# Check if temperature is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 {temperature}"
    exit 1
fi

TEMP=$1

# Create logs directory if it doesn't exist
mkdir -p logs

# Traverse all combinations
for model in "${MODEL_TYPES[@]}"; do
    for prune in "${PRUNE_TYPES[@]}"; do
        device=${MODEL_TO_DEVICE[$model]}
        echo "Running with model: $model, prune: $prune, temp: $TEMP, device: $device"
        run_eval "$model" "$TEMP" "$prune" "$device"
    done
done