#!/bin/bash 

# LOG_NAME_FORMAT: {TYPE}-{MODEL}-{QUANT}-t{TEMPERATURE}-{DATE}
# TYPE: GENFLOW; EVAL
# MODEL: llama3.1; qwen; mistral
# QUANT: gptq; smooth; awq;
# TEMPERATURE: from args
# DATE: current date

AVA_CUDA_DEVICES=(0 1 2 3 4)

# DEMO
# bash ./scripts/gen_flow.sh {model} {temp} {quant} {device}
# bash ./scripts/gen_flow.sh /data2/share/llama3.2/Llama-3.2-1B-Instruct-awq-w4-g128-zp 0 gptq 1 

# MODEL+QUANT=llama3.1+gptq/smooth/awq
# ROOT_DIR=/data2/share/llama3.1
# gptq: llama-3.1-8B-Instruct-W8A8-gptq
# smooth: llama-3.1-8B-Instruct-W8A8-smooth0.8
# awq: llama-3.1-8B-Instruct-awq-w4-g128-zp

# MODEL+QUANT=qwen2.5+gptq/smooth/awq
# ROOT_DIR=/data2/share/Qwen2.5 
# gptq: Qwen2.5-7B-Instruct-W8A8-gptq 
# smooth: Qwen2.5-7B-Instruct-W8A8-smooth0.8
# awq: Qwen2.5-7B-Instruct-awq-w4-g128-zp

# MODEL+QUANT=mistral0.3+gptq/smooth/awq
# ROOT_DIR=/data2/share/mistral-7B
# gptq: Mistral-7B-Instruct-v0.3-W8A8-gptq
# smooth: Mistral-7B-Instruct-v0.3-W8A8-smooth0.8
# awq: Mistral-7B-Instruct-v0.3-awq-w4-g128-zp


AVA_CUDA_DEVICES=(0 1 2 3 4)
declare -A MODEL_TO_DEVICE=(
    ["llama3.1"]="1"
    ["qwen2.5"]="2"
    ["mistral0.3"]="3"
)
MODEL_TYPES=("llama3.1" "qwen2.5" "mistral0.3")
QUANT_TYPES=("gptq" "smooth" "awq")

function generate_log_name() {
    local model=$1
    local quant=$2
    local temp=$3
    local current_date=$(date +%Y%m%d)
    echo "GENFLOW-${model}-${quant}-t${temp}-${current_date}-llmc"
}

function run_genflow() {
    local model=$1
    local temp=$2  
    local quant=$3
    local device=${MODEL_TO_DEVICE[$model]}
    local log_name=$(generate_log_name "$model" "$quant" "$temp")

    echo "Generating with log name: $log_name"

    case "$model" in
        "llama3.1")
            case "$quant" in
                "gptq")
                    bash ./scripts/gen_flow.sh /data2/share/llama3.1/llama-3.1-8B-Instruct-W8A8-gptq "$temp" "$quant" "$device" > "logs/${log_name}.log" 2>&1
                    ;;
                "smooth")
                    bash ./scripts/gen_flow.sh /data2/share/llama3.1/llama-3.1-8B-Instruct-W8A8-smooth0.8 "$temp" "$quant" "$device" > "logs/${log_name}.log" 2>&1
                    ;;
                "awq")
                    bash ./scripts/gen_flow.sh /data2/share/llama3.1/llama-3.1-8B-Instruct-awq-w4-g128-zp "$temp" "$quant" "$device" > "logs/${log_name}.log" 2>&1
                    ;;
            esac
            ;;
        "qwen2.5")
            case "$quant" in
                "gptq")
                    bash ./scripts/gen_flow.sh /data2/share/Qwen2.5/Qwen2.5-7B-Instruct-W8A8-gptq "$temp" "$quant" "$device" > "logs/${log_name}.log" 2>&1
                    ;;
                "smooth")
                    bash ./scripts/gen_flow.sh /data2/share/Qwen2.5/Qwen2.5-7B-Instruct-W8A8-smooth0.8 "$temp" "$quant" "$device" > "logs/${log_name}.log" 2>&1
                    ;;
                "awq")
                    bash ./scripts/gen_flow.sh /data2/share/Qwen2.5/Qwen2.5-7B-Instruct-awq-w4-g128-zp "$temp" "$quant" "$device" > "logs/${log_name}.log" 2>&1
                    ;;
            esac
            ;;
        "mistral0.3")
            case "$quant" in
                "gptq")
                    bash ./scripts/gen_flow.sh /data2/share/mistral-7B/Mistral-7B-Instruct-v0.3-W8A8-gptq "$temp" "$quant" "$device" > "logs/${log_name}.log" 2>&1
                    ;;
                "smooth")
                    bash ./scripts/gen_flow.sh /data2/share/mistral-7B/Mistral-7B-Instruct-v0.3-W8A8-smooth0.8 "$temp" "$quant" "$device" > "logs/${log_name}.log" 2>&1
                    ;;
                "awq")
                    bash ./scripts/gen_flow.sh /data2/share/mistral-7B/Mistral-7B-Instruct-v0.3-awq-w4-g128-zp "$temp" "$quant" "$device" > "logs/${log_name}.log" 2>&1
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
    for quant in "${QUANT_TYPES[@]}"; do
        device=${MODEL_TO_DEVICE[$model]}
        echo "Running with model: $model, quant: $quant, temp: $TEMP, device: $device"
        run_genflow "$model" "$TEMP" "$quant"
    done
done