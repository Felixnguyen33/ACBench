#!/bin/bash 

# LOG_NAME_FORMAT: {TYPE}-{MODEL}-{QUANT}-t{TEMPERATURE}-{DATE}
# TYPE: GENFLOW; EVAL
# MODEL: llama3.1; qwen; mistral
# QUANT: gptq; smooth; awq;
# TEMPERATURE: from args
# DATE: current date

# AVA_CUDA_DEVICES=(0 1 2 3 4)

# DEMO
# bash ./scripts/gen_flow.sh {model} {temp} {quant} {device}
# bash ./scripts/gen_flow.sh /data2/share/llama3.2/Llama-3.2-1B-Instruct-awq-w4-g128-zp 0 gptq 1 

# Small Language Model 
# Qwen2.5 3B/1.5B
# /data2/share/deepseek/DeepSeek-R1-Distill-Qwen-1.5B
# /data2/share/deepseek/DeepSeek-R1-Distill-Qwen-7B
# /data2/share/deepseek/DeepSeek-R1-Distill-Llama-8B
# /data2/share/openbmb/MiniCPM3-4B
# /data2/share/megrez/Megrez-3B-Instruct
# /data2/share/Qwen2.5/Qwen2.5-3B-Instruct-GPTQ-Int4
# /data2/share/Qwen2.5/Qwen2.5-3B-Instruct-GPTQ-Int8
# /data2/share/Qwen2.5/Qwen2.5-3B-Instruct-AWQ
# /data2/share/Qwen2.5/Qwen2.5-1.5B-Instruct-GPTQ-Int4
# /data2/share/Qwen2.5/Qwen2.5-1.5B-Instruct-GPTQ-Int8
# /data2/share/Qwen2.5/Qwen2.5-1.5B-Instruct-AWQ
# /data2/share/gemma/gemma-2-2b-it
# /data2/share/phi/Phi-3.5-mini-instruct

export CUDA_VISIBLE_DEVICES=1,2
export NUMEXPR_MAX_THREADS=40  # 或在代码中设置 os.environ["NUMEXPR_MAX_THREADS"] = "80"

# Define small language models
declare -A SLM_MODELS=(
    # ["deepseek-qwen-1.5b"]="/data2/share/deepseek/DeepSeek-R1-Distill-Qwen-1.5B"
    # ["deepseek-qwen-7b"]="/data2/share/deepseek/DeepSeek-R1-Distill-Qwen-7B"
    # ["deepseek-llama-8b"]="/data2/share/deepseek/DeepSeek-R1-Distill-Llama-8B"
    # ["minicpm-4b"]="/data2/share/openbmb/MiniCPM3-4B"
    # ["megrez-3b"]="/data2/share/megrez/Megrez-3B-Instruct"
    # ["qwen-3b-gptq-int4"]="/data2/share/Qwen2.5/Qwen2.5-3B-Instruct-GPTQ-Int4"
    # ["qwen-3b-gptq-int8"]="/data2/share/Qwen2.5/Qwen2.5-3B-Instruct-GPTQ-Int8"
    # ["qwen-3b-awq"]="/data2/share/Qwen2.5/Qwen2.5-3B-Instruct-AWQ"
    # ["qwen-1.5b-gptq-int4"]="/data2/share/Qwen2.5/Qwen2.5-1.5B-Instruct-GPTQ-Int4"
    # ["qwen-1.5b-gptq-int8"]="/data2/share/Qwen2.5/Qwen2.5-1.5B-Instruct-GPTQ-Int8"
    # ["qwen-1.5b-awq"]="/data2/share/Qwen2.5/Qwen2.5-1.5B-Instruct-AWQ"
    # ["gemma-2b"]="/data2/share/gemma/gemma-2-2b-it"
    # ["phi-3.5"]="/data2/share/phi/Phi-3.5-mini-instruct"
    ["qwen-14b"]="/data2/share/Qwen2.5/Qwen2.5-14B-Instruct"
    ["qwen-32b"]="/data2/share/Qwen2.5/Qwen2.5-32B-Instruct"
    ["qwen-14b-awq"]="/data2/share/Qwen2.5/Qwen2.5-14B-Instruct-AWQ"
    ["qwen-32b-awq"]="/data2/share/Qwen2.5/Qwen2.5-32B-Instruct-AWQ"
    ["qwen-14b-gptq-int4"]="/data2/share/Qwen2.5/Qwen2.5-14B-Instruct-GPTQ-Int4"
    ["qwen-32b-gptq-int4"]="/data2/share/Qwen2.5/Qwen2.5-32B-Instruct-GPTQ-Int4"
    ["internlm3-8b"]="/data2/share/internlm/internlm3-8b-instruct"
    ["internlm3-8b-awq"]="/data2/share/internlm/internlm3-8b-instruct-awq"
    ["internlm3-8b-gptq-int4"]="/data2/share/internlm/internlm3-8b-instruct-gptq-int4"
    ["internlm2.5-20b-awq"]="/data2/share/internlm/internlm2_5-20b-chat-4bit-awq"
    ["internlm2.5-20b"]="/data2/share/internlm/internlm2_5-20b-chat"
)

function generate_log_name() {
    local model=$1
    local quant=$2
    local temp=$3
    local current_date=$(date +%Y%m%d)
    echo "GENFLOW-${model}-${quant}-t${temp}-${current_date}-llmc"
}

function generate_slm_log_name() {
    local model=$1
    local temp=$2
    local current_date=$(date +%Y%m%d)
    echo "GENFLOW-SLM-${model}-t${temp}-${current_date}"
}

function run_slm_genflow() {
    local model_key=$1
    local temp=$2
    local model_path=${SLM_MODELS[$model_key]}
    local log_name=$(generate_slm_log_name "$model_key" "$temp")

    echo "Generating with model: $model_key, path: $model_path"
    echo "Log name: $log_name"

    bash ./scripts/gen_flow.sh "$model_path" "$temp" "base" "1"
    #  > "logs/${log_name}.log" 2>&1
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
    run_slm_genflow "$model_key" "$TEMP"
done
