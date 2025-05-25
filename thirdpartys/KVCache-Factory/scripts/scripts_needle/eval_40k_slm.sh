# This script is adapted from 
# https://github.com/FranxYao/Long-Context-Data-Engineering.git

export CUDA_VISIBLE_DEVICES=1,2

mkdir -p ./results_needle/logs/
mkdir -p ./results_needle/img/


METHOD='full'       # ['full', 'pyramidkv', 'snapkv', 'streamingllm', 'h2o', 'cam']
MAX_CAPACITY_PROMPT=64
# (1024 2048 4096 8192 16384 32768 65536)
attn_implementation="flash_attention_2" # Support "flash_attention_2", "sdpa", "None".
TAG=40k_slm


# # For Llama3-8b
# (
# python -u run_needle_in_haystack.py --s_len 1000 --e_len 40001\
#     --model_provider LLaMA3 \
#     --model_name /data2/share/LLaMA3/LLaMA3-8B-Instruct \
#     --attn_implementation ${attn_implementation} \
#     --step 1000 \
#     --method $METHOD \
#     --max_capacity_prompt $MAX_CAPACITY_PROMPT \
#     --model_version LlaMA3_${METHOD}_${MAX_CAPACITY_PROMPT}_${TAG}
# ) 2>&1  | tee results_needle/logs/LlaMA3_${METHOD}_${MAX_CAPACITY_PROMPT}_${TAG}.log

# # For Qwen2.5-7B 
# (
# python -u run_needle_in_haystack.py --s_len 1000 --e_len 40001 \
#     --model_provider Qwen \
#     --model_name /data2/share/Qwen2.5/Qwen2.5-7B-Instruct \
#     --attn_implementation ${attn_implementation} \
#     --step 1000 \
#     --method $METHOD \
#     --max_capacity_prompt $MAX_CAPACITY_PROMPT \
#     --model_version Qwen2.5_${METHOD}_${MAX_CAPACITY_PROMPT}_${TAG}
# ) 2>&1  | tee results_needle/logs/Qwen2.5_${METHOD}_${MAX_CAPACITY_PROMPT}_${TAG}.log

# # For Qwen2.5-7B-AWQ 
# (
# python -u run_needle_in_haystack.py --s_len 1000 --e_len 40001 \
#     --model_provider Qwen \
#     --model_name /data2/share/Qwen2.5/Qwen2.5-7B-Instruct-AWQ-W4-G128 \
#     --attn_implementation ${attn_implementation} \
#     --step 1000 \
#     --method $METHOD \
#     --max_capacity_prompt $MAX_CAPACITY_PROMPT \
#     --model_version Qwen2.5_7b_AWQ_${METHOD}_${MAX_CAPACITY_PROMPT}_${TAG}
# ) 2>&1  | tee results_needle/logs/Qwen2.5_7b_AWQ_${METHOD}_${MAX_CAPACITY_PROMPT}_${TAG}.log

# # For Qwen2.5-7B-GPTQ
# (
# python -u run_needle_in_haystack.py --s_len 1000 --e_len 40001 \
#     --model_provider Qwen \
#     --model_name /data2/share/Qwen2.5/Qwen2.5-7B-Instruct-GPTQ-w4a16 \
#     --attn_implementation ${attn_implementation} \
#     --step 1000 \
#     --method $METHOD \
#     --max_capacity_prompt $MAX_CAPACITY_PROMPT \
#     --model_version Qwen2.5_7b_GPTQ_${METHOD}_${MAX_CAPACITY_PROMPT}_${TAG}
# ) 2>&1  | tee results_needle/logs/Qwen2.5_7b_GPTQ_${METHOD}_${MAX_CAPACITY_PROMPT}_${TAG}.log

# # For Qwen2.5-7B-RTN
# (
# python -u run_needle_in_haystack.py --s_len 1000 --e_len 40001 \
#     --model_provider Qwen \
#     --model_name /data2/share/Qwen2.5/Qwen2.5-7B-Instruct-RTN-w4 \
#     --attn_implementation ${attn_implementation} \
#     --step 1000 \
#     --method $METHOD \
#     --max_capacity_prompt $MAX_CAPACITY_PROMPT \
#     --model_version Qwen2.5_7b_RTN_${METHOD}_${MAX_CAPACITY_PROMPT}_${TAG}
# ) 2>&1  | tee results_needle/logs/Qwen2.5_7b_RTN_${METHOD}_${MAX_CAPACITY_PROMPT}_${TAG}.log


# # Define root directory
# root_dir=/path/to/out_pruned_llm/qwen_7b

# # For Qwen2.5-7B-mag-un-0.5
# (
# python -u run_needle_in_haystack.py --s_len 1000 --e_len 40001 \
#     --model_provider Qwen \
#     --model_name ${root_dir}/unstructured/magnitude/qwen-2.5-7b-chat-magnitude-un0.5 \
#     --attn_implementation ${attn_implementation} \
#     --step 1000 \
#     --method $METHOD \
#     --max_capacity_prompt $MAX_CAPACITY_PROMPT \
#     --model_version Qwen2.5_7b_mag_un_${METHOD}_${MAX_CAPACITY_PROMPT}_${TAG}
# ) 2>&1  | tee results_needle/logs/Qwen2.5_7b_mag_un_${METHOD}_${MAX_CAPACITY_PROMPT}_${TAG}.log

# # For Qwen2.5-7B-mag-2-4-0.5
# (
# python -u run_needle_in_haystack.py --s_len 1000 --e_len 40001 \
#     --model_provider Qwen \
#     --model_name ${root_dir}/2-4/magnitude/qwen-2.5-7b-chat-magnitude-2-4-0.5 \
#     --attn_implementation ${attn_implementation} \
#     --step 1000 \
#     --method $METHOD \
#     --max_capacity_prompt $MAX_CAPACITY_PROMPT \
#     --model_version Qwen2.5_7b_mag_2_4_${METHOD}_${MAX_CAPACITY_PROMPT}_${TAG}
# ) 2>&1  | tee results_needle/logs/Qwen2.5_7b_mag_2_4_${METHOD}_${MAX_CAPACITY_PROMPT}_${TAG}.log

# # For Qwen2.5-7B-wanda-un-0.5
# (
# python -u run_needle_in_haystack.py --s_len 1000 --e_len 40001 \
#     --model_provider Qwen \
#     --model_name ${root_dir}/unstructured/wanda/qwen-2.5-7b-chat-wanda-un0.5 \
#     --attn_implementation ${attn_implementation} \
#     --step 1000 \
#     --method $METHOD \
#     --max_capacity_prompt $MAX_CAPACITY_PROMPT \
#     --model_version Qwen2.5_7b_wanda_un_${METHOD}_${MAX_CAPACITY_PROMPT}_${TAG}
# ) 2>&1  | tee results_needle/logs/Qwen2.5_7b_wanda_un_${METHOD}_${MAX_CAPACITY_PROMPT}_${TAG}.log

# # For Qwen2.5-7B-wanda-2-4-0.5
# (
# python -u run_needle_in_haystack.py --s_len 1000 --e_len 40001 \
#     --model_provider Qwen \
#     --model_name ${root_dir}/2-4/wanda/qwen-2.5-7b-chat-wanda-2-4-0.5 \
#     --attn_implementation ${attn_implementation} \
#     --step 1000 \
#     --method $METHOD \
#     --max_capacity_prompt $MAX_CAPACITY_PROMPT \
#     --model_version Qwen2.5_7b_wanda_2_4_${METHOD}_${MAX_CAPACITY_PROMPT}_${TAG}
# ) 2>&1  | tee results_needle/logs/Qwen2.5_7b_wanda_2_4_${METHOD}_${MAX_CAPACITY_PROMPT}_${TAG}.log

# # For Qwen2.5-7B-sparsegpt-un-0.5
# (
# python -u run_needle_in_haystack.py --s_len 1000 --e_len 40001 \
#     --model_provider Qwen \
#     --model_name ${root_dir}/unstructured/sparsegpt/qwen-2.5-7b-chat-sparsegpt-un0.5 \
#     --attn_implementation ${attn_implementation} \
#     --step 1000 \
#     --method $METHOD \
#     --max_capacity_prompt $MAX_CAPACITY_PROMPT \
#     --model_version Qwen2.5_7b_sparsegpt_un_${METHOD}_${MAX_CAPACITY_PROMPT}_${TAG}
# ) 2>&1  | tee results_needle/logs/Qwen2.5_7b_sparsegpt_un_${METHOD}_${MAX_CAPACITY_PROMPT}_${TAG}.log

# # For Qwen2.5-7B-sparsegpt-2-4-0.5
# (
# python -u run_needle_in_haystack.py --s_len 1000 --e_len 40001 \
#     --model_provider Qwen \
#     --model_name ${root_dir}/2-4/sparsegpt/qwen-2.5-7b-chat-sparsegpt-2-4-0.5 \
#     --attn_implementation ${attn_implementation} \
#     --step 1000 \
#     --method $METHOD \
#     --max_capacity_prompt $MAX_CAPACITY_PROMPT \
#     --model_version Qwen2.5_7b_sparsegpt_2_4_${METHOD}_${MAX_CAPACITY_PROMPT}_${TAG}
# ) 2>&1  | tee results_needle/logs/Qwen2.5_7b_sparsegpt_2_4_${METHOD}_${MAX_CAPACITY_PROMPT}_${TAG}.log


# # For InternLM2.5-7B-Instruct

# # origin 
# # /data2/share/internlm/internlm2_5-7b-chat
# (
# python -u run_needle_in_haystack.py --s_len 1000 --e_len 40001 \
#     --model_provider InternLM \
#     --model_name /data2/share/internlm/internlm2_5-7b-chat \
#     --attn_implementation ${attn_implementation} \
#     --step 1000 \
#     --method $METHOD \
#     --max_capacity_prompt $MAX_CAPACITY_PROMPT \
#     --model_version InternLM2.5_7b_Instruct_${METHOD}_${MAX_CAPACITY_PROMPT}_${TAG}
# ) 2>&1  | tee results_needle/logs/InternLM2.5_7b_Instruct_${METHOD}_${MAX_CAPACITY_PROMPT}_${TAG}.log


# # Define root directory for InternLM
# root_dir_internlm=/data2/share/peijiedong/AgentBench/thirdparty/wanda/out/internlm_7b

# # For InternLM2.5-7B-mag-un-0.5
# (
# python -u run_needle_in_haystack.py --s_len 1000 --e_len 40001 \
#     --model_provider InternLM \
#     --model_name ${root_dir_internlm}/unstructured/magnitude/internlm-2.5-7b-chat-magnitude-un0.5 \
#     --attn_implementation ${attn_implementation} \
#     --step 1000 \
#     --method $METHOD \
#     --max_capacity_prompt $MAX_CAPACITY_PROMPT \
#     --model_version InternLM2.5_7b_mag_un_${METHOD}_${MAX_CAPACITY_PROMPT}_${TAG}
# ) 2>&1  | tee results_needle/logs/InternLM2.5_7b_mag_un_${METHOD}_${MAX_CAPACITY_PROMPT}_${TAG}.log

# # For InternLM2.5-7B-mag-2-4-0.5
# (
# python -u run_needle_in_haystack.py --s_len 1000 --e_len 40001 \
#     --model_provider InternLM \
#     --model_name ${root_dir_internlm}/2-4/magnitude/internlm-2.5-7b-chat-magnitude-2-4-0.5 \
#     --attn_implementation ${attn_implementation} \
#     --step 1000 \
#     --method $METHOD \
#     --max_capacity_prompt $MAX_CAPACITY_PROMPT \
#     --model_version InternLM2.5_7b_mag_2_4_${METHOD}_${MAX_CAPACITY_PROMPT}_${TAG}
# ) 2>&1  | tee results_needle/logs/InternLM2.5_7b_mag_2_4_${METHOD}_${MAX_CAPACITY_PROMPT}_${TAG}.log

# # For InternLM2.5-7B-wanda-un-0.5
# (
# python -u run_needle_in_haystack.py --s_len 1000 --e_len 40001 \
#     --model_provider InternLM \
#     --model_name ${root_dir_internlm}/unstructured/wanda/internlm-2.5-7b-chat-wanda-un0.5 \
#     --attn_implementation ${attn_implementation} \
#     --step 1000 \
#     --method $METHOD \
#     --max_capacity_prompt $MAX_CAPACITY_PROMPT \
#     --model_version InternLM2.5_7b_wanda_un_${METHOD}_${MAX_CAPACITY_PROMPT}_${TAG}
# ) 2>&1  | tee results_needle/logs/InternLM2.5_7b_wanda_un_${METHOD}_${MAX_CAPACITY_PROMPT}_${TAG}.log

# # For InternLM2.5-7B-wanda-2-4-0.5
# (
# python -u run_needle_in_haystack.py --s_len 1000 --e_len 40001 \
#     --model_provider InternLM \
#     --model_name ${root_dir_internlm}/2-4/wanda/internlm-2.5-7b-chat-wanda-2-4-0.5 \
#     --attn_implementation ${attn_implementation} \
#     --step 1000 \
#     --method $METHOD \
#     --max_capacity_prompt $MAX_CAPACITY_PROMPT \
#     --model_version InternLM2.5_7b_wanda_2_4_${METHOD}_${MAX_CAPACITY_PROMPT}_${TAG}
# ) 2>&1  | tee results_needle/logs/InternLM2.5_7b_wanda_2_4_${METHOD}_${MAX_CAPACITY_PROMPT}_${TAG}.log

# # For InternLM2.5-7B-sparsegpt-un-0.5
# (
# python -u run_needle_in_haystack.py --s_len 1000 --e_len 40001 \
#     --model_provider InternLM \
#     --model_name ${root_dir_internlm}/unstructured/sparsegpt/internlm-2.5-7b-chat-sparsegpt-un0.5 \
#     --attn_implementation ${attn_implementation} \
#     --step 1000 \
#     --method $METHOD \
#     --max_capacity_prompt $MAX_CAPACITY_PROMPT \
#     --model_version InternLM2.5_7b_sparsegpt_un_${METHOD}_${MAX_CAPACITY_PROMPT}_${TAG}
# ) 2>&1  | tee results_needle/logs/InternLM2.5_7b_sparsegpt_un_${METHOD}_${MAX_CAPACITY_PROMPT}_${TAG}.log

# # For InternLM2.5-7B-sparsegpt-2-4-0.5
# (
# python -u run_needle_in_haystack.py --s_len 1000 --e_len 40001 \
#     --model_provider InternLM \
#     --model_name ${root_dir_internlm}/2-4/sparsegpt/internlm-2.5-7b-chat-sparsegpt-2-4-0.5 \
#     --attn_implementation ${attn_implementation} \
#     --step 1000 \
#     --method $METHOD \
#     --max_capacity_prompt $MAX_CAPACITY_PROMPT \
#     --model_version InternLM2.5_7b_sparsegpt_2_4_${METHOD}_${MAX_CAPACITY_PROMPT}_${TAG}
# ) 2>&1  | tee results_needle/logs/InternLM2.5_7b_sparsegpt_2_4_${METHOD}_${MAX_CAPACITY_PROMPT}_${TAG}.log

# # For InternLM2.5-7B-AWQ
# (
# python -u run_needle_in_haystack.py --s_len 1000 --e_len 40001 \
#     --model_provider InternLM \
#     --model_name /data2/share/internlm/internlm2_5-7b-chat-AWQ-W4-G128 \
#     --attn_implementation ${attn_implementation} \
#     --step 1000 \
#     --method $METHOD \
#     --max_capacity_prompt $MAX_CAPACITY_PROMPT \
#     --model_version InternLM2.5_7b_AWQ_${METHOD}_${MAX_CAPACITY_PROMPT}_${TAG}
# ) 2>&1  | tee results_needle/logs/InternLM2.5_7b_AWQ_${METHOD}_${MAX_CAPACITY_PROMPT}_${TAG}.log

# # For InternLM2.5-7B-GPTQ
# (
# python -u run_needle_in_haystack.py --s_len 1000 --e_len 40001 \
#     --model_provider InternLM \
#     --model_name /data2/share/internlm/internlm2_5-7b-chat-GPTQ-w4a16 \
#     --attn_implementation ${attn_implementation} \
#     --step 1000 \
#     --method $METHOD \
#     --max_capacity_prompt $MAX_CAPACITY_PROMPT \
#     --model_version InternLM2.5_7b_GPTQ_${METHOD}_${MAX_CAPACITY_PROMPT}_${TAG}
# ) 2>&1  | tee results_needle/logs/InternLM2.5_7b_GPTQ_${METHOD}_${MAX_CAPACITY_PROMPT}_${TAG}.log

# # For InternLM2.5-7B-RTN
# (
# python -u run_needle_in_haystack.py --s_len 1000 --e_len 40001 \
#     --model_provider InternLM \
#     --model_name /data2/share/internlm/internlm2_5-7b-chat-RTN-w4 \
#     --attn_implementation ${attn_implementation} \
#     --step 1000 \
#     --method $METHOD \
#     --max_capacity_prompt $MAX_CAPACITY_PROMPT \
#     --model_version InternLM2.5_7b_RTN_${METHOD}_${MAX_CAPACITY_PROMPT}_${TAG}
# ) 2>&1  | tee results_needle/logs/InternLM2.5_7b_RTN_${METHOD}_${MAX_CAPACITY_PROMPT}_${TAG}.log

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


# Iterate through all SLM models and run evaluation
for model_key in "${!SLM_MODELS[@]}"; do
    model_path="${SLM_MODELS[$model_key]}"
    
    # Determine model provider based on model name
    if [[ $model_key == deepseek* ]]; then
        provider="DeepSeek"
    elif [[ $model_key == qwen* ]]; then
        provider="Qwen"
    elif [[ $model_key == minicpm* ]]; then
        provider="MiniCPM"
    elif [[ $model_key == megrez* ]]; then
        provider="Megrez"
    elif [[ $model_key == gemma* ]]; then
        provider="Gemma"
    elif [[ $model_key == phi* ]]; then
        provider="Phi"
    elif [[ $model_key == internlm* ]]; then
        provider="InternLM"
    else
        echo "Warning: Unknown model provider for $model_key, skipping..."
        continue
    fi

    echo "Evaluating model: $model_key (Provider: $provider)"
    (
    python -u run_needle_in_haystack.py --s_len 1000 --e_len 40001 \
        --model_provider $provider \
        --model_name $model_path \
        --attn_implementation ${attn_implementation} \
        --step 1000 \
        --method $METHOD \
        --max_capacity_prompt $MAX_CAPACITY_PROMPT \
        --model_version ${model_key}_${METHOD}_${MAX_CAPACITY_PROMPT}_${TAG}
    ) 2>&1 | tee results_needle/logs/${model_key}_${METHOD}_${MAX_CAPACITY_PROMPT}_${TAG}.log

    # Add a small delay between runs to prevent potential resource conflicts
    sleep 2
done