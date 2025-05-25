# This script is adapted from 
# https://github.com/FranxYao/Long-Context-Data-Engineering.git

export CUDA_VISIBLE_DEVICES=6

mkdir -p ./results_needle/logs/
mkdir -p ./results_needle/img/


METHOD='full'       # ['full', 'pyramidkv', 'snapkv', 'streamingllm', 'h2o', 'cam']
MAX_CAPACITY_PROMPT=64
# (1024 2048 4096 8192 16384 32768 65536)
attn_implementation="flash_attention_2" # Support "flash_attention_2", "sdpa", "None".
TAG=test


# For Llama3-8b

# (
# python -u run_needle_in_haystack.py --s_len 1000 --e_len 8001\
#     --model_provider LLaMA3 \
#     --model_name YOU_PATH_TO_LLAMA_3 \
#     --attn_implementation ${attn_implementation} \
#     --step 100 \
#     --method $METHOD \
#     --max_capacity_prompt $MAX_CAPACITY_PROMPT \
#     --model_version LlaMA3_${METHOD}_${MAX_CAPACITY_PROMPT}_${TAG}
# ) 2>&1  | tee results_needle/logs/LlaMA3_${METHOD}_${MAX_CAPACITY_PROMPT}_${TAG}.log

# For Qwen2.5-7B 
# (
# python -u run_needle_in_haystack.py --s_len 1000 --e_len 8001 \
#     --model_provider Qwen \
#     --model_name /data2/share/Qwen2.5/Qwen2.5-7B-Instruct \
#     --attn_implementation ${attn_implementation} \
#     --step 100 \
#     --method $METHOD \
#     --max_capacity_prompt $MAX_CAPACITY_PROMPT \
#     --model_version Qwen2.5_${METHOD}_${MAX_CAPACITY_PROMPT}_${TAG}
# ) 2>&1  | tee results_needle/logs/Qwen2.5_${METHOD}_${MAX_CAPACITY_PROMPT}_${TAG}.log

# For Qwen2.5-7B-AWQ 
# (
# python -u run_needle_in_haystack.py --s_len 1000 --e_len 8001 \
#     --model_provider Qwen \
#     --model_name /data2/share/Qwen2.5/Qwen2.5-7B-Instruct-AWQ-W4-G128 \
#     --attn_implementation ${attn_implementation} \
#     --step 100 \
#     --method $METHOD \
#     --max_capacity_prompt $MAX_CAPACITY_PROMPT \
#     --model_version Qwen2.5_7b_AWQ_${METHOD}_${MAX_CAPACITY_PROMPT}_${TAG}
# ) 2>&1  | tee results_needle/logs/Qwen2.5_7b_AWQ_${METHOD}_${MAX_CAPACITY_PROMPT}_${TAG}.log


# Define root directory
root_dir=/path/to/out_pruned_llm/qwen_7b

# For Qwen2.5-7B-mag-un-0.5
(
python -u run_needle_in_haystack.py --s_len 1000 --e_len 8001 \
    --model_provider Qwen \
    --model_name ${root_dir}/unstructured/magnitude/qwen-2.5-7b-chat-magnitude-un0.5 \
    --attn_implementation ${attn_implementation} \
    --step 100 \
    --method $METHOD \
    --max_capacity_prompt $MAX_CAPACITY_PROMPT \
    --model_version Qwen2.5_7b_mag_un_${METHOD}_${MAX_CAPACITY_PROMPT}_${TAG}
) 2>&1  | tee results_needle/logs/Qwen2.5_7b_mag_un_${METHOD}_${MAX_CAPACITY_PROMPT}_${TAG}.log

# For Qwen2.5-7B-mag-2-4-0.5
(
python -u run_needle_in_haystack.py --s_len 1000 --e_len 8001 \
    --model_provider Qwen \
    --model_name ${root_dir}/2-4/magnitude/qwen-2.5-7b-chat-magnitude-2-4-0.5 \
    --attn_implementation ${attn_implementation} \
    --step 100 \
    --method $METHOD \
    --max_capacity_prompt $MAX_CAPACITY_PROMPT \
    --model_version Qwen2.5_7b_mag_2_4_${METHOD}_${MAX_CAPACITY_PROMPT}_${TAG}
) 2>&1  | tee results_needle/logs/Qwen2.5_7b_mag_2_4_${METHOD}_${MAX_CAPACITY_PROMPT}_${TAG}.log

# For Qwen2.5-7B-wanda-un-0.5
(
python -u run_needle_in_haystack.py --s_len 1000 --e_len 8001 \
    --model_provider Qwen \
    --model_name ${root_dir}/unstructured/wanda/qwen-2.5-7b-chat-wanda-un0.5 \
    --attn_implementation ${attn_implementation} \
    --step 100 \
    --method $METHOD \
    --max_capacity_prompt $MAX_CAPACITY_PROMPT \
    --model_version Qwen2.5_7b_wanda_un_${METHOD}_${MAX_CAPACITY_PROMPT}_${TAG}
) 2>&1  | tee results_needle/logs/Qwen2.5_7b_wanda_un_${METHOD}_${MAX_CAPACITY_PROMPT}_${TAG}.log

# For Qwen2.5-7B-wanda-2-4-0.5
(
python -u run_needle_in_haystack.py --s_len 1000 --e_len 8001 \
    --model_provider Qwen \
    --model_name ${root_dir}/2-4/wanda/qwen-2.5-7b-chat-wanda-2-4-0.5 \
    --attn_implementation ${attn_implementation} \
    --step 100 \
    --method $METHOD \
    --max_capacity_prompt $MAX_CAPACITY_PROMPT \
    --model_version Qwen2.5_7b_wanda_2_4_${METHOD}_${MAX_CAPACITY_PROMPT}_${TAG}
) 2>&1  | tee results_needle/logs/Qwen2.5_7b_wanda_2_4_${METHOD}_${MAX_CAPACITY_PROMPT}_${TAG}.log

# For Qwen2.5-7B-sparsegpt-un-0.5
(
python -u run_needle_in_haystack.py --s_len 1000 --e_len 8001 \
    --model_provider Qwen \
    --model_name ${root_dir}/unstructured/sparsegpt/qwen-2.5-7b-chat-sparsegpt-un0.5 \
    --attn_implementation ${attn_implementation} \
    --step 100 \
    --method $METHOD \
    --max_capacity_prompt $MAX_CAPACITY_PROMPT \
    --model_version Qwen2.5_7b_sparsegpt_un_${METHOD}_${MAX_CAPACITY_PROMPT}_${TAG}
) 2>&1  | tee results_needle/logs/Qwen2.5_7b_sparsegpt_un_${METHOD}_${MAX_CAPACITY_PROMPT}_${TAG}.log

# For Qwen2.5-7B-sparsegpt-2-4-0.5
(
python -u run_needle_in_haystack.py --s_len 1000 --e_len 8001 \
    --model_provider Qwen \
    --model_name ${root_dir}/2-4/sparsegpt/qwen-2.5-7b-chat-sparsegpt-2-4-0.5 \
    --attn_implementation ${attn_implementation} \
    --step 100 \
    --method $METHOD \
    --max_capacity_prompt $MAX_CAPACITY_PROMPT \
    --model_version Qwen2.5_7b_sparsegpt_2_4_${METHOD}_${MAX_CAPACITY_PROMPT}_${TAG}
) 2>&1  | tee results_needle/logs/Qwen2.5_7b_sparsegpt_2_4_${METHOD}_${MAX_CAPACITY_PROMPT}_${TAG}.log


# For Mistral

# (
# python -u run_needle_in_haystack.py --s_len 400 --e_len 32001\
#     --model_provider Mistral \
#     --model_name YOU_PATH_TO_MISTRAL_2 \
#     --step 400 \
#     --method $METHOD \
#     --max_capacity_prompt $MAX_CAPACITY_PROMPT \
#     --model_version Mistral2_${METHOD}_${MAX_CAPACITY_PROMPT}_${TAG}
# ) 2>&1  | tee logs/Mistral2_${METHOD}_${MAX_CAPACITY_PROMPT}_${TAG}.log