export CUDA_VISIBLE_DEVICES=6,7

# Define models and their paths in an array
# declare -A models=(
#     ["Qwen2.5-7B-Instruct-AWQ"]="/data2/share/Qwen2.5/Qwen2.5-7B-Instruct-AWQ"
#     ["Qwen2.5-7B-Instruct"]="/data2/share/Qwen2.5/Qwen2.5-7B-Instruct"
#     ["Qwen2.5-7B-Instruct-Mag-Un-0.5"]="/mnt/sdd/dongpeijie/out_pruned_llm/qwen_7b/unstructured/magnitude/qwen-2.5-7b-chat-magnitude-un0.5"
#     ["Qwen2.5-7B-Instruct-Mag-2-4-0.5"]="/mnt/sdd/dongpeijie/out_pruned_llm/qwen_7b/2-4/magnitude/qwen-2.5-7b-chat-magnitude-2-4-0.5"
#     ["Qwen2.5-7B-Instruct-Wanda-Un-0.5"]="/mnt/sdd/dongpeijie/out_pruned_llm/qwen_7b/unstructured/wanda/qwen-2.5-7b-chat-wanda-un0.5"
#     ["Qwen2.5-7B-Instruct-Wanda-2-4-0.5"]="/mnt/sdd/dongpeijie/out_pruned_llm/qwen_7b/2-4/wanda/qwen-2.5-7b-chat-wanda-2-4-0.5"
#     ["Qwen2.5-7B-Instruct-SparseGPT-Un-0.5"]="/mnt/sdd/dongpeijie/out_pruned_llm/qwen_7b/unstructured/sparsegpt/qwen-2.5-7b-chat-sparsegpt-un0.5"
#     ["Qwen2.5-7B-Instruct-SparseGPT-2-4-0.5"]="/mnt/sdd/dongpeijie/out_pruned_llm/qwen_7b/2-4/sparsegpt/qwen-2.5-7b-chat-sparsegpt-2-4-0.5"
#     ["InternLM2.5-7B-Instruct"]="/data2/share/internlm/internlm2_5-7b-chat"
#     ["InternLM2.5-7B-Instruct-Mag-Un-0.5"]="/mnt/sdd/dongpeijie/out_pruned_llm/internlm_7b/unstructured/magnitude/internlm-2.5-7b-chat-magnitude-un0.5"
#     ["InternLM2.5-7B-Instruct-Mag-2-4-0.5"]="/mnt/sdd/dongpeijie/out_pruned_llm/internlm_7b/2-4/magnitude/internlm-2.5-7b-chat-magnitude-2-4-0.5"
#     ["InternLM2.5-7B-Instruct-Wanda-Un-0.5"]="/mnt/sdd/dongpeijie/out_pruned_llm/internlm_7b/unstructured/wanda/internlm-2.5-7b-chat-wanda-un0.5"
#     ["InternLM2.5-7B-Instruct-Wanda-2-4-0.5"]="/mnt/sdd/dongpeijie/out_pruned_llm/internlm_7b/2-4/wanda/internlm-2.5-7b-chat-wanda-2-4-0.5"
#     ["InternLM2.5-7B-Instruct-SparseGPT-Un-0.5"]="/mnt/sdd/dongpeijie/out_pruned_llm/internlm_7b/unstructured/sparsegpt/internlm-2.5-7b-chat-sparsegpt-un0.5"
#     ["InternLM2.5-7B-Instruct-SparseGPT-2-4-0.5"]="/mnt/sdd/dongpeijie/out_pruned_llm/internlm_7b/2-4/sparsegpt/internlm-2.5-7b-chat-sparsegpt-2-4-0.5"
#     ["InternLM2.5-7B-Instruct-AWQ"]="/data2/share/internlm/internlm2_5-7b-chat-AWQ-W4-G128"
#     ["InternLM2.5-7B-Instruct-GPTQ-w4a16"]="/data2/share/internlm/internlm2_5-7b-chat-GPTQ-w4a16"
#     ["InternLM2.5-7B-Instruct-RTN-w4"]="/data2/share/internlm/internlm2_5-7b-chat-RTN-w4"
#     ["Qwen2.5-7B-Instruct-RTN-w4"]="/data2/share/Qwen2.5/Qwen2.5-7B-Instruct-RTN-w4"
#     ["Qwen2.5-7B-Instruct-GPTQ-w4a16"]="/data2/share/Qwen2.5/Qwen2.5-7B-Instruct-GPTQ-w4a16"
# )

declare -A models=(
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
    ["internlm2.5-20b-awq"]="/data2/share/internlm/internlm2_5-20b-chat-4bit-awq"
    ["internlm2.5-20b"]="/data2/share/internlm/internlm2_5-20b-chat"
)

K=30 # Number of questions in parallel
T=1319 # Max number of examples
dataset="gsm8k"

# Loop through all models
for model_name in "${!models[@]}"; do
    echo "Evaluating model: $model_name"
    save_dir="./outputs/LongGenBench_GSM8K_${model_name}_topk"
    
    python3 run_longgenbench.py \
        --model_name "${model_name}" \
        --save_dir "${save_dir}" \
        --K ${K} \
        --max_num_examples ${T} \
        --eval_batch_size 1 \
        --seed 42 \
        --sample_method topk \
        --dataset ${dataset}
done