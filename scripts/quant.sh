

# autoawq 


# MODELS
# /data2/share/llama3.2/Llama-3.2-1B-Instruct
# /data2/share/llama3.1/llama-3.1-8B-Instruct
# /data2/share/mistral-7B/Mistral-7B-Instruct-v0.3
# /data2/share/Qwen2.5/Qwen2.5-7B-Instruct

# QUANT_METHODS
# smoothquant
# gptq


# version: gemm,marlin,gemv;gemv_fast
# python quant/autoawq.py \
#     --model_path /data2/share/llama3.2/Llama-3.2-1B-Instruct \
#     --zero_point True \
#     --q_group_size 128 \
#     --w_bit 4 \
#     --version gemm

# llama3.1
# python quant/autoawq.py \
#     --model_path /data2/share/llama3.1/llama-3.1-8B-Instruct \
#     --zero_point True \
#     --q_group_size 128 \
#     --w_bit 4 \
#     --version gemm 

# python benchtools/quant/llm_compressor_vllm.py \
#      --model_path /data2/share/llama3.2/Llama-3.2-1B-Instruct \
#      --quant_method smoothquant > ./logs/llama3.2-smoothquant.log
    
# python benchtools/quant/llm_compressor_vllm.py \
#     --model_path /data2/share/llama3.1/llama-3.1-8B-Instruct \
#     --quant_method smoothquant > ./logs/llama3.1-smoothquant.log

# python benchtools/quant/llm_compressor_vllm.py \
#     --model_path /data2/share/llama3.1/llama-3.1-8B-Instruct \
#     --quant_method gptq > ./logs/llama3.1-gptq.log

# python benchtools/quant/llm_compressor_vllm.py \
#     --model_path /data2/share/mistral-7B/Mistral-7B-Instruct-v0.3 \
#     --quant_method smoothquant > ./logs/mistral-7B-smoothquant.log

CUDA_VISIBLE_DEVICES=1 python benchtools/quant/llm_compressor_vllm.py \
    --model_path /data2/share/mistral-7B/Mistral-7B-Instruct-v0.3 \
    --quant_method gptq > ./logs/mistral-7B-gptq.log &

CUDA_VISIBLE_DEVICES=2 python benchtools/quant/llm_compressor_vllm.py \
    --model_path /data2/share/Qwen2.5/Qwen2.5-7B-Instruct \
    --quant_method gptq > ./logs/Qwen2.5-7B-gptq.log &

CUDA_VISIBLE_DEVICES=3 python benchtools/quant/llm_compressor_vllm.py \
    --model_path /data2/share/Qwen2.5/Qwen2.5-7B-Instruct \
    --quant_method smoothquant > ./logs/Qwen2.5-7B-smoothquant.log &

CUDA_VISIBLE_DEVICES=4 python benchtools/quant/llm_autoawq.py \
    --model_path /data2/share/llama3.2/Llama-3.2-1B-Instruct \
    --zero_point True \
    --q_group_size 128 \
    --w_bit 4 \
    --version gemm &

CUDA_VISIBLE_DEVICES=5 python benchtools/quant/llm_autoawq.py \
    --model_path /data2/share/llama3.1/llama-3.1-8B-Instruct \
    --zero_point True \
    --q_group_size 128 \
    --w_bit 4 \
    --version gemm &

CUDA_VISIBLE_DEVICES=6 python benchtools/quant/llm_autoawq.py \
    --model_path /data2/share/Qwen2.5/Qwen2.5-7B-Instruct \
    --zero_point True \
    --q_group_size 128 \
    --w_bit 4 \
    --version gemm 

CUDA_VISIBLE_DEVICES=6 python benchtools/quant/llm_autoawq.py \
    --model_path /data2/share/mistral-7B/Mistral-7B-Instruct-v0.3 \
    --zero_point True \
    --q_group_size 128 \
    --w_bit 4 \
    --version gemm
