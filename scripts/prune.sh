
# pruning algo
# sparsegpt, wanda, magnitude 
# pruning setting
# 2:4 50% 

# MODELS - internlm-2.5 7b 
# /data2/share/peijiedong/AgentBench/thirdparty/wanda/out/internlm_7b/2-4/magnitude/internlm-2.5-7b-chat-magnitude-2-4-0.5
# /data2/share/peijiedong/AgentBench/thirdparty/wanda/out/internlm_7b/2-4/sparsegpt/internlm-2.5-7b-chat-sparsegpt-2-4-0.5
# /data2/share/peijiedong/AgentBench/thirdparty/wanda/out/internlm_7b/2-4/wanda/internlm-2.5-7b-chat-wanda-2-4-0.5

# /data2/share/peijiedong/AgentBench/thirdparty/wanda/out/internlm_7b/unstructured/magnitude/internlm-2.5-7b-chat-magnitude-un0.5
# /data2/share/peijiedong/AgentBench/thirdparty/wanda/out/internlm_7b/unstructured/sparsegpt/internlm-2.5-7b-chat-sparsegpt-un0.5
# /data2/share/peijiedong/AgentBench/thirdparty/wanda/out/internlm_7b/unstructured/wanda/internlm-2.5-7b-chat-wanda-un0.5

# MODELS - qwen-2.5 7b
# /data2/share/peijiedong/AgentBench/thirdparty/wanda/out/qwen_7b/2-4/magnitude/qwen-2.5-7b-chat-magnitude-2-4-0.5
# /data2/share/peijiedong/AgentBench/thirdparty/wanda/out/qwen_7b/2-4/sparsegpt/qwen-2.5-7b-chat-sparsegpt-2-4-0.5
# /data2/share/peijiedong/AgentBench/thirdparty/wanda/out/qwen_7b/2-4/wanda/qwen-2.5-7b-chat-wanda-2-4-0.5

# /data2/share/peijiedong/AgentBench/thirdparty/wanda/out/qwen_7b/unstructured/magnitude/qwen-2.5-7b-chat-magnitude-un0.5
# /data2/share/peijiedong/AgentBench/thirdparty/wanda/out/qwen_7b/unstructured/sparsegpt/qwen-2.5-7b-chat-sparsegpt-un0.5
# /data2/share/peijiedong/AgentBench/thirdparty/wanda/out/qwen_7b/unstructured/wanda/qwen-2.5-7b-chat-wanda-un0.5


# version: gemm,marlin,gemv;gemv_fast
python quant/autoawq.py \
    --model_path /data2/share/llama3.2/Llama-3.2-1B-Instruct \
    --zero_point True \
    --q_group_size 128 \
    --w_bit 4 \
    --version gemm

# llama3.1
python quant/autoawq.py \
    --model_path /data2/share/llama3.1/llama-3.1-8B-Instruct \
    --zero_point True \
    --q_group_size 128 \
    --w_bit 4 \
    --version gemm 

python agentbench/quant/llm_compressor_vllm.py \
     --model_path /data2/share/llama3.2/Llama-3.2-1B-Instruct \
     --quant_method smoothquant > ./logs/llama3.2-smoothquant.log
    
python agentbench/quant/llm_compressor_vllm.py \
    --model_path /data2/share/llama3.1/llama-3.1-8B-Instruct \
    --quant_method smoothquant > ./logs/llama3.1-smoothquant.log

CUDA_VISIBLE_DEVICES=3 python agentbench/quant/llm_compressor_vllm.py \
    --model_path /data2/share/llama3.1/llama-3.1-8B-Instruct \
    --quant_method gptq > ./logs/llama3.1-gptq.log &

python agentbench/quant/llm_compressor_vllm.py \
    --model_path /data2/share/mistral-7B/Mistral-7B-Instruct-v0.3 \
    --quant_method smoothquant > ./logs/mistral-7B-smoothquant.log

CUDA_VISIBLE_DEVICES=4 python agentbench/quant/llm_compressor_vllm.py \
    --model_path /data2/share/mistral-7B/Mistral-7B-Instruct-v0.3 \
    --quant_method gptq > ./logs/mistral-7B-gptq.log &

CUDA_VISIBLE_DEVICES=5 python agentbench/quant/llm_compressor_vllm.py \
    --model_path /data2/share/Qwen2.5/Qwen2.5-7B-Instruct \
    --quant_method gptq > ./logs/Qwen2.5-7B-gptq.log &

tail -f ./logs/Qwen2.5-7B-gptq.log

CUDA_VISIBLE_DEVICES=3 python agentbench/quant/llm_compressor_vllm.py \
    --model_path /data2/share/Qwen2.5/Qwen2.5-7B-Instruct \
    --quant_method smoothquant > ./logs/Qwen2.5-7B-smoothquant.log &

CUDA_VISIBLE_DEVICES=4 python agentbench/quant/llm_autoawq.py \
    --model_path /data2/share/llama3.2/Llama-3.2-1B-Instruct \
    --zero_point True \
    --q_group_size 128 \
    --w_bit 4 \
    --version gemm &

CUDA_VISIBLE_DEVICES=5 python agentbench/quant/llm_autoawq.py \
    --model_path /data2/share/llama3.1/llama-3.1-8B-Instruct \
    --zero_point True \
    --q_group_size 128 \
    --w_bit 4 \
    --version gemm &

CUDA_VISIBLE_DEVICES=6 python agentbench/quant/llm_autoawq.py \
    --model_path /data2/share/Qwen2.5/Qwen2.5-7B-Instruct \
    --zero_point True \
    --q_group_size 128 \
    --w_bit 4 \
    --version gemm 

CUDA_VISIBLE_DEVICES=6 python agentbench/quant/llm_autoawq.py \
    --model_path /data2/share/mistral-7B/Mistral-7B-Instruct-v0.3 \
    --zero_point True \
    --q_group_size 128 \
    --w_bit 4 \
    --version gemm
