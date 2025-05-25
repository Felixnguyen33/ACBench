#!/bin/bash 

# CUDA_VISIBLE_DEVICES=1 python examples/backend/vllm/infer_with_vllm.py --model_path '/path/to/workspace/AgentBench/thirdparty/llmc/save/vllm_awq_w4a16/vllm_quant_model' > ./infer_vllm_awq_w4a16.log &

# CUDA_VISIBLE_DEVICES=2 python examples/backend/vllm/infer_with_vllm.py --model_path '/path/to/workspace/AgentBench/thirdparty/llmc/save/vllm_awq_w8a8/vllm_quant_model' > ./infer_vllm_awq_w8a8.log &

CUDA_VISIBLE_DEVICES=3 python examples/backend/vllm/infer_with_vllm.py --model_path '/data2/share/Qwen2.5/Qwen2.5-7B-Instruct-AWQ-W4-G128/vllm_quant_model' 