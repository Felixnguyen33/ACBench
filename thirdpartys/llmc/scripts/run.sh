#!/bin/bash 

# awq_qwen2.5_7b 

CUDA_VISIBLE_DEVICES=4 bash scripts/run_llmc.sh awq_qwen2.5_7b quantization/methods/Awq/awq_w_only.yml 

# gptq qwen2.5_7b
CUDA_VISIBLE_DEVICES=5 bash scripts/run_llmc.sh gptq_qwen2.5_7b quantization/methods/GPTQ/gptq_w_only.yml