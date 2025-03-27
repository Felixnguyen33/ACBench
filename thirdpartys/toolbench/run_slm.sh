# # agent conda 

# # DeepSeek Qwen 1.5B
# MODEL_NAME_OR_PATH=/data2/share/deepseek/DeepSeek-R1-Distill-Qwen-1.5B
# TYPE=vllm 
# QUANT_TYPE=fp16
# EXP_NAME=deepseek-qwen-1.5b_${QUANT_TYPE}_en_vllm
# META_TEMPLATE=qwen
# CUDA_VISIBLE_DEVICES=2 bash test_all_en.sh $TYPE $MODEL_NAME_OR_PATH $EXP_NAME $META_TEMPLATE $QUANT_TYPE

# # DeepSeek Qwen 7B
# MODEL_NAME_OR_PATH=/data2/share/deepseek/DeepSeek-R1-Distill-Qwen-7B
# TYPE=vllm
# QUANT_TYPE=fp16
# EXP_NAME=deepseek-qwen-7b_${QUANT_TYPE}_en_vllm
# META_TEMPLATE=qwen
# CUDA_VISIBLE_DEVICES=2 bash test_all_en.sh $TYPE $MODEL_NAME_OR_PATH $EXP_NAME $META_TEMPLATE $QUANT_TYPE

# # DeepSeek LLaMA 8B
# MODEL_NAME_OR_PATH=/data2/share/deepseek/DeepSeek-R1-Distill-Llama-8B
# TYPE=vllm
# QUANT_TYPE=fp16
# EXP_NAME=deepseek-llama-8b_${QUANT_TYPE}_en_vllm
# META_TEMPLATE=llama2
# CUDA_VISIBLE_DEVICES=2 bash test_all_en.sh $TYPE $MODEL_NAME_OR_PATH $EXP_NAME $META_TEMPLATE $QUANT_TYPE

# # MiniCPM 4B
# MODEL_NAME_OR_PATH=/data2/share/openbmb/MiniCPM3-4B
# TYPE=vllm
# QUANT_TYPE=fp16
# EXP_NAME=minicpm-4b_${QUANT_TYPE}_en_vllm
# META_TEMPLATE=minicpm
# CUDA_VISIBLE_DEVICES=2 bash test_all_en.sh $TYPE $MODEL_NAME_OR_PATH $EXP_NAME $META_TEMPLATE $QUANT_TYPE

# # Megrez 3B
# MODEL_NAME_OR_PATH=/data2/share/megrez/Megrez-3B-Instruct
# TYPE=vllm
# QUANT_TYPE=fp16
# EXP_NAME=megrez-3b_${QUANT_TYPE}_en_vllm
# META_TEMPLATE=megrez
# CUDA_VISIBLE_DEVICES=2 bash test_all_en.sh $TYPE $MODEL_NAME_OR_PATH $EXP_NAME $META_TEMPLATE $QUANT_TYPE

# # Qwen 3B GPTQ Int4
# MODEL_NAME_OR_PATH=/data2/share/Qwen2.5/Qwen2.5-3B-Instruct-GPTQ-Int4
# TYPE=vllm
# QUANT_TYPE=int4
# EXP_NAME=qwen-3b-gptq-int4_${QUANT_TYPE}_en_vllm
# META_TEMPLATE=qwen
# CUDA_VISIBLE_DEVICES=2 bash test_all_en.sh $TYPE $MODEL_NAME_OR_PATH $EXP_NAME $META_TEMPLATE $QUANT_TYPE

# # Gemma 2B
# MODEL_NAME_OR_PATH=/data2/share/gemma/gemma-2-2b-it
# TYPE=vllm
# QUANT_TYPE=fp16
# EXP_NAME=gemma-2b_${QUANT_TYPE}_en_vllm
# META_TEMPLATE=gemma
# CUDA_VISIBLE_DEVICES=2 bash test_all_en.sh $TYPE $MODEL_NAME_OR_PATH $EXP_NAME $META_TEMPLATE $QUANT_TYPE

# # Phi 3.5
# MODEL_NAME_OR_PATH=/data2/share/phi/Phi-3.5-mini-instruct
# TYPE=vllm
# QUANT_TYPE=fp16
# EXP_NAME=phi-3.5_${QUANT_TYPE}_en_vllm
# META_TEMPLATE=phi
# CUDA_VISIBLE_DEVICES=2 bash test_all_en.sh $TYPE $MODEL_NAME_OR_PATH $EXP_NAME $META_TEMPLATE $QUANT_TYPE

# # Qwen 14B
# MODEL_NAME_OR_PATH=/data2/share/Qwen2.5/Qwen2.5-14B-Instruct
# TYPE=vllm
# QUANT_TYPE=fp16
# EXP_NAME=qwen-14b_${QUANT_TYPE}_en_vllm
# META_TEMPLATE=qwen
# CUDA_VISIBLE_DEVICES=6,7 bash test_all_en.sh $TYPE $MODEL_NAME_OR_PATH $EXP_NAME $META_TEMPLATE $QUANT_TYPE

# # Qwen 32B
# MODEL_NAME_OR_PATH=/data2/share/Qwen2.5/Qwen2.5-32B-Instruct
# TYPE=vllm
# QUANT_TYPE=fp16
# EXP_NAME=qwen-32b_${QUANT_TYPE}_en_vllm
# META_TEMPLATE=qwen
# CUDA_VISIBLE_DEVICES=6,7 bash test_all_en.sh $TYPE $MODEL_NAME_OR_PATH $EXP_NAME $META_TEMPLATE $QUANT_TYPE

# # Qwen 14B AWQ
# MODEL_NAME_OR_PATH=/data2/share/Qwen2.5/Qwen2.5-14B-Instruct-AWQ
# TYPE=vllm
# QUANT_TYPE=awq
# EXP_NAME=qwen-14b-awq_${QUANT_TYPE}_en_vllm
# META_TEMPLATE=qwen
# CUDA_VISIBLE_DEVICES=6,7 bash test_all_en.sh $TYPE $MODEL_NAME_OR_PATH $EXP_NAME $META_TEMPLATE $QUANT_TYPE

# # Qwen 32B AWQ
# MODEL_NAME_OR_PATH=/data2/share/Qwen2.5/Qwen2.5-32B-Instruct-AWQ
# TYPE=vllm
# QUANT_TYPE=awq
# EXP_NAME=qwen-32b-awq_${QUANT_TYPE}_en_vllm
# META_TEMPLATE=qwen
# CUDA_VISIBLE_DEVICES=6,7 bash test_all_en.sh $TYPE $MODEL_NAME_OR_PATH $EXP_NAME $META_TEMPLATE $QUANT_TYPE

# # Qwen 14B GPTQ Int4
# MODEL_NAME_OR_PATH=/data2/share/Qwen2.5/Qwen2.5-14B-Instruct-GPTQ-Int4
# TYPE=vllm
# QUANT_TYPE=int4
# EXP_NAME=qwen-14b-gptq-int4_${QUANT_TYPE}_en_vllm
# META_TEMPLATE=qwen
# CUDA_VISIBLE_DEVICES=6,7 bash test_all_en.sh $TYPE $MODEL_NAME_OR_PATH $EXP_NAME $META_TEMPLATE $QUANT_TYPE

# # Qwen 32B GPTQ Int4
# MODEL_NAME_OR_PATH=/data2/share/Qwen2.5/Qwen2.5-32B-Instruct-GPTQ-Int4
# TYPE=vllm
# QUANT_TYPE=int4
# EXP_NAME=qwen-32b-gptq-int4_${QUANT_TYPE}_en_vllm
# META_TEMPLATE=qwen
# CUDA_VISIBLE_DEVICES=6,7 bash test_all_en.sh $TYPE $MODEL_NAME_OR_PATH $EXP_NAME $META_TEMPLATE $QUANT_TYPE

# # InternLM3 8B
# MODEL_NAME_OR_PATH=/data2/share/internlm/internlm3-8b-instruct
# TYPE=vllm
# QUANT_TYPE=fp16
# EXP_NAME=internlm3-8b_${QUANT_TYPE}_en_vllm
# META_TEMPLATE=internlm
# CUDA_VISIBLE_DEVICES=6,7 bash test_all_en.sh $TYPE $MODEL_NAME_OR_PATH $EXP_NAME $META_TEMPLATE $QUANT_TYPE

# # InternLM3 8B AWQ
# MODEL_NAME_OR_PATH=/data2/share/internlm/internlm3-8b-instruct-awq
# TYPE=vllm
# QUANT_TYPE=awq
# EXP_NAME=internlm3-8b-awq_${QUANT_TYPE}_en_vllm
# META_TEMPLATE=internlm
# CUDA_VISIBLE_DEVICES=6,7 bash test_all_en.sh $TYPE $MODEL_NAME_OR_PATH $EXP_NAME $META_TEMPLATE $QUANT_TYPE

# # InternLM3 8B GPTQ Int4
# MODEL_NAME_OR_PATH=/data2/share/internlm/internlm3-8b-instruct-gptq-int4
# TYPE=vllm
# QUANT_TYPE=int4
# EXP_NAME=internlm3-8b-gptq-int4_${QUANT_TYPE}_en_vllm
# META_TEMPLATE=internlm
# CUDA_VISIBLE_DEVICES=6,7 bash test_all_en.sh $TYPE $MODEL_NAME_OR_PATH $EXP_NAME $META_TEMPLATE $QUANT_TYPE

# # InternLM2.5 20B
# MODEL_NAME_OR_PATH=/data2/share/internlm/internlm2_5-20b-chat
# TYPE=vllm
# QUANT_TYPE=fp16
# EXP_NAME=internlm2.5-20b_${QUANT_TYPE}_en_vllm
# META_TEMPLATE=internlm
# CUDA_VISIBLE_DEVICES=6,7 bash test_all_en.sh $TYPE $MODEL_NAME_OR_PATH $EXP_NAME $META_TEMPLATE $QUANT_TYPE

# # InternLM2.5 20B AWQ
# MODEL_NAME_OR_PATH=/data2/share/internlm/internlm2_5-20b-chat-4bit-awq
# TYPE=vllm
# QUANT_TYPE=awq
# EXP_NAME=internlm2.5-20b-awq_${QUANT_TYPE}_en_vllm
# META_TEMPLATE=internlm
# CUDA_VISIBLE_DEVICES=6,7 bash test_all_en.sh $TYPE $MODEL_NAME_OR_PATH $EXP_NAME $META_TEMPLATE $QUANT_TYPE


# Qwen2.5 1.5B
MODEL_NAME_OR_PATH=/data2/share/Qwen2.5/Qwen2.5-1.5B-Instruct
TYPE=vllm
QUANT_TYPE=fp16
EXP_NAME=qwen-1.5b_${QUANT_TYPE}_en_vllm
META_TEMPLATE=qwen
CUDA_VISIBLE_DEVICES=7 bash test_all_en.sh $TYPE $MODEL_NAME_OR_PATH $EXP_NAME $META_TEMPLATE $QUANT_TYPE
