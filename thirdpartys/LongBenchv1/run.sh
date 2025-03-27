# CUDA_VISIBLE_DEVICES=3 python pred.py --model Qwen2.5-7B-Instruct-Mag-Un-0.5 &
# CUDA_VISIBLE_DEVICES=4 python pred.py --model Qwen2.5-7B-Instruct-Mag-2-4-0.5 &
# CUDA_VISIBLE_DEVICES=5 python pred.py --model Qwen2.5-7B-Instruct-Wanda-Un-0.5 &
# CUDA_VISIBLE_DEVICES=6 python pred.py --model Qwen2.5-7B-Instruct-Wanda-2-4-0.5 


# CUDA_VISIBLE_DEVICES=3 python pred.py --model Qwen2.5-7B-Instruct-SparseGPT-Un-0.5 &
# CUDA_VISIBLE_DEVICES=4 python pred.py --model Qwen2.5-7B-Instruct-SparseGPT-2-4-0.5 &


CUDA_VISIBLE_DEVICES=2 python pred.py --model InternLM2.5-7B-Instruct-Mag-Un-0.5 
CUDA_VISIBLE_DEVICES=2 python pred.py --model InternLM2.5-7B-Instruct-Mag-2-4-0.5 
CUDA_VISIBLE_DEVICES=2 python pred.py --model InternLM2.5-7B-Instruct-Wanda-Un-0.5 
CUDA_VISIBLE_DEVICES=2 python pred.py --model InternLM2.5-7B-Instruct-Wanda-2-4-0.5 
CUDA_VISIBLE_DEVICES=2 python pred.py --model InternLM2.5-7B-Instruct-SparseGPT-Un-0.5 
CUDA_VISIBLE_DEVICES=2 python pred.py --model InternLM2.5-7B-Instruct-SparseGPT-2-4-0.5 
CUDA_VISIBLE_DEVICES=2 python pred.py --model InternLM2.5-7B-Instruct 

# CUDA_VISIBLE_DEVICES=3 python pred.py --model Qwen2.5-7B-Instruct-RTN-w4
# CUDA_VISIBLE_DEVICES=4 python pred.py --model Qwen2.5-7B-Instruct-GPTQ-w4a16

# CUDA_VISIBLE_DEVICES=5 python pred.py --model InternLM2.5-7B-Instruct-AWQ
# CUDA_VISIBLE_DEVICES=6 python pred.py --model InternLM2.5-7B-Instruct-GPTQ-w4a16
# CUDA_VISIBLE_DEVICES=7 python pred.py --model InternLM2.5-7B-Instruct-RTN-w4


# "InternLM2.5-7B-Instruct-AWQ": "/data2/share/internlm/internlm2_5-7b-chat-AWQ-W4-G128",
# "InternLM2.5-7B-Instruct-GPTQ-w4a16": "/data2/share/internlm/internlm2_5-7b-chat-GPTQ-w4a16",
# "InternLM2.5-7B-Instruct-RTN-w4": "/data2/share/internlm/internlm2_5-7b-chat-RTN-w4",
# "Qwen2.5-7B-Instruct-RTN-w4": "/data2/share/Qwen2.5/Qwen2.5-7B-Instruct-RTN-w4",
# "Qwen2.5-7B-Instruct-GPTQ-w4a16": "/data2/share/Qwen2.5/Qwen2.5-7B-Instruct-GPTQ-w4a16"


# CUDA_VISIBLE_DEVICES=1,2 python eval.py --model Qwen2.5-7B-Instruct


