export NUMEXPR_MAX_THREADS=80  # 或在代码中设置 os.environ["NUMEXPR_MAX_THREADS"] = "80"


# CUDA_VISIBLE_DEVICES=3 python pred.py --model Qwen2.5-7B-Instruct-Mag-Un-0.5 &
# CUDA_VISIBLE_DEVICES=4 python pred.py --model Qwen2.5-7B-Instruct-Mag-2-4-0.5 &
# CUDA_VISIBLE_DEVICES=5 python pred.py --model Qwen2.5-7B-Instruct-Wanda-Un-0.5 &
# CUDA_VISIBLE_DEVICES=6 python pred.py --model Qwen2.5-7B-Instruct-Wanda-2-4-0.5 


# CUDA_VISIBLE_DEVICES=3 python pred.py --model Qwen2.5-7B-Instruct-SparseGPT-Un-0.5 &
# CUDA_VISIBLE_DEVICES=4 python pred.py --model Qwen2.5-7B-Instruct-SparseGPT-2-4-0.5 &


# CUDA_VISIBLE_DEVICES=2 python pred.py --model InternLM2.5-7B-Instruct-Mag-Un-0.5 
# CUDA_VISIBLE_DEVICES=2 python pred.py --model InternLM2.5-7B-Instruct-Mag-2-4-0.5 
# CUDA_VISIBLE_DEVICES=2 python pred.py --model InternLM2.5-7B-Instruct-Wanda-Un-0.5 
# CUDA_VISIBLE_DEVICES=2 python pred.py --model InternLM2.5-7B-Instruct-Wanda-2-4-0.5 
# CUDA_VISIBLE_DEVICES=2 python pred.py --model InternLM2.5-7B-Instruct-SparseGPT-Un-0.5 
# CUDA_VISIBLE_DEVICES=2 python pred.py --model InternLM2.5-7B-Instruct-SparseGPT-2-4-0.5 
# CUDA_VISIBLE_DEVICES=2 python pred.py --model InternLM2.5-7B-Instruct 

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


# CUDA_VISIBLE_DEVICES=6 python pred.py --model deepseek-qwen-1.5b
# CUDA_VISIBLE_DEVICES=6 python pred.py --model deepseek-qwen-7b
# CUDA_VISIBLE_DEVICES=6 python pred.py --model deepseek-llama-8b
# CUDA_VISIBLE_DEVICES=6 python pred.py --model megrez-3b
# CUDA_VISIBLE_DEVICES=6 python pred.py --model qwen-3b-gptq-int4
# CUDA_VISIBLE_DEVICES=6 python pred.py --model qwen-3b-gptq-int8


# ["qwen-14b"]="/data2/share/Qwen2.5/Qwen2.5-14B-Instruct"
# ["qwen-32b"]="/data2/share/Qwen2.5/Qwen2.5-32B-Instruct"
# ["qwen-14b-awq"]="/data2/share/Qwen2.5/Qwen2.5-14B-Instruct-AWQ"
# ["qwen-32b-awq"]="/data2/share/Qwen2.5/Qwen2.5-32B-Instruct-AWQ"
# ["qwen-14b-gptq-int4"]="/data2/share/Qwen2.5/Qwen2.5-14B-Instruct-GPTQ-Int4"
# ["qwen-32b-gptq-int4"]="/data2/share/Qwen2.5/Qwen2.5-32B-Instruct-GPTQ-Int4"
# ["internlm3-8b"]="/data2/share/internlm/internlm3-8b-instruct"
# ["internlm3-8b-awq"]="/data2/share/internlm/internlm3-8b-instruct-awq"
# ["internlm3-8b-gptq-int4"]="/data2/share/internlm/internlm3-8b-instruct-gptq-int4"
# ["internlm2.5-20b-awq"]="/data2/share/internlm/internlm2_5-20b-chat-4bit-awq"
# ["internlm2.5-20b"]="/data2/share/internlm/internlm2_5-20b-chat"


# failed ones should re-run

export NUMEXPR_MAX_THREADS=80  # 或在代码中设置 os.environ["NUMEXPR_MAX_THREADS"] = "80"
# CUDA_VISIBLE_DEVICES=1 python pred.py --model minicpm-4b
# CUDA_VISIBLE_DEVICES=1 python pred.py --model qwen-3b-awq
# CUDA_VISIBLE_DEVICES=1 python pred.py --model qwen-1.5b-gptq-int4
# CUDA_VISIBLE_DEVICES=1 python pred.py --model qwen-1.5b-gptq-int8
# CUDA_VISIBLE_DEVICES=1 python pred.py --model qwen-1.5b-awq
# CUDA_VISIBLE_DEVICES=1 python pred.py --model gemma-2b > gemma-2b.log &
# CUDA_VISIBLE_DEVICES=2 python pred.py --model InternLM2.5-7B-Instruct > InternLM2.5-7B-Instruct.log &
# CUDA_VISIBLE_DEVICES=7 python pred.py --model InternLM2.5-7B-Instruct-GPTQ-w4a16 > InternLM2.5-7B-Instruct-GPTQ-w4a16.log &
# CUDA_VISIBLE_DEVICES=1 python pred.py --model phi-3.5
# CUDA_VISIBLE_DEVICES=7 python pred.py --model InternLM2.5-7B-Instruct-SparseGPT-2-4-0.5 > InternLM2.5-7B-SparseGPT-2-4-0.5.log &
# CUDA_VISIBLE_DEVICES=2 python pred.py --model qwen-1.5b > qwen-1.5b.log 
# CUDA_VISIBLE_DEVICES=2 python pred.py --model qwen-3b > qwen-3b.log 

CUDA_VISIBLE_DEVICES=2 python pred.py --model InternLM2.5-7B-Instruct-SparseGPT-2-4-0.5 
# > InternLM2.5-7B-SparseGPT-2-4-0.5.log &
# CUDA_VISIBLE_DEVICES=1 python pred.py --model InternLM2.5-7B-Instruct