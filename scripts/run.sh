


# TEST using SMALL LLM
bash ./scripts/gen_flow.sh /data2/share/llama3.2/Llama-3.2-1B-Instruct-awq-w4-g128-zp full AWQ 2

# GEN FLOW
# #GPTQ
# bash gen_flow.sh llama3.1 half gptq > gen_flow_llama3.1_half_gptq.out
# bash gen_flow.sh mistral_7b half gptq > gen_flow_mistral_7b_half_gptq.out
# bash gen_flow.sh qwen2_7b half gptq > gen_flow_qwen2_7b_half_gptq.out


# #AWQ
# bash gen_flow.sh llama3.1 half awq > gen_flow_llama3.1_half_awq.out
# bash gen_flow.sh mistral_7b half awq > gen_flow_mistral_7b_half_awq.out
# bash gen_flow.sh qwen2_7b half awq > gen_flow_qwen2_7b_half_awq.out

# #FP8
# bash gen_flow.sh llama3.1 half fp8 > gen_flow_llama3.1_half_fp8.out
# bash gen_flow.sh mistral_7b half fp8 > gen_flow_mistral_7b_half_fp8.out
# bash gen_flow.sh qwen2_7b half fp8 > gen_flow_qwen2_7b_half_fp8.out

# EVAL FLOW
# GPTQ
# bash eval.sh llama3.1 half > eval_llama3.1_half.out
# bash eval.sh mistral_7b half > eval_mistral_7b_half.out
# bash eval.sh qwen2_7b half > eval_qwen2_7b_half.out

# AWQ
# bash eval.sh llama3.1 half awq > eval_llama3.1_half_awq.out
# bash eval.sh mistral_7b half awq > eval_mistral_7b_half_awq.out
# bash eval.sh qwen2_7b half awq > eval_qwen2_7b_half_awq.out

# FP8
# bash eval.sh llama3.1 half fp8 > eval_llama3.1_half_fp8.out
# bash eval.sh mistral_7b half fp8 > eval_mistral_7b_half_fp8.out
# bash eval.sh qwen2_7b half fp8 > eval_qwen2_7b_half_fp8.out

