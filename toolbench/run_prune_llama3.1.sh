# agent conda

# prune -> mag, sparsegpt, wanda
# type: un, 2-4

# LLAMA 3.1 8B
# # 2-4 magnitude
# MODEL_NAME_OR_PATH=/data2/share/peijiedong/AgentBench/thirdparty/wanda/out/llama_3.1_8b/2-4/magnitude/llama-3.1-8b-instruct-magnitude-2-4-0.5
# TYPE=vllm
# PRUNE_TYPE=mag_2-4
# EXP_NAME=llama3.1-8b_${PRUNE_TYPE}_en_vllm
# META_TEMPLATE=llama
# CUDA_VISIBLE_DEVICES=6 bash test_all_en.sh $TYPE $MODEL_NAME_OR_PATH $EXP_NAME $META_TEMPLATE $PRUNE_TYPE &

# # un magnitude 
# MODEL_NAME_OR_PATH=/data2/share/peijiedong/AgentBench/thirdparty/wanda/out/llama_3.1_8b/unstructured/magnitude/llama-3.1-8b-instruct-magnitude-un0.5
# TYPE=vllm
# PRUNE_TYPE=mag_un
# EXP_NAME=llama3.1-8b_${PRUNE_TYPE}_en_vllm
# META_TEMPLATE=llama
# CUDA_VISIBLE_DEVICES=1 bash test_all_en.sh $TYPE $MODEL_NAME_OR_PATH $EXP_NAME $META_TEMPLATE $PRUNE_TYPE &

# # 2-4 sparsegpt
# MODEL_NAME_OR_PATH=/data2/share/peijiedong/AgentBench/thirdparty/wanda/out/llama_3.1_8b/2-4/sparsegpt/llama-3.1-8b-instruct-sparsegpt-2-4-0.5
# TYPE=vllm
# PRUNE_TYPE=sparsegpt_2-4
# EXP_NAME=llama3.1-8b_${PRUNE_TYPE}_en_vllm
# META_TEMPLATE=llama
# CUDA_VISIBLE_DEVICES=2 bash test_all_en.sh $TYPE $MODEL_NAME_OR_PATH $EXP_NAME $META_TEMPLATE $PRUNE_TYPE &

# un sparsegpt
MODEL_NAME_OR_PATH=/data2/share/peijiedong/AgentBench/thirdparty/wanda/out/llama_3.1_8b/unstructured/sparsegpt/llama-3.1-8b-instruct-sparsegpt-un0.5
TYPE=vllm
PRUNE_TYPE=sparsegpt_un
EXP_NAME=llama3.1-8b_${PRUNE_TYPE}_en_vllm
META_TEMPLATE=llama
CUDA_VISIBLE_DEVICES=3 bash test_all_en.sh $TYPE $MODEL_NAME_OR_PATH $EXP_NAME $META_TEMPLATE $PRUNE_TYPE &

# # 2-4 wanda
# MODEL_NAME_OR_PATH=/data2/share/peijiedong/AgentBench/thirdparty/wanda/out/llama_3.1_8b/2-4/wanda/llama-3.1-8b-instruct-wanda-2-4-0.5
# TYPE=vllm
# PRUNE_TYPE=wanda_2-4
# EXP_NAME=llama3.1-8b_${PRUNE_TYPE}_en_vllm
# META_TEMPLATE=llama
# CUDA_VISIBLE_DEVICES=4 bash test_all_en.sh $TYPE $MODEL_NAME_OR_PATH $EXP_NAME $META_TEMPLATE $PRUNE_TYPE &

# # un wanda
# MODEL_NAME_OR_PATH=/data2/share/peijiedong/AgentBench/thirdparty/wanda/out/llama_3.1_8b/unstructured/wanda/llama-3.1-8b-instruct-wanda-un0.5
# TYPE=vllm
# PRUNE_TYPE=wanda_un
# EXP_NAME=llama3.1-8b_${PRUNE_TYPE}_en_vllm
# META_TEMPLATE=llama
# CUDA_VISIBLE_DEVICES=5 bash test_all_en.sh $TYPE $MODEL_NAME_OR_PATH $EXP_NAME $META_TEMPLATE $PRUNE_TYPE 
