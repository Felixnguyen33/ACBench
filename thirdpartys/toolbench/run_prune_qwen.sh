# agent conda

# MODEL+PRUNE=internlm2.5+2-4/unstructured
# ROOT_DIR=/data2/share/peijiedong/AgentBench/thirdparty/wanda/out/internlm_7b
# 2-4:
# magnitude: internlm-2.5-7b-chat-magnitude-2-4-0.5
# sparsegpt: internlm-2.5-7b-chat-sparsegpt-2-4-0.5  
# wanda: internlm-2.5-7b-chat-wanda-2-4-0.5
# unstructured:
# magnitude: internlm-2.5-7b-chat-magnitude-un0.5
# sparsegpt: internlm-2.5-7b-chat-sparsegpt-un0.5
# wanda: internlm-2.5-7b-chat-wanda-un0.5

# MODEL+PRUNE=qwen2.5+2-4/unstructured
# ROOT_DIR=/data2/share/peijiedong/AgentBench/thirdparty/wanda/out/qwen_7b
# 2-4:
# magnitude: qwen-2.5-7b-chat-magnitude-2-4-0.5
# sparsegpt: qwen-2.5-7b-chat-sparsegpt-2-4-0.5
# wanda: qwen-2.5-7b-chat-wanda-2-4-0.5
# unstructured:
# magnitude: qwen-2.5-7b-chat-magnitude-un0.5
# sparsegpt: qwen-2.5-7b-chat-sparsegpt-un0.5
# wanda: qwen-2.5-7b-chat-wanda-un0.5

# prune -> mag, sparsegpt, wanda
# type: un, 2-4

# QWEN 2.5 7B
# # 2-4 magnitude
# MODEL_NAME_OR_PATH=/data2/share/peijiedong/AgentBench/thirdparty/wanda/out/qwen_7b/2-4/magnitude/qwen-2.5-7b-chat-magnitude-2-4-0.5
# TYPE=vllm
# PRUNE_TYPE=mag_2-4
# EXP_NAME=qwen2.5-7b_${PRUNE_TYPE}_en_vllm
# META_TEMPLATE=qwen
# CUDA_VISIBLE_DEVICES=6 bash test_all_en.sh $TYPE $MODEL_NAME_OR_PATH $EXP_NAME $META_TEMPLATE $PRUNE_TYPE &

# # un magnitude
# MODEL_NAME_OR_PATH=/data2/share/peijiedong/AgentBench/thirdparty/wanda/out/qwen_7b/unstructured/magnitude/qwen-2.5-7b-chat-magnitude-un0.5
# TYPE=vllm
# PRUNE_TYPE=mag_un
# EXP_NAME=qwen2.5-7b_${PRUNE_TYPE}_en_vllm
# META_TEMPLATE=qwen
# CUDA_VISIBLE_DEVICES=5 bash test_all_en.sh $TYPE $MODEL_NAME_OR_PATH $EXP_NAME $META_TEMPLATE $PRUNE_TYPE &

# # 2-4 sparsegpt
# MODEL_NAME_OR_PATH=/data2/share/peijiedong/AgentBench/thirdparty/wanda/out/qwen_7b/2-4/sparsegpt/qwen-2.5-7b-chat-sparsegpt-2-4-0.5
# TYPE=vllm
# PRUNE_TYPE=sparsegpt_2-4
# EXP_NAME=qwen2.5-7b_${PRUNE_TYPE}_en_vllm
# META_TEMPLATE=qwen
# CUDA_VISIBLE_DEVICES=1 bash test_all_en.sh $TYPE $MODEL_NAME_OR_PATH $EXP_NAME $META_TEMPLATE $PRUNE_TYPE &

# # un sparsegpt
# MODEL_NAME_OR_PATH=/data2/share/peijiedong/AgentBench/thirdparty/wanda/out/qwen_7b/unstructured/sparsegpt/qwen-2.5-7b-chat-sparsegpt-un0.5
# TYPE=vllm
# PRUNE_TYPE=sparsegpt_un
# EXP_NAME=qwen2.5-7b_${PRUNE_TYPE}_en_vllm
# META_TEMPLATE=qwen
# CUDA_VISIBLE_DEVICES=2 bash test_all_en.sh $TYPE $MODEL_NAME_OR_PATH $EXP_NAME $META_TEMPLATE $PRUNE_TYPE &

# # 2-4 wanda
# MODEL_NAME_OR_PATH=/data2/share/peijiedong/AgentBench/thirdparty/wanda/out/qwen_7b/2-4/wanda/qwen-2.5-7b-chat-wanda-2-4-0.5
# TYPE=vllm
# PRUNE_TYPE=wanda_2-4
# EXP_NAME=qwen2.5-7b_${PRUNE_TYPE}_en_vllm
# META_TEMPLATE=qwen
# CUDA_VISIBLE_DEVICES=3 bash test_all_en.sh $TYPE $MODEL_NAME_OR_PATH $EXP_NAME $META_TEMPLATE $PRUNE_TYPE &

# # un wanda
# MODEL_NAME_OR_PATH=/data2/share/peijiedong/AgentBench/thirdparty/wanda/out/qwen_7b/unstructured/wanda/qwen-2.5-7b-chat-wanda-un0.5
# TYPE=vllm
# PRUNE_TYPE=wanda_un
# EXP_NAME=qwen2.5-7b_${PRUNE_TYPE}_en_vllm
# META_TEMPLATE=qwen
# CUDA_VISIBLE_DEVICES=4 bash test_all_en.sh $TYPE $MODEL_NAME_OR_PATH $EXP_NAME $META_TEMPLATE $PRUNE_TYPE &


# QWEN 2.5 14B
# # 2-4 magnitude
# MODEL_NAME_OR_PATH=/mnt/sdd/dongpeijie/out_pruned_llm/qwen_14b/2-4/magnitude/qwen-2.5-14b-instruct-magnitude-2-4-0.5
# TYPE=vllm
# PRUNE_TYPE=mag_2-4
# EXP_NAME=qwen2.5-14b_${PRUNE_TYPE}_en_vllm
# META_TEMPLATE=qwen
# CUDA_VISIBLE_DEVICES=6 bash test_all_en.sh $TYPE $MODEL_NAME_OR_PATH $EXP_NAME $META_TEMPLATE $PRUNE_TYPE &

# un magnitude
MODEL_NAME_OR_PATH=/mnt/sdd/dongpeijie/out_pruned_llm/qwen_14b/unstructured/magnitude/qwen-2.5-14b-instruct-magnitude-un0.5
TYPE=vllm
PRUNE_TYPE=mag_un
EXP_NAME=qwen2.5-14b_${PRUNE_TYPE}_en_vllm
META_TEMPLATE=qwen
CUDA_VISIBLE_DEVICES=5 bash test_all_en.sh $TYPE $MODEL_NAME_OR_PATH $EXP_NAME $META_TEMPLATE $PRUNE_TYPE &

# 2-4 sparsegpt
MODEL_NAME_OR_PATH=/mnt/sdd/dongpeijie/out_pruned_llm/qwen_14b/2-4/sparsegpt/qwen-2.5-14b-instruct-sparsegpt-2-4-0.5
TYPE=vllm
PRUNE_TYPE=sparsegpt_2-4
EXP_NAME=qwen2.5-14b_${PRUNE_TYPE}_en_vllm
META_TEMPLATE=qwen
CUDA_VISIBLE_DEVICES=1 bash test_all_en.sh $TYPE $MODEL_NAME_OR_PATH $EXP_NAME $META_TEMPLATE $PRUNE_TYPE &

# un sparsegpt
MODEL_NAME_OR_PATH=/mnt/sdd/dongpeijie/out_pruned_llm/qwen_14b/unstructured/sparsegpt/qwen-2.5-14b-instruct-sparsegpt-un0.5
TYPE=vllm
PRUNE_TYPE=sparsegpt_un
EXP_NAME=qwen2.5-14b_${PRUNE_TYPE}_en_vllm
META_TEMPLATE=qwen
CUDA_VISIBLE_DEVICES=2 bash test_all_en.sh $TYPE $MODEL_NAME_OR_PATH $EXP_NAME $META_TEMPLATE $PRUNE_TYPE &

# 2-4 wanda
MODEL_NAME_OR_PATH=/mnt/sdd/dongpeijie/out_pruned_llm/qwen_14b/2-4/wanda/qwen-2.5-14b-instruct-wanda-2-4-0.5
TYPE=vllm
PRUNE_TYPE=wanda_2-4
EXP_NAME=qwen2.5-14b_${PRUNE_TYPE}_en_vllm
META_TEMPLATE=qwen
CUDA_VISIBLE_DEVICES=3 bash test_all_en.sh $TYPE $MODEL_NAME_OR_PATH $EXP_NAME $META_TEMPLATE $PRUNE_TYPE &


# un sparsegpt
MODEL_NAME_OR_PATH=/mnt/sdd/dongpeijie/out_pruned_llm/qwen_14b/unstructured/sparsegpt/qwen-2.5-14b-instruct-sparsegpt-un0.5
TYPE=vllm
PRUNE_TYPE=sparsegpt_un
EXP_NAME=qwen2.5-14b_${PRUNE_TYPE}_en_vllm
META_TEMPLATE=qwen
CUDA_VISIBLE_DEVICES=4 bash test_all_en.sh $TYPE $MODEL_NAME_OR_PATH $EXP_NAME $META_TEMPLATE $PRUNE_TYPE &

