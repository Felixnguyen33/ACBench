#!/bin/bash 

MODEL=$1
DTYPE=$2
QUANT=$3
DEVICE=${4:-6}

export CUDA_VISIBLE_DEVICES=$DEVICE

tasks=(os)
# (wikihow toolbench toolalpaca lumos alfworld webshop os)
# MODEL=/data2/share/llama3.2/Llama-3.2-1B-Instruct-awq-w4-g128-zp
# MODEL_NAME=Llama-3.2-1B-Instruct-awq-w4-g128-zp

MODEL_NAME=$(basename $MODEL)

for task in ${tasks[@]}; do
    python src/node_eval.py \
        --task gen_workflow \
        --model_name ${MODEL} \
        --gold_path ./data/gold_traj/${task}/graph_eval.json \
        --pred_path ./data/pred_traj/${MODEL_NAME}_${DTYPE}/${task}/${MODEL_NAME}/graph_eval_two_shot.json\
        --task_type ${task} \
        --few_shot \
        --quantization ${QUANT} 
done
