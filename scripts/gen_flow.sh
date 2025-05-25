#!/bin/bash 

MODEL=$1
TEMP=$2
QUANT=$3

DEVICE=${4:-6}

export CUDA_VISIBLE_DEVICES=$DEVICE

tasks=(wikihow toolbench toolalpaca lumos alfworld webshop os)

MODEL_NAME=$(basename $MODEL)

for task in ${tasks[@]}; do
    python agentbench/node_eval.py \
        --task gen_workflow \
        --model_name ${MODEL} \
        --gold_path ./data/gold_traj/${task}/graph_eval.json \
        --pred_path ./data/pred_traj/${MODEL_NAME}/${task}/${MODEL_NAME}/graph_eval_two_shot.json \
        --task_type ${task} \
        --few_shot \
        --temperature ${TEMP} \
        --quantization ${QUANT} 
done
