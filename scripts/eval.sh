#!/bin/bash 

MODEL=$1
DTYPE=$2
QUANT=$3
DEVICE=${4:-6}

export CUDA_VISIBLE_DEVICES=$DEVICE

tasks=(os)
# (wikihow toolbench toolalpaca lumos alfworld webshop os)

MODEL_NAME=$(basename $MODEL)

for task in ${tasks[@]}; do
    python src/node_eval.py \
        --task eval_workflow \
        --model_name ${MODEL} \
        --gold_path ./data/gold_traj/${task}/graph_eval.json \
        --pred_path ./data/pred_traj/${MODEL_NAME}_${DTYPE}/${task}/${MODEL_NAME}/graph_eval_two_shot.json \
        --eval_model all-mpnet-base-v2 \
        --eval_output ./data/eval_result/${MODEL_NAME}_${DTYPE}/${MODEL_NAME}_${task}_graph_eval_two_shot.json \
        --eval_type node \
        --task_type ${task} \
        --few_shot \
        --quantization ${QUANT}
done
