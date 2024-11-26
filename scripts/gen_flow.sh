#!/bin/bash 

MODEL=$1
DTYPE=$2
QUANT=$3

tasks=(wikihow toolbench toolalpaca lumos alfworld webshop os)
if [ ${MODEL} == "llama3.1" ]; then
    model_name=/data2/share/llama3.1/llama-3.1-8B-Instruct
elif [ ${MODEL} == "mistral_7b" ]; then
    model_name=/data2/share/mistral-7B/Mistral-7B-v0.1
elif [ ${MODEL} == "qwen2_7b" ]; then
    model_name=/data2/share/Qwen2/Qwen2-7B-Instruct
fi

for task in ${tasks[@]}; do
    python node_eval.py \
        --task gen_workflow \
        --model_name ${model_name} \
        --gold_path ./gold_traj/${task}/graph_eval.json \
        --pred_path ./pred_traj/${MODEL}_${DTYPE}/${task}/${model_name}/graph_eval_two_shot.json\
        --task_type ${task} \
        --few_shot \
        --quantization ${QUANT} 
done
