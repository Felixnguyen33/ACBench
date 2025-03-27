#!/bin/bash

TASKS_LIST=("alfworld" "jericho" "pddl" "webshop" "webarena" "tool-query" "tool-operation" "babyai" "scienceworld")
# webshop failed
# webarena failed
# babyai failed
# MODEL_LIST=("Qwen2.5-7B-Instruct" "Qwen2.5-7B-Instruct-AWQ" "Qwen2.5-7B-Instruct-Mag-Un-0.5" \
# "Qwen2.5-7B-Instruct-Mag-2-4-0.5" "Qwen2.5-7B-Instruct-Wanda-Un-0.5" "Qwen2.5-7B-Instruct-Wanda-2-4-0.5" \
# "Qwen2.5-7B-Instruct-SparseGPT-Un-0.5" "Qwen2.5-7B-Instruct-SparseGPT-2-4-0.5" "InternLM2.5-7B-Instruct" \
# "InternLM2.5-7B-Instruct-Mag-Un-0.5" "InternLM2.5-7B-Instruct-Mag-2-4-0.5" "InternLM2.5-7B-Instruct-Wanda-Un-0.5" \
# "InternLM2.5-7B-Instruct-Wanda-2-4-0.5" "InternLM2.5-7B-Instruct-SparseGPT-Un-0.5" "InternLM2.5-7B-Instruct-SparseGPT-2-4-0.5" \
# "InternLM2.5-7B-Instruct-AWQ" "InternLM2.5-7B-Instruct-GPTQ-w4a16" "InternLM2.5-7B-Instruct-RTN-w4" \
# "Qwen2.5-7B-Instruct-RTN-w4" "Qwen2.5-7B-Instruct-GPTQ-w4a16")



CUDA_VISIBLE_DEVICES=6 python agentboard/eval_main.py \
    --cfg eval_configs/main_results_all_tasks.yaml \
    --tasks scienceworld \
    --model Qwen2.5-7B-Instruct \
    --log_path results/Qwen2.5-7B-Instruct \
    --wandb \
    --project_name evaluate-Qwen2.5-7B-Instruct \
    --baseline_dir data/baseline_results