export CUDA_VISIBLE_DEVICES=5,6

METHOD="FullKV"       # Support PyramidKV, SnapKV, H2O, StreamingLLM, CAM
MAX_CAPACITY_PROMPT=128 # 128,2048 in paper
attn_implementation="flash_attention_2" # Support "flash_attention_2", "sdpa", "eager".
quant_method=None # Support kivi and kvquant, default None.
nbits=8 # Quantization bit-width support 8,4,2. Need to set quant_method first.

mkdir -p ./results_ruler/logs/

# For InternLM2.5-7B-Instruct
(
python3 run_ruler.py \
    --method ${METHOD} \
    --model_path /data2/share/internlm/internlm2_5-7b-chat \
    --max_capacity_prompts ${MAX_CAPACITY_PROMPT} \
    --attn_implementation ${attn_implementation} \
    --save_dir results_ruler \
    --use_cache True \
    --nbits ${nbits} 
) 2>&1 | tee results_ruler/logs/InternLM2.5_7b_Instruct_${METHOD}_${MAX_CAPACITY_PROMPT}.log

# Define root directory for InternLM
root_dir_internlm=/data2/share/peijiedong/AgentBench/thirdparty/wanda/out/internlm_7b

# For InternLM2.5-7B pruned models
for prune_method in magnitude wanda sparsegpt; do
    for struct in un 2-4; do
        model_path_suffix=""
        if [ "$struct" = "un" ]; then
            model_path_suffix="unstructured/${prune_method}/internlm-2.5-7b-chat-${prune_method}-un0.5"
        else
            model_path_suffix="2-4/${prune_method}/internlm-2.5-7b-chat-${prune_method}-2-4-0.5"
        fi
        
        (
        python3 run_ruler.py \
            --method ${METHOD} \
            --model_path ${root_dir_internlm}/${model_path_suffix} \
            --max_capacity_prompts ${MAX_CAPACITY_PROMPT} \
            --attn_implementation ${attn_implementation} \
            --save_dir results_ruler \
            --use_cache True \
            --nbits ${nbits} 
        ) 2>&1 | tee results_ruler/logs/InternLM2.5_7b_${prune_method}_${struct}_${METHOD}_${MAX_CAPACITY_PROMPT}.log
    done
done
