# agent conda 

# MISTRAL
MODEL_NAME_OR_PATH=/data2/share/mistral/Mistral-7B-v0.1
TYPE=vllm 
QUANT_TYPE=fp16
EXP_NAME=mistral-v0.1-7b_${QUANT_TYPE}_en_vllm
META_TEMPLATE=mistral

CUDA_VISIBLE_DEVICES=0 bash test_all_en.sh $TYPE $MODEL_NAME_OR_PATH $EXP_NAME $META_TEMPLATE $QUANT_TYPE &


# QWEN2.5 7B Instruct
MODEL_NAME_OR_PATH=/data2/share/Qwen2.5/Qwen2.5-7B-Instruct
TYPE=vllm
QUANT_TYPE=fp16
EXP_NAME=qwen2.5-7b_${QUANT_TYPE}_en_vllm
META_TEMPLATE=qwen

CUDA_VISIBLE_DEVICES=1 bash test_all_en.sh $TYPE $MODEL_NAME_OR_PATH $EXP_NAME $META_TEMPLATE $QUANT_TYPE &

# InternLM 2.5 7B
MODEL_NAME_OR_PATH=/data2/share/internlm/internlm2_5-7b-chat
TYPE=vllm 
QUANT_TYPE=fp16
EXP_NAME=internlm2.5-7b-intr_${QUANT_TYPE}_en_vllm
META_TEMPLATE=internlm
CUDA_VISIBLE_DEVICES=2 bash test_all_en.sh $TYPE $MODEL_NAME_OR_PATH $EXP_NAME $META_TEMPLATE $QUANT_TYPE &


# display_name=test_llama2
# model_path=/data2/share/llama-2/Llama-2-7b-hf
# meta_template=llama2
# batch_size=32
# QUANT_TYPE=fp16
# CUDA_VISIBLE_DEVICES=5 python test.py --model_type vllm --resume --out_name retrieve_str_$display_name.json --out_dir work_dirs/$display_name/ --dataset_path data/retrieve_str_v2_subset.json --eval retrieve --prompt_type str --model_path $model_path --model_display_name $display_name --meta_template $meta_template --batch_size $batch_size --quantization $QUANT_TYPE
