

# autoawq 

export CUDA_VISIBLE_DEVICES=1 
# version: gemm,marlin,gemv;gemv_fast
# python quant/autoawq.py \
#     --model_path /data2/share/llama3.2/Llama-3.2-1B-Instruct \
#     --zero_point True \
#     --q_group_size 128 \
#     --w_bit 4 \
#     --version gemm

# llama3
python quant/autoawq.py \
    --model_path /data2/share/llama3.1/llama-3.1-8B-Instruct \
    --zero_point True \
    --q_group_size 128 \
    --w_bit 4 \
    --version gemm 