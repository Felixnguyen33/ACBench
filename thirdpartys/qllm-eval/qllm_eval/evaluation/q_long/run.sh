MODEL_PATH=/data2/share/gemma/gemma-7b
# /data2/share/deepseek/DeepSeek-R1-Distill-Qwen-1.5B

mkdir -p logs

# weight-only quant on GPU 2:
# FP16;W8;W4;W3;W2 
for w_bit in 8 4 3 2; do
  LOG_NAME="w${w_bit}_a16_kv16"
  CUDA_VISIBLE_DEVICES=2 python main_longeval.py \
      --model-name-or-path ${MODEL_PATH} \
      --use_flash_attn \
      --task lines \
      --test_dir new_cases \
      --w_group_size 128 \
      --w_bit ${w_bit} \
      --a_group_size 128 \
      --a_bit 16 \
      --kv_group_size 128 \
      --kv_bit 16 \
      2>&1 | tee "logs/${LOG_NAME}.log"
done &

# weight-activation quant on GPU 6:
# FP16;W8A8;W4A8;W4A4;
(
for w_bit in 8 4; do
  for a_bit in 8 4; do
    LOG_NAME="w${w_bit}_a${a_bit}_kv16"
    CUDA_VISIBLE_DEVICES=6 python main_longeval.py \
        --model-name-or-path ${MODEL_PATH} \
        --use_flash_attn \
        --task lines \
        --test_dir new_cases \
        --w_group_size 128 \
        --w_bit ${w_bit} \
        --a_group_size 128 \
        --a_bit ${a_bit} \
        --kv_group_size 128 \
        --kv_bit 16 \
        2>&1 | tee "logs/${LOG_NAME}.log"
  done
done
) &

# kv-only quant on GPU 7:(W16A16)
# FP16;KV8;KV4;KV2
(
for kv_bit in 8 4 2; do
  LOG_NAME="w16_a16_kv${kv_bit}"
  CUDA_VISIBLE_DEVICES=7 python main_longeval.py \
      --model-name-or-path ${MODEL_PATH} \
      --use_flash_attn \
      --task lines \
      --test_dir new_cases \
      --w_group_size 128 \
      --w_bit 16 \
      --a_group_size 128 \
      --a_bit 16 \
      --kv_group_size 128 \
      --kv_bit ${kv_bit} \
      2>&1 | tee "logs/${LOG_NAME}.log"
done
) &

# Wait for all background processes to complete
wait
