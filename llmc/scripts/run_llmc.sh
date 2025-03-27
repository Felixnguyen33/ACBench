#!/bin/bash

export CUDA_VISIBLE_DEVICES=4,5
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export NCCL_TIMEOUT=80
# 添加以下环境变量
# export NCCL_IB_DISABLE=1  # 如果不使用InfiniBand，建议禁用
# export NCCL_P2P_DISABLE=1  # 如果GPU之间通信不稳定，可以禁用P2P
# export NCCL_SOCKET_IFNAME=eth0  # 指定网络接口，根据实际情况修改
# export NCCL_ASYNC_ERROR_HANDLING=1  # 启用异步错误处理


llmc=/data2/share/peijiedong/AgentBench/thirdparty/llmc
export PYTHONPATH=$llmc:$PYTHONPATH

task_name=$1
# config=${llmc}/configs/quantization/methods/Awq/awq_w_only.yml
# config=${llmc}/configs/quantization/backend/vllm/awq_w4a16.yml
# config=${llmc}/configs/quantization/backend/vllm/rtn_w4a16.yml
# config=${llmc}/configs/quantization/backend/vllm/gptq_w4a16.yml

# sparsification
config=${llmc}/configs/sparsification/methods/Wanda/wanda.yml

nnodes=1
nproc_per_node=2


find_unused_port() {
    while true; do
        port=$(shuf -i 10000-60000 -n 1)
        if ! ss -tuln | grep -q ":$port "; then
            echo "$port"
            return 0
        fi
    done
}
UNUSED_PORT=$(find_unused_port)


MASTER_ADDR=127.0.0.1
MASTER_PORT=$UNUSED_PORT
task_id=$UNUSED_PORT

set -e 
exec 2>&1 

# 记录开始时间和环境信息
echo "Start time: $(date)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "MASTER_PORT: $MASTER_PORT"

nohup \
torchrun \
--nnodes $nnodes \
--nproc_per_node $nproc_per_node \
--rdzv_id $task_id \
--rdzv_backend c10d \
--rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
${llmc}/llmc/__main__.py --config $config --task_id $task_id \
> ${task_name}.log 2>&1 &

sleep 2
ps aux | grep '__main__.py' | grep $task_id | awk '{print $2}' > ${task_name}.pid


tail -f ${task_name}.log

# You can kill this program by 
# xargs kill -9 < xxx.pid
# xxx.pid is ${task_name}.pid file
