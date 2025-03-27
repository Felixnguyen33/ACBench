#!/bin/bash 

ROOT_DIR=/data2/share/peijiedong/AgentBench/thirdparty/LongBenchv1/pred

if [ ! -d "$ROOT_DIR" ]; then
    echo "Error: Directory $ROOT_DIR does not exist"
    exit 1
fi

for subfolder in "$ROOT_DIR"/*; do
    if [ ! -d "$subfolder" ]; then
        continue
    fi
    
    echo "Evaluating $subfolder"
    subfolder_name=$(basename "$subfolder")
    
    if ! python eval.py --model "$subfolder_name"; then
        echo "Error evaluating $subfolder_name"
    fi
done