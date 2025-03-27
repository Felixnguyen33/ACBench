#!/bin/bash

# Find all subdirectories in outputs
for dir in ./outputs/*/; do
    if [ -d "$dir" ]; then
        # Get full path
        full_path=$(realpath "$dir")
        echo "Evaluating results in: $full_path"
        
        # Run evaluation script on each directory
        python3 eval_longgenbench.py --results_dir "$full_path"
    fi
done
