#!/bin/bash

# Traverse through all subdirectories in wandb folder and sync each one
for dir in wandb/*/; do
    if [ -d "$dir" ]; then
        echo "Syncing $dir..."
        wandb sync "$dir"
    fi
done

