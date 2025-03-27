# Create directories if they don't exist
save_dir="./saved_csv/"
out_dir="./work_dirs/"
mkdir -p $save_dir
echo "Created directories: save_dir=$save_dir, out_dir=$out_dir"

# Traverse work_dirs and process each model's results
echo "Starting to traverse work_dirs directory..."
for model_dir in $out_dir/*/; do
    if [ -d "$model_dir" ]; then
        # Get model name from directory path
        model_name=$(basename $model_dir)
        echo "Found model directory: $model_dir"
        
        # Check if results file exists
        result_file="$model_dir/${model_name}_-1.json"
        if [ -f "$result_file" ]; then
            echo "Processing results for model: $model_name"
            echo "Result file path: $result_file"
            echo "Will save to: $save_dir/${model_name}.csv"
            
            # Run conversion script
            python teval/utils/convert_results.py \
                --result_path "$result_file" \
                --save_path "$save_dir/${model_name}.csv"
            
            echo "Finished processing $model_name"
        else
            echo "No results file found at: $result_file"
        fi
    fi
done
echo "Finished processing all models"