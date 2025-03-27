import os
import json
import pandas as pd

def gather_results():
    # Initialize empty list to store all results
    all_results = []
    
    # Walk through results directory
    results_dir = "./results"
    for model_dir in os.listdir(results_dir):
        model_path = os.path.join(results_dir, model_dir)
        
        # Skip if not a directory
        if not os.path.isdir(model_path):
            continue
            
        results_file = os.path.join(model_path, "all_results.txt")
        
        # Skip if results file doesn't exist
        if not os.path.exists(results_file):
            continue
            
        # Read and parse each line
        with open(results_file, "r") as f:
            model_result = {}
            model_result["model"] = model_dir
            
            for line in f:
                try:
                    result = json.loads(line.strip())
                    task = result["task_name"]
                    model_result[f"{task}_progress_rate"] = result["progress_rate"]
                    model_result[f"{task}_success_rate"] = result["success_rate"]
                except:
                    continue
                    
            all_results.append(model_result)

    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    
    # Save to CSV
    df.to_csv("all_model_results_agentboard.csv", index=False)

if __name__ == "__main__":
    gather_results()
