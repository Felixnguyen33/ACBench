import os
import json

root_dir = "/data2/share/peijiedong/AgentBench/thirdparty/LongBenchv1/pred"
print(f"Looking for results in directory: {root_dir}")

all_results = {}
for folder in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, folder)
    if os.path.isdir(folder_path):
        print(f"Processing folder: {folder}")
        result_path = os.path.join(folder_path, "result.json")
        if os.path.exists(result_path):
            print(f"Found result.json in {folder}")
            with open(result_path, 'r') as f:
                result = json.load(f)
                all_results[folder] = result
                print(f"Successfully loaded results for {folder}")
        else:
            print(f"No result.json found in {folder}")

output_path = os.path.join(root_dir, "gathered_results.json")
print(f"Writing gathered results to: {output_path}")
with open(output_path, 'w') as f:
    json.dump(all_results, f, indent=4)
print(f"Successfully gathered results from {len(all_results)} models")
