import os
import json
import pandas as pd

root_path = "./data/eval_result/"

results = []
for root, dirs, files in os.walk(root_path):
    for file in files:
        if file.endswith('.json'):
            file_path = os.path.join(root, file)
            with open(file_path, 'r') as f:
                data = json.load(f)
                model_name = os.path.basename(os.path.dirname(file_path))
                dataset = file.split('_graph_eval')[0].replace(model_name + '_', '')
                results.append({
                    'model': model_name,
                    'dataset': dataset,
                    'precision': data['precision'] * 100,
                    'recall': data['recall'] * 100, 
                    'f1_score': data['f1_score'] * 100
                })

df = pd.DataFrame(results)
df.to_csv('results.csv', index=False)
print(df)