import pandas as pd
import numpy as np
import os
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gather_results_v2.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def extract_dataset_name(filename):
    datasets = ['alfworld', 'lumos', 'os', 'toolalpaca', 'toolbench', 'webshop']
    for dataset in datasets:
        if dataset in filename.lower():
            return dataset
    return None

def gather_results(root_dir='./data/eval_result'):
    logger.info(f"Starting to gather results from {root_dir}")
    
    # Dictionary to store all results
    results = {
        'model': []
    }
    
    # Initialize columns for each dataset-metric combination
    datasets = ['alfworld', 'lumos', 'os', 'toolalpaca', 'toolbench', 'webshop']
    metrics = ['precision', 'recall', 'f1_score']
    for dataset in datasets:
        for metric in metrics:
            results[f'{dataset}_{metric}'] = []
    
    # Walk through all subdirectories
    for model_dir in os.listdir(root_dir):
        model_path = os.path.join(root_dir, model_dir)
        
        if not os.path.isdir(model_path):
            continue
            
        logger.info(f"Processing model directory: {model_dir}")
        
        # Initialize temporary storage for this model's results
        model_results = {'model': model_dir}
        for dataset in datasets:
            for metric in metrics:
                model_results[f'{dataset}_{metric}'] = np.nan
        
        # Process all JSON files in the model directory
        for json_file in os.listdir(model_path):
            if not json_file.endswith('.json'):
                continue
                
            dataset_name = extract_dataset_name(json_file)
            if not dataset_name:
                continue
                
            file_path = os.path.join(model_path, json_file)
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                    # Add results to temporary storage
                    for metric in metrics:
                        column_name = f'{dataset_name}_{metric}'
                        model_results[column_name] = data.get(metric, np.nan)
                    
                logger.info(f"Successfully processed {file_path}")
                    
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
        
        # Add this model's results to the main results dictionary
        for key, value in model_results.items():
            results[key].append(value)
                
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    output_file = 'results_summary.csv'
    df.to_csv(output_file, index=False)
    logger.info(f"Results saved to {output_file}")
    
    return df

if __name__ == "__main__":
    results_df = gather_results()
    print("\nResults Summary:")
    print(results_df)
