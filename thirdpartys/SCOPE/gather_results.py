import os
import csv
import json
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('gather_results.log')
    ]
)

ROOT_DIR = "./outputs"

def gather_results():
    logging.info("Starting to gather results")
    all_results = {}
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(ROOT_DIR):
        # Look for results.csv in each directory
        if 'results.csv' in files:
            csv_path = os.path.join(root, 'results.csv')
            folder_name = os.path.relpath(root, ROOT_DIR)
            logging.info(f"Processing {csv_path}")
            
            # Read the CSV file
            results = {}
            try:
                with open(csv_path, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        model_name = row.pop('dataset')
                        results[model_name] = {
                            'gsm8k': float(row.get('gsm8k', -1)),
                            'mmlu': float(row.get('mmlu', -1))
                        }
                logging.info(f"Successfully processed {folder_name}")
            except Exception as e:
                logging.error(f"Error processing {csv_path}: {e}")
                continue
            
            all_results[folder_name] = results
    
    # Save all results to a CSV file
    try:
        with open('all_results.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            # Write header
            writer.writerow(['folder', 'model', 'gsm8k', 'mmlu'])
            # Write data
            for folder_name, models in all_results.items():
                for model_name, scores in models.items():
                    writer.writerow([
                        folder_name,
                        model_name,
                        scores['gsm8k'],
                        scores['mmlu']
                    ])
        logging.info("Successfully saved results to all_results.csv")
    except Exception as e:
        logging.error(f"Error saving results to CSV: {e}")

if __name__ == '__main__':
    gather_results()
