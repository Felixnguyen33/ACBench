import pandas as pd
import numpy as np

# Read the CSV file
df = pd.read_csv('tmp.csv')

def process_model_metrics():
    # Get all column names except 'eval_result'
    cols = [col for col in df.columns if col != 'eval_result']
    
    # Create a dictionary to store results
    results = {}
    
    for col in cols:
        # Split column name to get model name and metric type
        parts = col.split('_')
        
        # Skip columns that don't have at least 3 parts (model_dataset_metric)
        if len(parts) < 3:
            print(f"Warning: Skipping column '{col}' - invalid format")
            continue
            
        model = '_'.join(parts[:-2])  # Everything except last two parts
        dataset = parts[-2]
        metric = parts[-1]
        
        if model not in results:
            results[model] = {}
            
        if dataset not in results[model]:
            results[model][dataset] = {}
            
        # Store the metric value
        results[model][dataset][metric] = df[col].iloc[0]
    
    # Create a summary DataFrame
    rows = []
    for model in results:
        for dataset in results[model]:
            row = {
                'Model': model,
                'Dataset': dataset,
                'Precision': results[model][dataset].get('precision', np.nan),
                'Recall': results[model][dataset].get('recall', np.nan),
                'F1 Score': results[model][dataset].get('f1_score', np.nan)
            }
            rows.append(row)
    
    summary_df = pd.DataFrame(rows)
    
    # Calculate average metrics per model
    model_avg = summary_df.groupby('Model').mean()
    
    # Sort models by average F1 score
    model_avg = model_avg.sort_values('F1 Score', ascending=False)
    
    # Print results
    print("\n=== Model Performance Summary ===")
    print("\nTop 5 Models by Average F1 Score:")
    print(model_avg[['Precision', 'Recall', 'F1 Score']].head().round(3))
    
    # Save detailed results to CSV
    summary_df.to_csv('model_performance_summary.csv', index=False)
    print("\nDetailed results saved to 'model_performance_summary.csv'")

if __name__ == "__main__":
    process_model_metrics()
