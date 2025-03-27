import os
import pandas as pd

# Define fixed tasks and columns 
TASKS = ['Instruct', 'Plan', 'Reason', 'Retrieve', 'Understand', 'Review']

# Define column mapping for each task
COLUMN_MAPPING = {
    'Instruct': ['Instruct_String', 'Instruct_JSON'],
    'Plan': ['Plan_String', 'Plan_JSON'],
    'Reason': ['Reason_String', 'Reason_JSON'],
    'Retrieve': ['Retrieve_String', 'Retrieve_JSON'],
    'Understand': ['Understand_String', 'Understand_JSON'],
    'Review': ['Review_String', 'Review_JSON']
}

root_dir = "saved_csv"

# Get all CSV files in directory
csv_files = [f for f in os.listdir(root_dir) if f.endswith('.csv')]

# Initialize empty list to store all results
all_results = []

# Read each CSV file
for csv_file in csv_files:
    model_name = os.path.splitext(csv_file)[0]  # Remove .csv extension
    file_path = os.path.join(root_dir, csv_file)
    
    try:
        # Read CSV and add model name
        df = pd.read_csv(file_path)
        df['Model'] = model_name
        
        # Create a single row dictionary for all tasks
        row_data = {'Model': model_name}
        for task in TASKS:
            task_data = df[df['Task'] == task].iloc[0] if not df[df['Task'] == task].empty else None
            if task_data is not None:
                row_data[f'{task}_String'] = task_data['String']
                row_data[f'{task}_JSON'] = task_data['JSON']
            else:
                row_data[f'{task}_String'] = None
                row_data[f'{task}_JSON'] = None
        
        # Add the combined row to results
        all_results.append(pd.DataFrame([row_data]))
            
    except Exception as e:
        print(f"Error processing {csv_file}: {str(e)}")
        continue

# Combine all results
if all_results:
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Define column order
    columns = []
    for task in TASKS:
        columns.extend([f'{task}_String', f'{task}_JSON'])
    columns.append('Model')
    
    # Reorder columns
    combined_df = combined_df[columns]
    
    # Save combined results
    combined_df.to_csv('combined_results.csv', index=False)
    print(f"Successfully combined {len(all_results)} CSV files")
else:
    print("No valid CSV files found to combine")
