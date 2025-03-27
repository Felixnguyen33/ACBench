import pandas as pd

# Read data from CSV file
df = pd.read_csv('combined_results.csv')

# Separate tasks and overall scores
tasks_df = df[df['Task'] != 'Overall'].copy()
overall_df = df[df['Task'] == 'Overall'][['Model', 'Overall']].rename(columns={'Overall': 'Overall'})

# Pivot tasks data
tasks_pivoted = tasks_df.pivot(index='Model', columns='Task', values=['String', 'JSON'])

# Rename columns to match task names with sub-columns
tasks_pivoted.columns = [f"{task}_{col}" for task, col in tasks_pivoted.columns]

# For Review, use String as Choice (since String and JSON are identical)
tasks_pivoted['Review_Choice'] = tasks_pivoted['Review_String']

# Merge with overall scores
final_df = pd.merge(tasks_pivoted, overall_df, on='Model')

# Reorder columns to match the desired structure
tasks = ['Instruct', 'Plan', 'Reason', 'Retrieve', 'Understand', 'Review']
columns = []
for task in tasks:
    if task != 'Review':
        columns.append(f"{task}_String")
        columns.append(f"{task}_JSON")
    else:
        columns.append('Review_Choice')
columns.append('Overall')

final_df = final_df[columns]

# Rename columns to match the second table's headers
final_df.columns = [col.replace('_', ' ').title() for col in final_df.columns]
final_df.columns = [col.replace(' Json', ' JSON') for col in final_df.columns]
final_df.columns = [col.replace(' String', ' String') for col in final_df.columns]
final_df.columns = [col.replace(' Choice', ' Choice') for col in final_df.columns]

# Create MultiIndex for column headers
tasks = ['INSTRUCT', 'PLAN', 'REASON', 'RETRIEVE', 'UNDERSTAND', 'REVIEW']
sub_columns = ['STRING', 'JSON']
review_sub_col = ['CHOICE']

columns = []
for task in tasks:
    if task != 'REVIEW':
        columns.append((task, 'STRING'))
        columns.append((task, 'JSON'))
    else:
        columns.append((task, 'CHOICE'))
columns.append(('OVERALL', ''))

final_df.columns = pd.MultiIndex.from_tuples(columns)

# Display the final DataFrame
print(final_df)