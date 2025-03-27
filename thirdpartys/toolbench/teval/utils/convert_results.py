import mmengine
import os
import argparse
import numpy as np
from tabulate import tabulate
import pandas as pd

np.set_printoptions(precision=1)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path', type=str)
    parser.add_argument('--save_path', type=str)
    args = parser.parse_args()
    return args

def convert_results(result_path, save_path):
    result = mmengine.load(result_path)
    
    instruct_list = [
        (result['instruct_json']['json_format_metric'] + result['instruct_json']['json_args_em_metric']) / 2,
        (result['instruct_json']['string_format_metric'] + result['instruct_json']['string_args_em_metric']) / 2
    ]
    plan_list = [result['plan_str']['f1_score'], result['plan_json']['f1_score']]
    reason_list = [result['reason_str']['thought'], result['rru_json']['thought']]
    retrieve_list = [result['retrieve_str']['name'], result['rru_json']['name']]
    understand_list = [result['understand_str']['args'], result['rru_json']['args']]
    review_list = [result['review_str']['review_quality'], result['review_str']['review_quality']]

    final_score = [
        np.mean(instruct_list), 
        np.mean(plan_list), 
        np.mean(reason_list), 
        np.mean(retrieve_list), 
        np.mean(understand_list), 
        np.mean(review_list)
    ]
    overall = np.mean(final_score)
    final_score.insert(0, overall)
    name_list = ['Overall', 'Instruct', 'Plan', 'Reason', 'Retrieve', 'Understand', 'Review']
    
    # Create a table for the results
    table_data = [
        ['Task', 'String', 'JSON', 'Overall'],
        [name_list[1], f"{instruct_list[0]*100:.1f}", f"{instruct_list[1]*100:.1f}", f"{final_score[1]*100:.1f}"],
        [name_list[2], f"{plan_list[0]*100:.1f}", f"{plan_list[1]*100:.1f}", f"{final_score[2]*100:.1f}"],
        [name_list[3], f"{reason_list[0]*100:.1f}", f"{reason_list[1]*100:.1f}", f"{final_score[3]*100:.1f}"],
        [name_list[4], f"{retrieve_list[0]*100:.1f}", f"{retrieve_list[1]*100:.1f}", f"{final_score[4]*100:.1f}"],
        [name_list[5], f"{understand_list[0]*100:.1f}", f"{understand_list[1]*100:.1f}", f"{final_score[5]*100:.1f}"],
        [name_list[6], f"{review_list[0]*100:.1f}", f"{review_list[1]*100:.1f}", f"{final_score[6]*100:.1f}"],
        [name_list[0], '-', '-', f"{final_score[0]*100:.1f}"]
    ]
    
    print(tabulate(table_data, headers="firstrow", tablefmt="grid"))

    # Save as CSV
    if save_path:
        df = pd.DataFrame(table_data[1:], columns=table_data[0])
        df.to_csv(save_path, index=False)

if __name__ == '__main__':
    args = parse_args()
    convert_results(args.result_path, args.save_path)