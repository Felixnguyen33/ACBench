import logging
import os
import json
import argparse
import numpy as np
import re
import csv 

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('eval_longgenbench.log')
    ]
)

MODEL_PATHS = {
    "Qwen2.5-7B-Instruct-AWQ": "/data2/share/Qwen2.5/Qwen2.5-7B-Instruct-AWQ",
    "Qwen2.5-7B-Instruct": "/data2/share/Qwen2.5/Qwen2.5-7B-Instruct",
    "Qwen2.5-7B-Instruct-Mag-Un-0.5": "/path/to/out_pruned_llm/qwen_7b/unstructured/magnitude/qwen-2.5-7b-chat-magnitude-un0.5",
    "Qwen2.5-7B-Instruct-Mag-2-4-0.5": "/path/to/out_pruned_llm/qwen_7b/2-4/magnitude/qwen-2.5-7b-chat-magnitude-2-4-0.5",
    "Qwen2.5-7B-Instruct-Wanda-Un-0.5": "/path/to/out_pruned_llm/qwen_7b/unstructured/wanda/qwen-2.5-7b-chat-wanda-un0.5",
    "Qwen2.5-7B-Instruct-Wanda-2-4-0.5": "/path/to/out_pruned_llm/qwen_7b/2-4/wanda/qwen-2.5-7b-chat-wanda-2-4-0.5",
    "Qwen2.5-7B-Instruct-SparseGPT-Un-0.5": "/path/to/out_pruned_llm/qwen_7b/unstructured/sparsegpt/qwen-2.5-7b-chat-sparsegpt-un0.5",
    "Qwen2.5-7B-Instruct-SparseGPT-2-4-0.5": "/path/to/out_pruned_llm/qwen_7b/2-4/sparsegpt/qwen-2.5-7b-chat-sparsegpt-2-4-0.5",
    "InternLM2.5-7B-Instruct": "/data2/share/internlm/internlm2_5-7b-chat",
    "InternLM2.5-7B-Instruct-Mag-Un-0.5": "/path/to/out_pruned_llm/internlm_7b/unstructured/magnitude/internlm-2.5-7b-chat-magnitude-un0.5",
    "InternLM2.5-7B-Instruct-Mag-2-4-0.5": "/path/to/out_pruned_llm/internlm_7b/2-4/magnitude/internlm-2.5-7b-chat-magnitude-2-4-0.5",
    "InternLM2.5-7B-Instruct-Wanda-Un-0.5": "/path/to/out_pruned_llm/internlm_7b/unstructured/wanda/internlm-2.5-7b-chat-wanda-un0.5",
    "InternLM2.5-7B-Instruct-Wanda-2-4-0.5": "/path/to/out_pruned_llm/internlm_7b/2-4/wanda/internlm-2.5-7b-chat-wanda-2-4-0.5",
    "InternLM2.5-7B-Instruct-SparseGPT-Un-0.5": "/path/to/out_pruned_llm/internlm_7b/unstructured/sparsegpt/internlm-2.5-7b-chat-sparsegpt-un0.5",
    "InternLM2.5-7B-Instruct-SparseGPT-2-4-0.5": "/path/to/out_pruned_llm/internlm_7b/2-4/sparsegpt/internlm-2.5-7b-chat-sparsegpt-2-4-0.5",
    "InternLM2.5-7B-Instruct-AWQ": "/data2/share/internlm/internlm2_5-7b-chat-AWQ-W4-G128",
    "InternLM2.5-7B-Instruct-GPTQ-w4a16": "/data2/share/internlm/internlm2_5-7b-chat-GPTQ-w4a16",
    "InternLM2.5-7B-Instruct-RTN-w4": "/data2/share/internlm/internlm2_5-7b-chat-RTN-w4",
    "Qwen2.5-7B-Instruct-RTN-w4": "/data2/share/Qwen2.5/Qwen2.5-7B-Instruct-RTN-w4",
    "Qwen2.5-7B-Instruct-GPTQ-w4a16": "/data2/share/Qwen2.5/Qwen2.5-7B-Instruct-GPTQ-w4a16",
    "deepseek-qwen-1.5b": "/data2/share/deepseek/DeepSeek-R1-Distill-Qwen-1.5B",
    "deepseek-qwen-7b": "/data2/share/deepseek/DeepSeek-R1-Distill-Qwen-7B", 
    "deepseek-llama-8b": "/data2/share/deepseek/DeepSeek-R1-Distill-Llama-8B",
    "minicpm-4b": "/data2/share/openbmb/MiniCPM3-4B",
    "megrez-3b": "/data2/share/megrez/Megrez-3B-Instruct",
    "qwen-3b-gptq-int4": "/data2/share/Qwen2.5/Qwen2.5-3B-Instruct-GPTQ-Int4",
    "qwen-3b-gptq-int8": "/data2/share/Qwen2.5/Qwen2.5-3B-Instruct-GPTQ-Int8", 
    "qwen-3b-awq": "/data2/share/Qwen2.5/Qwen2.5-3B-Instruct-AWQ",
    "qwen-1.5b-gptq-int4": "/data2/share/Qwen2.5/Qwen2.5-1.5B-Instruct-GPTQ-Int4",
    "qwen-1.5b-gptq-int8": "/data2/share/Qwen2.5/Qwen2.5-1.5B-Instruct-GPTQ-Int8",
    "qwen-1.5b-awq": "/data2/share/Qwen2.5/Qwen2.5-1.5B-Instruct-AWQ",
    "gemma-2b": "/data2/share/gemma/gemma-2-2b-it",
    "phi-3.5": "/data2/share/phi/Phi-3.5-mini-instruct"
}

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default=None)
    parser.add_argument('--model_names', nargs='+', choices=list(MODEL_PATHS.keys()), 
                      help='List of model names to evaluate')
    return parser.parse_args(args)

def extract_final_answer(answer):
    """Extracts the final answer from the answer string."""
    match = re.search(r'####\s*(\d+)', answer)
    return match.group(1) if match else None

def extract_predicted_answers(pred):
    """Extracts the predicted answers from the pred string."""
    matches = re.findall(r'Answer_\d+:\s*.*?answer is (\d+)', pred, re.DOTALL)
    return matches

def compare_answers(pred, answers):
    # Extract final answers from the answers list
    expected_answers = [extract_final_answer(ans) for ans in answers]
    
    # Extract predicted answers from the pred string
    predicted_answers = extract_predicted_answers(pred)

    # print(f'Expected: {expected_answers}')
    # print(f'Predicted: {predicted_answers}')
    
    # Compare the two lists and calculate accuracy
    results = {}
    correct_count = 0
    for i, (expected, predicted) in enumerate(zip(expected_answers, predicted_answers)):
        is_correct = expected == predicted
        results[f'Answer_{i+9}'] = {
            'expected': expected,
            'predicted': predicted,
            'correct': is_correct
        }
        if is_correct:
            correct_count += 1

    # Calculate accuracy
    total_questions = len(expected_answers)
    accuracy = correct_count / total_questions if total_questions > 0 else 0.0
    # print(f'Accuracy: {accuracy:.4f}')
    return accuracy


def extract_predicted_choices(pred):
    """Extracts the predicted answers (A, B, C, D, E, etc.) from the pred string, considering Answer_... patterns."""
    matches = re.findall(r'Answer_\d+:\s*.*?answer is \((.*?)\)', pred)
    return matches


def compare_choices(pred, answers):
    # Extract final answers from the answers list
    expected_answers = answers
    
    # Extract predicted answers from the pred string
    predicted_answers = extract_predicted_choices(pred)

    # print(f'Expected: {expected_answers}')
    # print(f'Predicted: {predicted_answers}')
    
    # Compare the two lists and calculate accuracy
    results = {}
    correct_count = 0
    for i, (expected, predicted) in enumerate(zip(expected_answers, predicted_answers)):
        is_correct = expected == predicted
        results[f'Answer_{i+6}'] = {
            'expected': expected,
            'predicted': predicted,
            'correct': is_correct
        }
        if is_correct:
            correct_count += 1

    # Calculate accuracy
    total_questions = len(expected_answers)
    accuracy = correct_count / total_questions if total_questions > 0 else 0.0
    # print(f'Accuracy: {accuracy:.4f}')
    return accuracy

def scorer(dataset, predictions, answers):
    scores = []
    for (prediction, ground_truths) in zip(predictions, answers):
        if dataset in ["gsm8k"]:
            scores.append(compare_answers(prediction, ground_truths))
        elif dataset in ["mmlu","csqa"]:
            scores.append(compare_choices(prediction, ground_truths))

    return round(100 * np.mean(scores), 4)

if __name__ == '__main__':
    args = parse_args()
    
    dataset_list = [
        "gsm8k",
        "mmlu",
        # "csqa",
    ]
    
    if not args.model_names:
        args.model_names = list(MODEL_PATHS.keys())
    
    # Initialize results list with method names
    results_list = [
        ["dataset"] + dataset_list,  # Header row with all datasets
    ]
    
    # Add a row for each model with initial -1 scores
    for model_name in args.model_names:
        results_list.append([model_name] + [-1] * len(dataset_list))
    
    logging.info(f"Initialized results_list: {results_list}")
    
    for dataset_idx, dataset in enumerate(dataset_list):
        logging.info(f"Processing dataset: {dataset}")
        
        for model_idx, model_name in enumerate(args.model_names):
            try:
                logging.info(f"Processing model: {model_name}")
                model_path = MODEL_PATHS[model_name]
                args.model_path = model_path
                args.dataset = dataset
                args.eval_file = os.path.join(args.results_dir, model_name, dataset, "results.json")
                
                logging.info(f"Reading evaluation file: {args.eval_file}")
                
                predictions, answers, lengths = [], [], []
                
                with open(args.eval_file, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            predictions.append(data["pred"])
                            answers.append(data["answers"])
                            if "length" in data:
                                lengths.append(data["length"])
                        except Exception as e:
                            logging.error(f"Error processing line: {e}")

                score = scorer(args.dataset, predictions, answers)
                avg_length = round(np.mean(lengths), 2) if lengths else 0
                
                logging.info(f"Score: {score}, Average length: {avg_length}")

                # Update the score in results_list (dataset_idx + 1 because first column is model name)
                results_list[model_idx + 1][dataset_idx + 1] = score

                output_dir = os.path.dirname(args.eval_file)
                os.makedirs(output_dir, exist_ok=True)
                with open(os.path.join(output_dir, "metrics.json"), "w") as f:
                    json.dump({args.dataset: score}, f, ensure_ascii=False, indent=4)
                
                logging.info(f"Dataset {args.dataset} model {model_name} scores {score}")
            
            except Exception as e:
                logging.error(f"Error processing dataset {dataset} with model {model_name}: {e}")
                logging.info(f"Dataset {args.dataset} model {model_name} scores None")
    
    logging.info(f"Final results_list: {results_list}")
    
    # Write results to CSV
    try:
        csv_path = os.path.join(args.results_dir, "results.csv")
        with open(csv_path, 'w') as fp:
            writer = csv.writer(fp)
            # Filter out rows where all values except first column are -1
            filtered_results = [row for row in results_list if not all(val == -1 for val in row[1:])]
            writer.writerows(filtered_results)
        logging.info(f"Successfully wrote filtered results to {csv_path}")
    except Exception as e:
        logging.error(f"Error writing CSV file: {e}")
