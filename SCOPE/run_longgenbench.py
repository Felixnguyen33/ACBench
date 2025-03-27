import os
import json
import random
import argparse

import numpy as np
from tqdm import tqdm

import torch
from vllm import LLM, SamplingParams

datasets = ["gsm8k", "mmlu"]
# ["gsm8k", "mmlu"]
# , "csqa", "mmlu"]

MODEL_PATHS = {
    "Qwen2.5-7B-Instruct-AWQ": "/data2/share/Qwen2.5/Qwen2.5-7B-Instruct-AWQ",
    "Qwen2.5-7B-Instruct": "/data2/share/Qwen2.5/Qwen2.5-7B-Instruct",
    "Qwen2.5-7B-Instruct-Mag-Un-0.5": "/mnt/sdd/dongpeijie/out_pruned_llm/qwen_7b/unstructured/magnitude/qwen-2.5-7b-chat-magnitude-un0.5",
    "Qwen2.5-7B-Instruct-Mag-2-4-0.5": "/mnt/sdd/dongpeijie/out_pruned_llm/qwen_7b/2-4/magnitude/qwen-2.5-7b-chat-magnitude-2-4-0.5",
    "Qwen2.5-7B-Instruct-Wanda-Un-0.5": "/mnt/sdd/dongpeijie/out_pruned_llm/qwen_7b/unstructured/wanda/qwen-2.5-7b-chat-wanda-un0.5",
    "Qwen2.5-7B-Instruct-Wanda-2-4-0.5": "/mnt/sdd/dongpeijie/out_pruned_llm/qwen_7b/2-4/wanda/qwen-2.5-7b-chat-wanda-2-4-0.5",
    "Qwen2.5-7B-Instruct-SparseGPT-Un-0.5": "/mnt/sdd/dongpeijie/out_pruned_llm/qwen_7b/unstructured/sparsegpt/qwen-2.5-7b-chat-sparsegpt-un0.5",
    "Qwen2.5-7B-Instruct-SparseGPT-2-4-0.5": "/mnt/sdd/dongpeijie/out_pruned_llm/qwen_7b/2-4/sparsegpt/qwen-2.5-7b-chat-sparsegpt-2-4-0.5",
    "InternLM2.5-7B-Instruct": "/data2/share/internlm/internlm2_5-7b-chat",
    "InternLM2.5-7B-Instruct-Mag-Un-0.5": "/mnt/sdd/dongpeijie/out_pruned_llm/internlm_7b/unstructured/magnitude/internlm-2.5-7b-chat-magnitude-un0.5",
    "InternLM2.5-7B-Instruct-Mag-2-4-0.5": "/mnt/sdd/dongpeijie/out_pruned_llm/internlm_7b/2-4/magnitude/internlm-2.5-7b-chat-magnitude-2-4-0.5",
    "InternLM2.5-7B-Instruct-Wanda-Un-0.5": "/mnt/sdd/dongpeijie/out_pruned_llm/internlm_7b/unstructured/wanda/internlm-2.5-7b-chat-wanda-un0.5",
    "InternLM2.5-7B-Instruct-Wanda-2-4-0.5": "/mnt/sdd/dongpeijie/out_pruned_llm/internlm_7b/2-4/wanda/internlm-2.5-7b-chat-wanda-2-4-0.5",
    "InternLM2.5-7B-Instruct-SparseGPT-Un-0.5": "/mnt/sdd/dongpeijie/out_pruned_llm/internlm_7b/unstructured/sparsegpt/internlm-2.5-7b-chat-sparsegpt-un0.5",
    "InternLM2.5-7B-Instruct-SparseGPT-2-4-0.5": "/mnt/sdd/dongpeijie/out_pruned_llm/internlm_7b/2-4/sparsegpt/internlm-2.5-7b-chat-sparsegpt-2-4-0.5",
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

dataset2maxlen_8K = {
    "gsm8k": 7950,
    "mmlu": 7950,
    "csqa": 7950,
}

dataset2maxlen_4K = {
    "gsm8k": 4096,
    "mmlu": 4096,
    "csqa": 4096,
}

model2prompt = {
    "gsm8k": "Answer each question step by step, adhering to the format shown in the examples provided. Start each response with 'Answer_' and introduce the final response with 'The answer is'. Do not repeat the question. Ensure that you respond to all the questions presented, regardless of their number.",
    "mmlu": "Answer each question step by step, adhering to the format shown in the examples provided. Start each response with 'Answer_' and introduce the final response with 'The answer is'. Do not repeat the question. Ensure that you respond to all the questions presented, regardless of their number. The following are multiple choice questions (with answers) about ",
    "csqa": "Answer each question step by step, adhering to the format shown in the examples provided. Start each response with 'Answer_' and introduce the final response with 'The answer is'. Do not repeat the question. Ensure that you respond to all the questions presented, regardless of their number.",
}

model2maxlen = {
    "llama2": 3950,
    "llama-2": 3950,
    "llama3-": 7950,
    "llama-3-": 7950,
    "llama3.1": 130000,
    "llama-3.1": 130000,
    "llama3.2": 130000,
    "llama-3.2": 130000,
    "mistral": 31500,
    "qwen": 32768,
    "internlm": 32768
}

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def build_chat_llama2(system_prompt, prompt):
    return f"[INST] <<SYS>>\n {system_prompt} \n<</SYS>>\n\n{prompt} [/INST]"

def build_chat_llama3_modify(system_prompt, prompt):
    return f"<<SYS>>\n {system_prompt} \n<</SYS>>\n\n{prompt}"

def build_chat_llama3(system_prompt, prompt):
    return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

def build_chat_llama3_wo_system(system_prompt, prompt):
    return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{system_prompt}\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

def build_chat_qwen(system_prompt, prompt):
    return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

def build_chat_internlm(system_prompt, prompt):
    return f"<|System|>: {system_prompt}\n<|User|>: {prompt}\n<|Assistant|>: "

def build_chat_megrez(system_prompt, prompt):
    return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

def build_chat_gemma(system_prompt, prompt):
    return f"<start_of_turn>system\n{system_prompt}<end_of_turn>\n<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model"

def build_chat_phi(system_prompt, prompt):
    return f"Instruct: {system_prompt}\nInput: {prompt}\nOutput: "

def build_chat_minicpm(system_prompt, prompt):
    return f"<|system|>{system_prompt}</s><|user|>{prompt}</s><|assistant|>"

def main(args):
    print("Loading data...")
    test_data = []
    prompts = []
    questionss = []
    answerss = []
    lengths = []
    
    model_path = MODEL_PATHS.get(args.model_name, args.model_name)
    model_path_lower = model_path.lower()

    for key in model2maxlen:
        if key in model_path_lower:
            model_max_len = model2maxlen[key]
    if args.K == 30:
        output_max_len = dataset2maxlen_4K[args.dataset]
    else:
        output_max_len = dataset2maxlen_8K[args.dataset]
    
    with open(args.data_file) as fp:
        for line in fp:
            example = json.loads(line)
            
            template = model2prompt[args.dataset]
            system_prompt = template.format(**example)
            
            # mmlu
            if "task" in example:
                template = template + example["task"] + "."
            
            if "llama2" in args.model_name.lower():
                prompt = build_chat_llama2(system_prompt, example["prompt"])
            elif "llama-3" in args.model_name.lower() and (
                args.dataset not in [""]
            ):
                print(f"using template for {args.dataset}")
                prompt = build_chat_llama3_modify(system_prompt, example["prompt"])
            elif "qwen" in args.model_name.lower():
                prompt = build_chat_qwen(system_prompt, example["prompt"])
            elif "internlm" in args.model_name.lower():
                prompt = build_chat_internlm(system_prompt, example["prompt"])
            elif "megrez" in args.model_name.lower():
                prompt = build_chat_megrez(system_prompt, example["prompt"])
            elif "gemma" in args.model_name.lower():
                prompt = build_chat_gemma(system_prompt, example["prompt"])
            elif "phi" in args.model_name.lower():
                prompt = build_chat_phi(system_prompt, example["prompt"])
            elif "minicpm" in args.model_name.lower():
                prompt = build_chat_minicpm(system_prompt, example["prompt"])
            else:
                print(f"NOT using template for {args.dataset}")
                prompt = system_prompt + "\n\n" + example["prompt"]

            example["prompt"] = prompt
                
            test_data.append(example)

    if args.max_num_examples and len(test_data) > args.max_num_examples:
        if args.sample_method == "random":
            test_data = random.sample(test_data, args.max_num_examples)
        elif args.sample_method == "topk":
            test_data = test_data[:args.max_num_examples]
    
    for example in test_data:
        prompts.append(example["prompt"])
        questionss.append(example["questions"])
        answerss.append(example["answers"])
        lengths.append(len(example["prompt"]))

    print("Loading model...")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        trust_remote_code=True
    )
    
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=output_max_len,
        stop_token_ids=[llm.get_tokenizer().eos_token_id],
    )
    
    os.makedirs(os.path.join(args.save_dir, f"{args.model_name}", args.dataset), exist_ok=True)

    fout = open(os.path.join(args.save_dir, f"{args.model_name}", args.dataset, "results.json"), "w")
     
    for i in tqdm(range(0, len(prompts), args.eval_batch_size)):
        batch_prompts = prompts[i:i+args.eval_batch_size]
        batch_questionss = questionss[i:i+args.eval_batch_size]
        batch_answerss = answerss[i:i+args.eval_batch_size]
        batch_lengths = lengths[i:i+args.eval_batch_size]

        outputs = llm.generate(batch_prompts, sampling_params)
        batch_generations = [output.outputs[0].text for output in outputs]

        for j in range(len(batch_prompts)):
            example = {}
            example["prompt"] = batch_prompts[j]
            example["questions"] = batch_questionss[j]
            example["answers"] = batch_answerss[j]
            example["length"] = batch_lengths[j]
            example["pred"] = batch_generations[j]

            fout.write(json.dumps(example) + "\n")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--seed", type=int, default=42, help="")
    parser.add_argument("--base_dir", type=str, default="")
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="")    
    parser.add_argument("--model_name", type=str, required=True, help="name of the model to use (key in MODEL_PATHS)")
    parser.add_argument("--max_num_examples", type=int, default=None, help="maximum number of examples to evaluate per task.")
    parser.add_argument("--sample_method", type=str, default="topk", choices=["random", "topk"], help="how to sample the examples.")
    parser.add_argument("--eval_batch_size", type=int, default=1, help="batch size for evaluation.")
    parser.add_argument("--K", type=int, default=-1, help="number of questions parallel")
    
    args = parser.parse_args()
    
    set_seed(args.seed)

    for idx, dataset in enumerate(datasets):
        print(f"Working on dataset {dataset} - {idx}/{len(datasets)}")
        
        args.dataset = dataset
        if args.dataset == "csqa":
            args.K = int(args.K / 3 * 4) # GSM8K/MMLU has 30,60 questions in a single long input;CSQA has 40,80 questions
        elif args.dataset == 'mmlu':
            args.K = 20 
            args.data_file = f"longgenbench_examples/{args.dataset}_{args.K}.jsonl"
        elif args.dataset == 'gsm8k':
            args.K = 30 
            args.data_file = f"longgenbench_examples/{args.dataset}_{args.K}.jsonl"    
        
        main(args)
