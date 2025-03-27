# -*- coding: utf-8 -*-
import torch
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
from awq import AutoAWQForCausalLM
import os

# Create save directory if it doesn't exist
os.makedirs('./save_imgs', exist_ok=True)

def load_wikitext_samples(num_samples=100, max_length=2048):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    samples = []
    for text in dataset["text"][:num_samples]:
        if len(text) > 10:  # 过滤空文本
            samples.append(text[:max_length])  # 截断至模型最大长度
    return samples

# ====================== 配置参数 ======================
MODEL_CONFIGS = {
    "model_sizes": {
        "1.5B-AWQ": ("/data2/share/Qwen2.5/Qwen2.5-1.5B-Instruct-AWQ", "awq"),
        "3B-AWQ": ("/data2/share/Qwen2.5/Qwen2.5-3B-Instruct-AWQ", "awq"),
        "7B-AWQ": ("/data2/share/Qwen2.5/Qwen2.5-7B-Instruct-AWQ", "awq"),
        "14B-AWQ": ("/data2/share/Qwen2.5/Qwen2.5-14B-Instruct-AWQ", "awq"),
        "32B-AWQ": ("/data2/share/Qwen2.5/Qwen2.5-32B-Instruct-AWQ", "awq"),
        # "7B-GPTQ": ("/data2/share/Qwen2.5/Qwen2.5-7B-Instruct-GPTQ-w4a16", "gptq"),
        # "14B-GPTQ": ("/data2/share/Qwen2.5/Qwen2.5-14B-Instruct-GPTQ-Int4", "gptq"),
        # "32B-GPTQ": ("/data2/share/Qwen2.5/Qwen2.5-32B-Instruct-GPTQ-Int4", "gptq")
    }
}

NUM_SAMPLES = 50  # 每个实验的样本数
RESULTS = []

# ====================== 工具函数 ====================== 
def load_model(model_path, quant_type=None):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="left"
    )
    
    if quant_type == "awq":
        model = LLM(
            model=model_path,
            trust_remote_code=True,
            dtype="auto",
            gpu_memory_utilization=0.8,  # Reduced from 0.4
            # max_sequence_length=512,  # Changed from max_seq_len to max_sequence_length
            tensor_parallel_size=1
        )
        return model, tokenizer, "vllm"
    else:
        # Add memory optimization for non-AWQ models
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16,  # Use FP16 to reduce memory
            low_cpu_mem_usage=True
        )
        return model, tokenizer, "hf"

def get_hidden_states(model, tokenizer, text, backend):
    if backend == "vllm":
        # For VLLM, we'll only get logits and save them
        sampling_params = SamplingParams(temperature=0, max_tokens=1)
        outputs = model.generate(
            prompts=[text],
            sampling_params=sampling_params,
            use_tqdm=False
        )
        logits = outputs[0].outputs[0].logits
        
        # Save logits to file
        os.makedirs("./save_logits", exist_ok=True)
        save_path = f"./save_logits/{hash(text)}.pt"
        torch.save(logits, save_path)
        
        # Return dummy hidden states for now
        return torch.zeros(1, 768)  # Adjust size based on your model
    else:
        device = next(model.parameters()).device
        inputs = tokenizer(text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            model.eval()
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                output_hidden_states=True
            )
        
        hidden_states = outputs[0][0, :, :]
        return hidden_states.detach()

def compute_erank(hidden_states):
    with torch.no_grad():
        # Convert to torch tensor if needed
        if isinstance(hidden_states, np.ndarray):
            hidden_states = torch.from_numpy(hidden_states).cuda()
            
        # Convert to float32 if needed
        if hidden_states.dtype != torch.float32:
            hidden_states = hidden_states.float()
            
        # Ensure hidden states are 2D
        if hidden_states.ndim > 2:
            hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
            
        # Normalize: center and scale
        mean = hidden_states.mean(dim=0)
        R = hidden_states - mean
        R = R / torch.norm(R, p=2, dim=1, keepdim=True)
        
        # Compute Gram matrix instead of covariance matrix
        Z = torch.nn.functional.normalize(R, dim=1)  # Shape: [n_tokens, hidden_dim]
        B = torch.matmul(Z, Z.T) / Z.shape[0]        # Shape: [n_tokens, n_tokens]
        
        # Calculate effective rank using eigenvalues of B
        eig_val = torch.linalg.eigvalsh(B)  # Eigenvalues of B (same as non-zero eigenvalues of A)
        eig_val = eig_val[eig_val > 1e-8]   # Filter out near-zero eigenvalues
        eig_val = eig_val / eig_val.sum()   # Normalize eigenvalues
        
        # Compute entropy and effective rank
        entropy = -(eig_val * torch.log(eig_val)).nansum().item()
        erank = math.exp(entropy)
        
        return erank

def compute_metrics(model, tokenizer, texts, backend, fp16_erank=None):
    device = "cuda" if backend != "vllm" else None
    eranks, losses = [], []
    
    # Process texts in smaller batches
    batch_size = 5  # Reduce batch size
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        for text in batch_texts:
            hidden_state = get_hidden_states(model, tokenizer, text, backend)
            
            if backend == "vllm":
                # Load saved logits
                save_path = f"./save_logits/{hash(text)}.pt"
                logits = torch.load(save_path)
                
                inputs = tokenizer(text, return_tensors="pt")
                target = inputs.input_ids[0, -1].item()
                loss = torch.nn.functional.cross_entropy(
                    logits.float(),
                    torch.tensor([target]).long()
                ).item()
                losses.append(loss)
            else:
                eranks.append(compute_erank(hidden_state))
                inputs = tokenizer(text, return_tensors="pt").to(device)
                with torch.cuda.amp.autocast():  # Add mixed precision
                    outputs = model(**inputs)
                logits = outputs.logits[0, -1, :].unsqueeze(0)
                target = inputs.input_ids[0, -1].item()
                loss = torch.nn.functional.cross_entropy(
                    logits.float(),
                    torch.tensor([target], device=device).long()
                ).item()
                losses.append(loss)

    avg_erank = np.mean(eranks) if eranks else 0
    avg_ppl = math.exp(np.mean(losses))
    
    if fp16_erank is not None:
        return {"Diff-eRank": avg_erank - fp16_erank, "Perplexity": avg_ppl}
    else:
        return {"eRank": avg_erank, "Perplexity": avg_ppl}

# ====================== 实验执行 ======================
def run_experiments():
    texts = load_wikitext_samples(NUM_SAMPLES)
    
    # 任务3: 模型规模对比
    for name, (path, quant) in MODEL_CONFIGS["model_sizes"].items():
        size = name.split("-")[0]
        model, tokenizer, backend = load_model(path, quant)
        metrics = compute_metrics(model, tokenizer, texts, backend)
        
        # 获取对应规模的 FP16 基准
        fp16_path = f"/data2/share/Qwen2.5/Qwen2.5-{size}-Instruct"  # This path doesn't exist
        fp16_model, fp16_tok, _ = load_model(fp16_path, None)
        fp16_metrics = compute_metrics(fp16_model, fp16_tok, texts, "transformers")
        
        metrics["erank_growth"] = (metrics["eRank"] / fp16_metrics["eRank"] - 1) * 100
        RESULTS.append({"Task": "model_sizes", "Model": name, **metrics})

# ====================== 可视化 ======================
def plot_task1(df):
    """任务1：双Y轴柱状图+折线图"""
    fig, ax1 = plt.subplots(figsize=(10,6))
    
    # 柱状图 (eRank)
    sns.barplot(data=df, x="Model", y="erank", ax=ax1, color='skyblue')
    ax1.set_ylabel('eRank', color='skyblue')
    ax1.tick_params(axis='y', labelcolor='skyblue')
    
    # 折线图 (困惑度)
    ax2 = ax1.twinx()
    sns.lineplot(data=df, x="Model", y="ppl", ax=ax2, color='orange', marker='o')
    ax2.set_ylabel('Perplexity', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')
    
    plt.title("Quantization Methods vs eRank/Perplexity (Task1)")
    plt.savefig('./save_imgs/task1.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_task2(df):
    """任务2：量化位宽敏感性折线图"""
    plt.figure(figsize=(10,6))
    order = ["FP16", "4-bit"]
    sns.lineplot(
        data=df, x=pd.Categorical(df["Model"], categories=order, ordered=True), 
        y="erank_pct", marker="o"
    )
    plt.axhline(100, color='red', linestyle='--', label='FP16 Baseline')
    plt.title("Bit-width Sensitivity Analysis (Task2)")
    plt.xlabel("Quantization Bit-width")
    plt.ylabel("eRank (% of FP16)")
    plt.savefig('./save_imgs/task2.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_task3(df):
    """任务3：分组柱状图"""
    df["Size"] = df["Model"].str.extract('(\d+B)')
    df["Method"] = df["Model"].str.extract('-(AWQ|GPTQ)')
    
    plt.figure(figsize=(12,6))
    sns.barplot(
        data=df, x="Size", y="erank_growth", hue="Method",
        palette="viridis", order=["7B", "14B", "32B"]
    )

    
    plt.title("Model Size vs Quantization Robustness (Task3)")
    plt.ylabel("eRank Growth (%)")
    plt.axhline(0, color='black', linestyle='--')
    plt.savefig('./save_imgs/task3.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_task4(df):
    """任务4：相关性散点图"""
    plt.figure(figsize=(10,6))
    sns.regplot(
        data=df, x="erank_change", y="ppl_change", 
        scatter_kws={'s': 100, 'alpha': 0.6},
        line_kws={'color': 'red'}
    )
    plt.title("Correlation: eRank Change vs Perplexity Increase (Task4)")
    plt.xlabel("eRank Change (%)")
    plt.ylabel("Perplexity Increase (%)")
    plt.grid(True)
    plt.savefig('./save_imgs/task4.png', bbox_inches='tight', dpi=300)
    plt.close()

# ====================== 主程序 ======================
if __name__ == "__main__":
    run_experiments()
    df = pd.DataFrame(RESULTS)
    
    # 绘制各任务图表并保存
    # plot_task1(df[df["Task"] == "quant_methods"])
    plot_task3(df[df["Task"] == "model_sizes"]) 
    # plot_task4(df[df["Task"] == "quant_methods"])