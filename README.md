<h1 align="center"> ACBench </h1>
<h3 align="center"> Can Compressed LLMs Truly Act? An Empirical Evaluation of Agentic Capabilities in LLM Compression </h3>

<p align="center">
  <a href="https://arxiv.org/abs/2403.xxxxx">📄arXiv</a> •
  <a href="https://huggingface.co/papers/2403.xxxxx">🤗HFPaper</a> •
  <a href="https://github.com/pprp/ACBench">🌐GitHub</a> •
</p>

[![Awesome](https://awesome.re/badge.svg)](https://github.com/pprp/ACBench) 
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
![](https://img.shields.io/github/last-commit/pprp/ACBench?color=green) 

## Table of Contents

- 
- 🌻[Overview](#🌻overview)
- 🔧[Installation](#🔧installation)
- 📊[Benchmark Structure](#📊benchmark-structure)
- 🧪[Evaluation](#🧪evaluation)

---

## Abstract 

Post-training compression reduces the computational and memory costs of large language models (LLMs), enabling resource-efficient deployment. However, existing compression benchmarks focus narrowly on language modeling (e.g., perplexity) and natural language understanding tasks (e.g., GLUE accuracy), ignoring the agentic capabilities—*workflow*, *tool use/function call*, *long-context understanding* and *real-world application*. We introduce the **Agent Compression Benchmark (ACBench)**, the first comprehensive benchmark for evaluating how compression impacts LLMs' agentic abilities. ACBench spans (1) 12 tasks across 4 capabilities (e.g., WorfBench for workflow generation, Needle-in-Haystack for long-context retrieval), (2) 4-bit quantization (GPTQ, AWQ) and 50% pruning (Wanda, SparseGPT), and (3) 15 models, including small (Gemma-2B), standard (Qwen2.5-7B), and distilled reasoning LLMs (DeepSeek-R1-Distill). Our experiments reveal compression tradeoffs: 4-bit quantization preserves workflow generation and tool use (1%--3% drop) but degrades real-world application accuracy by 10%--15%. We introduce *ERank*, *Top-k Ranking Correlation* and *Energy* to systematize analysis. ACBench provides actionable insights for optimizing LLM compression in agentic scenarios, bridging the gap between algorithmic efficiency and real-world applicability.


## 🌻Overview

Post-training compression reduces the computational and memory costs of large language models (LLMs), enabling resource-efficient deployment. However, existing compression benchmarks focus narrowly on language modeling (e.g., perplexity) and natural language understanding tasks (e.g., GLUE accuracy), ignoring the agentic capabilities—workflow, tool use/function call, long-context understanding and real-world application.

![](./misc/main.png)

We introduce the Agent Compression Benchmark (ACBench), the first comprehensive benchmark for evaluating how compression impacts LLMs' agentic abilities. ACBench spans:
- 12 tasks across 4 capabilities (e.g., WorfBench for workflow generation, Needle-in-Haystack for long-context retrieval)
- 4-bit quantization (GPTQ, AWQ) and 50% pruning (Wanda, SparseGPT)
- 15 models, including small (Gemma-2B), standard (Qwen2.5-7B), and distilled reasoning LLMs (DeepSeek-R1-Distill)

Our experiments reveal compression tradeoffs: 4-bit quantization preserves workflow generation and tool use (1%--3% drop) but degrades real-world application accuracy by 10%--15%. We introduce ERank, Top-k Ranking Correlation and Energy to systematize analysis. ACBench provides actionable insights for optimizing LLM compression in agentic scenarios, bridging the gap between algorithmic efficiency and real-world applicability.

## 🔧Installation

```bash
git clone https://github.com/pprp/ACBench
cd ACBench
pip install -r requirements.txt
```
## 🧪Evaluation

ACBench builds upon and extends several excellent agentic benchmarks and compression toolkits. We integrate these benchmarks into our evaluation pipeline while preserving their original settings. For efficient model serving and evaluation, we utilize VLLM to deploy the compressed language models.

For detailed implementation and usage instructions, please refer to the corresponding subfolders in the `thirdpartys` directory. Each subfolder contains the original benchmark code along with our modifications to support compressed model evaluation. For experiment result on WorfBench, we have integrated it in acbench. 

Taking ALFWorld as example, we can run using the following scripts:

```
#!/bin/bash 

MODEL=$1
TEMP=$2
QUANT=$3

DEVICE=${4:-6}

export CUDA_VISIBLE_DEVICES=$DEVICE

tasks=(wikihow toolbench toolalpaca lumos alfworld webshop os)

MODEL_NAME=$(basename $MODEL)

for task in ${tasks[@]}; do
    python agentbench/node_eval.py \
        --task gen_workflow \
        --model_name ${MODEL} \
        --gold_path ./data/gold_traj/${task}/graph_eval.json \
        --pred_path ./data/pred_traj/${MODEL_NAME}/${task}/${MODEL_NAME}/graph_eval_two_shot.json \
        --task_type ${task} \
        --few_shot \
        --temperature ${TEMP} \
        --quantization ${QUANT} 
done
```


For Agentic Tasks:
- WorfBench: A Benchmark for Grounding Language in Vision and Robotics [![arXiv](https://img.shields.io/badge/arXiv-2410.07869-b31b1b.svg)](https://arxiv.org/abs/2410.07869) [![GitHub](https://img.shields.io/github/stars/zjunlp/WorfBench?style=social)](https://github.com/zjunlp/WorfBench)
- AgentBoard: A Comprehensive Benchmark for LLM-as-Agent Evaluation [![arXiv](https://img.shields.io/badge/arXiv-2401.13178-b31b1b.svg)](https://arxiv.org/abs/2401.13178) [![GitHub](https://img.shields.io/github/stars/hkust-nlp/agentboard?style=social)](https://github.com/hkust-nlp/agentboard)
- KVCache-Factory: Evaluating LLMs' Efficiency and Quality [![arXiv](https://img.shields.io/badge/arXiv-2406.02069-b31b1b.svg)](https://arxiv.org/abs/2406.02069) [![GitHub](https://img.shields.io/github/stars/Zefan-Cai/KVCache-Factory?style=social)](https://github.com/Zefan-Cai/KVCache-Factory)
- LongBench: A Comprehensive Benchmark for LLMs' Long-Context Understanding [![arXiv](https://img.shields.io/badge/arXiv-2308.14508-b31b1b.svg)](https://arxiv.org/abs/2308.14508) [![GitHub](https://img.shields.io/github/stars/THUDM/LongBench?style=social)](https://github.com/THUDM/LongBench)
- SCOPE: A Task-Centric Benchmark for Autonomous Agents [![arXiv](https://img.shields.io/badge/arXiv-2412.13649-b31b1b.svg)](https://arxiv.org/pdf/2412.13649) [![GitHub](https://img.shields.io/github/stars/Linking-ai/SCOPE?style=social)](https://github.com/Linking-ai/SCOPE)
- T-Eval: Evaluating Tool Utilization Capability of LLMs Step by Step [![arXiv](https://img.shields.io/badge/arXiv-2312.14033-b31b1b.svg)](https://arxiv.org/abs/2312.14033) [![GitHub](https://img.shields.io/github/stars/open-compass/T-Eval?style=social)](https://github.com/open-compass/T-Eval)

For Compression:
- Wanda: Weight-Adaptive Neural Network Pruning [![arXiv](https://img.shields.io/badge/arXiv-2306.11695-b31b1b.svg)](https://arxiv.org/abs/2306.11695) [![GitHub](https://img.shields.io/github/stars/locuslab/wanda?style=social)](https://github.com/locuslab/wanda)
- LLMC: Language Model Compression Benchmark [![arXiv](https://img.shields.io/badge/arXiv-2405.06001-b31b1b.svg)](https://arxiv.org/abs/2405.06001) [![GitHub](https://img.shields.io/github/stars/ModelTC/llmc?style=social)](https://github.com/ModelTC/llmc)
- QLLM-Eval: Evaluating Quantized Large Language Models [![arXiv](https://img.shields.io/badge/arXiv-2402.18158-b31b1b.svg)](https://arxiv.org/abs/2402.18158) [![GitHub](https://img.shields.io/github/stars/thu-nics/qllm-eval?style=social)](https://github.com/thu-nics/qllm-eval)


For fast serving, we employ ![VLLM](https://github.com/vllm-project/vllm) for evaluation. 

## Citation

If you use our work, please cite:

```
@inproceedings{dong2025compressed,
  title     = {Can Compressed LLMs Truly Act? An Empirical Evaluation of Agentic Capabilities in LLM Compression},
  author    = {Peijie Dong and Zhenheng Tang and Xiang Liu and Lujun Li and Xiaowen Chu and Bo Li},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2025}
}
```
