<h1 align="center">🤖 ACBench 🔬</h1>
<h3 align="center">🤔 Can Compressed LLMs Truly Act? An Empirical Evaluation of Agentic Capabilities in LLM Compression 📊</h3>

<p align="center">
  <a href="https://arxiv.org/abs/2505.xxxxx">📄arXiv</a> •
  <a href="https://github.com/pprp/ACBench">🌐GitHub</a>
</p>

<p align="center">
  <a href="https://github.com/pprp/ACBench"><img src="https://awesome.re/badge.svg" alt="Awesome"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"></a>
</p>

## Table of Contents

- [Overview](#overview)
- [Abstract](#abstract)
- [Installation](#installation)
- [Evaluation](#evaluation)
- [Citation](#citation)

![](./assets/main_figure_v2.png)

## Overview

Post-training compression reduces the computational and memory costs of large language models (LLMs), enabling resource-efficient deployment. However, existing compression benchmarks focus narrowly on language modeling (e.g., perplexity) and natural language understanding tasks (e.g., GLUE accuracy), ignoring the agentic capabilities—workflow, tool use/function call, long-context understanding and real-world application.

We introduce the Agent Compression Benchmark (ACBench), the first comprehensive benchmark for evaluating how compression impacts LLMs' agentic abilities. ACBench spans:

- 12 tasks across 4 capabilities (e.g., WorfBench for workflow generation, Needle-in-Haystack for long-context retrieval)
- 4-bit quantization (GPTQ, AWQ) and 50% pruning (Wanda, SparseGPT)
- 15 models, including small (Gemma-2B), standard (Qwen2.5-7B), and distilled reasoning LLMs (DeepSeek-R1-Distill)

## Abstract

Post-training compression reduces the computational and memory costs of large language models (LLMs), enabling resource-efficient deployment. However, existing compression benchmarks focus narrowly on language modeling (e.g., perplexity) and natural language understanding tasks (e.g., GLUE accuracy), ignoring the agentic capabilities—_workflow_, _tool use/function call_, _long-context understanding_ and _real-world application_. We introduce the **Agent Compression Benchmark (ACBench)**, the first comprehensive benchmark for evaluating how compression impacts LLMs' agentic abilities. ACBench spans (1) 12 tasks across 4 capabilities (e.g., WorfBench for workflow generation, Needle-in-Haystack for long-context retrieval), (2) 4-bit quantization (GPTQ, AWQ) and 50% pruning (Wanda, SparseGPT), and (3) 15 models, including small (Gemma-2B), standard (Qwen2.5-7B), and distilled reasoning LLMs (DeepSeek-R1-Distill). Our experiments reveal compression tradeoffs: 4-bit quantization preserves workflow generation and tool use (1%--3% drop) but degrades real-world application accuracy by 10%--15%. We introduce _ERank_, _Top-k Ranking Correlation_ and _Energy_ to systematize analysis. ACBench provides actionable insights for optimizing LLM compression in agentic scenarios, bridging the gap between algorithmic efficiency and real-world applicability.

## Installation

```bash
git clone https://github.com/pprp/ACBench
cd ACBench
pip install -r requirements.txt
pip install -e .
```

## Evaluation

ACBench builds upon and extends several excellent agentic benchmarks and compression toolkits. We integrate these benchmarks into our evaluation pipeline while preserving their original settings. For efficient model serving and evaluation, we utilize VLLM to deploy the compressed language models.

For detailed implementation and usage instructions, please refer to the corresponding subfolders in the `thirdpartys` directory. Each subfolder contains the original benchmark code along with our modifications to support compressed model evaluation. For experiment result on WorfBench, we have integrated it in acbench.

### A Demo for WorfBench

Taking WorfBench as example, we can run using the following scripts:

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

### Agentic Tasks

For Agentic Tasks:

- WorfBench: Benchmarking Agentic Workflow Generation [![arXiv](https://img.shields.io/badge/arXiv-2410.07869-b31b1b.svg)](https://arxiv.org/abs/2410.07869) [![GitHub](https://img.shields.io/github/stars/zjunlp/WorfBench?style=social)](https://github.com/zjunlp/WorfBench)
- AgentBoard: An Analytical Evaluation Board of Multi-turn LLM Agents [![arXiv](https://img.shields.io/badge/arXiv-2401.13178-b31b1b.svg)](https://arxiv.org/abs/2401.13178) [![GitHub](https://img.shields.io/github/stars/hkust-nlp/agentboard?style=social)](https://github.com/hkust-nlp/agentboard)
- KVCache-Factory: Unified KV Cache Compression Methods for Auto-Regressive Models [![arXiv](https://img.shields.io/badge/arXiv-2406.02069-b31b1b.svg)](https://arxiv.org/abs/2406.02069) [![GitHub](https://img.shields.io/github/stars/Zefan-Cai/KVCache-Factory?style=social)](https://github.com/Zefan-Cai/KVCache-Factory)
- LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding [![arXiv](https://img.shields.io/badge/arXiv-2308.14508-b31b1b.svg)](https://arxiv.org/abs/2308.14508) [![GitHub](https://img.shields.io/github/stars/THUDM/LongBench?style=social)](https://github.com/THUDM/LongBench)
- SCOPE: Optimizing Key-Value Cache Compression in Long-context Generation [![arXiv](https://img.shields.io/badge/arXiv-2412.13649-b31b1b.svg)](https://arxiv.org/pdf/2412.13649) [![GitHub](https://img.shields.io/github/stars/Linking-ai/SCOPE?style=social)](https://github.com/Linking-ai/SCOPE)
- T-Eval: Evaluating Tool Utilization Capability of LLMs Step by Step [![arXiv](https://img.shields.io/badge/arXiv-2312.14033-b31b1b.svg)](https://arxiv.org/abs/2312.14033) [![GitHub](https://img.shields.io/github/stars/open-compass/T-Eval?style=social)](https://github.com/open-compass/T-Eval)

### Compression

For Compression:

- Wanda: A Simple and Effective Pruning Approach for Large Language Models [![arXiv](https://img.shields.io/badge/arXiv-2306.11695-b31b1b.svg)](https://arxiv.org/abs/2306.11695) [![GitHub](https://img.shields.io/github/stars/locuslab/wanda?style=social)](https://github.com/locuslab/wanda)
- LLMC: Benchmarking Large Language Model Quantization with a Versatile Compression Toolkit [![arXiv](https://img.shields.io/badge/arXiv-2405.06001-b31b1b.svg)](https://arxiv.org/abs/2405.06001) [![GitHub](https://img.shields.io/github/stars/ModelTC/llmc?style=social)](https://github.com/ModelTC/llmc)
- QLLM-Eval: Evaluating Quantized Large Language Models [![arXiv](https://img.shields.io/badge/arXiv-2402.18158-b31b1b.svg)](https://arxiv.org/abs/2402.18158) [![GitHub](https://img.shields.io/github/stars/thu-nics/qllm-eval?style=social)](https://github.com/thu-nics/qllm-eval)

For fast serving, we employ [![Github](https://img.shields.io/github/stars/vllm-project/vllm?style=social)](https://github.com/vllm-project/vllm) for evaluation.

## Visualization

Energy-based analysis:

| ![Combined Position 1](./assets/energy_vis/combined_position_01.png)  | ![Combined Position 2](./assets/energy_vis/combined_position_64.png)  | ![Combined Position 3](./assets/energy_vis/combined_position_117.png) |
| :-------------------------------------------------------------------: | :-------------------------------------------------------------------: | :-------------------------------------------------------------------: |
| ![Combined Position 1](./assets/energy_vis/combined_position_175.png) | ![Combined Position 2](./assets/energy_vis/combined_position_244.png) | ![Combined Position 3](./assets/energy_vis/combined_position_252.png) |

Logits Visualization:

| ![Logits Visualization 1](./assets/logits_vis/combined_position_02.png)  | ![Logits Visualization 2](./assets/logits_vis/combined_position_82.png)  | ![Logits Visualization 3](./assets/logits_vis/combined_position_134.png) |
| :----------------------------------------------------------------------: | :----------------------------------------------------------------------: | :----------------------------------------------------------------------: |
| ![Logits Visualization 1](./assets/logits_vis/combined_position_184.png) | ![Logits Visualization 2](./assets/logits_vis/combined_position_225.png) | ![Logits Visualization 3](./assets/logits_vis/combined_position_246.png) |

Needle Visualization:

### InternLM Models

|     ![Needle Vis 1](./assets/needle/InternLM2.5_7b_AWQ_full_64_40k.png)      |   ![Needle Vis 2](./assets/needle/InternLM2.5_7b_GPTQ_full_64_40k.png)    |   ![Needle Vis 3](./assets/needle/InternLM2.5_7b_Instruct_full_64_40k.png)    |
| :--------------------------------------------------------------------------: | :-----------------------------------------------------------------------: | :---------------------------------------------------------------------------: |
|   ![Needle Vis 4](./assets/needle/InternLM2.5_7b_mag_2_4_full_64_40k.png)    |  ![Needle Vis 5](./assets/needle/InternLM2.5_7b_mag_un_full_64_40k.png)   | ![Needle Vis 6](./assets/needle/InternLM2.5_7b_sparsegpt_2_4_full_64_40k.png) |
| ![Needle Vis 7](./assets/needle/InternLM2.5_7b_sparsegpt_un_full_64_40k.png) | ![Needle Vis 8](./assets/needle/InternLM2.5_7b_wanda_2_4_full_64_40k.png) |   ![Needle Vis 9](./assets/needle/InternLM2.5_7b_wanda_un_full_64_40k.png)    |

### Qwen Models

|     ![Needle Vis 10](./assets/needle/Qwen2.5_7b_AWQ_full_64_40k.png)      |     ![Needle Vis 11](./assets/needle/Qwen2.5_7b_GPTQ_full_64_40k.png)     |      ![Needle Vis 12](./assets/needle/Qwen2.5_7b_RTN_full_64_40k.png)      |
| :-----------------------------------------------------------------------: | :-----------------------------------------------------------------------: | :------------------------------------------------------------------------: |
|   ![Needle Vis 13](./assets/needle/Qwen2.5_7b_mag_2_4_full_64_40k.png)    |    ![Needle Vis 14](./assets/needle/Qwen2.5_7b_mag_un_full_64_40k.png)    | ![Needle Vis 15](./assets/needle/Qwen2.5_7b_sparsegpt_2_4_full_64_40k.png) |
| ![Needle Vis 16](./assets/needle/Qwen2.5_7b_sparsegpt_un_full_64_40k.png) |  ![Needle Vis 17](./assets/needle/Qwen2.5_7b_wanda_2_4_full_64_40k.png)   |   ![Needle Vis 18](./assets/needle/Qwen2.5_7b_wanda_un_full_64_40k.png)    |
|    ![Needle Vis 24](./assets/needle/qwen-1.5b-awq_full_64_40k_slm.png)    | ![Needle Vis 25](./assets/needle/qwen-1.5b-gptq-int4_full_64_40k_slm.png) | ![Needle Vis 26](./assets/needle/qwen-1.5b-gptq-int8_full_64_40k_slm.png)  |
|     ![Needle Vis 27](./assets/needle/qwen-3b-awq_full_64_40k_slm.png)     |  ![Needle Vis 28](./assets/needle/qwen-3b-gptq-int4_full_64_40k_slm.png)  |  ![Needle Vis 29](./assets/needle/qwen-3b-gptq-int8_full_64_40k_slm.png)   |

### Distilled/Megrez Models

| ![Needle Vis 20](./assets/needle/deepseek-llama-8b_full_64_40k_slm.png) | ![Needle Vis 21](./assets/needle/deepseek-qwen-1.5b_full_64_40k_slm.png) |
| :---------------------------------------------------------------------: | :----------------------------------------------------------------------: |
| ![Needle Vis 22](./assets/needle/deepseek-qwen-7b_full_64_40k_slm.png)  |     ![Needle Vis 23](./assets/needle/megrez-3b_full_64_40k_slm.png)      |

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
