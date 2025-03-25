<h1 align="center"> ACBench </h1>
<h3 align="center"> Benchmarking Agentic Capabilities of Compressed LLMs </h3>

<p align="center">
  <a href="https://arxiv.org/abs/2403.xxxxx">📄arXiv</a> •
  <a href="https://huggingface.co/papers/2403.xxxxx">🤗HFPaper</a> •
  <a href="https://github.com/yourusername/ACBench">🌐GitHub</a> •
  <a href="https://huggingface.co/datasets/yourusername/ACBench">📊Dataset</a>
</p>

[![Awesome](https://awesome.re/badge.svg)](https://github.com/yourusername/ACBench) 
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
![](https://img.shields.io/github/last-commit/yourusername/ACBench?color=green) 

## Table of Contents
- 🌻[Overview](#🌻overview)
- 🔧[Installation](#🔧installation)
- 📊[Benchmark Structure](#📊benchmark-structure)
- 🧪[Evaluation](#🧪evaluation)
- 📈[Results](#📈results)

---

## 🌻Overview

Post-training compression reduces the computational and memory costs of large language models (LLMs), enabling resource-efficient deployment. However, existing compression benchmarks focus narrowly on language modeling (e.g., perplexity) and natural language understanding tasks (e.g., GLUE accuracy), ignoring the agentic capabilities—workflow, tool use/function call, long-context understanding and real-world application.

We introduce the Agent Compression Benchmark (ACBench), the first comprehensive benchmark for evaluating how compression impacts LLMs' agentic abilities. ACBench spans:
- 12 tasks across 4 capabilities (e.g., WorfBench for workflow generation, Needle-in-Haystack for long-context retrieval)
- 4-bit quantization (GPTQ, AWQ) and 50% pruning (Wanda, SparseGPT)
- 15 models, including small (Gemma-2B), standard (Qwen2.5-7B), and distilled reasoning LLMs (DeepSeek-R1-Distill)

Our experiments reveal compression tradeoffs: 4-bit quantization preserves workflow generation and tool use (1%--3% drop) but degrades real-world application accuracy by 10%--15%. We introduce ERank, Top-k Ranking Correlation and Energy to systematize analysis. ACBench provides actionable insights for optimizing LLM compression in agentic scenarios, bridging the gap between algorithmic efficiency and real-world applicability.

## 🔧Installation

```bash
git clone https://github.com/yourusername/ACBench
cd ACBench
pip install -r requirements.txt
```

## 📊Benchmark Structure

ACBench evaluates LLM compression across four key agentic capabilities:

1. **Workflow Generation**
   - WorfBench tasks
   - Complex planning scenarios
   - Multi-step reasoning

2. **Tool Use & Function Calling**
   - API interaction tasks
   - Tool selection and execution
   - Parameter handling

3. **Long-Context Understanding**
   - Needle-in-Haystack tasks
   - Document analysis
   - Context retrieval

4. **Real-World Applications**
   - Practical scenarios
   - Domain-specific tasks
   - Real-world problem solving

## 🧪Evaluation

ACBench supports evaluation of various compression methods:

```bash
# Evaluate 4-bit quantization
python evaluate.py \
    --model_name your_model \
    --compression_method gptq \
    --tasks workflow tool_use long_context real_world

# Evaluate pruning
python evaluate.py \
    --model_name your_model \
    --compression_method wanda \
    --tasks workflow tool_use long_context real_world
```

## 📈Results

Our comprehensive evaluation reveals:

1. **Quantization Impact**
   - Workflow generation: 1-3% performance drop
   - Tool use: Minimal degradation
   - Real-world applications: 10-15% accuracy reduction

2. **Pruning Effects**
   - Selective impact on different capabilities
   - Trade-offs between efficiency and performance

3. **Model-Specific Findings**
   - Small models (Gemma-2B)
   - Standard models (Qwen2.5-7B)
   - Distilled models (DeepSeek-R1-Distill)

For detailed results and analysis, please refer to our paper.

