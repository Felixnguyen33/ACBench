from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

import argparse

parser = argparse.ArgumentParser(description='Inference with VLLM')
parser.add_argument('--model_path', type=str, default='/mnt/sdb/dongpeijie/workspace/AgentBench/thirdparty/llmc/save/awq_w4a16/vllm_quant_model', help='Path to the model')
args = parser.parse_args()

model_path = args.model_path
model = LLM(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

prompts = [
    'Hello, my name is',
    'The president of the United States is',
    'The capital of France is',
    'The future of AI is',
    '完成24点游戏:使用数字17、19、23、29，每个数字仅使用一次，如何利用加减乘除和括号的简单运算法则计算出24?'
]
sampling_params = SamplingParams(temperature=0.1, top_p=0.5, max_tokens=4096, stop_token_ids=[128009], stop=["END", "---", "\n\n"])

outputs = model.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f'Prompt: {prompt!r}, Generated text: {generated_text!r}')
