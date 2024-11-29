from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from llmcompressor.transformers import oneshot
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from vllm import LLM

def main(args):
    # Configure the quantization algorithms
    recipe = []
    if "smoothquant" in args.quant_method:
        recipe.append(SmoothQuantModifier(smoothing_strength=args.smoothing_strength))
    if "gptq" in args.quant_method:
        recipe.append(GPTQModifier(targets="Linear", scheme=args.quant_scheme, ignore=["lm_head"]))

    # Apply quantization using open_platypus dataset
    save_dir = args.model_path + f"-{args.quant_scheme}"
    if "smoothquant" in args.quant_method:
        save_dir += f"-smooth{args.smoothing_strength}"
    if "gptq" in args.quant_method:
        save_dir += "-gptq"

    oneshot(
        model=args.model_path,
        dataset="open_platypus",
        recipe=recipe,
        output_dir=save_dir,
        max_seq_length=args.max_sequence_length,
        num_calibration_samples=args.num_calibration_samples,
    )
    
    # model = LLM(save_dir)
    # output = model.generate("My name is")
    # print('=============OUTPUT START=================')
    # for o in output:
    #     print(f"Request ID: {o.request_id}")
    #     print("Generated text:")
    #     for completion in o.outputs:
    #         print(completion.text)
    # print('=============OUTPUT END ==================')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--num_calibration_samples", type=int, default=1024)
    parser.add_argument("--max_sequence_length", type=int, default=2048)
    parser.add_argument("--quant_method", nargs="+", choices=["smoothquant", "gptq"], default=["smoothquant", "gptq"])
    parser.add_argument("--smoothing_strength", type=float, default=0.8)
    parser.add_argument("--quant_scheme", type=str, default="W8A8")
    args = parser.parse_args()
    main(args)