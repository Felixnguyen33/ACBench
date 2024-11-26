from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import argparse

def main(args):
    model_path = args.model_path
    # Create a path that includes quantization settings
    quant_path = f"{model_path}-awq-w{args.w_bit}-g{args.q_group_size}"
    if args.zero_point:
        quant_path += "-zp"
    
    quant_config = {
        "zero_point": args.zero_point,
        "q_group_size": args.q_group_size,
        "w_bit": args.w_bit,
        "version": args.version
    }

    # Load model with device_map for automatic GPU allocation
    model = AutoAWQForCausalLM.from_pretrained(
        model_path,
        device_map="auto",  # Add automatic device mapping
        low_cpu_mem_usage=True,
        use_cache=False
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Quantize - model.device will now point to the correct device for each part
    model.quantize(tokenizer, quant_config=quant_config)

    # Save quantized model
    model.save_quantized(quant_path)
    tokenizer.save_pretrained(quant_path)

    print(f'Model is quantized and saved at "{quant_path}"')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--quant_path", type=str, default="mistral-instruct-v0.2-awq")
    parser.add_argument("--zero_point", type=bool, default=True)
    parser.add_argument("--q_group_size", type=int, default=128)
    parser.add_argument("--w_bit", type=int, default=4)
    parser.add_argument("--version", type=str, default="gemm")
    args = parser.parse_args()
    main(args)