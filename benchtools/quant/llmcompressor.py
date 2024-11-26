from llmcompressor.transformers import SparseAutoModelForCausalLM
from transformers import AutoTokenizer
from llmcompressor.transformers import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from datasets import load_dataset
import argparse

def main(args):
    model = SparseAutoModelForCausalLM.from_pretrained(
        args.model_path, device_map="auto", torch_dtype="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # Load and preprocess the dataset
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    ds = ds.shuffle(seed=42).select(range(args.num_calibration_samples))

    def preprocess(example):
        return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}
    ds = ds.map(preprocess)

    def tokenize(sample):
        return tokenizer(sample["text"], padding=False, max_length=args.max_sequence_length, truncation=True, add_special_tokens=False)
    ds = ds.map(tokenize, remove_columns=ds.column_names)

    # Configure the quantization algorithms
    recipe = []
    if "smoothquant" in args.quant_method:
        recipe.append(SmoothQuantModifier(smoothing_strength=args.smoothing_strength))
    if "gptq" in args.quant_method:
        recipe.append(GPTQModifier(targets="Linear", scheme=args.quant_scheme, ignore=["lm_head"]))

    # Apply quantization
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=args.max_sequence_length,
        num_calibration_samples=args.num_calibration_samples,
    )

    # Save the compressed model
    save_dir = args.model_path.split("/")[-1] + f"-{args.quant_scheme}"
    model.save_pretrained(save_dir, save_compressed=True)
    tokenizer.save_pretrained(save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--num_calibration_samples", type=int, default=512)
    parser.add_argument("--max_sequence_length", type=int, default=2048)
    parser.add_argument("--quant_method", nargs="+", choices=["smoothquant", "gptq"], default=["smoothquant", "gptq"])
    parser.add_argument("--smoothing_strength", type=float, default=0.8)
    parser.add_argument("--quant_scheme", type=str, default="W8A8")
    args = parser.parse_args()
    main(args)