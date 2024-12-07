import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datasets import load_dataset
from modeling_llama import LlamaForCausalLM


def calculate_perplexity(model, tokenizer, text, device, max_length=128):
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = encodings.input_ids.to(device)
    target_ids = input_ids.clone()

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)
        loss = outputs.loss

    ppl = torch.exp(loss)
    return ppl.item()

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    model_path = "/data2/share/llama3.2/Llama-3.2-1B-Instruct"
    logging.info(f"Loading model and tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = LlamaForCausalLM.from_pretrained(model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    model.to(device)

    logging.info("Loading WikiText-2 dataset")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = " ".join(dataset["text"])

    logging.info("Calculating perplexity")
    perplexity = calculate_perplexity(model, tokenizer, text, device)
    logging.info(f"Perplexity: {perplexity}")

if __name__ == "__main__":
    main()
