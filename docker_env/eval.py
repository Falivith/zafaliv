import argparse
import re
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--data_path", type=str, default="data/dataset.jsonl")
    parser.add_argument("--split", type=float, default=0.2)
    parser.add_argument("--max_new_tokens", type=int, default=5)
    return parser.parse_args()


def format_example(example):
    return f"""### Instrução:
{example["instruction"]}

### Entrada:
{example["input"]}

### Resposta:
"""


def extract_answer(text):
    """
    Extrai a alternativa (A–E) da saída do modelo.
    """
    match = re.search(r"\b([A-E])\b", text)
    return match.group(1) if match else None


def load_model(model_id, adapter_path):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )

    if adapter_path:
        print(f"▶ Carregando LoRA de {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model


def evaluate(model, tokenizer, dataset, max_new_tokens):
    correct = 0
    total = len(dataset)

    for example in dataset:
        prompt = format_example(example)

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=False
        ).to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )

        decoded = tokenizer.decode(output[0], skip_special_tokens=True)

        pred = extract_answer(decoded)
        gold = extract_answer(example["output"])

        if pred == gold:
            correct += 1

    return correct / total if total > 0 else 0.0


def main():
    args = parse_args()

    print("▶ Carregando tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token

    print("▶ Carregando dataset")
    raw_dataset = load_dataset("json", data_files=args.data_path)["train"]

    split = raw_dataset.train_test_split(test_size=args.split, seed=42)
    test_dataset = split["test"]

    print(f"▶ Test set size: {len(test_dataset)}")

    print("▶ Carregando modelo")
    model = load_model(args.model_id, args.adapter_path)

    print("▶ Avaliando...")
    acc = evaluate(
        model,
        tokenizer,
        test_dataset,
        args.max_new_tokens
    )

    print("\n==============================")
    print(f"Accuracy: {acc:.4f}")
    print("==============================\n")


if __name__ == "__main__":
    main()
