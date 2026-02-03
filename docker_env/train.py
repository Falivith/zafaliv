import argparse
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="data/dataset.jsonl")
    parser.add_argument("--output_dir", type=str, default="outputs/model-ft")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument(
        "--mode",
        choices=["local", "remote"],
        default="local",
        help="local = GTX 1660 | remote = RTX 4090"
    )
    return parser.parse_args()


def format_example(example):
    return {
        "text": f"""### Instrução:
{example["instruction"]}

### Entrada:
{example["input"]}

### Resposta:
{example["output"]}"""
    }


def load_model(args):
    print(f"Loading model ({args.mode} mode): {args.model_id}")

    if args.mode == "remote":
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            load_in_4bit=True,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_id)

    return model


def maybe_apply_lora(model, args):
    if args.mode != "remote":
        return model

    print("Applying LoRA")

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token

    model = load_model(args)
    model = maybe_apply_lora(model, args)

    print("Loading dataset")
    
    raw_dataset = load_dataset("json", data_files=args.data_path)["train"]
    split = raw_dataset.train_test_split(test_size=0.2, seed=42)

    train_dataset = split["train"].map(format_example)
    test_dataset  = split["test"].map(format_example)

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        max_length=2048 if args.mode == "remote" else 1024,
        packing=True,
        optim="paged_adamw_8bit" if args.mode == "remote" else "adamw_torch",
        bf16=True if args.mode == "remote" else False,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
    )

    print("Starting training")
    trainer.train()

    print("Saving model")
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
