from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
)
from trl import SFTConfig, SFTTrainer

from .prompting import render_training_text
from .runtime import clear_memory, get_hf_token, preferred_dtype


def load_tokenizer(model_id: str, trust_remote_code: bool) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        token=get_hf_token(),
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _build_quantization_config(compute_dtype: torch.dtype):
    if not torch.cuda.is_available():
        return None

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )


def train_lora_adapter(
    *,
    model_id: str,
    trust_remote_code: bool,
    train_records: list[dict],
    val_records: list[dict],
    output_dir: str | Path,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    eval_batch_size: int,
    gradient_accumulation_steps: int,
    max_length: int,
    seed: int,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    early_stopping_patience: int,
) -> dict:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(model_id, trust_remote_code=trust_remote_code)
    train_dataset = Dataset.from_list(
        [{"text": render_training_text(record, tokenizer)} for record in train_records]
    )
    val_dataset = Dataset.from_list(
        [{"text": render_training_text(record, tokenizer)} for record in val_records]
    )

    compute_dtype = preferred_dtype()
    quantization_config = _build_quantization_config(compute_dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        token=get_hf_token(),
        quantization_config=quantization_config,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=None if quantization_config else compute_dtype,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False

    if quantization_config is not None:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
        )

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    model = get_peft_model(model, lora_config)

    config_kwargs = {
        "output_dir": str(output_path),
        "dataset_text_field": "text",
        "max_length": max_length,
        "num_train_epochs": epochs,
        "learning_rate": learning_rate,
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "warmup_ratio": 0.1,
        "lr_scheduler_type": "cosine",
        "logging_steps": 1,
        "logging_first_step": True,
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "report_to": "none",
        "packing": False,
        "seed": seed,
        "data_seed": seed,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "max_grad_norm": 0.3,
        "optim": "paged_adamw_8bit" if quantization_config else "adamw_torch",
        "bf16": torch.cuda.is_available() and compute_dtype == torch.bfloat16,
        "fp16": torch.cuda.is_available() and compute_dtype == torch.float16,
    }
    if len(val_dataset) > 0:
        config_kwargs.update(
            {
                "per_device_eval_batch_size": eval_batch_size,
                "eval_strategy": "epoch",
                "load_best_model_at_end": True,
                "metric_for_best_model": "eval_loss",
                "greater_is_better": False,
            }
        )
    else:
        config_kwargs["eval_strategy"] = "no"

    train_args = SFTConfig(**config_kwargs)

    callbacks = []
    if len(val_dataset) > 0 and early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=early_stopping_patience
            )
        )

    trainer = SFTTrainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset if len(val_dataset) > 0 else None,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    train_result = trainer.train()
    eval_metrics = trainer.evaluate() if len(val_dataset) > 0 else {}
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    summary = {
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_metrics,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "trainable_params": {
            "total": sum(p.numel() for p in model.parameters()),
            "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        },
    }
    if len(train_dataset) > 0:
        summary["effective_train_examples"] = len(train_dataset)
        summary["expected_train_steps"] = math.ceil(
            len(train_dataset) * epochs / max(1, batch_size * gradient_accumulation_steps)
        )

    metrics_path = output_path / "training_summary.json"
    metrics_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    del trainer
    del model
    del tokenizer
    clear_memory()
    return summary
