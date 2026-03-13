from __future__ import annotations

import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

from .metrics import compute_accuracy, extract_choice
from .prompting import render_inference_prompt
from .runtime import clear_memory, get_hf_token, preferred_dtype
from .training import _build_quantization_config, load_tokenizer


def _load_model(
    *,
    model_id: str,
    trust_remote_code: bool,
    adapter_path: str | Path | None,
):
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

    if adapter_path:
        model = PeftModel.from_pretrained(model, str(adapter_path))

    model.eval()
    return model


def evaluate_model(
    *,
    model_id: str,
    trust_remote_code: bool,
    records: list[dict],
    split_name: str,
    output_dir: str | Path,
    adapter_path: str | Path | None = None,
    max_length: int = 1024,
    max_new_tokens: int = 8,
) -> dict:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(model_id, trust_remote_code=trust_remote_code)
    model = _load_model(
        model_id=model_id,
        trust_remote_code=trust_remote_code,
        adapter_path=adapter_path,
    )

    rows = []
    for index, record in enumerate(records):
        prompt = render_inference_prompt(record, tokenizer)
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(model.device)

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_tokens = outputs[0][inputs["input_ids"].shape[-1] :]
        generated_text = tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        ).strip()
        predicted_choice = extract_choice(generated_text)
        expected_choice = extract_choice(record["output"])

        rows.append(
            {
                "example_id": index,
                "split": split_name,
                "instruction": record["instruction"],
                "input": record["input"],
                "expected_choice": expected_choice,
                "generated_text": generated_text,
                "predicted_choice": predicted_choice,
                "is_correct": predicted_choice == expected_choice,
            }
        )

    summary = compute_accuracy(rows)
    report = {
        "summary": summary,
        "predictions": rows,
    }

    report_path = output_path / f"{split_name}_predictions.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    del model
    del tokenizer
    clear_memory()
    return report
