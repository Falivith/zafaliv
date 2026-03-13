from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline.dataset_utils import build_split_datasets, dataset_stats, load_records
from pipeline.evaluation import evaluate_model
from pipeline.model_registry import resolve_model_spec
from pipeline.training import train_lora_adapter


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Executa treino e teste para um modelo."
    )
    parser.add_argument("--model", required=True, help="Alias ou model_id no HF.")
    parser.add_argument(
        "--data-path",
        default=str(root / "data" / "datasets" / "dataset.jsonl"),
    )
    parser.add_argument(
        "--output-root",
        default=str(root / "artifacts"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--val-ratio", type=float, default=0.0)
    parser.add_argument("--test-ratio", type=float, default=0.20)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    return parser.parse_args()


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def run_experiment(args: argparse.Namespace) -> dict:
    model_spec = resolve_model_spec(args.model)
    max_length = args.max_length or model_spec.default_max_length

    records = load_records(args.data_path)
    datasets, split_manifest = build_split_datasets(
        records,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    train_records = list(datasets["train"])
    val_records = list(datasets["val"])
    test_records = list(datasets["test"])

    run_dir = (
        Path(args.output_root)
        / model_spec.alias
        / f"seed-{args.seed}"
    )
    reports_dir = run_dir / "reports"
    adapter_dir = run_dir / "adapter"

    _write_json(
        run_dir / "run_config.json",
        {
            "model_alias": model_spec.alias,
            "model_id": model_spec.model_id,
            "trust_remote_code": model_spec.trust_remote_code,
            "requires_auth": model_spec.requires_auth,
            "notes": model_spec.notes,
            "data_path": str(Path(args.data_path).resolve()),
            "seed": args.seed,
            "dataset_stats": dataset_stats(records),
            "split_manifest": split_manifest,
            "training": {
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "eval_batch_size": args.eval_batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "max_length": max_length,
                "max_new_tokens": args.max_new_tokens,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "early_stopping_patience": args.early_stopping_patience,
            },
        },
    )

    baseline_test = evaluate_model(
        model_id=model_spec.model_id,
        trust_remote_code=model_spec.trust_remote_code,
        records=test_records,
        split_name="test_before",
        output_dir=reports_dir,
        max_length=max_length,
        max_new_tokens=args.max_new_tokens,
    )

    training_summary = train_lora_adapter(
        model_id=model_spec.model_id,
        trust_remote_code=model_spec.trust_remote_code,
        train_records=train_records,
        val_records=val_records,
        output_dir=adapter_dir,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=max_length,
        seed=args.seed,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        early_stopping_patience=args.early_stopping_patience,
    )

    finetuned_test = evaluate_model(
        model_id=model_spec.model_id,
        trust_remote_code=model_spec.trust_remote_code,
        records=test_records,
        split_name="test_after",
        output_dir=reports_dir,
        adapter_path=adapter_dir,
        max_length=max_length,
        max_new_tokens=args.max_new_tokens,
    )

    baseline_test_summary = baseline_test["summary"]
    finetuned_test_summary = finetuned_test["summary"]
    summary = {
        "model_alias": model_spec.alias,
        "model_id": model_spec.model_id,
        "seed": args.seed,
        "train_count": len(train_records),
        "test_count": len(test_records),
        "baseline_test_accuracy": baseline_test_summary["accuracy"],
        "finetuned_test_accuracy": finetuned_test_summary["accuracy"],
        "delta_test_accuracy": (
            finetuned_test_summary["accuracy"] - baseline_test_summary["accuracy"]
        ),
        "baseline_test_correct": baseline_test_summary["num_correct"],
        "finetuned_test_correct": finetuned_test_summary["num_correct"],
        "baseline_test_answer_rate": baseline_test_summary["answer_rate"],
        "finetuned_test_answer_rate": finetuned_test_summary["answer_rate"],
        "training_summary": training_summary,
    }
    if val_records:
        baseline_val = evaluate_model(
            model_id=model_spec.model_id,
            trust_remote_code=model_spec.trust_remote_code,
            records=val_records,
            split_name="val_before",
            output_dir=reports_dir,
            max_length=max_length,
            max_new_tokens=args.max_new_tokens,
        )
        finetuned_val = evaluate_model(
            model_id=model_spec.model_id,
            trust_remote_code=model_spec.trust_remote_code,
            records=val_records,
            split_name="val_after",
            output_dir=reports_dir,
            adapter_path=adapter_dir,
            max_length=max_length,
            max_new_tokens=args.max_new_tokens,
        )
        baseline_val_summary = baseline_val["summary"]
        finetuned_val_summary = finetuned_val["summary"]
        summary["val_count"] = len(val_records)
        summary["baseline_val_accuracy"] = baseline_val_summary["accuracy"]
        summary["finetuned_val_accuracy"] = finetuned_val_summary["accuracy"]
        summary["delta_val_accuracy"] = (
            finetuned_val_summary["accuracy"] - baseline_val_summary["accuracy"]
        )

    _write_json(run_dir / "summary.json", summary)
    return summary


def main() -> None:
    summary = run_experiment(parse_args())
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
