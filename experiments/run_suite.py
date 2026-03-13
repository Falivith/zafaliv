from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_experiment import run_experiment


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Roda uma bateria de modelos com o pipeline de fine-tuning."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "qwen2.5-1.5b",
            "gemma-3-1b",
            "amadeus-verbo-1.5b",
        ],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument(
        "--output-root",
        default=str(root / "artifacts"),
    )
    parser.add_argument(
        "--data-path",
        default=str(root / "data" / "datasets" / "dataset.jsonl"),
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--val-ratio", type=float, default=0.0)
    parser.add_argument("--test-ratio", type=float, default=0.20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_runs = []

    for model in args.models:
        for seed in args.seeds:
            experiment_args = argparse.Namespace(
                model=model,
                data_path=args.data_path,
                output_root=args.output_root,
                seed=seed,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                batch_size=args.batch_size,
                eval_batch_size=args.eval_batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                max_length=args.max_length,
                max_new_tokens=args.max_new_tokens,
                lora_r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                early_stopping_patience=args.early_stopping_patience,
            )
            summary = run_experiment(experiment_args)
            all_runs.append(summary)
            print(
                json.dumps(
                    {
                        "model_alias": summary["model_alias"],
                        "seed": seed,
                        "test_before": summary["baseline_test_accuracy"],
                        "test_after": summary["finetuned_test_accuracy"],
                        "delta_test_accuracy": summary["delta_test_accuracy"],
                    },
                    ensure_ascii=False,
                )
            )

    aggregated_by_model = {}
    for run in all_runs:
        model_alias = run["model_alias"]
        aggregated_by_model.setdefault(model_alias, []).append(run)

    aggregate_summary = {}
    for model_alias, runs in aggregated_by_model.items():
        deltas = [run["delta_test_accuracy"] for run in runs]
        before = [run["baseline_test_accuracy"] for run in runs]
        after = [run["finetuned_test_accuracy"] for run in runs]
        aggregate_summary[model_alias] = {
            "num_runs": len(runs),
            "mean_test_accuracy_before": sum(before) / len(before),
            "mean_test_accuracy_after": sum(after) / len(after),
            "mean_delta_test_accuracy": sum(deltas) / len(deltas),
            "runs_with_positive_delta": sum(1 for delta in deltas if delta > 0),
        }

    summary_path = Path(args.output_root) / "suite_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "models": args.models,
                "seeds": args.seeds,
                "aggregate_summary": aggregate_summary,
                "runs": all_runs,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
