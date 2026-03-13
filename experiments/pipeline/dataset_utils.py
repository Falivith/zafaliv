from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from datasets import Dataset

VALID_CHOICES = {"A", "B", "C", "D", "E"}


def normalize_choice(value: str) -> str:
    normalized = str(value).strip().upper()
    if normalized not in VALID_CHOICES:
        raise ValueError(f"Alternativa invalida: {value!r}")
    return normalized


def load_records(data_path: str | Path) -> list[dict]:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset nao encontrado: {path}")

    if path.suffix == ".jsonl":
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    elif path.suffix == ".json":
        records = json.loads(path.read_text(encoding="utf-8"))
    else:
        raise ValueError("Use um arquivo .json ou .jsonl")

    normalized_records = []
    for index, record in enumerate(records):
        missing = {"instruction", "input", "output"} - record.keys()
        if missing:
            raise ValueError(
                f"Exemplo {index} sem os campos obrigatorios: {sorted(missing)}"
            )

        normalized_records.append(
            {
                "instruction": str(record["instruction"]).strip(),
                "input": str(record["input"]).strip(),
                "output": normalize_choice(record["output"]),
            }
        )

    return normalized_records


def dataset_stats(records: list[dict]) -> dict:
    labels = Counter(record["output"] for record in records)
    return {
        "num_examples": len(records),
        "label_distribution": dict(sorted(labels.items())),
    }


def _allocate_group_counts(
    group_size: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[int, int, int]:
    ratios = [train_ratio, val_ratio, test_ratio]
    raw_counts = [group_size * ratio for ratio in ratios]
    counts = [math.floor(value) for value in raw_counts]
    remainder = group_size - sum(counts)

    by_fraction = sorted(
        range(len(raw_counts)),
        key=lambda idx: raw_counts[idx] - counts[idx],
        reverse=True,
    )
    for idx in by_fraction[:remainder]:
        counts[idx] += 1

    active_splits = [idx for idx, ratio in enumerate(ratios) if ratio > 0]
    for idx in active_splits:
        if counts[idx] == 0 and group_size >= len(active_splits):
            donor = max(active_splits, key=lambda split_idx: counts[split_idx])
            if counts[donor] > 1:
                counts[donor] -= 1
                counts[idx] += 1

    if sum(counts) != group_size:
        raise RuntimeError("Falha ao dividir a amostra por classe.")

    return counts[0], counts[1], counts[2]


def build_split_datasets(
    records: list[dict],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[dict[str, Dataset], dict]:
    total_ratio = train_ratio + val_ratio + test_ratio
    if not math.isclose(total_ratio, 1.0, rel_tol=0, abs_tol=1e-9):
        raise ValueError("train_ratio + val_ratio + test_ratio precisa ser 1.0")

    grouped_indices: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        grouped_indices[record["output"]].append(index)

    rng = random.Random(seed)
    split_indices = {"train": [], "val": [], "test": []}

    for label in sorted(grouped_indices):
        indices = grouped_indices[label][:]
        rng.shuffle(indices)
        train_count, val_count, test_count = _allocate_group_counts(
            len(indices),
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )
        split_indices["train"].extend(indices[:train_count])
        split_indices["val"].extend(indices[train_count : train_count + val_count])
        split_indices["test"].extend(indices[train_count + val_count :])

    split_records = {}
    for split_name, indices in split_indices.items():
        rng.shuffle(indices)
        split_records[split_name] = [records[index] for index in indices]

    manifest = {
        "seed": seed,
        "ratios": {
            "train": train_ratio,
            "val": val_ratio,
            "test": test_ratio,
        },
        "indices": split_indices,
        "split_stats": {
            split_name: dataset_stats(split_records[split_name])
            for split_name in ("train", "val", "test")
        },
    }

    datasets = {
        split_name: Dataset.from_list(items)
        for split_name, items in split_records.items()
    }
    return datasets, manifest
